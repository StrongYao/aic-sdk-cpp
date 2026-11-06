#!/usr/bin/env python3
"""Pure Python/PyTorch reconstruction of the quail-vf AIC model.

This module parses the .aicmodel tensor payload directly. It never imports or
loads libaic.dylib. The implementation is specialized for the recovered
quail-vf-2.1-s-16khz architecture bundled with this repository.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

SAMPLE_RATE = 16_000
FFT_SIZE = 480
HOP_SIZE = 240
FREQ_BINS = 241
ERB_BANDS = 32
DF_BINS = 96
DF_ORDER = 3
CONV_CHANNELS = 48
HIDDEN_SIZE = 192
LINEAR_GROUPS = 8
VAD_SPEECH_SAMPLE = 30_000
OUTPUT_DELAY_SAMPLES = 480
DEMO_BLOCK_SIZE = 160


_DF_ON_NOISY = os.environ.get("AIC_DF_ON_NOISY", "0") == "1"

# Import-time snapshot retained for callers inspecting this module. Processing
# reads the environment at call time, just as each native processor does.
_SKIP_POSTPROCESSING = "AIC_SKIP_MODEL_POSTPROCESSING" in os.environ


def _decode_uint(data: bytes, offset: int) -> tuple[int, int]:
    prefix = data[offset]
    if prefix <= 250:
        return prefix, offset + 1
    width = {251: 2, 252: 4, 253: 8, 254: 16}.get(prefix)
    if width is None:
        raise ValueError(f"invalid bincode integer prefix 0x{prefix:02x}")
    start = offset + 1
    end = start + width
    if end > len(data):
        raise ValueError("truncated bincode integer")
    return int.from_bytes(data[start:end], "little"), end


def _tensor_descriptors(
    data: bytes,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for descriptor_offset in range(min(len(data), 64 * 1024)):
        try:
            values: list[int] = []
            cursor = descriptor_offset
            for _ in range(11):
                value, cursor = _decode_uint(data, cursor)
                values.append(value)
        except (IndexError, ValueError, OverflowError):
            continue
        start, end, tag, *dims = values
        shape = tuple(dims[:4])
        strides = tuple(dims[4:])
        if (
            tag in (1, 2)
            and all(0 < dim <= 1_000_000 for dim in shape)
            and end > start
            and end - start == math.prod(shape) * 4
            and strides
            == (
                shape[1] * shape[2] * shape[3],
                shape[2] * shape[3],
                shape[3],
                1,
            )
        ):
            candidates.append((start, end, shape))

    chain: list[tuple[int, int, tuple[int, int, int, int]]] = []
    previous_end = 0
    for candidate in candidates:
        start, end, _ = candidate
        if not chain:
            if start != 0:
                continue
        elif start < previous_end or start - previous_end >= 64 or start % 64:
            continue
        chain.append(candidate)
        previous_end = end
    if len(chain) != 220:
        raise ValueError(f"expected 220 model tensors, found {len(chain)}")
    return chain


class AicModelTensors:
    """Read-only Float32 tensors recovered from an .aicmodel file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        data = path.read_bytes()
        descriptors = _tensor_descriptors(data)
        data_base = len(data) - descriptors[-1][1]
        if data_base % 64:
            raise ValueError("model tensor payload is not 64-byte aligned")
        self.arrays = [
            np.frombuffer(
                data,
                dtype="<f4",
                count=math.prod(shape),
                offset=data_base + start,
            )
            .reshape(shape)
            .copy()
            for start, _, shape in descriptors
        ]
        expected = {
            0: (1, 1, FREQ_BINS, ERB_BANDS),
            68: (1, 1, HIDDEN_SIZE, 3 * HIDDEN_SIZE),
            150: (1, 1, 1, 3 * CONV_CHANNELS),
            219: (1, 1, ERB_BANDS, FREQ_BINS),
        }
        for index, shape in expected.items():
            if self.arrays[index].shape != shape:
                raise ValueError(
                    f"tensor {index} has shape {self.arrays[index].shape}, "
                    f"expected {shape}"
                )

    def tensor(self, index: int) -> Tensor:
        return torch.from_numpy(self.arrays[index].copy()).squeeze(0).squeeze(0)


class InferenceBatchNorm2d(nn.Module):
    def __init__(self, tensors: AicModelTensors, start: int) -> None:
        super().__init__()
        self.register_buffer("weight", tensors.tensor(start).flatten())
        self.register_buffer("bias", tensors.tensor(start + 1).flatten())
        self.register_buffer("mean", tensors.tensor(start + 2).flatten())
        self.register_buffer("variance", tensors.tensor(start + 3).flatten())

    def forward(self, x: Tensor) -> Tensor:
        shape = (1, -1, 1, 1)
        return (x - self.mean.view(shape)) * torch.rsqrt(
            self.variance.view(shape) + 1e-5
        ) * self.weight.view(shape) + self.bias.view(shape)


class ConvBlock(nn.Module):
    """AirTen Conv2dNormAct using [batch, channel, frequency, time]."""

    def __init__(
        self,
        tensors: AicModelTensors,
        weight_indices: Sequence[int],
        bn_start: int | None,
        *,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        groups: int,
        frequency_stride: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        main = tensors.tensor(weight_indices[0])
        stored_shape = (
            out_channels,
            in_channels // groups,
            kernel[1],
            kernel[0],
        )
        self.register_buffer(
            "main_weight", main.reshape(stored_shape).transpose(-1, -2)
        )
        self.groups = groups
        self.frequency_stride = frequency_stride
        self.frequency_padding = kernel[0] // 2
        self.pointwise_weight: Tensor | None
        if len(weight_indices) == 2:
            pointwise = tensors.tensor(weight_indices[1])
            self.register_buffer(
                "pointwise_weight",
                pointwise.reshape(out_channels, out_channels, 1, 1),
            )
        else:
            self.pointwise_weight = None
        self.norm = (
            InferenceBatchNorm2d(tensors, bn_start) if bn_start is not None else None
        )
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(
            x,
            self.main_weight,
            stride=(self.frequency_stride, 1),
            padding=(self.frequency_padding, 0),
            groups=self.groups,
        )
        if self.pointwise_weight is not None:
            x = F.conv2d(x, self.pointwise_weight)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation != "identity":
            raise ValueError(f"unsupported activation {self.activation}")
        return x


class FrequencyPixelShuffle(nn.Module):
    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frequency, time = x.shape
        if channels % self.factor:
            raise ValueError("pixel shuffle channels are not divisible by factor")
        out_channels = channels // self.factor
        return (
            x.reshape(batch, self.factor, out_channels, frequency, time)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, out_channels, frequency * self.factor, time)
        )


class PixelShuffleBlock(nn.Module):
    def __init__(
        self,
        tensors: AicModelTensors,
        depthwise_index: int,
        pointwise_index: int,
        bn_start: int,
    ) -> None:
        super().__init__()
        depthwise = tensors.tensor(depthwise_index).reshape(CONV_CHANNELS, 1, 3, 1)
        pointwise = tensors.tensor(pointwise_index).reshape(
            2 * CONV_CHANNELS, CONV_CHANNELS, 1, 1
        )
        self.register_buffer("depthwise", depthwise)
        self.register_buffer("pointwise", pointwise)
        self.norm = InferenceBatchNorm2d(tensors, bn_start)
        self.shuffle = FrequencyPixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(
            x,
            self.depthwise,
            padding=(1, 0),
            groups=CONV_CHANNELS,
        )
        x = F.conv2d(x, self.pointwise)
        x = F.relu(self.norm(x))
        return self.shuffle(x)


class GroupedLinear(nn.Module):
    def __init__(
        self,
        tensors: AicModelTensors,
        start: int,
        groups: int = LINEAR_GROUPS,
        *,
        activation: str = "identity",
    ) -> None:
        super().__init__()
        self.groups = groups
        self.activation = activation
        for group in range(groups):
            self.register_buffer(f"weight_{group}", tensors.tensor(start + group * 2))
            self.register_buffer(
                f"bias_{group}", tensors.tensor(start + group * 2 + 1).flatten()
            )

    def forward(self, x: Tensor) -> Tensor:
        chunks = torch.chunk(x, self.groups, dim=-1)
        outputs = []
        for group, chunk in enumerate(chunks):
            weight = getattr(self, f"weight_{group}")
            bias = getattr(self, f"bias_{group}")
            outputs.append(chunk @ weight + bias)
        x = torch.cat(outputs, dim=-1)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation != "identity":
            raise ValueError(f"unsupported activation {self.activation}")
        return x


class StatefulGru(nn.Module):
    """Single-layer GRU with the tensor orientation used by AirTen."""

    def __init__(self, tensors: AicModelTensors, start: int) -> None:
        super().__init__()
        self.register_buffer("weight_ih", tensors.tensor(start))
        self.register_buffer("weight_hh", tensors.tensor(start + 1))
        self.register_buffer("bias_ih", tensors.tensor(start + 2).flatten())
        self.register_buffer("bias_hh", tensors.tensor(start + 3).flatten())
        self.hidden: Tensor | None = None

    def reset(self) -> None:
        self.hidden = None

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, _ = x.shape
        hidden = self.hidden
        if hidden is None or hidden.shape[0] != batch:
            hidden = x.new_zeros((batch, HIDDEN_SIZE))
        outputs = []
        for step in range(steps):
            input_gates = x[:, step] @ self.weight_ih + self.bias_ih
            hidden_gates = hidden @ self.weight_hh + self.bias_hh
            ir, iz, inn = input_gates.chunk(3, dim=-1)
            hr, hz, hn = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            hidden = candidate + update * (hidden - candidate)
            outputs.append(hidden)
        self.hidden = hidden.detach()
        return torch.stack(outputs, dim=1)


class SqueezedGru(nn.Module):
    def __init__(
        self,
        tensors: AicModelTensors,
        linear_in: int,
        gru_starts: Iterable[int],
        linear_out: int | None = None,
        *,
        skip: bool = False,
    ) -> None:
        super().__init__()
        self.linear_in = GroupedLinear(tensors, linear_in, activation="relu")
        self.grus = nn.ModuleList([StatefulGru(tensors, start) for start in gru_starts])
        self.linear_out = (
            GroupedLinear(tensors, linear_out, activation="relu")
            if linear_out is not None
            else None
        )
        self.skip = skip

    def reset(self) -> None:
        for gru in self.grus:
            gru.reset()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_in(x)
        residual = x
        for gru in self.grus:
            x = gru(x)
        if self.skip:
            x = x + residual
        if self.linear_out is not None:
            x = self.linear_out(x)
        return x


class Encoder(nn.Module):
    def __init__(self, tensors: AicModelTensors) -> None:
        super().__init__()
        self.erb_conv0 = ConvBlock(
            tensors,
            [1],
            2,
            in_channels=1,
            out_channels=48,
            kernel=(3, 3),
            groups=1,
        )
        self.erb_conv1 = ConvBlock(
            tensors,
            [6, 7],
            8,
            in_channels=48,
            out_channels=48,
            kernel=(3, 1),
            groups=48,
            frequency_stride=2,
        )
        self.erb_conv2 = ConvBlock(
            tensors,
            [12, 13],
            14,
            in_channels=48,
            out_channels=48,
            kernel=(3, 1),
            groups=48,
            frequency_stride=2,
        )
        self.erb_conv3 = ConvBlock(
            tensors,
            [18, 19],
            20,
            in_channels=48,
            out_channels=48,
            kernel=(3, 1),
            groups=48,
        )
        self.df_conv0 = ConvBlock(
            tensors,
            [24, 25],
            26,
            in_channels=2,
            out_channels=48,
            kernel=(3, 3),
            groups=2,
        )
        self.df_conv1 = ConvBlock(
            tensors,
            [30, 31],
            32,
            in_channels=48,
            out_channels=48,
            kernel=(3, 1),
            groups=48,
            frequency_stride=2,
        )
        self.df_embedding = GroupedLinear(tensors, 36, activation="relu")
        self.embedding = SqueezedGru(tensors, 52, [68])

    def reset(self) -> None:
        self.embedding.reset()

    def forward(
        self, erb: Tensor, spectrum: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        e0 = self.erb_conv0(erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        c0 = self.df_conv0(spectrum)
        c1 = self.df_conv1(c0)
        batch = erb.shape[0]
        erb_embedding = e3.permute(0, 3, 2, 1).reshape(batch, 1, -1)
        df_embedding = c1.permute(0, 3, 2, 1).reshape(batch, 1, -1)
        df_embedding = self.df_embedding(df_embedding)
        embedding = self.embedding(torch.cat((erb_embedding, df_embedding), -1))
        return e0, e1, e2, e3, c0, embedding


class ErbDecoder(nn.Module):
    def __init__(self, tensors: AicModelTensors) -> None:
        super().__init__()
        self.embedding = SqueezedGru(tensors, 72, [88, 92], 96)
        self.conv3p = ConvBlock(
            tensors,
            [112],
            113,
            in_channels=48,
            out_channels=48,
            kernel=(1, 1),
            groups=48,
        )
        self.convt3 = ConvBlock(
            tensors,
            [117, 118],
            119,
            in_channels=48,
            out_channels=48,
            kernel=(3, 1),
            groups=48,
        )
        self.conv2p = ConvBlock(
            tensors,
            [123],
            124,
            in_channels=48,
            out_channels=48,
            kernel=(1, 1),
            groups=48,
        )
        self.convt2 = PixelShuffleBlock(tensors, 128, 129, 130)
        self.conv1p = ConvBlock(
            tensors,
            [134],
            135,
            in_channels=48,
            out_channels=48,
            kernel=(1, 1),
            groups=48,
        )
        self.convt1 = PixelShuffleBlock(tensors, 139, 140, 141)
        self.conv0p = ConvBlock(
            tensors,
            [145],
            146,
            in_channels=48,
            out_channels=48,
            kernel=(1, 1),
            groups=48,
        )
        self.output = ConvBlock(
            tensors,
            [150],
            151,
            in_channels=48,
            out_channels=1,
            kernel=(3, 1),
            groups=1,
            activation="sigmoid",
        )

    def reset(self) -> None:
        self.embedding.reset()

    def forward(
        self,
        embedding: Tensor,
        e0: Tensor,
        e1: Tensor,
        e2: Tensor,
        e3: Tensor,
    ) -> Tensor:
        batch = embedding.shape[0]
        x = self.embedding(embedding)
        x = x.reshape(batch, 1, 8, 48).permute(0, 3, 2, 1)
        x = self.convt3(self.conv3p(e3) + x)
        x = self.convt2(self.conv2p(e2) + x)
        x = self.convt1(self.conv1p(e1) + x)
        return self.output(self.conv0p(e0) + x)


class DfDecoder(nn.Module):
    def __init__(self, tensors: AicModelTensors) -> None:
        super().__init__()
        self.embedding = SqueezedGru(tensors, 155, [171, 175])
        self.pathway = ConvBlock(
            tensors,
            [179],
            180,
            in_channels=48,
            out_channels=6,
            kernel=(1, 1),
            groups=6,
        )
        self.output = GroupedLinear(tensors, 184, activation="tanh")
        self.register_buffer("alpha_weight", tensors.tensor(205))
        self.register_buffer("alpha_bias", tensors.tensor(206).flatten())

    def reset(self) -> None:
        self.embedding.reset()

    def forward(self, embedding: Tensor, c0: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding(embedding)
        alpha = torch.sigmoid(x @ self.alpha_weight + self.alpha_bias)
        coefficients = self.output(x).reshape(x.shape[0], 1, DF_BINS, 6)
        pathway = self.pathway(c0).permute(0, 3, 2, 1)
        return coefficients + pathway, alpha


class VadDecoder(nn.Module):
    """Temporal convolutional VAD head recovered from the native forward path."""

    def __init__(self, tensors: AicModelTensors) -> None:
        super().__init__()
        self.register_buffer(
            "conv0_weight",
            tensors.tensor(200).reshape(16, 1, 3, 1).transpose(-1, -2),
        )
        self.norm0 = InferenceBatchNorm2d(tensors, 201)
        self.register_buffer(
            "conv1_weight",
            tensors.tensor(209).reshape(16, 16, 3, 1).transpose(-1, -2),
        )
        self.norm1 = InferenceBatchNorm2d(tensors, 210)
        self.register_buffer(
            "conv2_weight",
            tensors.tensor(214).reshape(16, 16, 3, 1).transpose(-1, -2),
        )
        self.norm2 = InferenceBatchNorm2d(tensors, 215)
        self.register_buffer("hidden_weight", tensors.tensor(205))
        self.register_buffer("hidden_bias", tensors.tensor(206).flatten())
        self.register_buffer("output_weight", tensors.tensor(207))
        self.register_buffer("output_bias", tensors.tensor(208).flatten())

    def forward(self, embedding: Tensor) -> Tensor:
        x = embedding.reshape(embedding.shape[0], 1, 1, HIDDEN_SIZE)
        x = F.relu(self.norm0(F.conv2d(x, self.conv0_weight, padding=(0, 1))))
        x = F.relu(self.norm1(F.conv2d(x, self.conv1_weight, padding=(0, 1))))
        x = F.relu(self.norm2(F.conv2d(x, self.conv2_weight, padding=(0, 1))))
        x = x @ self.hidden_weight + self.hidden_bias
        x = x.permute(0, 2, 3, 1)
        x = x @ self.output_weight + self.output_bias
        return torch.sigmoid(x.reshape(embedding.shape[0]))


class QuailVoiceFocus(nn.Module):
    """Streaming neural network reconstructed from the model tensor sequence."""

    def __init__(
        self, model_path: Path, *, spectrum_scale: float = 1.0 / FFT_SIZE
    ) -> None:
        super().__init__()
        tensors = AicModelTensors(model_path)
        self.encoder = Encoder(tensors)
        self.erb_decoder = ErbDecoder(tensors)
        self.df_decoder = DfDecoder(tensors)
        self.vad_decoder = VadDecoder(tensors)
        self.register_buffer("erb_analysis", tensors.tensor(0))
        self.register_buffer("erb_synthesis", tensors.tensor(219))
        self.spectrum_scale = spectrum_scale

    def reset(self) -> None:
        self.encoder.reset()
        self.erb_decoder.reset()
        self.df_decoder.reset()

    def _features(self, spectrum: Tensor) -> tuple[Tensor, Tensor]:
        scaled = spectrum * self.spectrum_scale
        power = scaled.square().sum(dim=1)
        erb = torch.einsum("bft,fe->bet", power, self.erb_analysis)
        erb = torch.log10(erb + 1e-10) / 4.0
        complex_features = scaled[:, :, :DF_BINS]
        return erb.unsqueeze(1), complex_features

    def forward(self, spectrum: Tensor) -> tuple[Tensor, Tensor]:
        if spectrum.ndim != 4 or spectrum.shape[1:] != (2, FREQ_BINS, 3):
            raise ValueError(
                "network input must have shape [batch, 2, 241, 3], got "
                f"{tuple(spectrum.shape)}"
            )
        erb, complex_features = self._features(spectrum)
        e0, e1, e2, e3, c0, embedding = self.encoder(erb, complex_features)
        mask_erb = self.erb_decoder(embedding, e0, e1, e2, e3)
        coefficients, _alpha = self.df_decoder(embedding, c0)

        mask = torch.einsum("boet,ef->boft", mask_erb, self.erb_synthesis)
        masked_spectrum = spectrum * mask
        masked_center = masked_spectrum[:, :, :, 1:2]

        df_source = spectrum if _DF_ON_NOISY else masked_spectrum
        history = df_source[:, :, :DF_BINS, :]
        real = history[:, 0].permute(0, 2, 1)
        imag = history[:, 1].permute(0, 2, 1)
        coefficients = coefficients[:, 0].reshape(
            spectrum.shape[0], DF_BINS, DF_ORDER, 2
        )
        coef_real = coefficients[..., 0].permute(0, 2, 1)
        coef_imag = coefficients[..., 1].permute(0, 2, 1)
        filtered_real = (real * coef_real - imag * coef_imag).sum(dim=1)
        filtered_imag = (real * coef_imag + imag * coef_real).sum(dim=1)
        low = torch.stack((filtered_real, filtered_imag), dim=1).unsqueeze(-1)
        enhanced = torch.cat((low, masked_center[:, :, DF_BINS:]), dim=2)
        vad = self.vad_decoder(embedding)
        return enhanced, vad


def periodic_hann(size: int) -> np.ndarray:
    return np.hanning(size + 1)[:-1].astype(np.float32)


def canonical_dual_window(analysis_window: np.ndarray, hop_size: int) -> np.ndarray:
    """Return the least-squares WOLA synthesis window for 50% overlap."""
    if analysis_window.ndim != 1 or len(analysis_window) != 2 * hop_size:
        raise ValueError("analysis window must contain exactly two hops")
    shifted = np.roll(analysis_window, hop_size)
    denominator = np.square(analysis_window) + np.square(shifted)
    if np.any(denominator == 0):
        raise ValueError("analysis window has an uncovered sample")
    return (analysis_window / denominator).astype(np.float32)


class StreamingEnhancer:
    # STFT front-end recovered from libaic.dylib (verified via a Frida capture of
    # the SDK's own per-frame analysis/enhanced spectra and output time samples on
    # demo_input.wav). The analysis window is a PERIODIC Hann at unit scale: pure
    # ``rfft(frame * periodic_hann)`` reproduces the SDK analysis spectrum to a
    # relative residual of 1.5e-7. The SDK synthesis window is the canonical dual
    # of that periodic Hann for a 50%-overlap WOLA. Recovering the 480 coefficients
    # from 2694 SDK frames reproduces this formula at relative L2 1.6e-7; using a
    # rectangular synthesis window instead caused the previous 70-LSB no-post gap.
    def __init__(self, network: QuailVoiceFocus) -> None:
        self.network = network
        self.analysis_window = periodic_hann(FFT_SIZE)
        self.synthesis_window = canonical_dual_window(self.analysis_window, HOP_SIZE)
        self.input_history = np.zeros(HOP_SIZE, dtype=np.float32)
        self.spectrum_history = np.zeros((2, FREQ_BINS, 3), dtype=np.float32)
        self.overlap = np.zeros(HOP_SIZE, dtype=np.float32)

    def reset(self) -> None:
        self.network.reset()
        self.input_history.fill(0)
        self.spectrum_history.fill(0)
        self.overlap.fill(0)

    @torch.inference_mode()
    def process_hop(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        if samples.shape != (HOP_SIZE,):
            raise ValueError(f"expected {HOP_SIZE} samples, got {samples.shape}")
        frame = np.concatenate((self.input_history, samples))
        self.input_history[:] = samples
        spectrum = np.fft.rfft(frame * self.analysis_window).astype(np.complex64)
        self.spectrum_history[..., :-1] = self.spectrum_history[..., 1:]
        self.spectrum_history[0, :, -1] = spectrum.real
        self.spectrum_history[1, :, -1] = spectrum.imag
        network_input = torch.from_numpy(self.spectrum_history[None].copy())
        enhanced, vad = self.network(network_input)
        ri = enhanced[0, :, :, 0].cpu().numpy()
        enhanced_complex = ri[0] + 1j * ri[1]
        time_frame = np.fft.irfft(enhanced_complex, n=FFT_SIZE).astype(np.float32)
        time_frame *= self.synthesis_window
        output = self.overlap + time_frame[:HOP_SIZE]
        self.overlap[:] = time_frame[HOP_SIZE:]
        return output, float(vad.flatten()[0])


def read_pcm16_wav(
    path: Path, *, respect_data_chunk: bool = False
) -> tuple[np.ndarray, int]:
    wav = path.read_bytes()
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError(f"expected RIFF/WAVE input: {path}")
    fmt: tuple[int, int, int, int, int] | None = None
    data_offset: int | None = None
    data_size = 0
    offset = 12
    while offset + 8 <= len(wav):
        chunk_id = wav[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", wav, offset + 4)[0]
        payload = offset + 8
        if payload + chunk_size > len(wav) and chunk_id != b"data":
            raise ValueError(f"truncated {chunk_id!r} WAV chunk")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("WAV fmt chunk must contain at least 16 bytes")
            values = struct.unpack_from("<HHIIHH", wav, payload)
            fmt = (values[0], values[1], values[2], values[4], values[5])
        elif chunk_id == b"data":
            data_offset = payload
            data_size = min(chunk_size, len(wav) - payload)
            break
        offset = payload + chunk_size + (chunk_size & 1)
    if fmt is None or data_offset is None:
        raise ValueError("missing WAV fmt or data chunk")
    audio_format, channels, sample_rate, block_align, bits = fmt
    if (
        audio_format != 1
        or bits != 16
        or channels < 1
        or sample_rate < 1
        or block_align != channels * 2
    ):
        raise ValueError("only integer PCM16 WAV input is supported")
    data_end = data_offset + data_size if respect_data_chunk else len(wav)
    raw = memoryview(wav)[data_offset:data_end]
    usable = len(raw) // block_align * block_align
    audio = np.frombuffer(raw[:usable], dtype="<i2").reshape(-1, channels)
    return audio.astype(np.float32) / 32768.0, sample_rate


def _pcm16(audio: np.ndarray) -> np.ndarray:
    return np.trunc(np.clip(audio, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2")


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim == 1:
        audio = audio[:, None]
    pcm = _pcm16(audio)
    payload = pcm.tobytes()
    channels = pcm.shape[1]
    byte_rate = sample_rate * channels * 2
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(payload))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, channels * 2, 16
        )
        + b"data"
        + struct.pack("<I", len(payload))
    )
    path.write_bytes(header + payload)


@dataclass(frozen=True)
class SmoothGateConfig:
    """Native Stage-A SmoothGate parameters recovered from live state."""

    # Wet-signal absolute amplitude required to open a closed gate.
    # -35 dBFS is the native opening level; 20 converts dB amplitude to linear.
    threshold_open: float = float(np.float32(10.0 ** (-35.0 / 20.0)))
    # Decayed peak level below which an open gate enters hold/release.
    # -40 dBFS is the native hold level; 20 is the dB-to-amplitude factor.
    hold_threshold: float = float(np.float32(10.0 ** (-40.0 / 20.0)))
    # Linear envelope increments/decrements applied per output sample.
    # 6 ms is attack duration; 1000 converts milliseconds to seconds.
    attack_increment: float = float(np.float32(1.0 / (SAMPLE_RATE * (6.0 / 1000.0))))
    # 400 ms is release duration; 1000 converts milliseconds to seconds.
    release_decrement: float = float(np.float32(1.0 / (SAMPLE_RATE * (400.0 / 1000.0))))
    # 1000 is milliseconds per second; SAMPLE_RATE is samples per second.
    hold_step: float = float(np.float32(1000.0 / SAMPLE_RATE))
    # Native hold limit is an explicit 0.1 ms state-machine parameter.
    hold_limit: float = float(np.float32(0.1))
    # Per-sample decay applied after max(previous_peak, abs(sample)).
    # 5 dB is the open/hold dynamic range; 0.013333334 s is native decay time.
    peak_decay: float = float(
        np.float32(
            (np.float32(-35.0) - np.float32(-40.0))
            / (np.float32(SAMPLE_RATE) * np.float32(0.013333334))
        )
    )


class SmoothGate:
    """Sample-ordered reconstruction of the SDK Stage-A wet-signal gate."""

    def __init__(self, config: SmoothGateConfig | None = None) -> None:
        self.config = config or SmoothGateConfig()
        self.reset()

    def reset(self) -> None:
        self.envelope = np.float32(0.0)
        self.peak = np.float32(0.0)
        self.hold_accumulator = np.float32(0.0)
        self.gate_open = False

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim != 1:
            raise ValueError("SmoothGate input must be one-dimensional")
        output = np.empty(samples.shape, dtype=np.float32)
        threshold_open = np.float32(self.config.threshold_open)
        hold_threshold = np.float32(self.config.hold_threshold)
        attack_increment = np.float32(self.config.attack_increment)
        release_decrement = np.float32(self.config.release_decrement)
        hold_step = np.float32(self.config.hold_step)
        hold_limit = np.float32(self.config.hold_limit)
        peak_decay = np.float32(self.config.peak_decay)
        envelope = self.envelope
        peak = self.peak
        hold = self.hold_accumulator
        gate_open = self.gate_open
        zero = np.float32(0.0)
        one = np.float32(1.0)
        half = np.float32(0.5)
        pi = np.float32(np.pi)

        for index, value in enumerate(samples):
            sample = np.float32(value)
            amplitude = np.float32(abs(sample))
            if amplitude > threshold_open and not gate_open:
                gate_open = True

            peak = np.float32(max(peak, amplitude) - peak_decay)
            if gate_open and peak < hold_threshold:
                gate_open = False
                hold = hold_step
                if hold > hold_limit:
                    envelope = np.float32(
                        max(np.float32(envelope - release_decrement), zero)
                    )
            elif gate_open:
                envelope = np.float32(min(np.float32(envelope + attack_increment), one))
            else:
                hold = np.float32(hold + hold_step)
                if hold > hold_limit:
                    envelope = np.float32(
                        max(np.float32(envelope - release_decrement), zero)
                    )

            phase = np.float32(envelope * pi)
            cosine = np.float32(math.cos(float(phase)))
            gain = np.float32(np.float32(one - cosine) * half)
            output[index] = np.float32(sample * gain)

        self.envelope = envelope
        self.peak = peak
        self.hold_accumulator = hold
        self.gate_open = gate_open
        return output


@dataclass(frozen=True)
class SmartMixerConfig:
    """Native Stage-B constants recovered from the live processor state."""

    # 65% is the native maximum VAD probability threshold.
    outer_threshold: float = float(np.float32(65.0 / 100.0))
    outer_counter_limit: int = 7
    ring_length: int = 26
    # Negative decision requires half of the 26-entry history.
    negative_limit: int = ring_length // 2
    # 0.8 is default enhancement level; 0.65 is the native max threshold.
    vad_threshold: float = float(
        np.float32(np.sqrt(np.float32(0.8)) * np.float32(0.65))
    )
    # 0.01 is the target residual gain; 80 is the 5-ms ramp in samples.
    smoothing_coefficient: float = float(np.float32(math.pow(0.01, 1.0 / 80.0)))
    # 0.005 s is smoothing time; SAMPLE_RATE converts seconds to samples.
    smoothing_samples: int = math.floor(float(np.float32(0.005 * SAMPLE_RATE)) + 0.5)
    # 0.8 is default enhancement level; 1.0 is maximum wet-path ratio.
    dry_wet_maximum: float = float(np.float32(0.8 * 1.0))

    @classmethod
    def from_environment(cls, enhancement_level: float) -> "SmartMixerConfig":
        """Derive live parameters, rather than freezing a level=0.8 snapshot.

        Native 0x10484 / process_sequential: threshold=max(low_clip,
        sqrt(level)*threshold_max), wet_fraction=level*max_dry_wet.
        The 26-entry ring and 80-sample ramp are model defaults at 16 kHz.
        """

        def value(name: str, default: float) -> np.float32:
            result = np.float32(os.environ.get(name, default))
            if not np.isfinite(result):
                raise ValueError(f"{name} must be finite")
            return result

        level = np.float32(enhancement_level)
        if not np.isfinite(level) or not 0 <= level <= 1:
            raise ValueError("enhancement_level must be finite and in [0, 1]")
        prefix = "AIC_SMOOTH_MIXER_"
        low = value(prefix + "VAD_THRESHOLD_LOW_CLIP", 0.05)
        maximum = value(prefix + "VAD_THRESHOLD_MAX", 0.65)
        wet_maximum = value(prefix + "MAX_DRY_WET", 1.0)
        seconds = value(prefix + "SMOOTHING_TIME", 0.005)
        if seconds < 0:
            raise ValueError("SmartMixer smoothing time must be nonnegative")
        # Rust Duration rounds the Float32 seconds to nanoseconds; the native
        # constructor then rounds sr*duration to nearest, with ties away.
        duration = np.float32(round(float(seconds) * 1e9) / 1e9)
        samples = math.floor(float(np.float32(duration * SAMPLE_RATE)) + 0.5)
        coefficient = (
            np.float32(
                math.pow(
                    float(np.float32(0.01)),
                    float(np.float32(1.0) / np.float32(samples)),
                )
            )
            if samples
            else np.float32(0.0)
        )
        return cls(
            vad_threshold=float(max(low, np.float32(np.sqrt(level) * maximum))),
            dry_wet_maximum=float(np.float32(level * wet_maximum)),
            smoothing_samples=samples,
            smoothing_coefficient=float(coefficient),
        )


def mixback(wet: np.ndarray, dry: np.ndarray, level: float) -> np.ndarray:
    """Native separate Float32 multiplies/add; preserve each dry channel."""
    wet_gain = np.float32(level)
    dry_gain = np.float32(np.float32(1.0) - wet_gain)
    if dry.ndim == 2 and wet.ndim == 1:
        wet = wet[:, None]
    return wet * wet_gain + dry * dry_gain


def downmix(audio: np.ndarray) -> np.ndarray:
    """Native channel-order sum followed by a Float32 reciprocal multiply."""
    mono = audio[:, 0].copy()
    for channel in range(1, audio.shape[1]):
        mono += audio[:, channel]
    mono *= np.float32(1.0) / np.float32(audio.shape[1])
    return mono


class SmartMixer:
    """Native VAD ring, hysteresis, mixback and per-hop gain smoothing."""

    def __init__(self, config: SmartMixerConfig | None = None) -> None:
        self.config = config or SmartMixerConfig()
        self.reset()

    def reset(self) -> None:
        self.ring = np.zeros(self.config.ring_length, dtype=np.uint8)
        self.ring_index = 0
        self.positive_run = 0
        self.smoothed_gain = np.float32(0.0)

    def process(
        self, vad_probability: float, dry: np.ndarray, wet: np.ndarray
    ) -> np.ndarray:
        if dry.shape != (HOP_SIZE,) or wet.shape != (HOP_SIZE,):
            raise ValueError(f"SmartMixer inputs must contain {HOP_SIZE} samples")

        probability = np.float32(vad_probability)
        threshold = np.float32(self.config.vad_threshold)
        self.ring_index = (self.ring_index + 1) % len(self.ring)
        self.ring[self.ring_index] = int(probability > threshold)
        ring_sum = int(np.sum(self.ring, dtype=np.int64))
        if probability > threshold:
            self.positive_run += 1
            decision = True
        else:
            self.positive_run = 0
            decision = ring_sum >= self.config.negative_limit

        dry_wet = np.float32(self.config.dry_wet_maximum)
        dry_gain = np.float32(np.float32(1.0) - dry_wet)
        mixed = wet * dry_wet + dry * dry_gain

        target = np.float32(1.0 if decision else 0.0)
        coefficient = np.float32(self.config.smoothing_coefficient)
        one_minus_coefficient = np.float32(np.float32(1.0) - coefficient)
        countdown = self.config.smoothing_samples
        output = np.empty(HOP_SIZE, dtype=np.float32)
        for index, sample in enumerate(mixed):
            next_countdown = countdown - 1
            if countdown > 0:
                countdown = next_countdown
                updated = target
                if next_countdown != 0:
                    updated = np.float32(
                        np.float32(self.smoothed_gain * coefficient)
                        + np.float32(target * one_minus_coefficient)
                    )
                self.smoothed_gain = updated
            output[index] = np.float32(sample * self.smoothed_gain)
        return output


class NativePostprocessor:
    """SDK outer routing latch plus the inner Stage-B SmartMixer."""

    def __init__(
        self,
        config: SmartMixerConfig | None = None,
        *,
        force_smart_mixer: bool = False,
    ) -> None:
        self.config = config or SmartMixerConfig()
        self.force_smart_mixer = force_smart_mixer
        self.mixer = SmartMixer(self.config)
        self.reset()

    def reset(self) -> None:
        self.outer_counter = 0
        # AIC_DISABLE_SMART_MIXER_ROUTING bypasses the *outer routing latch*:
        # it enters the inner SmartMixer immediately, not plain mixback.
        self.outer_latched = self.force_smart_mixer
        self.mixer.reset()

    def process(
        self, vad_probability: float, dry: np.ndarray, wet: np.ndarray
    ) -> np.ndarray:
        if wet.shape != (HOP_SIZE,) or dry.ndim not in (1, 2) or len(dry) != HOP_SIZE:
            raise ValueError(
                "postprocessor requires one wet hop and matching dry frames"
            )
        probability = np.float32(vad_probability)
        threshold = np.float32(self.config.outer_threshold)
        route_to_mixer = self.outer_latched
        if not self.outer_latched:
            if probability > threshold:
                self.outer_counter += 1
                if self.outer_counter > self.config.outer_counter_limit:
                    self.outer_latched = True
                    route_to_mixer = True
            else:
                self.outer_counter = 0

        if route_to_mixer:
            mono = downmix(dry) if dry.ndim == 2 else dry
            output = self.mixer.process(probability, mono, wet)
            if dry.ndim == 2:
                output = np.repeat(output[:, None], dry.shape[1], axis=1)
            return output
        return mixback(wet, dry, 0.5)


def postprocess_smart_mixer(
    wet: np.ndarray,
    dry: np.ndarray,
    vad_probabilities: Sequence[float],
    *,
    config: SmartMixerConfig | None = None,
    force_smart_mixer: bool = False,
) -> np.ndarray:
    if wet.ndim != 1 or dry.ndim not in (1, 2) or len(wet) != len(dry):
        raise ValueError(
            "wet must be mono with matching mono or multichannel dry frames"
        )
    if len(wet) != len(vad_probabilities) * HOP_SIZE:
        raise ValueError("one VAD probability is required for each audio hop")
    postprocessor = NativePostprocessor(config, force_smart_mixer=force_smart_mixer)
    output = np.empty_like(dry, dtype=np.float32)
    for frame_index, probability in enumerate(vad_probabilities):
        start = frame_index * HOP_SIZE
        stop = start + HOP_SIZE
        output[start:stop] = postprocessor.process(
            probability, dry[start:stop], wet[start:stop]
        )
    return output


class NativeVadContext:
    """Public VAD context fed by Stage-A gated wet hops.

    Durations are quantized to the model's 15-ms hop with positive half-up
    rounding. Speech starts after the configured number of consecutive
    positive hops. On a negative hop, the decision remains active while at
    least half of the 2*hold_hops history is positive.
    """

    def __init__(
        self,
        *,
        speech_hold_duration: float = 0.1,
        sensitivity: float = 8.0,
        minimum_speech_duration: float = 0.0,
    ) -> None:
        if not math.isfinite(speech_hold_duration) or not (
            0.0 <= speech_hold_duration <= 4.5
        ):
            raise ValueError("vad_speech_hold_duration must be finite and in [0, 4.5]")
        if not math.isfinite(sensitivity) or not 1.0 <= sensitivity <= 15.0:
            raise ValueError("vad_sensitivity must be finite and in [1, 15]")
        if not math.isfinite(minimum_speech_duration) or not (
            0.0 <= minimum_speech_duration <= 1.0
        ):
            raise ValueError("vad_minimum_speech_duration must be finite and in [0, 1]")
        self.gate = SmoothGate()
        hold_hops = self._duration_hops(speech_hold_duration)
        minimum_hops = self._duration_hops(minimum_speech_duration)
        self.ring = np.zeros(min(2 * hold_hops, 600), dtype=np.uint8)
        self.negative_limit = max(len(self.ring) // 2, 1)
        self.positive_limit = max(minimum_hops, 1)
        self.energy_threshold = np.float32(
            math.pow(10.0, -float(np.float32(sensitivity)))
        )
        self.ring_index = 0
        self.positive_run = 0

    @staticmethod
    def _duration_hops(seconds: float) -> int:
        # The SDK converts the public Float32 seconds value to integer
        # nanoseconds before quantizing by the 15-ms model hop.
        nanoseconds = int(math.floor(float(np.float32(seconds)) * 1_000_000_000 + 0.5))
        hops = nanoseconds * SAMPLE_RATE / (1_000_000_000 * HOP_SIZE)
        return int(math.floor(hops + 0.5))

    def process(self, wet: np.ndarray) -> bool:
        gated = self.gate.process(wet)
        # Preserve the native left-to-right Float32 sum of squared samples.
        energy = np.cumsum(gated * gated, dtype=np.float32)[-1]
        positive = energy > self.energy_threshold
        if len(self.ring):
            self.ring_index = (self.ring_index + 1) % len(self.ring)
            self.ring[self.ring_index] = int(positive)
            ring_sum = int(np.sum(self.ring, dtype=np.int64))
        else:
            ring_sum = 0
        if positive:
            self.positive_run += 1
            return self.positive_run >= self.positive_limit
        self.positive_run = 0
        return ring_sum >= self.negative_limit


def schedule_hops(hops: np.ndarray, output_samples: int) -> np.ndarray:
    """160-sample host latency; discard unconsumed output without flushing.

    A host stream may end in any of the three 160/240 scheduler phases. Only
    the part actually consumed by complete host calls belongs in the WAV.
    """
    output = np.zeros((output_samples, *hops.shape[1:]), dtype=hops.dtype)
    count = min(len(hops), max(0, output_samples - DEMO_BLOCK_SIZE))
    output[DEMO_BLOCK_SIZE : DEMO_BLOCK_SIZE + count] = hops[:count]
    return output


def process_audio(
    model_path: Path,
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    bypass: float = 0.0,
    enhancement_level: float = 0.8,
    vad_speech_hold_duration: float = 0.1,
    vad_sensitivity: float = 8.0,
    vad_minimum_speech_duration: float = 0.0,
    spectrum_scale: float = 1.0 / FFT_SIZE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Process Float32 [frames, channels] without decoding or quantization.

    Returns raw output, mono VAD waveform (0 or 30000/32768), and processing
    seconds excluding model construction. Each call starts a fresh stream;
    state persists across the entire input. Only complete 160-frame host
    blocks are consumed, with no end-of-stream flush. Input is not modified.
    """
    if audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError("audio must be Float32 [frames, channels], channels >= 1")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    if not math.isfinite(bypass) or not 0.0 <= bypass <= 1.0:
        raise ValueError("bypass must be finite and in [0, 1]")
    if not math.isfinite(enhancement_level) or not (0.0 <= enhancement_level <= 1.0):
        raise ValueError("enhancement_level must be finite and in [0, 1]")
    config = SmartMixerConfig.from_environment(enhancement_level)
    skip_postprocessing = "AIC_SKIP_MODEL_POSTPROCESSING" in os.environ
    force_smart_mixer = "AIC_DISABLE_SMART_MIXER_ROUTING" in os.environ
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"model requires {SAMPLE_RATE} Hz input, got {sample_rate} Hz")
    vad_context = NativeVadContext(
        speech_hold_duration=vad_speech_hold_duration,
        sensitivity=vad_sensitivity,
        minimum_speech_duration=vad_minimum_speech_duration,
    )
    mono = downmix(audio)
    network = QuailVoiceFocus(model_path, spectrum_scale=spectrum_scale).eval()
    enhancer = StreamingEnhancer(network)
    output_samples = len(mono) // DEMO_BLOCK_SIZE * DEMO_BLOCK_SIZE
    model_samples = output_samples // HOP_SIZE * HOP_SIZE
    source = mono[:model_samples].copy()
    enhanced = np.zeros(model_samples, dtype=np.float32)
    probabilities: list[float] = []
    vad_decisions: list[bool] = []
    start = time.perf_counter()
    for offset in range(0, model_samples, HOP_SIZE):
        hop, probability = enhancer.process_hop(source[offset : offset + HOP_SIZE])
        enhanced[offset : offset + HOP_SIZE] = hop
        probabilities.append(probability)
        vad_decisions.append(vad_context.process(hop))
    delayed_source = np.zeros((model_samples, audio.shape[1]), dtype=np.float32)
    if len(source) > OUTPUT_DELAY_SAMPLES:
        delayed_source[OUTPUT_DELAY_SAMPLES:] = audio[
            : model_samples - OUTPUT_DELAY_SAMPLES
        ]
    if skip_postprocessing:
        effective_level = 0.0 if bypass > 0.0 else enhancement_level
        mixed = mixback(enhanced, delayed_source, effective_level)
    else:
        mixed = postprocess_smart_mixer(
            enhanced,
            delayed_source,
            probabilities,
            config=config,
            force_smart_mixer=force_smart_mixer,
        )
    output = schedule_hops(mixed, output_samples)
    elapsed = time.perf_counter() - start

    vad = np.zeros(output_samples, dtype=np.float32)
    buffered_samples = 0
    probability_index = 0
    current_vad = np.float32(0.0)
    for offset in range(0, output_samples, DEMO_BLOCK_SIZE):
        buffered_samples += DEMO_BLOCK_SIZE
        if buffered_samples >= HOP_SIZE and probability_index < len(probabilities):
            buffered_samples -= HOP_SIZE
            current_vad = np.float32(
                VAD_SPEECH_SAMPLE / 32768.0 if vad_decisions[probability_index] else 0.0
            )
            probability_index += 1
        vad[offset : offset + DEMO_BLOCK_SIZE] = current_vad
    return output, vad, elapsed


def process_wav(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    enhancement_level: float,
    spectrum_scale: float,
    respect_data_chunk: bool,
    bypass: float = 0.0,
    vad_speech_hold_duration: float = 0.1,
    vad_sensitivity: float = 8.0,
    vad_minimum_speech_duration: float = 0.0,
) -> tuple[float, float]:
    audio, sample_rate = read_pcm16_wav(
        input_path, respect_data_chunk=respect_data_chunk
    )
    output, vad, elapsed = process_audio(
        model_path,
        audio,
        sample_rate=sample_rate,
        bypass=bypass,
        enhancement_level=enhancement_level,
        vad_speech_hold_duration=vad_speech_hold_duration,
        vad_sensitivity=vad_sensitivity,
        vad_minimum_speech_duration=vad_minimum_speech_duration,
        spectrum_scale=spectrum_scale,
    )
    write_pcm16_wav(output_path, output, sample_rate)
    vad_path = output_path.with_name(output_path.stem + "_vad.wav")
    write_pcm16_wav(vad_path, vad, sample_rate)
    output_samples = len(output)
    duration = output_samples / sample_rate
    if output_samples == 0:
        return 0.0, 0.0
    return elapsed / duration, elapsed * 1e9 / (output_samples * audio.shape[1])


@torch.inference_mode()
def run_probe(
    model_path: Path,
    input_path: Path,
    reference_path: Path | None,
    spectrum_scale: float,
) -> None:
    values = np.fromfile(input_path, dtype="<f4")
    expected_size = 2 * FREQ_BINS * 3
    if values.size != expected_size:
        raise ValueError(
            f"probe input has {values.size} floats, expected {expected_size}"
        )
    network = QuailVoiceFocus(model_path, spectrum_scale=spectrum_scale).eval()
    input_tensor = torch.from_numpy(values.reshape(1, 2, FREQ_BINS, 3))
    output, vad = network(input_tensor)
    output_array = output.numpy()
    print(
        f"output min={output_array.min():.9g} max={output_array.max():.9g} "
        f"rms={np.sqrt(np.mean(output_array**2)):.9g} "
        f"vad={float(vad.flatten()[0]):.9g}"
    )
    if reference_path is not None:
        reference = np.fromfile(reference_path, dtype="<f4").reshape(1, 2, FREQ_BINS, 1)
        error = output_array - reference
        cosine = np.vdot(output_array, reference).real / (
            np.linalg.norm(output_array) * np.linalg.norm(reference)
        )
        print(
            f"reference max_abs={np.max(np.abs(error)):.9g} "
            f"rmse={np.sqrt(np.mean(error**2)):.9g} "
            f"cosine={cosine:.9g}"
        )


def main() -> None:
    default_model = Path(__file__).with_name(
        "quail_vf_2_1_s_16khz_5i8jb8of_v12.aicmodel"
    )
    parser = argparse.ArgumentParser(
        description="Pure PyTorch quail-vf speech enhancement"
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--enhancement-level", type=float, default=0.8)
    parser.add_argument(
        "--bypass",
        action="store_true",
        help="set the public processor bypass parameter",
    )
    parser.add_argument(
        "--vad-hold",
        type=float,
        default=0.1,
        help="public VAD speech hold duration in seconds",
    )
    parser.add_argument(
        "--vad-sensitivity",
        type=float,
        default=8.0,
        help="public VAD sensitivity in [1, 15]",
    )
    parser.add_argument(
        "--vad-minimum-speech-duration",
        type=float,
        default=0.0,
        help="public VAD minimum speech duration in seconds",
    )
    parser.add_argument("--spectrum-scale", type=float, default=1.0 / FFT_SIZE)
    parser.add_argument(
        "--respect-data-chunk",
        action="store_true",
        help="stop at the declared data length instead of matching demo EOF reads",
    )
    parser.add_argument("--probe-input", type=Path)
    parser.add_argument("--probe-reference", type=Path)
    args = parser.parse_args()

    if args.probe_input is not None:
        run_probe(
            args.model,
            args.probe_input,
            args.probe_reference,
            args.spectrum_scale,
        )
        return
    if args.input is None or args.output is None:
        parser.error("input and output WAV paths are required")
    rtf, ns_per_sample = process_wav(
        args.model,
        args.input,
        args.output,
        enhancement_level=args.enhancement_level,
        spectrum_scale=args.spectrum_scale,
        respect_data_chunk=args.respect_data_chunk,
        bypass=float(args.bypass),
        vad_speech_hold_duration=args.vad_hold,
        vad_sensitivity=args.vad_sensitivity,
        vad_minimum_speech_duration=args.vad_minimum_speech_duration,
    )
    print(f"RTF: {rtf:.6f}")
    print(f"Time: {ns_per_sample:.2f} ns/sample/channel")


if __name__ == "__main__":
    main()
