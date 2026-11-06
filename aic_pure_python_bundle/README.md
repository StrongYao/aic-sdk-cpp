# AIC Pure Python Bundle

This folder contains the files required to run the pure Python/PyTorch AIC reconstruction.

Files:

- `aic_pure_python.py`: algorithm implementation.
- `quail_vf_2_1_s_16khz_5i8jb8of_v12.aicmodel`: model weights loaded by default.

Python packages are not included. Install `numpy` and `torch` in the runtime environment.

Example:

```bash
python3 aic_pure_python.py input.wav output.wav
```

Input contract:

- 16000 Hz
- PCM16 WAV
- Complete 160-sample host blocks are processed; incomplete tail samples are dropped.

