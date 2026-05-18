#include "aic.hpp"
#include "audio_wave.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::cout << "ai-coustics SDK version: " << aic::get_sdk_version() << "\n";
    std::cout << "Compatible model version: " << aic::get_compatible_model_version() << "\n";

#ifdef AIC_SDK_LICENSE
    auto license_key = std::string(AIC_SDK_LICENSE);
#else
    auto license_env = std::getenv("AIC_SDK_LICENSE");
    if (!license_env || std::string(license_env).empty())
    {
        std::cerr << "Error: Environment variable AIC_SDK_LICENSE not set.\n";
        return 1;
    }
    auto license_key = std::string(license_env);
#endif

    std::string model_path =
        "/Users/bytedance/Work/aic-sdk-cpp/model/quail_vf_2_1_s_16khz_5i8jb8of_v12.aicmodel";
    std::string input_wav_path  = "/Users/bytedance/Downloads/an_example.wav";
    std::string output_wav_path = "/Users/bytedance/Downloads/aic_out.wav";
    std::string vad_wav_path    = "/Users/bytedance/Downloads/aic_vad.wav";

    (void) argc;
    (void) argv;

    auto model_result = aic::Model::create_from_file(model_path);
    auto err          = model_result.error;

    if (!model_result.ok())
    {
        std::cerr << "Model creation failed with error code: " << static_cast<int>(err) << "\n";
        return 1;
    }

    auto model = model_result.take();

    // Query optimal settings from the model
    auto sample_rate = model.get_optimal_sample_rate();
    auto num_frames  = model.get_optimal_num_frames(sample_rate);

    // Create configuration with optimal settings
    aic::ProcessorConfig config(sample_rate, num_frames);  // mono, fixed frames

    auto processor_result = aic::Processor::create(model, license_key);
    err                   = processor_result.error;

    if (!processor_result.ok())
    {
        std::cerr << "Processor creation failed with error code: " << static_cast<int>(err) << "\n";
        return 1;
    }

    auto processor = processor_result.take();
    err = processor.initialize(config.sample_rate, config.num_channels, config.num_frames,
                               config.allow_variable_frames);
    if (err != aic::ErrorCode::Success)
    {
        std::cerr << "Initialization failed\n";
        return 1;
    }

    auto ctx_result = processor.create_context();
    if (!ctx_result.ok())
    {
        std::cerr << "Processor context creation failed\n";
        return 1;
    }

    auto ctx          = ctx_result.take();
    auto output_delay = ctx.get_output_delay();
    std::cout << "Output delay: " << output_delay << " samples\n";

    auto vad_result = processor.create_vad_context();
    if (!vad_result.ok())
    {
        std::cerr << "VAD context creation failed\n";
        return 1;
    }

    auto vad = vad_result.take();
    err      = vad.set_parameter(aic::VadParameter::SpeechHoldDuration, 0.1f);
    if (err != aic::ErrorCode::Success)
    {
        std::cerr << "Failed to set VAD speech hold duration\n";
        return 1;
    }

    err = vad.set_parameter(aic::VadParameter::Sensitivity, 8.0f);
    if (err != aic::ErrorCode::Success)
    {
        std::cerr << "Failed to set VAD sensitivity\n";
        return 1;
    }

    err = ctx.set_parameter(aic::ProcessorParameter::EnhancementLevel, 0.8f);
    if (err != aic::ErrorCode::Success)
    {
        std::cerr << "Failed to set enhancement level\n";
        return 1;
    }

    // Open input wav
    SWavFile* wav_in = wav_open(input_wav_path.c_str(), "rb");
    if (!wav_in)
    {
        std::cerr << "Failed to open input wav: " << input_wav_path << "\n";
        return 1;
    }

    uint32_t wav_sample_rate = wav_get_sample_rate(wav_in);
    if (wav_sample_rate != sample_rate)
    {
        std::cerr << "Warning: input wav sample rate (" << wav_sample_rate
                  << ") does not match model optimal sample rate (" << sample_rate << ")\n";
    }

    // Open output wav
    SWavFile* wav_out = wav_open(output_wav_path.c_str(), "wb");
    if (!wav_out)
    {
        std::cerr << "Failed to open output wav: " << output_wav_path << "\n";
        wav_close(wav_in);
        return 1;
    }
    wav_set_format(wav_out, wav_get_format(wav_in));
    wav_set_sample_rate(wav_out, sample_rate);
    wav_set_num_channels(wav_out, config.num_channels);
    wav_set_sample_size(wav_out, 16);

    // Open vad output wav (same format as output wav)
    SWavFile* wav_vad = wav_open(vad_wav_path.c_str(), "wb");
    if (!wav_vad)
    {
        std::cerr << "Failed to open vad wav: " << vad_wav_path << "\n";
        wav_close(wav_in);
        wav_close(wav_out);
        return 1;
    }
    wav_set_format(wav_vad, wav_get_format(wav_in));
    wav_set_sample_rate(wav_vad, sample_rate);
    wav_set_num_channels(wav_vad, config.num_channels);
    wav_set_sample_size(wav_vad, 16);

    // Buffers for int16 <-> float conversion
    const size_t         frame_samples = config.num_frames * config.num_channels;
    const size_t         frame_bytes   = frame_samples * sizeof(int16_t);
    std::vector<int16_t> pcm_buf(frame_samples);
    std::vector<float>   float_buf(frame_samples);

    int                       frames_processed = 0;
    std::chrono::microseconds total_process_time(0);

    while (true)
    {
        size_t read_bytes = wav_read_interleave(wav_in, pcm_buf.data(), frame_bytes);
        if (read_bytes < frame_bytes)
        {
            break;
        }

        // Convert int16 -> float [-1.0, 1.0]
        for (size_t i = 0; i < frame_samples; ++i)
        {
            float_buf[i] = pcm_buf[i] / 32768.0f;
        }

        auto start = std::chrono::high_resolution_clock::now();

        err =
            processor.process_interleaved(float_buf.data(), config.num_channels, config.num_frames);

        auto stop = std::chrono::high_resolution_clock::now();
        total_process_time += std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        if (err != aic::ErrorCode::Success)
        {
            std::cerr << "process_interleaved failed at frame " << frames_processed
                      << ", error: " << static_cast<int>(err) << "\n";
            break;
        }

        // Convert float -> int16, clamp to avoid overflow
        for (size_t i = 0; i < frame_samples; ++i)
        {
            float s = float_buf[i] * 32768.0f;
            if (s > 32767.0f)
                s = 32767.0f;
            if (s < -32768.0f)
                s = -32768.0f;
            pcm_buf[i] = static_cast<int16_t>(s);
        }

        wav_write_interleave(wav_out, pcm_buf.data(), frame_bytes);

        auto speech_detected = vad.is_speech_detected();

        // Write vad result as a constant-value frame: 32767 for speech, 0 for silence
        int16_t vad_sample = speech_detected ? 30000 : 0;
        std::fill(pcm_buf.begin(), pcm_buf.end(), vad_sample);
        wav_write_interleave(wav_vad, pcm_buf.data(), frame_bytes);

        ++frames_processed;
    }

    wav_close(wav_in);
    wav_close(wav_out);
    wav_close(wav_vad);

    double audio_duration_ms =
        static_cast<double>(frames_processed) * num_frames * 1000.0 / sample_rate;
    double process_duration_ms = total_process_time.count() * 0.001;
    std::cout << "Processed " << frames_processed << " frames (" << audio_duration_ms << " ms)\n";
    std::cout << "Total processing time: " << process_duration_ms << " ms\n";
    std::cout << "RTF: " << process_duration_ms / audio_duration_ms << "\n";
    std::cout << "Output written to: " << output_wav_path << "\n";

    return 0;
}
