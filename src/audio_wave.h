/**
 * \file
 * Audio Wave IO functions
 * To do: support multichannel read/write
 */

#ifndef AUDIO_WAVE_H
#define AUDIO_WAVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SWavFile_t SWavFile;

SWavFile* wav_open(const char* filename, const char* mode);

void wav_close(SWavFile* self);

/**
 * Read single-channel wave file.
 *
 * \param[in,out] self       Pointer to SWavFile.
 * \param[out]    buffer     The prepared input buffer to read.
 * \param[in]     read_bytes The intended read bytes per-channel.
 * \return                   The actual amount of bytes read.
 */
size_t wav_read_mono(SWavFile* self, void* buffer, size_t read_bytes);

/**
 * Write single-channel wave file.
 *
 * \param[in,out] self        Pointer to SWavFile.
 * \param[in]     buffer      The prepared output buffer to write.
 * \param[in]     write_bytes The buffer length to write in bytes.
 * \return                    If 0 means write failure, otherwise equals to
 * write_bytes.
 */
size_t wav_write_mono(SWavFile* self, const void* buffer, size_t write_bytes);

// size_t wav_read_interleaved(SWavFile *self, void *buffer, size_t read_bytes);
#define wav_read_interleaved(self, ...) wav_read_mono(self, ##__VA_ARGS__)

// size_t wav_write_interleaved(SWavFile *self, void *buffer, size_t
// write_bytes);
#define wav_write_interleaved(self, ...) wav_write_mono(self, ##__VA_ARGS__)

/**
 * Read multi-channel sequential (non-interleaved) wave file.
 *
 * \param[in,out] self       Pointer to SWavFile.
 * \param[out]    buffer     The prepared input buffer to read.
 * \param[in]     read_bytes The intended read bytes in total (should be
 * frameSize * (bitDepth / 8)
 * * channelNum). \return                   The actual amount of bytes read.
 */
size_t wav_read_sequential(SWavFile* self, void* buffer, size_t read_bytes);
size_t wav_read_interleave(SWavFile* self, void* buffer, size_t read_bytes);
/**
 * Write multi-channel sequential (non-interleaved) wave file.
 *
 * \param[in,out] self        Pointer to SWavFile.
 * \param[in]     buffer      The prepared output buffer to write.
 * \param[in]     write_bytes The buffer length to write in bytes (should be
 * frameSize * (bitDepth / 8) * channelNum). \return                    If 0
 * means write failure, otherwise equals to write_bytes.
 */
size_t wav_write_sequential(SWavFile* self, const void* buffer,
                            size_t write_bytes);
size_t wav_write_interleave(SWavFile* self, const void* buffer,
                            size_t write_bytes);
size_t wav_write_interleave_stereo(SWavFile* self, const void* buffer1,
                                   const void* buffer2, size_t write_bytes);

/**
 * Set the wave sample rate, should set before writing.
 *
 * \param[in,out] self        Pointer to SWavFile.
 * \param[in]     sample_rate The sampling rate of the output file like 16000.
 */
void wav_set_sample_rate(SWavFile* self, uint32_t sample_rate);

/**
 * set the wave bits per sample, should set before writing.
 *
 * \param[in,out] self        Pointer to SWavFile.
 * \param[in]     sample_size Bits per sample in output file, like 16.
 */
void wav_set_sample_size(SWavFile* self, int16_t sample_size);

void wav_set_num_channels(SWavFile* self, uint16_t num_channels);

void wav_set_format(SWavFile* self, int format);

uint16_t wav_get_num_channels(const SWavFile* self);

uint32_t wav_get_sample_rate(const SWavFile* self);

size_t wav_get_sample_size(const SWavFile* self);

size_t wav_get_length(const SWavFile* self);

int32_t wav_get_data_length(const SWavFile* self);

int wav_get_format(const SWavFile* self);

/**
 * copy the header from another SWavFile
 *
 * \param[in,out] self        Pointer to SWavFile.
 * \param[in] other     the src SWavFile to be copied.
 */
void wav_copy_format(SWavFile* self, const SWavFile* other);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_WAVE_H */
