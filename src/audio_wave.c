#include "audio_wave.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct SWavHead_t {
  // Resource Interchange File Flag (0-3) "RIFF"
  char RIFF[4];
  // File Length ( not include 8 bytes from the beginning ) (4-7)
  int32_t FileLength;
  // WAVE File Flag (8-15) "WAVEfmt "
  char WAVEfmt_[8];
  // Transitory Byte ( normally it is 10H 00H 00H 00H ) (16-19)
  // Format chunk size - 8. (16-19)
  uint32_t fmtSize;
  // Format Category ( normally it is 1 means PCM-u Law ) (20-21)
  int16_t FormatCategory;
  // NChannels (22-23)
  int16_t NChannels;
  // Sample Rate (24-27)
  int32_t SampleRate;
  // l=NChannels*SampleRate*NBitsPerSample/8 (28-31)
  int32_t byteRate;
  // i=NChannels*NBitsPerSample/8 (32-33)
  int16_t BytesPerSample;
  // NBitsPerSample (34-35)
  int16_t NBitsPerSample;
  // Data Flag  "data"
  char data[4];
  // Raw Data File Length
  int32_t RawDataFileLength;
} SWavHead;

struct SWavFile_t {
  SWavHead m_head;
  FILE* m_fp;
  fpos_t m_DATApos;
  bool m_bInited;
  bool isRead;
};

static void scan2data(SWavFile* self) {
  int i;
  char chunktype[5], temp;
  for (i = 0; i < 4; i++) chunktype[i] = self->m_head.data[i];
  chunktype[4] = '\0';
  if (strcmp(chunktype, "data") == 0) {
    return;
  } else {
    for (i = 0; i < self->m_head.RawDataFileLength; i++)
      if (fread(&temp, 1, 1, self->m_fp) == 0) {
        return;
      }
    // padding
    if (self->m_head.RawDataFileLength % 2 != 0)
      if (fread(&temp, 1, 1, self->m_fp) == 0) {
        return;
      }
    if (fread(&self->m_head.data, 4, 1, self->m_fp) == 0) {
      return;
    }
    if (fread(&self->m_head.RawDataFileLength, 4, 1, self->m_fp) == 0) {
      return;
    }
    scan2data(self);
  }
}

static int wav_init(SWavFile* self, const char* filename, const char* mode) {
  self->m_bInited = false;
  self->m_fp = NULL;
  char riff[8], wavefmt[8];
  short i;

  self->m_fp = fopen(filename, mode);
  if (!self->m_fp)
    return -2;  // cann't open the file
  else if (strchr(mode, 'r')) {
    self->isRead = true;
    if (fread(&self->m_head.RIFF[0], 4, 1, self->m_fp) == 0) {
      return -1;
    }
    if (fread(&self->m_head.FileLength, sizeof(int32_t), 1, self->m_fp) == 0) {
      return -1;
    }
    if (fread(&self->m_head.WAVEfmt_[0], 8, 1, self->m_fp) == 0) {
      return -1;
    }
    for (i = 0; i < 4; i++) riff[i] = self->m_head.RIFF[i];
    for (i = 0; i < 8; i++) wavefmt[i] = self->m_head.WAVEfmt_[i];
    riff[4] = '\0';
    wavefmt[7] = '\0';
    if (strcmp(riff, "RIFF") == 0 && strcmp(wavefmt, "WAVEfmt") == 0) {
      if (fread(&self->m_head.fmtSize, sizeof(int32_t), 1, self->m_fp) == 0) {
        return -1;
      }
      if (fread(&self->m_head.FormatCategory, sizeof(int16_t), 1, self->m_fp) ==
          0) {
        return -1;
      }
      if (fread(&self->m_head.NChannels, sizeof(int16_t), 1, self->m_fp) == 0) {
        return -1;
      }
      if (fread(&self->m_head.SampleRate, sizeof(int32_t), 1, self->m_fp) ==
          0) {
        return -1;
      }
      if (fread(&self->m_head.byteRate, sizeof(int32_t), 1, self->m_fp) == 0) {
        return -1;
      }
      if (fread(&self->m_head.BytesPerSample, sizeof(int16_t), 1, self->m_fp) ==
          0) {
        return -1;
      }
      if (fread(&self->m_head.NBitsPerSample, sizeof(int16_t), 1, self->m_fp) ==
          0) {
        return -1;
      }
      unsigned int read_len = self->m_head.fmtSize - 16U;
      if (read_len > 0 && read_len <= 32) {  // Skip extra.
        char tmp[32];
        if (fread(&tmp[0], read_len, 1, self->m_fp) == 0) {
          return -1;
        }
      }
      if (fread(&self->m_head.data[0], 4, 1, self->m_fp) == 0) {
        return -1;
      }
      if (fread(&self->m_head.RawDataFileLength, sizeof(int32_t), 1,
                self->m_fp) == 0) {
        return -1;
      }
      scan2data(self);
      // save DATA position
      fgetpos(self->m_fp, &self->m_DATApos);
    } else
      return -1;
  } else if (strchr(mode, 'w')) {
    self->isRead = false;
    memset(&self->m_head, 0, sizeof(SWavHead));
    strncpy(self->m_head.RIFF, "RIFF", 4);
    strncpy(self->m_head.WAVEfmt_, "WAVEfmt ", 8);
    strncpy(self->m_head.data, "data", 4);

    self->m_head.fmtSize = 16;
    self->m_head.BytesPerSample =
        self->m_head.NChannels * self->m_head.NBitsPerSample / 8;
    self->m_head.byteRate =
        self->m_head.BytesPerSample * self->m_head.SampleRate;
    rewind(self->m_fp);
    fwrite(&self->m_head, sizeof(SWavHead), 1, self->m_fp);
  }
  self->m_bInited = true;
  return 0;
}

static void wav_update_header(SWavFile* self) {
  if (self->m_fp) {
    self->m_head.FileLength = self->m_head.RawDataFileLength + 36;
    self->m_head.BytesPerSample =
        self->m_head.NChannels * self->m_head.NBitsPerSample / 8;
    self->m_head.byteRate =
        self->m_head.BytesPerSample * self->m_head.SampleRate;
    rewind(self->m_fp);
    fwrite(&self->m_head, sizeof(SWavHead), 1, self->m_fp);
  }
}

SWavFile* wav_open(const char* filename, const char* mode) {
  SWavFile* wave = (SWavFile*)malloc(sizeof(SWavFile));
  if (wave == NULL) {
    return wave;
  }
  int ret = wav_init(wave, filename, mode);
  if (ret != 0) {
    if (wave->m_fp) {
      fclose(wave->m_fp);
      wave->m_fp = NULL;
    }
    if (wave != NULL) {
      free(wave);
    }
    return NULL;
  }
  return wave;
}

void wav_close(SWavFile* self) {
  if (!self->isRead) {
    wav_update_header(self);
  }
  if (self->m_fp) {
    fclose(self->m_fp);
    self->m_fp = NULL;
  }
  free(self);
}

size_t wav_read_mono(SWavFile* self, void* buffer, size_t read_bytes) {
  size_t read_len = 0;
  if (!self->isRead) {
    return read_len;
  }
  if (self->m_fp) {
    read_len = fread(buffer, 1, read_bytes, self->m_fp);
  }
  return read_len;
}

size_t wav_write_mono(SWavFile* self, const void* buffer, size_t write_bytes) {
  size_t write_len = 0;
  if (self->isRead) {
    return write_len;
  }
  if (self->m_fp) {
    fwrite(buffer, 1, write_bytes, self->m_fp);
  }
  self->m_head.RawDataFileLength += (int32_t)write_bytes;
  return write_bytes;
}

size_t wav_read_sequential(SWavFile* self, void* buffer, size_t read_bytes) {
  uint8_t* temp = (uint8_t*)malloc(read_bytes);
  if (NULL == temp) {
    return 0;
  }
  size_t res = wav_read_interleaved(self, temp, read_bytes);

  int bytes_per_sample = self->m_head.NBitsPerSample / 8;
  int channel_num = self->m_head.NChannels;
  int frame_size = (int)(res / (bytes_per_sample * channel_num));
  for (int i = 0; i < channel_num; i++) {
    for (int j = 0; j < frame_size; j++) {
      uint8_t* dst = (uint8_t*)buffer + (i * frame_size + j) * bytes_per_sample;
      uint8_t* src = temp + (j * channel_num + i) * bytes_per_sample;
      memcpy(dst, src, bytes_per_sample);
    }
  }
  free(temp);
  return res;
}

size_t wav_read_interleave(SWavFile* self, void* buffer, size_t read_bytes) {
  uint8_t* temp = (uint8_t*)malloc(read_bytes);
  if (temp == NULL) {
    return -1;
  }
  size_t res = wav_read_interleaved(self, temp, read_bytes);

  int bytes_per_sample = self->m_head.NBitsPerSample / 8;
  int channel_num = self->m_head.NChannels;
  int frame_size = (int)(res / (bytes_per_sample * channel_num));
  for (int i = 0; i < channel_num; i++) {
    for (int j = 0; j < frame_size; j++) {
      uint8_t* dst =
          (uint8_t*)buffer + (j * channel_num + i) * bytes_per_sample;
      uint8_t* src = temp + (j * channel_num + i) * bytes_per_sample;
      memcpy(dst, src, bytes_per_sample);
    }
  }
  free(temp);
  return res;
}

size_t wav_write_sequential(SWavFile* self, const void* buffer,
                            size_t write_bytes) {
  uint8_t* temp = (uint8_t*)malloc(write_bytes);
  int bytes_per_sample = self->m_head.NBitsPerSample / 8;
  int channel_num = self->m_head.NChannels;
  int frame_size = (int)(write_bytes / (bytes_per_sample * channel_num));
  for (int sampleCnt = 0; sampleCnt < frame_size; sampleCnt++) {
    for (int channelCnt = 0; channelCnt < channel_num; channelCnt++) {
      uint8_t* dst =
          temp + (sampleCnt * channel_num + channelCnt) * bytes_per_sample;
      uint8_t* src = (uint8_t*)buffer +
                     (channelCnt * frame_size + sampleCnt) * bytes_per_sample;
      memcpy(dst, src, bytes_per_sample);
    }
  }

  size_t res = wav_write_interleaved(self, temp, write_bytes);
  free(temp);
  return res;
}

size_t wav_write_interleave(SWavFile* self, const void* buffer,
                            size_t write_bytes) {
  uint8_t* temp = (uint8_t*)malloc(write_bytes);
  int bytes_per_sample = self->m_head.NBitsPerSample / 8;
  int channel_num = self->m_head.NChannels;
  int frame_size = (int)(write_bytes / (bytes_per_sample * channel_num));
  for (int sampleCnt = 0; sampleCnt < frame_size; sampleCnt++) {
    for (int channelCnt = 0; channelCnt < channel_num; channelCnt++) {
      uint8_t* dst =
          temp + (sampleCnt * channel_num + channelCnt) * bytes_per_sample;
      //            uint8_t *src = (uint8_t *)buffer + (channelCnt * frame_size
      //            + sampleCnt) * bytes_per_sample;
      uint8_t* src = (uint8_t*)buffer +
                     (sampleCnt * channel_num + channelCnt) * bytes_per_sample;
      memcpy(dst, src, bytes_per_sample);
    }
  }

  size_t res = wav_write_interleaved(self, temp, write_bytes);
  free(temp);
  return res;
}

size_t wav_write_interleave_stereo(SWavFile* self, const void* buffer1,
                                   const void* buffer2, size_t write_bytes) {
  // buffer1 is a mono buffer, buffer2 is a stereo buffer, output file is a
  // stereo file
  uint8_t* temp = (uint8_t*)malloc(write_bytes);

  int bytes_per_sample = self->m_head.NBitsPerSample / 8;
  int channel_num = self->m_head.NChannels;
  int frame_size = (int)(write_bytes / (bytes_per_sample * channel_num));
  uint8_t* src = NULL;
  for (int sampleCnt = 0; sampleCnt < frame_size; sampleCnt++) {
    for (int channelCnt = 0; channelCnt < channel_num; channelCnt++) {
      uint8_t* dst =
          temp + (sampleCnt * channel_num + channelCnt) * bytes_per_sample;
      if (channelCnt == 0) {
        src = (uint8_t*)buffer1 + sampleCnt * bytes_per_sample;
      } else {
        // src = (uint8_t *)buffer2 + sampleCnt * bytes_per_sample;
        src = (uint8_t*)buffer2 + (sampleCnt * channel_num) * bytes_per_sample;
      }
      memcpy(dst, src, bytes_per_sample);
    }
  }

  size_t res = wav_write_interleaved(self, temp, write_bytes);
  free(temp);
  return res;
}

void wav_set_sample_rate(SWavFile* self, uint32_t sample_rate) {
  if (self->isRead) {
    return;
  }
  self->m_head.SampleRate = sample_rate;
}

void wav_set_sample_size(SWavFile* self, int16_t sample_size) {
  if (self->isRead) {
    return;
  }
  self->m_head.NBitsPerSample = sample_size;
}

void wav_set_num_channels(SWavFile* self, uint16_t num_channels) {
  if (self->isRead) {
    return;
  }
  self->m_head.NChannels = num_channels;
}

void wav_set_format(SWavFile* self, int format) {
  if (self->isRead) {
    return;
  }
  self->m_head.FormatCategory = format;
}

uint16_t wav_get_num_channels(const SWavFile* self) {
  return self->m_head.NChannels;
}

uint32_t wav_get_sample_rate(const SWavFile* self) {
  return self->m_head.SampleRate;
}

size_t wav_get_sample_size(const SWavFile* self) {
  return self->m_head.NBitsPerSample;
}

size_t wav_get_length(const SWavFile* self) { return self->m_head.FileLength; }

int32_t wav_get_data_length(const SWavFile* self) {
  return self->m_head.RawDataFileLength;
}

int wav_get_format(const SWavFile* self) { return self->m_head.FormatCategory; }

void wav_copy_format(SWavFile* self, const SWavFile* other) {
  if (!self || !other) {
    return;
  }
  wav_set_format(self, wav_get_format(other));
  wav_set_sample_rate(self, wav_get_sample_rate(other));
  wav_set_sample_size(self, (int16_t)wav_get_sample_size(other));
  wav_set_num_channels(self, wav_get_num_channels(other));
}
