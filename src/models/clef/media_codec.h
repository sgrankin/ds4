#ifndef DS4_CLEF_MEDIA_CODEC_H
#define DS4_CLEF_MEDIA_CODEC_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *rgb;
    int width, height;
    double seconds;
} ds4_clef_media_frame;

typedef struct {
    ds4_clef_media_frame *frames;
    size_t frame_count;
    float *samples; /* mono, 16000 Hz */
    size_t sample_count;
    double source_fps, audio_start_seconds;
} ds4_clef_media;

/* Decode already-fetched bytes, with no filesystem or network access. Video
 * selects frames at target_fps and also decodes its soundtrack when present.
 * Hard request limits fail visibly rather than silently truncating media. */
int ds4_clef_media_decode(const uint8_t *bytes, size_t length, int video,
                         double target_fps, size_t max_frames,
                         size_t max_samples, size_t max_frame_pixels,
                         ds4_clef_media *out, char *error, size_t error_capacity);
void ds4_clef_media_free(ds4_clef_media *media);
#endif
