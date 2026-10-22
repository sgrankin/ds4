#ifndef DS4_SESSION_FRONTIER_H
#define DS4_SESSION_FRONTIER_H

#include "ds4.h"

/* Existing image identity record, moved once from ds4.c. The embedding itself
 * is borrowed only during sync; a committed checkpoint owns this small copy. */
typedef struct {
    uint32_t token_start;
    uint32_t token_count;
    uint8_t fingerprint[32];
} ds4_vision_identity;

/* The actual common session frontier, embedded ONCE in ds4_session. This is
 * neither a Session mirror nor a second checkpoint. C fallback backends and
 * native controllers read/write these same fields during the transition.
 *
 * checkpoint, checkpoint_images and the three host logit/probability buffers
 * keep their existing libc ownership. The outer session still allocates/frees
 * the host buffers. sync_images is borrowed for a synchronous call only.
 * A valid checkpoint includes matching logits AND all compressor frontiers;
 * truncating checkpoint tokens alone never makes a DeepSeek rewind valid. */
typedef struct {
    ds4_tokens checkpoint;
    ds4_vision_identity *checkpoint_images;
    size_t checkpoint_image_count;
    const ds4_vision_span *sync_images;
    size_t sync_image_count;
    float *logits;
    float *sampling_logits;
    int overthink_prompt_len;
    float *sample_probs;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    bool approximate_suffix;
    bool approximate_decoder_replay; /* Not a relocatable suffix-cache source. */
    uint32_t decoder_replay_rows;
    bool decoder_replay_selected;
    bool mtp_draft_valid;
} ds4_session_frontier;

#endif
