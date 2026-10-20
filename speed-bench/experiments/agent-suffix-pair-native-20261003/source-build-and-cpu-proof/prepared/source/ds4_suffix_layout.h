#ifndef DS4_SUFFIX_LAYOUT_H
#define DS4_SUFFIX_LAYOUT_H

#include <stdbool.h>
#include <stdint.h>

/* DS4.1 ratio-two pools require both boundaries to remain even. */
static inline bool ds4_suffix_even_left_shift(uint32_t source, uint32_t destination) {
    return source >= destination && !((source | destination) & 1u);
}

typedef struct {
    uint32_t source_slot, destination_slot, rows;
} ds4_suffix_raw_segment;

/* Source bytes belong to an immutable snapshot. Split at both physical wraps;
 * only the trailing min(suffix_length,128) source rows may be copied. */
static inline bool ds4_suffix_raw_next(uint32_t position, uint32_t end,
                                      uint32_t displacement,
                                      ds4_suffix_raw_segment *segment) {
    if (!segment || position >= end || displacement > position) return false;
    segment->source_slot = position % 128u;
    segment->destination_slot = (position - displacement) % 128u;
    segment->rows = end - position;
    if (segment->rows > 128u - segment->source_slot)
        segment->rows = 128u - segment->source_slot;
    if (segment->rows > 128u - segment->destination_slot)
        segment->rows = 128u - segment->destination_slot;
    return true;
}

#endif
