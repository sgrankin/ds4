/* Model-free verification of the production suffix relocation layout. */
#include "ds4_suffix_layout.h"
#include <stdio.h>
#include <string.h>

static int verify(uint32_t start, int64_t displacement, uint32_t count) {
    uint32_t source = (uint32_t)((int64_t)start + displacement), source_end = source + count;
    uint32_t end = start + count, raw = count < 128u ? count : 128u;
    uint32_t original[128], actual[128], expected[128];
    for (uint32_t i = 0; i < 128; i++) original[i] = actual[i] = expected[i] = UINT32_MAX;
    for (uint32_t p = source_end > 128 ? source_end - 128 : 0; p < source_end; p++)
        original[p % 128] = 10000000u + p;
    for (uint32_t p = start > 128 ? start - 128 : 0; p < start; p++)
        actual[p % 128] = expected[p % 128] = 20000000u + p;
    /* Independent token-by-token reference; no shared segment arithmetic. */
    for (uint32_t p = end - raw; p < end; p++)
        expected[p % 128] = 10000000u + source + (p - start);
    uint32_t chunks = 0, copied = 0;
    for (uint32_t p = source_end - raw; p < source_end;) {
        ds4_suffix_raw_segment segment;
        if (!ds4_suffix_raw_next(p, source_end, displacement, &segment) || !segment.rows ||
            segment.source_slot + segment.rows > 128 || segment.destination_slot + segment.rows > 128)
            return 1;
        for (uint32_t j = 0; j < segment.rows; j++) {
            if ((int64_t)p + j - displacement != end - raw + copied + j) return 1;
        }
        memcpy(actual + segment.destination_slot, original + segment.source_slot,
               segment.rows * sizeof(*actual));
        copied += segment.rows; p += segment.rows; chunks++;
    }
    if (copied != raw || chunks > 3 || memcmp(actual, expected, sizeof(actual))) return 1;
    /* Fresh writes must overwrite only their own cells, retaining mixed prefix. */
    for (uint32_t p = end; p < end + 129u; p++) {
        actual[p % 128] = expected[p % 128] = 30000000u + p;
        if (memcmp(actual, expected, sizeof(actual))) return 1;
    }
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
        uint32_t source_first = source / ratio, destination_first = start / ratio;
        uint32_t rows = source_end / ratio - source_first;
        if (rows != end / ratio - destination_first || (source_end & 1u) != (end & 1u)) return 1;
        for (uint32_t r = 0; r < rows; r++) {
            if ((int64_t)(source_first + r) * ratio - displacement != (destination_first + r) * ratio) return 1;
        }
    }
    return 0;
}

int main(void) {
    if (!ds4_suffix_even_left_shift(386, 384) || !ds4_suffix_even_left_shift(384, 384) ||
        ds4_suffix_even_left_shift(385, 384) || ds4_suffix_even_left_shift(386, 385) ||
        ds4_suffix_even_left_shift(382, 384)) return 1;
    if (!ds4_suffix_even_shift(382, 384) || !ds4_suffix_even_shift(386, 384) ||
        ds4_suffix_even_shift(383, 384) || ds4_suffix_even_shift(382, 385)) return 1;
    ds4_suffix_raw_segment segment;
    if (ds4_suffix_raw_next(1, 1, 0, &segment) || ds4_suffix_raw_next(1, 2, 2, &segment) ||
        ds4_suffix_raw_next(1, 2, 0, NULL) ||
        ds4_suffix_raw_next(UINT32_MAX - 1, UINT32_MAX, -2, &segment)) return 1;
    uint32_t total = 0;
    const int64_t shifts[] = {0, 2, 4, 62, 64, 66, 126, 128, 130, 256, 258,
                             -2, -4, -62, -64, -66, -126, -128, -130, -256, -258};
    const uint32_t counts[] = {1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257};
    for (uint32_t start = 0; start <= 1024; start += 2)
        for (uint32_t d = 0; d < sizeof(shifts) / sizeof(*shifts); d++)
            for (uint32_t c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
                if ((int64_t)start + shifts[d] < 0) continue;
                if (verify(start, shifts[d], counts[c])) {
                    fprintf(stderr, "suffix-layout FAIL start=%u shift=%lld count=%u\n",
                            start, (long long)shifts[d], counts[c]); return 1;
                }
                total++;
            }
    uint32_t seed = 0x5f4a13u;
    for (uint32_t i = 0; i < 50000; i++) {
        seed = seed * 1664525u + 1013904223u; uint32_t start = (seed % 4096u) * 2u;
        seed = seed * 1664525u + 1013904223u; uint32_t shift = (seed % 1024u) * 2u;
        seed = seed * 1664525u + 1013904223u; uint32_t count = seed % 2048u + 1u;
        if (verify(start, shift, count)) return 1;
        total++;
        if (start >= shift) {
            if (verify(start, -(int64_t)shift, count)) return 1;
            total++;
        }
    }
    printf("suffix-layout: %u production segment/ring/pool ownership states PASS; no model opened\n", total);
    return 0;
}
