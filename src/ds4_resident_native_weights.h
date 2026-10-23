#ifndef DS4_RESIDENT_NATIVE_WEIGHTS_H
#define DS4_RESIDENT_NATIVE_WEIGHTS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Explicit experiment flag: unset and literal0 preserve the old path. Empty,
 * whitespace, signed or alternative spellings are rejected, not coerced. */
static inline bool ds4_resident_native_weights_parse(const char *value, bool *enabled) {
    if (!enabled) return false;
    *enabled = false;
    if (!value) return true;
    if (value[0] == '0' && value[1] == '\0') return true;
    if (value[0] == '1' && value[1] == '\0') {
        *enabled = true;
        return true;
    }
    return false;
}

/* Opt-in admission is checked even for native prefill. Bitwise finiteness
 * remains valid under the production finite-math optimizer assumptions. */
static inline bool ds4_resident_native_weights_gap(const char *value) {
    if (!value || !*value) return false;
    char *end = NULL;
    const float gap = strtof(value, &end);
    uint32_t bits;
    memcpy(&bits, &gap, sizeof(bits));
    return end != value && !*end && (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) &&
        gap > 0.0f && gap <= 1.0f;
}

#endif
