#ifndef DS4_SUFFIX_SNAPSHOT_COMPARE_H
#define DS4_SUFFIX_SNAPSHOT_COMPARE_H
#include "ds4.h"
#include <string.h>

/* Test-only comparison. Every byte except the deliberately distinct magic
 * must match; accepting arbitrary diagnostic/canonical tag pairs is forbidden. */
static uint32_t suffix_snapshot_magic(const ds4_session_snapshot *s) {
    if (!s || !s->ptr || s->len < 52) return 0;
    return (uint32_t)s->ptr[0] | (uint32_t)s->ptr[1] << 8 |
           (uint32_t)s->ptr[2] << 16 | (uint32_t)s->ptr[3] << 24;
}
static bool suffix_diagnostic_equal_canonical(const ds4_session_snapshot *diagnostic,
                                             const ds4_session_snapshot *canonical) {
    return suffix_snapshot_magic(diagnostic) == DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC &&
           suffix_snapshot_magic(canonical) == DS4_SESSION_PAYLOAD_MAGIC &&
           diagnostic->len == canonical->len &&
           !memcmp(diagnostic->ptr + 4, canonical->ptr + 4,
                   (size_t)canonical->len - 4);
}
#endif
