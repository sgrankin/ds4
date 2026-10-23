#ifndef DS4_SUFFIX_MATCH_H
#define DS4_SUFFIX_MATCH_H

#include "ds4_suffix_layout.h"
#include <string.h>

typedef struct {
    uint32_t source_start, destination_start, source_end, destination_end;
    uint32_t reused_tokens, fresh_tokens;
} ds4_suffix_match_plan;

/* Match one immutable endpoint; it cannot be truncated. fresh_cutoff is the
 * latest allowed cached end, normally min(latest_user_start, len - fresh_min).
 * Everything at and after that boundary remains fresh. Across endpoints the
 * caller keeps the plan with the most reused tokens. No source is modified. */
static inline bool ds4_suffix_match_shift(const int *source, uint32_t source_end,
                                    const int *edited, uint32_t edited_len,
                                    uint32_t fresh_cutoff, bool allow_right,
                                    ds4_suffix_match_plan *plan) {
    if (!plan) return false;
    *plan = (ds4_suffix_match_plan){0};
    if (!source || !edited || source_end < 64u || fresh_cutoff < 64u ||
        fresh_cutoff >= edited_len) return false;

    uint32_t max_end = allow_right || fresh_cutoff < source_end ? fresh_cutoff : source_end;
    for (uint32_t end = max_end; end >= 64u; end--) {
        if (((source_end ^ end) & 1u) ||
            memcmp(edited + end - 64u, source + source_end - 64u,
                   64u * sizeof(*source))) continue;
        uint32_t src = source_end - 64u, dst = end - 64u;
        while (src && dst && source[src - 1u] == edited[dst - 1u]) {
            src--; dst--;
        }
        if (src & 1u) { src++; dst++; }
        uint32_t reused = source_end - src;
        /* Token-zero matches belong to exact prefix lookup, not a graft. */
        if (!dst || !ds4_suffix_even_shift(src, dst) || (!allow_right && dst > src) || reused < 64u ||
            reused <= plan->reused_tokens) continue;
        *plan = (ds4_suffix_match_plan) {
            .source_start = src, .destination_start = dst,
            .source_end = source_end, .destination_end = end,
            .reused_tokens = reused, .fresh_tokens = edited_len - end,
        };
    }
    return plan->reused_tokens != 0;
}

/* Existing callers keep left-shift admission unless they explicitly opt in. */
static inline bool ds4_suffix_match(const int *source, uint32_t source_end,
                                    const int *edited, uint32_t edited_len,
                                    uint32_t fresh_cutoff,
                                    ds4_suffix_match_plan *plan) {
    return ds4_suffix_match_shift(source, source_end, edited, edited_len,
                                 fresh_cutoff, false, plan);
}

#endif
