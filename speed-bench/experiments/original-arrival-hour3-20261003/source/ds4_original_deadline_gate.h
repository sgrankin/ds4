#ifndef DS4_ORIGINAL_DEADLINE_GATE_H
#define DS4_ORIGINAL_DEADLINE_GATE_H
#include <string.h>
/* Deliberately fixed first experiment, including prefill validation. This
 * parser is independent of fast-math floating comparisons. Mode is still
 * interpreted by the literal parent resident configuration. */
static inline int ds4_od_flag(const char *s, int *enabled) {
    if (!enabled) return 0;
    if (!s || !strcmp(s,"0")) { *enabled=0; return 1; }
    if (!strcmp(s,"1")) { *enabled=1; return 1; }
    return 0;
}
static inline int ds4_od_fixed_policy(const char *gap,const char *mass,
        const char *rank,const char *subs,const char *weights,const char *policy) {
    return gap && (!strcmp(gap,".30") || !strcmp(gap,"0.30")) &&
        mass && (!strcmp(mass,".12") || !strcmp(mass,"0.12")) &&
        rank && !strcmp(rank,"24") && subs && !strcmp(subs,"1") &&
        weights && !strcmp(weights,"1") && policy && !strcmp(policy,"replace");
}
#endif
