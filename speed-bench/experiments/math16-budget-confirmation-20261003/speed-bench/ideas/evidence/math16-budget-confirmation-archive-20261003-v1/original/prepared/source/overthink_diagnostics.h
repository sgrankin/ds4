#ifndef DS4_OVERTHINK_DIAGNOSTICS_H
#define DS4_OVERTHINK_DIAGNOSTICS_H
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Stable before-filtering softmax mass. Fast-math builds still reject nonfinite
 * values by their representation rather than compiler-assumed isfinite(). */
static int overthink_finite(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int overthink_is_marker(int token, const int *ids, int count) {
    for (int i = 0; i < count; i++) if (ids[i] == token) return 1;
    return 0;
}

static double overthink_marker_mass(const float *logits, int vocab,
                                    const int *ids, int count, double temperature) {
    if (!(temperature > 0.0) || vocab < 1) return -1.0;
    double max = -INFINITY, total = 0.0, marked = 0.0;
    int any_finite = 0;
    for (int i = 0; i < vocab; i++)
        if (overthink_finite(logits[i])) {
            any_finite = 1;
            if (logits[i] > max) max = logits[i];
        }
    if (!any_finite) return -1.0;
    for (int i = 0; i < vocab; i++)
        if (overthink_finite(logits[i])) total += exp(((double)logits[i] - max) / temperature);
    for (int i = 0; i < count; i++) {
        const int id = ids[i];
        if (id < 0 || id >= vocab || !overthink_finite(logits[id])) return -1.0;
        marked += exp(((double)logits[id] - max) / temperature);
    }
    return marked / total;
}

/* P and Q share one forward-pass prefix. These formulas are pre-filtering,
 * temperature-scaled KL(P||Q) and change in selected-token NLL. They do not
 * estimate post-top-p probabilities or a corpus reference loss. */
static void overthink_policy_metrics(double mass, double effective_penalty,
                                     int selected_marker, double *kl, double *delta_nll) {
    const double log_z = log1p(mass * expm1(-effective_penalty));
    *kl = effective_penalty * mass + log_z;
    if (*kl < 0.0 && *kl > -1e-12) *kl = 0.0;
    *delta_nll = effective_penalty * selected_marker + log_z;
}
#endif
