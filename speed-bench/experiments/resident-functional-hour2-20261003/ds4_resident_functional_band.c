#include "ds4_resident_functional.h"
#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "functional map requires IEEE FP32");
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53,
               "functional feature arithmetic requires IEEE FP64");

/* File metadata has variable length: map payloads may be unaligned. Read
 * bytes without typed pointer casts or a duplicate decoded payload. */
static float fm_read_f32_le(const uint8_t *p) {
    const uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float x; memcpy(&x, &bits, sizeof x); return x;
}

static uint16_t fm_read_f16_le(const uint8_t *p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

static int fm_finite32(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof bits);
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int fm_finite64(double x) {
    uint64_t bits; memcpy(&bits, &x, sizeof bits);
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

static int fm_half(uint16_t bits, double *out) {
    const uint32_t exponent = (bits >> 10) & 31u;
    const uint32_t mantissa = bits & 1023u;
    if (exponent == 31u) return 0;
    double x = exponent ? ldexp((double)(1024u + mantissa), (int)exponent - 25) :
                          ldexp((double)mantissa, -24);
    *out = bits & 32768u ? -x : x;
    return 1;
}

int ds4_fm_flag(const char *value) {
    if (!value || !strcmp(value, "0")) return 0;
    return !strcmp(value, "1") ? 1 : -1;
}

int ds4_fm_rank_for_layer(const ds4_fm_layer_view *v,
        const float input[DS4_FM_INPUT], const int32_t original[6],
        uint32_t weak_slot, float original_mass,
        const float affinity[DS4_FM_EXPERTS], const float bias[DS4_FM_EXPERTS],
        const uint8_t resident[DS4_FM_EXPERTS], ds4_fm_choice *choice) {
    if (!v || !v->probes_le || v->probe_bytes != (uint64_t)DS4_FM_PROBES * DS4_FM_INPUT * 4u ||
        !v->probe_norms || !v->signatures_le ||
        v->signature_bytes != (uint64_t)DS4_FM_EXPERTS * DS4_FM_PROBES * DS4_FM_FEATURES * 2u || !input || !original ||
        !affinity || !bias || !resident || !choice || weak_slot >= 6u ||
        !fm_finite32(original_mass) || original_mass <= 0.0f || original_mass > .12f) return -1;
    for (uint32_t i = 0; i < 6u; i++) {
        if (original[i] < 0 || (uint32_t)original[i] >= DS4_FM_EXPERTS ||
            !fm_finite32(affinity[original[i]]) || affinity[original[i]] <= 0.0f) return -1;
        for (uint32_t j = 0; j < i; j++) if (original[i] == original[j]) return -1;
    }
    const uint32_t old = (uint32_t)original[weak_slot];
    for (uint32_t i = 0; i < 6u; i++) {
        if (affinity[original[i]] < affinity[old] ||
            (affinity[original[i]] == affinity[old] && i < weak_slot)) return -1;
    }
    if (resident[old]) return 0;
    float score[DS4_FM_EXPERTS];
    for (uint32_t e = 0; e < DS4_FM_EXPERTS; e++) {
        score[e] = affinity[e] + bias[e];
        if (!fm_finite32(affinity[e]) || !fm_finite32(bias[e]) || !fm_finite32(score[e])) return -1;
    }
    double norm2 = 0.0;
    for (uint32_t d = 0; d < DS4_FM_INPUT; d++) {
        if (!fm_finite32(input[d])) return -1;
        norm2 += (double)input[d] * (double)input[d];
    }
    if (!fm_finite64(norm2)) return -1;
    if (norm2 == 0.0) return 0;
    const double norm = sqrt(norm2);
    uint32_t probe = 0;
    double best_cosine = -2.0;
    for (uint32_t p = 0; p < DS4_FM_PROBES; p++) {
        if (!fm_finite64(v->probe_norms[p]) || v->probe_norms[p] <= 0.0) return -1;
        double dot = 0.0;
        for (uint32_t d = 0; d < DS4_FM_INPUT; d++) {
            const float q = fm_read_f32_le(v->probes_le + (p * DS4_FM_INPUT + d) * 4u);
            if (!fm_finite32(q)) return -1;
            dot += (double)input[d] * (double)q;
        }
        const double cosine = dot / (norm * v->probe_norms[p]);
        if (!fm_finite64(cosine)) return -1;
        if (cosine > best_cosine) { best_cosine = cosine; probe = p; }
    }
    const float old_p = affinity[old];
#ifdef DS4_FM_ROUTER_BAND
    /* Router-score slack is measured in units of the replaced original's
     * unbiased affinity. Choose the functional closest expert only inside
     * this band; the caller still preserves the six native GPU weight bits.
     * The highest-score eligible resident bounds every lower candidate, so
     * the prepass needs no global-rank scan. The final rank guard is unchanged. */
    float best_score = -FLT_MAX;
    for (uint32_t e = 0; e < DS4_FM_EXPERTS; e++) {
        if (!resident[e]) continue;
        int included = 0;
        for (uint32_t i = 0; i < 6u; i++) included |= original[i] == (int32_t)e;
        if (included) continue;
        const float p = affinity[e];
        if (p <= 0.0f || p < old_p * (1.0f - .30f) || p > old_p * (1.0f + .30f) ||
            score[old] - score[e] > .30f * old_p) continue;
        if (score[e] > best_score) best_score = score[e];
    }
#endif
    uint32_t candidate = DS4_FM_EXPERTS, best_rank = 0;
    double best_distance = 0.0;
    for (uint32_t e = 0; e < DS4_FM_EXPERTS; e++) {
        if (!resident[e]) continue;
        int included = 0;
        for (uint32_t i = 0; i < 6u; i++) included |= original[i] == (int32_t)e;
        if (included) continue;
        const float p = affinity[e];
        if (p <= 0.0f || p < old_p * (1.0f - .30f) || p > old_p * (1.0f + .30f) ||
            score[old] - score[e] > .30f * old_p) continue;
        uint32_t rank = 1;
        for (uint32_t k = 0; k < DS4_FM_EXPERTS; k++)
            rank += score[k] > score[e] || (score[k] == score[e] && k < e);
        if (rank > 24u) continue;
#ifdef DS4_FM_ROUTER_BAND
        if ((double)best_score - (double)score[e] >
            (double)DS4_FM_ROUTER_BAND * (double)old_p) continue;
#endif
        double distance = 0.0;
        for (uint32_t k = 0; k < DS4_FM_FEATURES; k++) {
            double x, y;
            const uint32_t at = (e * DS4_FM_PROBES + probe) * DS4_FM_FEATURES + k;
            const uint32_t old_at = (old * DS4_FM_PROBES + probe) * DS4_FM_FEATURES + k;
            if (!fm_half(fm_read_f16_le(v->signatures_le + at * 2u), &x) ||
                !fm_half(fm_read_f16_le(v->signatures_le + old_at * 2u), &y)) return -1;
            const double delta = x - y;
            distance += delta * delta;
        }
        if (!fm_finite64(distance)) return -1;
        if (candidate == DS4_FM_EXPERTS || distance < best_distance) {
            candidate = e; best_rank = rank; best_distance = distance;
        }
    }
    if (candidate == DS4_FM_EXPERTS) return 0;
    *choice = (ds4_fm_choice){weak_slot, old, candidate, best_rank, probe,
        original_mass, best_cosine, best_distance};
    return 1;
}
