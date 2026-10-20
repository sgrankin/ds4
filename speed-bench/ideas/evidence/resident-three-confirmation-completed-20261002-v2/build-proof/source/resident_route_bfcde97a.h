#ifndef DS4_RESIDENT_ROUTE_H
#define DS4_RESIDENT_ROUTE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#define DS4_RESIDENT_ROUTE_PROFILE_LAYERS 40u

typedef struct {
    uint64_t calls, eligible, snapshots, plans, planned_substitutions, substitutions;
    double router_wait_ms, selected_read_ms, lookup_ms, guard_ms, snapshot_ms;
    double plan_ms, commit_ms, begin_load_ms;
} ds4_resident_route_layer_metrics;

typedef struct {
    bool enabled;
    ds4_resident_route_layer_metrics layer[DS4_RESIDENT_ROUTE_PROFILE_LAYERS];
} ds4_resident_route_metrics;

#ifdef __APPLE__
/* Main-thread snapshots only. No reset, GPU synchronization or per-token output.
 * Subtract snapshots outside timed phases; PROFILE clocks are attribution-only. */
void ds4_resident_route_metrics_read(ds4_resident_route_metrics *out);
#endif

typedef struct {
    uint32_t slot, old_expert, new_expert, candidate_rank;
    float old_mass, score_gap, affinity_ratio;
} ds4_resident_route_change;

typedef struct {
    uint32_t count;
    float total_mass;
    ds4_resident_route_change changes[2];
} ds4_resident_route_plan;

typedef struct {
    uint32_t slot, old_expert, alias_slot, alias_expert;
    float old_mass;
} ds4_resident_drop_change;

/* The caller can reject an ineligible native route before examining the cache. */
static inline bool ds4_resident_route_weakest(const int32_t *selected,
        const float *affinity, uint32_t total, uint32_t used, float max_mass,
        uint32_t *slot, float *mass) {
    if (!selected || !affinity || !slot || !mass || !total || used != 6 ||
        max_mass <= 0.0f) return false;
    uint32_t weak = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < used; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= total) return false;
        const float p = affinity[(uint32_t)selected[i]];
        if (!isfinite(p) || p <= 0.0f) return false;
        for (uint32_t j = 0; j < i; j++) if (selected[j] == selected[i]) return false;
        sum += p;
        if (p < affinity[(uint32_t)selected[weak]]) weak = i;
    }
    const uint32_t old = (uint32_t)selected[weak];
    const float old_mass = affinity[old] / sum;
    if (old_mass > max_mass) return false;
    *slot = weak;
    *mass = old_mass;
    return true;
}

/* Replace only the weakest unbiased contribution among the native six routes.
 * A score neighbor need not compute the same function: this is intentionally
 * lossy and its caller must measure end-to-end quality. The five strongest
 * contributions, the learned correction bias, and six unique routes survive. */
static inline bool ds4_resident_route_choose(const int32_t *selected,
        const float *affinity, const float *bias, const uint8_t *resident,
        uint32_t total, uint32_t used, float max_gap, uint32_t max_rank, float max_mass,
        ds4_resident_route_change *change) {
    if (!bias || !resident || !change || max_gap <= 0.0f || max_rank <= used) return false;
    uint32_t weak;
    float old_mass;
    if (!ds4_resident_route_weakest(selected, affinity, total, used, max_mass,
            &weak, &old_mass)) return false;
    const uint32_t old = (uint32_t)selected[weak];
    if (resident[old]) return false;
    const float old_p = affinity[old], old_score = old_p + bias[old];
    uint32_t candidate = total;
    float best_score = 0.0f;
    for (uint32_t expert = 0; expert < total; expert++) {
        if (!resident[expert]) continue;
        bool included = false;
        for (uint32_t i = 0; i < used; i++) included |= (uint32_t)selected[i] == expert;
        if (included) continue;
        const float p = affinity[expert], score = p + bias[expert];
        if (!isfinite(p) || !isfinite(score) || p <= 0.0f ||
            p < old_p * (1.0f - max_gap) ||
            p > old_p * (1.0f + max_gap) ||
            old_score - score > max_gap * old_p) continue;
        if (candidate == total || score > best_score) {
            candidate = expert;
            best_score = score;
        }
    }
    if (candidate == total) return false;
    uint32_t rank = 1;
    for (uint32_t expert = 0; expert < total; expert++) {
        const float score = affinity[expert] + bias[expert];
        if (score > best_score || (score == best_score && expert < candidate)) rank++;
    }
    if (rank > max_rank) return false;
    *change = (ds4_resident_route_change) {
        .slot = weak, .old_expert = old, .new_expert = candidate,
        .candidate_rank = rank, .old_mass = old_mass,
        .score_gap = fmaxf(0.0f, old_score - best_score) / old_p,
        .affinity_ratio = affinity[candidate] / old_p,
    };
    return true;
}

/* For the two-slot experiment, only the two weakest ORIGINAL native routes
 * are eligible. Replacement affinities never change the protected four. */
static inline uint32_t ds4_resident_route_weak_pair(const int32_t *selected,
        const float *affinity, uint32_t total, uint32_t used, float max_mass,
        uint32_t *slots, float *masses) {
    if (!slots || !masses || !ds4_resident_route_weakest(selected, affinity,
            total, used, max_mass, &slots[0], &masses[0])) return 0;
    float sum = 0.0f;
    uint32_t second = slots[0] == 0 ? 1 : 0;
    for (uint32_t i = 0; i < used; i++) {
        sum += affinity[(uint32_t)selected[i]];
        if (i != slots[0] && affinity[(uint32_t)selected[i]] <
                affinity[(uint32_t)selected[second]]) second = i;
    }
    slots[1] = second;
    masses[1] = affinity[(uint32_t)selected[second]] / sum;
    return masses[1] <= max_mass ? 2 : 1;
}

static inline bool ds4_resident_route_choose_plan(const int32_t *selected,
        const float *affinity, const float *bias, const uint8_t *resident,
        uint32_t total, uint32_t used, float max_gap, uint32_t max_rank,
        float max_mass, uint32_t max_subs, float max_total_mass,
        ds4_resident_route_plan *plan) {
    if (!plan) return false;
    *plan = (ds4_resident_route_plan){0};
    if (max_subs == 1) {
        /* Preserve the v3 selector and its exact arithmetic for the default. */
        if (!ds4_resident_route_choose(selected, affinity, bias, resident,
                total, used, max_gap, max_rank, max_mass, &plan->changes[0])) return false;
        plan->count = 1;
        plan->total_mass = plan->changes[0].old_mass;
        return true;
    }
    if (max_subs != 2 || !bias || !resident || max_gap <= 0.0f ||
        max_rank <= used || max_total_mass <= 0.0f) return false;
    uint32_t slots[2];
    float masses[2];
    const uint32_t n_weak = ds4_resident_route_weak_pair(selected, affinity,
        total, used, max_mass, slots, masses);
    for (uint32_t w = 0; w < n_weak; w++) {
        const uint32_t old = (uint32_t)selected[slots[w]];
        if (resident[old] || plan->total_mass + masses[w] > max_total_mass) continue;
        const float old_p = affinity[old], old_score = old_p + bias[old];
        uint32_t candidate = total;
        float best_score = 0.0f;
        for (uint32_t expert = 0; expert < total; expert++) {
            if (!resident[expert]) continue;
            bool included = false;
            for (uint32_t i = 0; i < used; i++) included |= (uint32_t)selected[i] == expert;
            for (uint32_t i = 0; i < plan->count; i++)
                included |= plan->changes[i].new_expert == expert;
            if (included) continue;
            const float p = affinity[expert], score = p + bias[expert];
            if (!isfinite(p) || !isfinite(score) || p <= 0.0f ||
                p < old_p * (1.0f - max_gap) || p > old_p * (1.0f + max_gap) ||
                old_score - score > max_gap * old_p) continue;
            if (candidate == total || score > best_score) {
                candidate = expert;
                best_score = score;
            }
        }
        if (candidate == total) continue;
        uint32_t rank = 1;
        for (uint32_t expert = 0; expert < total; expert++) {
            const float score = affinity[expert] + bias[expert];
            if (score > best_score || (score == best_score && expert < candidate)) rank++;
        }
        if (rank > max_rank) continue;
        plan->changes[plan->count++] = (ds4_resident_route_change) {
            .slot = slots[w], .old_expert = old, .new_expert = candidate,
            .candidate_rank = rank, .old_mass = masses[w],
            .score_gap = fmaxf(0.0f, old_score - best_score) / old_p,
            .affinity_ratio = affinity[candidate] / old_p,
        };
        plan->total_mass += masses[w];
    }
    return plan->count != 0;
}

/* A deliberately lossy bound: drop only the weakest ORIGINAL contribution if
 * missing, and alias its zero-weight slot to an already resident native ID.
 * The other five IDs and their original affinity ratios remain unchanged. */
static inline bool ds4_resident_route_choose_drop(const int32_t *selected,
        const float *affinity, const uint8_t *selected_resident,
        uint32_t total, uint32_t used, float max_mass,
        ds4_resident_drop_change *change) {
    if (!selected_resident || !change) return false;
    uint32_t weak;
    float old_mass;
    if (!ds4_resident_route_weakest(selected, affinity, total, used, max_mass,
            &weak, &old_mass) || selected_resident[weak]) return false;
    for (uint32_t slot = 0; slot < used; slot++) {
        if (slot == weak || !selected_resident[slot]) continue;
        *change = (ds4_resident_drop_change) {
            .slot = weak, .old_expert = (uint32_t)selected[weak],
            .alias_slot = slot, .alias_expert = (uint32_t)selected[slot],
            .old_mass = old_mass,
        };
        return true;
    }
    return false;
}

static inline void ds4_resident_route_drop_weights(float *weights, const int32_t *original,
        const float *affinity, uint32_t used, uint32_t dropped_slot, float scale) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < used; i++) if (i != dropped_slot)
        sum += affinity[(uint32_t)original[i]];
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t i = 0; i < used; i++)
        weights[i] = i == dropped_slot ? 0.0f : affinity[(uint32_t)original[i]] / sum * scale;
}

/* The cache prior affects selection only. Mixture coefficients always use the
 * unmodified sqrt(softplus(logit)) affinities of the final chosen experts. */
static inline void ds4_resident_route_weights(float *weights, const int32_t *ids,
        const float *affinity, uint32_t used, float scale) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < used; i++) sum += affinity[(uint32_t)ids[i]];
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t i = 0; i < used; i++)
        weights[i] = affinity[(uint32_t)ids[i]] / sum * scale;
}

#endif
