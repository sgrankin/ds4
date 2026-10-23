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
    ds4_resident_route_change changes[3];
} ds4_resident_route_plan;

typedef struct {
    uint32_t slot, old_expert, alias_slot, alias_expert;
    float old_mass;
} ds4_resident_drop_change;

/* Native bodies use these exact shared records. Cache ownership, snapshots,
 * commit timing and metrics remain with the existing C engine. */
bool ds4_resident_route_weakest(const int32_t *selected,
        const float *affinity, uint32_t total, uint32_t used, float max_mass,
        uint32_t *slot, float *mass);
bool ds4_resident_route_choose(const int32_t *selected,
        const float *affinity, const float *bias, const uint8_t *resident,
        uint32_t total, uint32_t used, float max_gap, uint32_t max_rank, float max_mass,
        ds4_resident_route_change *change);
uint32_t ds4_resident_route_weak_pair(const int32_t *selected,
        const float *affinity, uint32_t total, uint32_t used, float max_mass,
        uint32_t *slots, float *masses);
uint32_t ds4_resident_route_weak_triple(const int32_t *selected,
        const float *affinity, uint32_t total, uint32_t used, float max_mass,
        uint32_t *slots, float *masses);
bool ds4_resident_route_choose_triple(const int32_t *selected,
        const float *affinity, const float *bias, const uint8_t *resident,
        uint32_t total, uint32_t used, float max_gap, uint32_t max_rank,
        float max_mass, float max_total_mass, ds4_resident_route_plan *plan);
bool ds4_resident_route_choose_plan(const int32_t *selected,
        const float *affinity, const float *bias, const uint8_t *resident,
        uint32_t total, uint32_t used, float max_gap, uint32_t max_rank,
        float max_mass, uint32_t max_subs, float max_total_mass,
        ds4_resident_route_plan *plan);
bool ds4_resident_route_choose_drop(const int32_t *selected,
        const float *affinity, const uint8_t *selected_resident,
        uint32_t total, uint32_t used, float max_mass,
        ds4_resident_drop_change *change);
void ds4_resident_route_drop_weights(float *weights, const int32_t *original,
        const float *affinity, uint32_t used, uint32_t dropped_slot, float scale);
void ds4_resident_route_weights(float *weights, const int32_t *ids,
        const float *affinity, uint32_t used, float scale);

#endif
