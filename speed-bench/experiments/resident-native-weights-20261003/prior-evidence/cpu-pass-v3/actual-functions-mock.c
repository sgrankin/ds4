
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_resident_route.h"
#include "ds4_resident_native_weights.h"
#define DS4_MAX_EXPERT 384
#define DS4_N_EXPERT 384
#define DS4_MAX_EXPERT_USED 6
#define DS4_N_EXPERT_USED 6
#define DS4_EXPERT_WEIGHT_SCALE 1.5f
enum {DS41_RESIDENT_ROUTE_APPLY,DS41_RESIDENT_ROUTE_NATIVE,DS41_RESIDENT_ROUTE_LOOKUP,DS41_RESIDENT_ROUTE_NOOP};
typedef struct {union {float values[384];unsigned char bytes[384*sizeof(float)];};int kind;} ds4_gpu_tensor;
typedef struct {float bias[384];} ds4_model;
typedef struct {const void *ffn_exp_probs_b;} ds4_layer_weights;
typedef struct {uint64_t gate_expert_bytes,down_expert_bytes;} ds4_gpu_stream_expert_table;
typedef struct {
    float resident_route_gap,resident_route_max_mass,resident_route_max_total_mass;
    uint32_t resident_route_max_rank,resident_route_max_subs,resident_route_mode;
    bool resident_route_profile,resident_route_drop_weak,resident_route_native_weights;
    bool streaming,quality,imatrix;uint32_t image_count,tp_world;
    ds4_gpu_tensor *route_probs,*selected,*route_weights;
} ds41_gpu_graph;
static ds4_resident_route_metrics ds41_resident_metrics;
static volatile float ds41_resident_noop_weights[6];
static struct {bool active;} ds41_route_oracle;
static struct {
    bool registered;uint64_t layers,substitutions,native_misses,skipped_loads,skipped_bytes,full_snapshots;
    uint64_t double_substitution_layers,triple_substitution_layers,planned_substitutions,drops;
    bool native_weights_requested;uint64_t native_weight_substitutions;
    double replaced_mass,score_gap,affinity_ratio,rank;uint64_t per_layer[40];
} ds41_resident_route_stats;
static struct {const char *key,*value;} envs[16];static unsigned env_count;
static jmp_buf fatal;static unsigned selected_writes,weight_writes,contents_reads,selected_snapshots,full_snapshots;
static bool fail_selected_write;static uint8_t resident[384];
static const char *mock_getenv(const char *key){for(unsigned i=0;i<env_count;i++)if(!strcmp(key,envs[i].key))return envs[i].value;return NULL;}
static void option(const char *key,const char *value){envs[env_count].key=key;envs[env_count++].value=value;}
static void ds4_die(const char *message){(void)message;longjmp(fatal,1);}
static bool ds41_resident_profile_enabled(void){return false;}
static void ds41_resident_route_report(void){}
static int mock_atexit(void (*fn)(void)){(void)fn;return 0;}
static double now_sec(void){return 0;}
static const void *ds4_gpu_tensor_contents(const ds4_gpu_tensor *t){assert(t->kind==1);contents_reads++;return t->values;}
static const float *tensor_data(const ds4_model *m,const void *t){(void)t;return m->bias;}
static bool ds4_gpu_stream_expert_cache_snapshot_selected(const ds4_gpu_stream_expert_table *t,const int32_t *ids,uint32_t n,uint8_t *out){
    (void)t;assert(n==6);selected_snapshots++;for(unsigned i=0;i<n;i++)out[i]=resident[ids[i]];return true;
}
static bool ds4_gpu_stream_expert_cache_snapshot(const ds4_gpu_stream_expert_table *t,uint8_t *out){(void)t;full_snapshots++;memcpy(out,resident,sizeof(resident));return true;}
static bool ds4_gpu_tensor_write(ds4_gpu_tensor *t,uint64_t off,const void *src,uint64_t n){
    assert(!off && n==6*sizeof(float));
    if(t->kind==2){selected_writes++;if(fail_selected_write)return false;}
    else {assert(t->kind==3);weight_writes++;}
    memcpy(t->bytes,src,(size_t)n);return true;
}
#define getenv mock_getenv
#define atexit mock_atexit

static void old_configure(ds41_gpu_graph *g, bool decode) {
    g->resident_route_gap = 0.0f;
    g->resident_route_max_rank = 12;
    g->resident_route_max_mass = 1.0f / DS4_N_EXPERT_USED;
    g->resident_route_max_subs = 1;
    g->resident_route_max_total_mass = 0.20f;
    g->resident_route_mode = DS41_RESIDENT_ROUTE_APPLY;
    g->resident_route_profile = decode && ds41_resident_profile_enabled();
    g->resident_route_drop_weak = false;
    const char *value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP");
    if (!decode || !value || !*value) return;
    char *end = NULL;
    const float gap = strtof(value, &end);
    if (end == value || *end || !isfinite(gap) || gap < 0.0f || gap > 1.0f)
        ds4_die("resident route gap must be in [0,1]");
    if (gap == 0.0f) return; /* Mode-off arithmetic and execution remain native. */
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MODE");
    if (value && *value) {
        if (!strcmp(value, "native")) g->resident_route_mode = DS41_RESIDENT_ROUTE_NATIVE;
        else if (!strcmp(value, "lookup")) g->resident_route_mode = DS41_RESIDENT_ROUTE_LOOKUP;
        else if (!strcmp(value, "noop")) g->resident_route_mode = DS41_RESIDENT_ROUTE_NOOP;
        else if (strcmp(value, "apply"))
            ds4_die("resident route mode must be native, lookup, noop or apply");
    }
    if (!g->streaming || g->quality || g->imatrix || g->image_count || g->tp_world != 1 ||
        getenv("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") ||
        getenv("DS4_METAL_V41_ASYNC_EXPERT_LOAD") ||
        getenv("DS4_METAL_V41_GATE_PREFETCH") || ds41_route_oracle.active ||
        getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD"))
        ds4_die("resident routing requires ordinary scalar text SSD early loading");
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_POLICY");
    if (value && *value) {
        if (!strcmp(value, "drop")) {
            g->resident_route_drop_weak = true;
            g->resident_route_max_mass = 0.10f;
        } else if (strcmp(value, "replace"))
            ds4_die("resident route policy must be replace or drop");
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK");
    if (value && *value) {
        const unsigned long rank = strtoul(value, &end, 10);
        if (end == value || *end || rank < 7 || rank > DS4_N_EXPERT)
            ds4_die("resident route max rank must be in [7,384]");
        g->resident_route_max_rank = (uint32_t)rank;
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS");
    if (value && *value) {
        const float mass = strtof(value, &end);
        if (end == value || *end || !isfinite(mass) || mass < 0.0f || mass > 1.0f)
            ds4_die("resident route max mass must be in [0,1]");
        g->resident_route_max_mass = mass;
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS");
    if (value && *value) {
        const unsigned long count = strtoul(value, &end, 10);
        if (end == value || *end || count < 1 || count > 3)
            ds4_die("resident route max substitutions must be 1, 2 or 3");
        g->resident_route_max_subs = (uint32_t)count;
    }
    if (g->resident_route_drop_weak && g->resident_route_max_subs != 1)
        ds4_die("resident route drop policy permits one original weak miss only");
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS");
    if (g->resident_route_max_subs >= 2 && value && *value) {
        const float mass = strtof(value, &end);
        if (end == value || *end || !isfinite(mass) || mass < 0.0f || mass > 1.0f)
            ds4_die("resident route maximum total mass must be in [0,1]");
        g->resident_route_max_total_mass = mass;
    }
    g->resident_route_gap = g->resident_route_mode == DS41_RESIDENT_ROUTE_NATIVE ? 0.0f : gap;
    if (!ds41_resident_route_stats.registered) {
        ds41_resident_route_stats.registered = true;
        atexit(ds41_resident_route_report);
    }
}

static bool old_apply(ds41_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il,
        const ds4_gpu_stream_expert_table *table, int32_t *selected) {
    if (g->resident_route_gap <= 0.0f) return true;
    ds4_resident_route_layer_metrics *profile = g->resident_route_profile ?
        &ds41_resident_metrics.layer[il] : NULL;
    double t0 = profile ? now_sec() * 1000.0 : 0.0;
    uint8_t resident[DS4_MAX_EXPERT];
    uint8_t selected_resident[DS4_MAX_EXPERT_USED];
    const float *affinity = ds4_gpu_tensor_contents(g->route_probs);
    const float *bias = tensor_data(m, l->ffn_exp_probs_b);
    if (!affinity || !bias || !ds4_gpu_stream_expert_cache_snapshot_selected(
            table, selected, DS4_N_EXPERT_USED, selected_resident)) return false;
    if (profile) profile->lookup_ms += now_sec() * 1000.0 - t0;
    ds41_resident_route_stats.layers++;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
        ds41_resident_route_stats.native_misses += !selected_resident[i];
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    bool eligible = false;
    ds4_resident_drop_change drop;
    if (g->resident_route_drop_weak) {
        eligible = ds4_resident_route_choose_drop(selected, affinity, selected_resident,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, &drop);
    } else if (g->resident_route_max_subs == 1) {
        uint32_t weak;
        float old_mass;
        eligible = ds4_resident_route_weakest(selected, affinity, DS4_N_EXPERT, DS4_N_EXPERT_USED,
                g->resident_route_max_mass, &weak, &old_mass) && !selected_resident[weak];
    } else if (g->resident_route_max_subs == 2) {
        uint32_t slots[2];
        float masses[2];
        const uint32_t n_weak = ds4_resident_route_weak_pair(selected, affinity,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, slots, masses);
        for (uint32_t i = 0; i < n_weak; i++)
            eligible |= !selected_resident[slots[i]] && masses[i] <= g->resident_route_max_total_mass;
    } else {
        uint32_t slots[3];
        float masses[3];
        const uint32_t n_weak = ds4_resident_route_weak_triple(selected, affinity,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, slots, masses);
        for (uint32_t i = 0; i < n_weak; i++)
            eligible |= !selected_resident[slots[i]] && masses[i] <= g->resident_route_max_total_mass;
    }
    if (profile) {
        profile->guard_ms += now_sec() * 1000.0 - t0;
        profile->eligible += eligible;
    }
    if (!eligible || g->resident_route_mode == DS41_RESIDENT_ROUTE_LOOKUP) return true;
    if (!g->resident_route_drop_weak) {
        t0 = profile ? now_sec() * 1000.0 : 0.0;
        if (!ds4_gpu_stream_expert_cache_snapshot(table, resident)) return false;
        if (profile) {
            profile->snapshot_ms += now_sec() * 1000.0 - t0;
            profile->snapshots++;
        }
        ds41_resident_route_stats.full_snapshots++;
    }
    ds4_resident_route_plan plan;
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    bool planned;
    if (g->resident_route_drop_weak) {
        plan = (ds4_resident_route_plan) {
            .count = 1, .total_mass = drop.old_mass,
            .changes = {{.slot = drop.slot, .old_expert = drop.old_expert,
                .new_expert = drop.alias_expert, .old_mass = drop.old_mass}},
        };
        planned = true;
    } else planned = ds4_resident_route_choose_plan(selected, affinity, bias, resident,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_gap,
            g->resident_route_max_rank, g->resident_route_max_mass,
            g->resident_route_max_subs, g->resident_route_max_total_mass, &plan);
    if (profile) {
        profile->plan_ms += now_sec() * 1000.0 - t0;
        profile->plans += planned;
        profile->planned_substitutions += plan.count;
    }
    if (!planned) return true;
    ds41_resident_route_stats.planned_substitutions += plan.count;
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    float weights[DS4_MAX_EXPERT_USED];
    int32_t final_ids[DS4_MAX_EXPERT_USED];
    memcpy(final_ids, selected, DS4_N_EXPERT_USED * sizeof(*selected));
    for (uint32_t i = 0; i < plan.count; i++)
        final_ids[plan.changes[i].slot] = (int32_t)plan.changes[i].new_expert;
    if (g->resident_route_drop_weak)
        ds4_resident_route_drop_weights(weights, selected, affinity, DS4_N_EXPERT_USED,
            drop.slot, DS4_EXPERT_WEIGHT_SCALE);
    else ds4_resident_route_weights(weights, final_ids, affinity, DS4_N_EXPERT_USED,
            DS4_EXPERT_WEIGHT_SCALE);
    /* Plan-noop computes the same proposal and normalization, while preserving
     * native GPU-produced IDs/weights and the exact loader/cache behavior. */
    if (g->resident_route_mode == DS41_RESIDENT_ROUTE_NOOP) {
        /* Keep all coefficient arithmetic observable to the optimizer without
         * modifying GPU tensors or cache state in the exact native control. */
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
            ds41_resident_noop_weights[i] = weights[i];
        if (profile) profile->commit_ms += now_sec() * 1000.0 - t0;
        return true;
    }
    memcpy(selected, final_ids, DS4_N_EXPERT_USED * sizeof(*selected));
    if (!ds4_gpu_tensor_write(g->selected, 0, selected, DS4_N_EXPERT_USED * sizeof(*selected)) ||
        !ds4_gpu_tensor_write(g->route_weights, 0, weights, DS4_N_EXPERT_USED * sizeof(*weights)))
        return false;
    if (profile) {
        profile->commit_ms += now_sec() * 1000.0 - t0;
        profile->substitutions += plan.count;
    }
    ds41_resident_route_stats.substitutions += plan.count;
    ds41_resident_route_stats.per_layer[il] += plan.count;
    ds41_resident_route_stats.double_substitution_layers += plan.count == 2;
    ds41_resident_route_stats.triple_substitution_layers += plan.count == 3;
    ds41_resident_route_stats.drops += g->resident_route_drop_weak;
    ds41_resident_route_stats.skipped_loads += plan.count;
    ds41_resident_route_stats.skipped_bytes += plan.count * (2 * table->gate_expert_bytes + table->down_expert_bytes);
    for (uint32_t i = 0; i < plan.count; i++) {
        ds41_resident_route_stats.replaced_mass += plan.changes[i].old_mass;
        ds41_resident_route_stats.score_gap += plan.changes[i].score_gap;
        ds41_resident_route_stats.affinity_ratio += plan.changes[i].affinity_ratio;
        ds41_resident_route_stats.rank += plan.changes[i].candidate_rank;
    }
    return true;
}

static void ds41_resident_route_configure(ds41_gpu_graph *g, bool decode) {
    g->resident_route_gap = 0.0f;
    g->resident_route_max_rank = 12;
    g->resident_route_max_mass = 1.0f / DS4_N_EXPERT_USED;
    g->resident_route_max_subs = 1;
    g->resident_route_max_total_mass = 0.20f;
    g->resident_route_mode = DS41_RESIDENT_ROUTE_APPLY;
    g->resident_route_profile = decode && ds41_resident_profile_enabled();
    g->resident_route_drop_weak = false;
    g->resident_route_native_weights = false;
    bool native_weights_requested;
    if (!ds4_resident_native_weights_parse(
            getenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS"), &native_weights_requested))
        ds4_die("resident route native weights must be literal 0 or 1");
    const char *value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP");
    if (native_weights_requested) {
        const char *policy = getenv("DS4_METAL_V41_RESIDENT_ROUTE_POLICY");
        if (policy && !strcmp(policy, "drop"))
            ds4_die("resident route native weights are incompatible with drop policy");
        if (!ds4_resident_native_weights_gap(value))
            ds4_die("resident route native weights require a positive configured gap");
    }
    if (!decode || !value || !*value) return;
    char *end = NULL;
    const float gap = strtof(value, &end);
    if (end == value || *end || !isfinite(gap) || gap < 0.0f || gap > 1.0f)
        ds4_die("resident route gap must be in [0,1]");
    if (gap == 0.0f) {
        if (native_weights_requested)
            ds4_die("resident route native weights require a positive configured gap");
        return; /* Mode-off arithmetic and execution remain native. */
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MODE");
    if (value && *value) {
        if (!strcmp(value, "native")) g->resident_route_mode = DS41_RESIDENT_ROUTE_NATIVE;
        else if (!strcmp(value, "lookup")) g->resident_route_mode = DS41_RESIDENT_ROUTE_LOOKUP;
        else if (!strcmp(value, "noop")) g->resident_route_mode = DS41_RESIDENT_ROUTE_NOOP;
        else if (strcmp(value, "apply"))
            ds4_die("resident route mode must be native, lookup, noop or apply");
    }
    if (!g->streaming || g->quality || g->imatrix || g->image_count || g->tp_world != 1 ||
        getenv("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") ||
        getenv("DS4_METAL_V41_ASYNC_EXPERT_LOAD") ||
        getenv("DS4_METAL_V41_GATE_PREFETCH") || ds41_route_oracle.active ||
        getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD"))
        ds4_die("resident routing requires ordinary scalar text SSD early loading");
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_POLICY");
    if (value && *value) {
        if (!strcmp(value, "drop")) {
            g->resident_route_drop_weak = true;
            g->resident_route_max_mass = 0.10f;
        } else if (strcmp(value, "replace"))
            ds4_die("resident route policy must be replace or drop");
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK");
    if (value && *value) {
        const unsigned long rank = strtoul(value, &end, 10);
        if (end == value || *end || rank < 7 || rank > DS4_N_EXPERT)
            ds4_die("resident route max rank must be in [7,384]");
        g->resident_route_max_rank = (uint32_t)rank;
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS");
    if (value && *value) {
        const float mass = strtof(value, &end);
        if (end == value || *end || !isfinite(mass) || mass < 0.0f || mass > 1.0f)
            ds4_die("resident route max mass must be in [0,1]");
        g->resident_route_max_mass = mass;
    }
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS");
    if (value && *value) {
        const unsigned long count = strtoul(value, &end, 10);
        if (end == value || *end || count < 1 || count > 3)
            ds4_die("resident route max substitutions must be 1, 2 or 3");
        g->resident_route_max_subs = (uint32_t)count;
    }
    if (g->resident_route_drop_weak && g->resident_route_max_subs != 1)
        ds4_die("resident route drop policy permits one original weak miss only");
    value = getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS");
    if (g->resident_route_max_subs >= 2 && value && *value) {
        const float mass = strtof(value, &end);
        if (end == value || *end || !isfinite(mass) || mass < 0.0f || mass > 1.0f)
            ds4_die("resident route maximum total mass must be in [0,1]");
        g->resident_route_max_total_mass = mass;
    }
    g->resident_route_gap = g->resident_route_mode == DS41_RESIDENT_ROUTE_NATIVE ? 0.0f : gap;
    g->resident_route_native_weights = native_weights_requested;
    ds41_resident_route_stats.native_weights_requested |= native_weights_requested;
    if (!ds41_resident_route_stats.registered) {
        ds41_resident_route_stats.registered = true;
        atexit(ds41_resident_route_report);
    }
}

static bool ds41_resident_route_apply(ds41_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il,
        const ds4_gpu_stream_expert_table *table, int32_t *selected) {
    if (g->resident_route_gap <= 0.0f) return true;
    ds4_resident_route_layer_metrics *profile = g->resident_route_profile ?
        &ds41_resident_metrics.layer[il] : NULL;
    double t0 = profile ? now_sec() * 1000.0 : 0.0;
    uint8_t resident[DS4_MAX_EXPERT];
    uint8_t selected_resident[DS4_MAX_EXPERT_USED];
    const float *affinity = ds4_gpu_tensor_contents(g->route_probs);
    const float *bias = tensor_data(m, l->ffn_exp_probs_b);
    if (!affinity || !bias || !ds4_gpu_stream_expert_cache_snapshot_selected(
            table, selected, DS4_N_EXPERT_USED, selected_resident)) return false;
    if (profile) profile->lookup_ms += now_sec() * 1000.0 - t0;
    ds41_resident_route_stats.layers++;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
        ds41_resident_route_stats.native_misses += !selected_resident[i];
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    bool eligible = false;
    ds4_resident_drop_change drop;
    if (g->resident_route_drop_weak) {
        eligible = ds4_resident_route_choose_drop(selected, affinity, selected_resident,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, &drop);
    } else if (g->resident_route_max_subs == 1) {
        uint32_t weak;
        float old_mass;
        eligible = ds4_resident_route_weakest(selected, affinity, DS4_N_EXPERT, DS4_N_EXPERT_USED,
                g->resident_route_max_mass, &weak, &old_mass) && !selected_resident[weak];
    } else if (g->resident_route_max_subs == 2) {
        uint32_t slots[2];
        float masses[2];
        const uint32_t n_weak = ds4_resident_route_weak_pair(selected, affinity,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, slots, masses);
        for (uint32_t i = 0; i < n_weak; i++)
            eligible |= !selected_resident[slots[i]] && masses[i] <= g->resident_route_max_total_mass;
    } else {
        uint32_t slots[3];
        float masses[3];
        const uint32_t n_weak = ds4_resident_route_weak_triple(selected, affinity,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, slots, masses);
        for (uint32_t i = 0; i < n_weak; i++)
            eligible |= !selected_resident[slots[i]] && masses[i] <= g->resident_route_max_total_mass;
    }
    if (profile) {
        profile->guard_ms += now_sec() * 1000.0 - t0;
        profile->eligible += eligible;
    }
    if (!eligible || g->resident_route_mode == DS41_RESIDENT_ROUTE_LOOKUP) return true;
    if (!g->resident_route_drop_weak) {
        t0 = profile ? now_sec() * 1000.0 : 0.0;
        if (!ds4_gpu_stream_expert_cache_snapshot(table, resident)) return false;
        if (profile) {
            profile->snapshot_ms += now_sec() * 1000.0 - t0;
            profile->snapshots++;
        }
        ds41_resident_route_stats.full_snapshots++;
    }
    ds4_resident_route_plan plan;
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    bool planned;
    if (g->resident_route_drop_weak) {
        plan = (ds4_resident_route_plan) {
            .count = 1, .total_mass = drop.old_mass,
            .changes = {{.slot = drop.slot, .old_expert = drop.old_expert,
                .new_expert = drop.alias_expert, .old_mass = drop.old_mass}},
        };
        planned = true;
    } else planned = ds4_resident_route_choose_plan(selected, affinity, bias, resident,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_gap,
            g->resident_route_max_rank, g->resident_route_max_mass,
            g->resident_route_max_subs, g->resident_route_max_total_mass, &plan);
    if (profile) {
        profile->plan_ms += now_sec() * 1000.0 - t0;
        profile->plans += planned;
        profile->planned_substitutions += plan.count;
    }
    if (!planned) return true;
    ds41_resident_route_stats.planned_substitutions += plan.count;
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    float weights[DS4_MAX_EXPERT_USED];
    int32_t final_ids[DS4_MAX_EXPERT_USED];
    memcpy(final_ids, selected, DS4_N_EXPERT_USED * sizeof(*selected));
    for (uint32_t i = 0; i < plan.count; i++)
        final_ids[plan.changes[i].slot] = (int32_t)plan.changes[i].new_expert;
    if (g->resident_route_drop_weak)
        ds4_resident_route_drop_weights(weights, selected, affinity, DS4_N_EXPERT_USED,
            drop.slot, DS4_EXPERT_WEIGHT_SCALE);
    else ds4_resident_route_weights(weights, final_ids, affinity, DS4_N_EXPERT_USED,
            DS4_EXPERT_WEIGHT_SCALE);
    /* Plan-noop computes the same proposal and normalization, while preserving
     * native GPU-produced IDs/weights and the exact loader/cache behavior. */
    if (g->resident_route_mode == DS41_RESIDENT_ROUTE_NOOP) {
        /* Keep all coefficient arithmetic observable to the optimizer without
         * modifying GPU tensors or cache state in the exact native control. */
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
            ds41_resident_noop_weights[i] = weights[i];
        if (profile) profile->commit_ms += now_sec() * 1000.0 - t0;
        return true;
    }
    memcpy(selected, final_ids, DS4_N_EXPERT_USED * sizeof(*selected));
    if (!ds4_gpu_tensor_write(g->selected, 0, selected, DS4_N_EXPERT_USED * sizeof(*selected)) ||
        !ds4_gpu_tensor_write(g->route_weights, 0, weights, DS4_N_EXPERT_USED * sizeof(*weights)))
        return false;
    if (profile) {
        profile->commit_ms += now_sec() * 1000.0 - t0;
        profile->substitutions += plan.count;
    }
    ds41_resident_route_stats.substitutions += plan.count;
    ds41_resident_route_stats.per_layer[il] += plan.count;
    ds41_resident_route_stats.double_substitution_layers += plan.count == 2;
    ds41_resident_route_stats.triple_substitution_layers += plan.count == 3;
    ds41_resident_route_stats.drops += g->resident_route_drop_weak;
    ds41_resident_route_stats.skipped_loads += plan.count;
    ds41_resident_route_stats.skipped_bytes += plan.count * (2 * table->gate_expert_bytes + table->down_expert_bytes);
    for (uint32_t i = 0; i < plan.count; i++) {
        ds41_resident_route_stats.replaced_mass += plan.changes[i].old_mass;
        ds41_resident_route_stats.score_gap += plan.changes[i].score_gap;
        ds41_resident_route_stats.affinity_ratio += plan.changes[i].affinity_ratio;
        ds41_resident_route_stats.rank += plan.changes[i].candidate_rank;
    }
    return true;
}

static bool ds41_resident_route_apply_native_weights(ds41_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il,
        const ds4_gpu_stream_expert_table *table, int32_t *selected) {
    if (g->resident_route_gap <= 0.0f) return true;
    if (g->resident_route_drop_weak) return false;
    ds4_resident_route_layer_metrics *profile = g->resident_route_profile ?
        &ds41_resident_metrics.layer[il] : NULL;
    double t0 = profile ? now_sec() * 1000.0 : 0.0;
    uint8_t resident[DS4_MAX_EXPERT];
    uint8_t selected_resident[DS4_MAX_EXPERT_USED];
    const float *affinity = ds4_gpu_tensor_contents(g->route_probs);
    const float *bias = tensor_data(m, l->ffn_exp_probs_b);
    if (!affinity || !bias || !ds4_gpu_stream_expert_cache_snapshot_selected(
            table, selected, DS4_N_EXPERT_USED, selected_resident)) return false;
    if (profile) profile->lookup_ms += now_sec() * 1000.0 - t0;
    ds41_resident_route_stats.layers++;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++)
        ds41_resident_route_stats.native_misses += !selected_resident[i];
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    bool eligible = false;
    ds4_resident_drop_change drop;
    if (g->resident_route_drop_weak) {
        eligible = ds4_resident_route_choose_drop(selected, affinity, selected_resident,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, &drop);
    } else if (g->resident_route_max_subs == 1) {
        uint32_t weak;
        float old_mass;
        eligible = ds4_resident_route_weakest(selected, affinity, DS4_N_EXPERT, DS4_N_EXPERT_USED,
                g->resident_route_max_mass, &weak, &old_mass) && !selected_resident[weak];
    } else if (g->resident_route_max_subs == 2) {
        uint32_t slots[2];
        float masses[2];
        const uint32_t n_weak = ds4_resident_route_weak_pair(selected, affinity,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, slots, masses);
        for (uint32_t i = 0; i < n_weak; i++)
            eligible |= !selected_resident[slots[i]] && masses[i] <= g->resident_route_max_total_mass;
    } else {
        uint32_t slots[3];
        float masses[3];
        const uint32_t n_weak = ds4_resident_route_weak_triple(selected, affinity,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_max_mass, slots, masses);
        for (uint32_t i = 0; i < n_weak; i++)
            eligible |= !selected_resident[slots[i]] && masses[i] <= g->resident_route_max_total_mass;
    }
    if (profile) {
        profile->guard_ms += now_sec() * 1000.0 - t0;
        profile->eligible += eligible;
    }
    if (!eligible || g->resident_route_mode == DS41_RESIDENT_ROUTE_LOOKUP) return true;
    if (!g->resident_route_drop_weak) {
        t0 = profile ? now_sec() * 1000.0 : 0.0;
        if (!ds4_gpu_stream_expert_cache_snapshot(table, resident)) return false;
        if (profile) {
            profile->snapshot_ms += now_sec() * 1000.0 - t0;
            profile->snapshots++;
        }
        ds41_resident_route_stats.full_snapshots++;
    }
    ds4_resident_route_plan plan;
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    bool planned;
    if (g->resident_route_drop_weak) {
        plan = (ds4_resident_route_plan) {
            .count = 1, .total_mass = drop.old_mass,
            .changes = {{.slot = drop.slot, .old_expert = drop.old_expert,
                .new_expert = drop.alias_expert, .old_mass = drop.old_mass}},
        };
        planned = true;
    } else planned = ds4_resident_route_choose_plan(selected, affinity, bias, resident,
            DS4_N_EXPERT, DS4_N_EXPERT_USED, g->resident_route_gap,
            g->resident_route_max_rank, g->resident_route_max_mass,
            g->resident_route_max_subs, g->resident_route_max_total_mass, &plan);
    if (profile) {
        profile->plan_ms += now_sec() * 1000.0 - t0;
        profile->plans += planned;
        profile->planned_substitutions += plan.count;
    }
    if (!planned) return true;
    ds41_resident_route_stats.planned_substitutions += plan.count;
    t0 = profile ? now_sec() * 1000.0 : 0.0;
    int32_t final_ids[DS4_MAX_EXPERT_USED];
    memcpy(final_ids, selected, DS4_N_EXPERT_USED * sizeof(*selected));
    for (uint32_t i = 0; i < plan.count; i++)
        final_ids[plan.changes[i].slot] = (int32_t)plan.changes[i].new_expert;
    /* ID-only noop plans the same slots without touching either GPU tensor
     * or recomputing any native mixture coefficient. */
    if (g->resident_route_mode == DS41_RESIDENT_ROUTE_NOOP) {
        if (profile) profile->commit_ms += now_sec() * 1000.0 - t0;
        return true;
    }
    memcpy(selected, final_ids, DS4_N_EXPERT_USED * sizeof(*selected));
    if (!ds4_gpu_tensor_write(g->selected, 0, selected, DS4_N_EXPERT_USED * sizeof(*selected)))
        return false;
    if (profile) {
        profile->commit_ms += now_sec() * 1000.0 - t0;
        profile->substitutions += plan.count;
    }
    ds41_resident_route_stats.substitutions += plan.count;
    ds41_resident_route_stats.native_weight_substitutions += plan.count;
    ds41_resident_route_stats.per_layer[il] += plan.count;
    ds41_resident_route_stats.double_substitution_layers += plan.count == 2;
    ds41_resident_route_stats.triple_substitution_layers += plan.count == 3;
    ds41_resident_route_stats.drops += g->resident_route_drop_weak;
    ds41_resident_route_stats.skipped_loads += plan.count;
    ds41_resident_route_stats.skipped_bytes += plan.count * (2 * table->gate_expert_bytes + table->down_expert_bytes);
    for (uint32_t i = 0; i < plan.count; i++) {
        ds41_resident_route_stats.replaced_mass += plan.changes[i].old_mass;
        ds41_resident_route_stats.score_gap += plan.changes[i].score_gap;
        ds41_resident_route_stats.affinity_ratio += plan.changes[i].affinity_ratio;
        ds41_resident_route_stats.rank += plan.changes[i].candidate_rank;
    }
    return true;
}

static bool ds41_resident_route_apply_dispatch(ds41_gpu_graph *g, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il,
        const ds4_gpu_stream_expert_table *table, int32_t *selected) {
    if (g->resident_route_native_weights)
        return ds41_resident_route_apply_native_weights(g, m, l, il, table, selected);
    return ds41_resident_route_apply(g, m, l, il, table, selected);
}


#undef getenv
#undef atexit
static ds4_gpu_tensor prob,ids_tensor,weight_tensor;
static ds4_model model;static ds4_layer_weights layer;
static ds4_gpu_stream_expert_table table={100,200};
static const int32_t native_ids[6]={0,1,2,3,4,5};
static const uint32_t native_weight_bits[6]={0x3ec00001,0x3e999999,0x3e8a3d70,0x3e75c290,0x3e3851ea,0x3e0a3d71};
static void reset(void){
    memset(&ds41_resident_route_stats,0,sizeof(ds41_resident_route_stats));memset(&ds41_resident_metrics,0,sizeof(ds41_resident_metrics));
    memset(&prob,0,sizeof(prob));memset(&model,0,sizeof(model));memset(resident,0,sizeof(resident));
    float *p=prob.values;float values[10]={.25f,.20f,.18f,.16f,.12f,.09f,.089f,.119f,.159f,.01f};
    memcpy(p,values,sizeof(values));for(unsigned i=10;i<384;i++)p[i]=.001f;
    model.bias[3]=-.04f;model.bias[4]=-.02f;model.bias[7]=-.031f;model.bias[8]=-.072f;
    resident[0]=resident[1]=resident[2]=resident[6]=resident[7]=resident[8]=1;
    layer.ffn_exp_probs_b=model.bias;prob.kind=1;ids_tensor.kind=2;weight_tensor.kind=3;
    memcpy(ids_tensor.bytes,native_ids,sizeof(native_ids));memcpy(weight_tensor.bytes,native_weight_bits,sizeof(native_weight_bits));
    for(unsigned i=0;i<6;i++)ds41_resident_noop_weights[i]=-123.0f;
    env_count=selected_writes=weight_writes=contents_reads=selected_snapshots=full_snapshots=0;fail_selected_write=false;ds41_route_oracle.active=false;
}
static ds41_gpu_graph graph(void){return (ds41_gpu_graph){.streaming=true,.tp_world=1,.route_probs=&prob,.selected=&ids_tensor,.route_weights=&weight_tensor};}
static void common(const char *subs,const char *mode,const char *flag){
    option("DS4_METAL_V41_RESIDENT_ROUTE_GAP",".30");option("DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS",".20");
    option("DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS",".40");option("DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK","48");
    option("DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS",subs);
    if(mode)option("DS4_METAL_V41_RESIDENT_ROUTE_MODE",mode);
    if(flag)option("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS",flag);
}
static void configured(ds41_gpu_graph *g,bool old,bool decode){
    assert(setjmp(fatal)==0);if(old)old_configure(g,decode);else ds41_resident_route_configure(g,decode);
}
static void strict_options(void){
    bool on=true;assert(ds4_resident_native_weights_parse(NULL,&on)&&!on);
    assert(ds4_resident_native_weights_parse("0",&on)&&!on);assert(ds4_resident_native_weights_parse("1",&on)&&on);
    const char *bad[]={"","00","01","+1","-0"," 1","1 ","true","nan","inf","2","1\n"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++){
        assert(!ds4_resident_native_weights_parse(bad[i],&on));reset();option("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS",bad[i]);
        ds41_gpu_graph g=graph();if(!setjmp(fatal)){ds41_resident_route_configure(&g,false);assert(!"invalid flag accepted");}
    }
    for(unsigned i=0;i<4;i++){
        reset();ds41_gpu_graph g=graph();option("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS","1");
        if(i==1)option("DS4_METAL_V41_RESIDENT_ROUTE_GAP","0");
        if(i>=2){option("DS4_METAL_V41_RESIDENT_ROUTE_GAP",".30");option("DS4_METAL_V41_RESIDENT_ROUTE_POLICY","drop");}
        if(!setjmp(fatal)){ds41_resident_route_configure(&g,i!=3);assert(!"incompatible preservation accepted");}
    }
    const char *bad_gap[]={NULL,"","0","-0","nan","NaN","inf","1e10000","broken",".30oops","-1","1.1"};
    for(unsigned i=0;i<sizeof(bad_gap)/sizeof(*bad_gap);i++){
        assert(!ds4_resident_native_weights_gap(bad_gap[i]));
        for(unsigned decode=0;decode<2;decode++){
            reset();option("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS","1");
            if(bad_gap[i])option("DS4_METAL_V41_RESIDENT_ROUTE_GAP",bad_gap[i]);
            ds41_gpu_graph g=graph();if(!setjmp(fatal)){ds41_resident_route_configure(&g,decode);assert(!"invalid positive gap accepted");}
            /* Flag0 preserves the old prefill return, even for ignored GAP. */
            reset();option("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS","0");
            if(bad_gap[i])option("DS4_METAL_V41_RESIDENT_ROUTE_GAP",bad_gap[i]);
            g=graph();configured(&g,false,false);assert(g.resident_route_gap==0 && !g.resident_route_native_weights);
        }
    }
    const char *good_gap[]={".001",".30","1"};
    for(unsigned i=0;i<sizeof(good_gap)/sizeof(*good_gap);i++)assert(ds4_resident_native_weights_gap(good_gap[i]));
}
static void apply_preserves_every_native_weight_slot(void){
    const char *counts[]={"1","2","3"};
    for(unsigned c=0;c<3;c++){
        reset();common(counts[c],"apply","1");ds41_gpu_graph g=graph();configured(&g,false,true);
        int32_t ids[6];memcpy(ids,native_ids,sizeof(ids));assert(ds41_resident_route_apply_dispatch(&g,&model,&layer,0,&table,ids));
        assert(ds41_resident_route_stats.substitutions==c+1 && ds41_resident_route_stats.native_weight_substitutions==c+1);
        assert(selected_writes==1 && weight_writes==0 && contents_reads==1 && selected_snapshots==1 && full_snapshots==1);
        assert(!memcmp(weight_tensor.bytes,native_weight_bits,sizeof(native_weight_bits)));assert(!memcmp(ids_tensor.bytes,ids,sizeof(ids)));
        for(unsigned i=0;i<6;i++)for(unsigned j=0;j<i;j++)assert(ids[i]!=ids[j]);
        for(unsigned i=0;i<6-(c+1);i++)assert(ids[i]==native_ids[i]);
    }
}
static void native_lookup_noop_do_not_commit(void){
    const char *modes[]={"native","lookup","noop"};
    for(unsigned m=0;m<3;m++){
        reset();common("3",modes[m],"1");ds41_gpu_graph g=graph();configured(&g,false,true);
        int32_t ids[6];memcpy(ids,native_ids,sizeof(ids));assert(ds41_resident_route_apply_dispatch(&g,&model,&layer,0,&table,ids));
        assert(selected_writes==0 && weight_writes==0);assert(!memcmp(ids,native_ids,sizeof(ids)));
        assert(!memcmp(ids_tensor.bytes,native_ids,sizeof(ids)));assert(!memcmp(weight_tensor.bytes,native_weight_bits,sizeof(native_weight_bits)));
        for(unsigned i=0;i<6;i++)assert(ds41_resident_noop_weights[i]==-123.0f);
        assert(contents_reads==(m==0?0:1));assert(full_snapshots==(m==2));
    }
    reset();common("3","apply","1");ds41_gpu_graph g=graph();configured(&g,false,false);
    int32_t ids[6];memcpy(ids,native_ids,sizeof(ids));assert(ds41_resident_route_apply_dispatch(&g,&model,&layer,0,&table,ids));
    assert(g.resident_route_gap==0 && !g.resident_route_native_weights && contents_reads==0 && !selected_writes && !weight_writes);
}
static void old_default_differential(void){
    const char *modes[]={"apply","native","lookup","noop"};const char *counts[]={"1","2","3"};
    for(unsigned m=0;m<4;m++)for(unsigned c=0;c<3;c++)for(unsigned flag=0;flag<2;flag++){
        unsigned char old_ids[24],old_weights[24];float old_noop[6];uint64_t old_subs;unsigned old_sw,old_ww,old_read,old_ss,old_fs;
        reset();common(counts[c],modes[m],flag?"0":NULL);ds41_gpu_graph g=graph();configured(&g,true,true);
        int32_t ids[6];memcpy(ids,native_ids,sizeof(ids));assert(old_apply(&g,&model,&layer,0,&table,ids));
        memcpy(old_ids,ids_tensor.bytes,24);memcpy(old_weights,weight_tensor.bytes,24);for(unsigned i=0;i<6;i++)old_noop[i]=ds41_resident_noop_weights[i];
        old_subs=ds41_resident_route_stats.substitutions;old_sw=selected_writes;old_ww=weight_writes;old_read=contents_reads;old_ss=selected_snapshots;old_fs=full_snapshots;
        reset();common(counts[c],modes[m],flag?"0":NULL);g=graph();configured(&g,false,true);memcpy(ids,native_ids,sizeof(ids));
        assert(ds41_resident_route_apply_dispatch(&g,&model,&layer,0,&table,ids));
        if(memcmp(old_ids,ids_tensor.bytes,24) || memcmp(old_weights,weight_tensor.bytes,24)){
            fprintf(stderr,"DIFFERENTIAL mode=%s subs=%s flag=%s\n",modes[m],counts[c],flag?"0":"unset");
            for(unsigned i=0;i<6;i++){
                int32_t oi,ni;uint32_t ow,nw;memcpy(&oi,old_ids+4*i,4);memcpy(&ni,ids_tensor.bytes+4*i,4);
                memcpy(&ow,old_weights+4*i,4);memcpy(&nw,weight_tensor.bytes+4*i,4);
                fprintf(stderr,"slot=%u old_id=%" PRId32 " new_id=%" PRId32 " old_weight=%08" PRIx32 " new_weight=%08" PRIx32 "\n",i,oi,ni,ow,nw);
            }
        }
        assert(!memcmp(old_ids,ids_tensor.bytes,24) && !memcmp(old_weights,weight_tensor.bytes,24));
        for(unsigned i=0;i<6;i++)assert(!memcmp(old_noop+i,(const void *)(ds41_resident_noop_weights+i),sizeof(float)));
        assert(ds41_resident_route_stats.substitutions==old_subs && ds41_resident_route_stats.native_weight_substitutions==0);
        assert(selected_writes==old_sw && weight_writes==old_ww && contents_reads==old_read && selected_snapshots==old_ss && full_snapshots==old_fs);
    }
}
static void declined_and_failed_commit(void){
    reset();common("3","apply","1");ds41_gpu_graph g=graph();configured(&g,false,true);resident[3]=resident[4]=resident[5]=1;
    int32_t ids[6];memcpy(ids,native_ids,sizeof(ids));assert(ds41_resident_route_apply_dispatch(&g,&model,&layer,0,&table,ids));assert(!selected_writes && !weight_writes && !full_snapshots);
    reset();common("3","apply","1");g=graph();configured(&g,false,true);fail_selected_write=true;memcpy(ids,native_ids,sizeof(ids));
    assert(!ds41_resident_route_apply_dispatch(&g,&model,&layer,0,&table,ids));assert(selected_writes==1 && !weight_writes);
    assert(!ds41_resident_route_stats.substitutions && !ds41_resident_route_stats.native_weight_substitutions);
    assert(!memcmp(weight_tensor.bytes,native_weight_bits,24));
}
int main(void){strict_options();apply_preserves_every_native_weight_slot();native_lookup_noop_do_not_commit();old_default_differential();declined_and_failed_commit();puts("PASS: strict option, untouched GPU coefficient bits, unchanged selector and off controls, no readback, decline/failure");return 0;}
