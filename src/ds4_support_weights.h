#ifndef DS4_SUPPORT_WEIGHTS_H
#define DS4_SUPPORT_WEIGHTS_H
#include "ds4_model_weights.h"
#include "ds4_dspark_weights.h"
#include "ds4_dspark_cycle.h"

/* Actual summary/kind/legacy records moved ONCE from the common Core header.
 * All tensor pointers borrow Model's mapping; these tables allocate nothing.
 * Summary preserves ordered metadata aliases and bounded target-layer reads. */
typedef struct {
    uint32_t stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    bool has_metadata;
    bool has_main_proj;
    bool has_main_norm;
    bool has_markov_head;
    bool has_confidence_head;
    bool has_final_head;
    bool has_block_size;
    bool has_markov_rank;
    bool has_noise_token_id;
    bool has_target_layers;
} ds4_dspark_summary;

typedef enum {
    DS4_SUPPORT_NONE = 0,
    DS4_SUPPORT_MTP_LEGACY,
    DS4_SUPPORT_DSPARK,
} ds4_support_kind;

typedef struct {
    ds4_tensor *e_proj;
    ds4_tensor *h_proj;
    ds4_tensor *enorm;
    ds4_tensor *hnorm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_layer_weights block;
} ds4_mtp_weights;


/* Whole library actions. Target shape/family are explicit immutable inputs;
 * no Engine pointer or global getter enters native policy. Legacy validation
 * remains fatal; DSpark fills its counters and lets Engine decide admission. */
void ds4_native_support_summary(const ds4_model *, ds4_dspark_summary *);
ds4_support_kind ds4_native_support_detect(const ds4_model *, uint32_t *, ds4_dspark_summary *);
bool ds4_native_support_compatible(const ds4_model *, ds4_model_family family);
void ds4_native_support_bind(ds4_dspark_weights *, const ds4_model *,
    const ds4_dspark_summary *, const ds4_shape *);
void ds4_native_support_bind_mtp(ds4_mtp_weights *, const ds4_model *, const ds4_shape *);
#endif
