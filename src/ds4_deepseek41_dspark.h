/* Internal V4.1 drafter graph. Included after the target graph in ds4.c. */
#ifndef DS4_DEEPSEEK41_DSPARK_H
#define DS4_DEEPSEEK41_DSPARK_H

#include "ds4_deepseek41_dspark_state.h"

static void ds41_dspark_reset(ds41_dspark *state) {
    ds4_native_dspark_reset41(state);
}
static void ds41_dspark_free(ds41_dspark *state) {
    ds4_native_dspark_free41(state);
}
static DS4_MAYBE_UNUSED ds41_dspark *ds41_dspark_alloc(const ds4_dspark_weights *weights) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_alloc41(&config, weights);
}
static bool ds41_dspark_capture(ds41_dspark *state, uint32_t layer, uint32_t start,
        const ds4_gpu_tensor *residual, uint32_t count) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_capture41(&config, state, layer, start, residual, count);
}
static bool DS4_MAYBE_UNUSED ds41_dspark_seed(ds41_dspark *state, const ds4_model *model, uint32_t position) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_seed41(state, model, &config, position, ds41_exact_prefill_rows);
}
static bool DS4_MAYBE_UNUSED ds41_dspark_stage(ds41_dspark *state, const ds4_model *model,
        uint32_t stage, uint32_t position) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_stage41(state, model, &config, stage, position, ds41_exact_prefill_rows);
}
static bool DS4_MAYBE_UNUSED ds41_dspark_forward(ds41_dspark *state, const ds4_model *target,
        const ds4_weights *weights, const ds4_model *draft, uint32_t position, int token,
        ds41_gpu_graph *graph) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_forward41(state, target, weights, draft, &config, position,
        token, graph, ds41_exact_prefill_rows);
}
static bool DS4_MAYBE_UNUSED ds41_dspark_markov(ds41_dspark *state, const ds4_model *model,
        uint32_t row, int previous, float *confidence) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_markov41(state, model, &config, row, previous, confidence, ds41_exact_prefill_rows);
}
static bool DS4_MAYBE_UNUSED ds41_dspark_verify(ds41_dspark *state, ds41_gpu_graph *graph,
        const ds4_model *model, const ds4_weights *weights, const int *tokens, uint32_t count) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_verify41(state, graph, model, weights, &config,
        g_tp_block_ctx, tokens, count, ds41_exact_prefill_rows);
}
static bool DS4_MAYBE_UNUSED ds41_dspark_commit(ds41_dspark *state, ds41_gpu_graph *graph,
        uint32_t keep, float *logits) {
    const ds4_model_config config = ds41_native_config();
    return ds4_native_dspark_commit41(state, graph, &config, keep, logits);
}


#endif
