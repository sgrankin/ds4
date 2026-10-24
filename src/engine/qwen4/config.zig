//! Qwen dimensions and layer policy captured with each graph. Metadata loading
//! supplies this immutable value; numerical owners never consult family globals.
const c = @import("gguf_c");
pub const Config = c.ds4_qwen4_config;
pub const Shape = c.ds4_shape;

pub fn trunkLayers(config: *const Config) u32 {
    return config.shape.n_layer - config.shape.n_nextn_predict;
}

pub fn isLinear(config: *const Config, layer: u32) bool {
    return layer < trunkLayers(config) and (layer + 1) % config.shape.n_full_attn_interval != 0;
}

pub fn isPle(config: *const Config, layer: u32) bool {
    return layer == config.shape.n_ple_layer;
}

pub fn isNextn(config: *const Config, layer: u32) bool {
    return config.shape.n_nextn_predict != 0 and layer >= trunkLayers(config);
}

pub fn convWidth(shape: *const Shape) u64 {
    return @as(u64, 2) * shape.n_lin_k_head * shape.n_lin_head_dim +
        @as(u64, shape.n_lin_v_head) * shape.n_lin_head_dim;
}

pub fn valueWidth(shape: *const Shape) u64 {
    return @as(u64, shape.n_lin_v_head) * shape.n_lin_head_dim;
}

pub fn stateFloats(shape: *const Shape) u64 {
    return valueWidth(shape) * shape.n_lin_head_dim;
}

pub fn historyFloats(shape: *const Shape) u64 {
    return @as(u64, shape.n_lin_conv - 1) * convWidth(shape);
}

pub fn pleHistoryFloats(shape: *const Shape) u64 {
    return @as(u64, shape.n_ple_conv - 1) * shape.n_ple_ngram * shape.n_embd * shape.n_hc;
}
