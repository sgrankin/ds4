//! CPU-only C compatibility exports over explicit model/view policy.
const std = @import("std");
const c = @import("gguf_c");
const model = @import("model_spans.zig");
const Weights = c.ds4_weights;
const Shape = c.ds4_shape;
const Policy = c.ds4_streaming_policy;
const Layer = c.ds4_layer_weights;
const Vec = c.ds4_model_map_span_vec;

export fn ds4_native_model_spans_full(weights: [*c]const Weights, shape: [*c]const Shape, first: u32, last: u32, output: bool, out: [*c]Vec) bool {
    if (weights == null or shape == null or out == null) return false;
    const policy = std.mem.zeroes(Policy);
    return model.build(weights, shape, &policy, first, last, first == 0, output, .full, out);
}
export fn ds4_native_model_spans_static(weights: [*c]const Weights, shape: [*c]const Shape, first: u32, last: u32, token: bool, output: bool, out: [*c]Vec) bool {
    if (weights == null or shape == null or out == null) return false;
    const policy = std.mem.zeroes(Policy);
    return model.build(weights, shape, &policy, first, last, token, output, .static, out);
}
export fn ds4_native_model_spans_runtime(weights: [*c]const Weights, shape: [*c]const Shape, policy: [*c]const Policy, first: u32, last: u32, token: bool, output: bool, out: [*c]Vec) bool {
    if (weights == null or shape == null or policy == null or out == null) return false;
    return model.build(weights, shape, policy, first, last, token, output, .runtime, out);
}
export fn ds4_native_model_spans_layer(weights: [*c]const Weights, shape: [*c]const Shape, policy: [*c]const Policy, index: u32, decode: bool, out: [*c]Vec) bool {
    if (weights == null or shape == null or policy == null or out == null) return false;
    return model.build(weights, shape, policy, index, index, !decode and index == 0, false, if (decode) .runtime else .full, out);
}
export fn ds4_native_model_spans_token(weights: [*c]const Weights, out: [*c]Vec) bool {
    if (weights == null or out == null) return false;
    return model.tokenOnly(weights, out);
}
export fn ds4_native_model_spans_output(weights: [*c]const Weights, out: [*c]Vec) bool {
    if (weights == null or out == null) return false;
    return model.outputOnly(weights, out);
}
export fn ds4_native_model_spans_total(spans: [*c]const Vec) u64 {
    return if (spans == null) 0 else model.total(spans);
}
export fn ds4_native_model_spans_static_bytes(weights: [*c]const Weights, shape: [*c]const Shape, token: bool, output: bool, bytes: [*c]u64) bool {
    if (bytes != null) bytes.* = 0;
    if (weights == null or shape == null or bytes == null) return false;
    return model.staticBytes(weights, shape, token, output, bytes);
}
export fn ds4_native_model_spans_uniform(weights: [*c]const Weights, shape: [*c]const Shape, index: u32) bool {
    return if (weights == null or shape == null) true else model.uniform(weights, shape, index);
}
export fn ds4_native_model_spans_resident_supported(shape: [*c]const Shape, layer: [*c]const Layer, index: u32) bool {
    return shape != null and layer != null and model.residentSupported(shape, layer, index);
}
export fn ds4_native_model_spans_resident_layer(shape: [*c]const Shape, policy: [*c]const Policy, layer: [*c]const Layer, index: u32) bool {
    return shape != null and policy != null and layer != null and model.residentLayer(shape, policy, layer, index);
}
export fn ds4_native_model_spans_static_supported(weights: [*c]const Weights, shape: [*c]const Shape, policy: [*c]const Policy) bool {
    return weights != null and shape != null and policy != null and model.staticSupported(weights, shape, policy);
}
export fn ds4_native_model_spans_glm_addr_layout(weights: [*c]const Weights, shape: [*c]const Shape, layer: [*c]const Layer, index: u32) bool {
    return weights != null and shape != null and layer != null and model.glmAddrLayout(weights, shape, layer, index);
}
export fn ds4_native_model_spans_glm_selected_layout(shape: [*c]const Shape, layer: [*c]const Layer, index: u32) bool {
    return shape != null and layer != null and model.glmSelectedLayout(shape, layer, index);
}
export fn ds4_native_model_spans_glm_streamed(weights: [*c]const Weights, shape: [*c]const Shape, layer: [*c]const Layer, index: u32) bool {
    return weights != null and shape != null and layer != null and model.glmStreamed(weights, shape, layer, index);
}
