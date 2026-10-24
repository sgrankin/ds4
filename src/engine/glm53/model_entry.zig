//! Model-loading ABI only. Numerical execution and graph resources live in their
//! typed owners; the caller retains fatal reporting and compatibility globals.
const c = @import("gguf_c");
const metadata = @import("metadata.zig");
const weights = @import("weights.zig");

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    const result: []u8 = if (capacity == 0) &.{} else pointer[0..capacity];
    if (result.len != 0) result[0] = 0;
    return result;
}

export fn ds4_native_glm53_model_config(model: *const c.ds4_model, output: *c.ds4_glm_model_config, text: [*c]u8, capacity: usize) bool {
    const config = metadata.load(model, errors(text, capacity)) catch return false;
    output.* = config;
    return true;
}

export fn ds4_native_glm53_weights_bind(output: *c.ds4_weights, model: *const c.ds4_model, config: *const c.ds4_glm_model_config, slice: bool, start: u32, end: u32, require_output: bool, optional_output: bool, text: [*c]u8, capacity: usize) bool {
    var binding = weights.Binding{ .table = output, .model = model, .config = config, .errors = errors(text, capacity) };
    binding.bind(.{ .slice = slice, .start = start, .end = end, .require_output = require_output, .optional_output = optional_output }) catch return false;
    return true;
}

export fn ds4_native_glm53_weights_validate(table: *const c.ds4_weights, config: *const c.ds4_glm_model_config, start: u32, end: u32, require_embedding: bool, require_output: bool, text: [*c]u8, capacity: usize) bool {
    const binding = weights.Binding{ .table = @constCast(table), .config = config, .errors = errors(text, capacity) };
    binding.validate(start, end, require_embedding, require_output) catch return false;
    return true;
}

export fn ds4_native_glm53_layer_has_required(layer: ?*const c.ds4_layer_weights, index: u32, config: *const c.ds4_glm_model_config) bool {
    return if (layer) |value| weights.layerComplete(value, index, config) else false;
}
