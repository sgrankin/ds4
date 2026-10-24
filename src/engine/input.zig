//! C facade for native model input. Owners and algorithms live in pure modules
//! so graph/session code can reuse them without exporting a second ABI copy.
const c = @import("gguf_c");
const mapping = @import("model.zig");
const configuration = @import("config.zig");
const binding = @import("weights.zig");
pub const Model = mapping.Model;
pub const ModelConfig = configuration.ModelConfig;
pub const Weights = binding.Weights;

export fn ds4_native_model_open(record: *c.ds4_model, path: [*c]const u8, metal_mapping: bool, prefetch_cpu: bool) void {
    mapping.openRecord(record, path, metal_mapping, prefetch_cpu);
}
export fn ds4_native_model_close(record: ?*c.ds4_model) void {
    mapping.closeRecordPointer(record);
}
export fn ds4_model_config_deepseek41(model: *const c.ds4_model, output: *c.ds4_model_config, error_text: [*c]u8, error_capacity: usize) bool {
    return configuration.readRecord(model, output, error_text, error_capacity);
}
export fn ds4_native_weights_bind41(output: *c.ds4_weights, model: *const c.ds4_model, config: *const c.ds4_model_config, load_slice: bool, start: u32, end: u32, require_output: bool, optional_output: bool, error_text: [*c]u8, error_capacity: usize) bool {
    return binding.bindRecord(output, model, config, load_slice, start, end, require_output, optional_output, error_text, error_capacity);
}

// Append to the ONE existing input facade, not a second graph export root.
// Core imports the pure owner directly through this same gguf_c type identity.
const support_owner = @import("support_weights.zig");
export fn ds4_native_support_summary(model: *const c.ds4_model, out: *c.ds4_dspark_summary) void {
    out.* = support_owner.summarize(model);
}
export fn ds4_native_support_detect(model: ?*const c.ds4_model, stages: ?*u32, summary: ?*c.ds4_dspark_summary) c.ds4_support_kind {
    return support_owner.detect(model, stages, summary);
}
export fn ds4_native_support_compatible(model: *const c.ds4_model, family: c.ds4_model_family) bool {
    return support_owner.compatible(model, family);
}
export fn ds4_native_support_bind(weights: *c.ds4_dspark_weights, model: ?*const c.ds4_model, summary: ?*const c.ds4_dspark_summary, shape: *const c.ds4_shape) void {
    support_owner.bind(weights, model, summary, shape);
}
export fn ds4_native_support_bind_mtp(weights: *c.ds4_mtp_weights, model: *const c.ds4_model, shape: *const c.ds4_shape) void {
    support_owner.bindLegacy(weights, model, shape);
}

// Append to the ONE existing input facade/shared gguf_c Module.
const vision_binding = @import("vision_weights.zig");
export fn ds4_native_vision_weights_bind(weights: *c.ds4_deepseek4_vision_weights, model: *const c.ds4_model, shape: *const c.ds4_shape) void {
    vision_binding.bind(weights, model, shape);
}
