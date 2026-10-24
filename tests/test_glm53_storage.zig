//! CPU map/cache transactions over real descriptors; only the Metal budget and
//! final map installation are stand-ins. Span construction remains production.
const std = @import("std");
const c = @import("gguf_c");
const Storage = @import("glm_storage").Storage;
pub const Map = struct {
    count: u32 = 0,
    max_tensor_bytes: u64 = 0,
    offsets: [64]u64 = @splat(0),
    sizes: [64]u64 = @splat(0),
};
pub var installed = Map{};
pub var succeeds = true;
pub var budget: u32 = 128;
const q2: c.gguf_type_info = .{ .name = "q2_k", .block_elems = 256, .block_bytes = 84 };
const q4: c.gguf_type_info = .{ .name = "q4_k", .block_elems = 256, .block_bytes = 144 };
const iq2: c.gguf_type_info = .{ .name = "iq2_xxs", .block_elems = 256, .block_bytes = 66 };
export fn ds4_gguf_tensor_type(kind: u32) [*c]const c.gguf_type_info {
    return switch (kind) {
        c.DS4_TENSOR_Q2_K => &q2,
        c.DS4_TENSOR_Q4_K => &q4,
        c.DS4_TENSOR_IQ2_XXS => &iq2,
        else => null,
    };
}
export fn ds4_engine_model_die(message: [*c]const u8) noreturn {
    std.debug.panic("unexpected model diagnostic: {s}", .{std.mem.span(message)});
}
export fn ds4_gpu_stream_expert_cache_budget_for_expert_size(_: u64, _: u64) u32 {
    return budget;
}
export fn ds4_gpu_set_model_map_spans(_: ?*const anyopaque, _: u64, offsets: [*c]const u64, sizes: [*c]const u64, count: u32, maximum: u64) c_int {
    installed = .{ .count = count, .max_tensor_bytes = maximum };
    @memcpy(installed.offsets[0..count], offsets[0..count]);
    @memcpy(installed.sizes[0..count], sizes[0..count]);
    return @intFromBool(succeeds);
}
fn weight(offset: u64, bytes: u64) c.ds4_tensor {
    var result = std.mem.zeroes(c.ds4_tensor);
    result.type = c.DS4_TENSOR_F32;
    result.abs_offset = offset;
    result.bytes = bytes;
    return result;
}
fn expert(kind: u32, offset: u64) c.ds4_tensor {
    var result = weight(offset, @as(u64, ds4_gguf_tensor_type(kind).*.block_bytes) * 2 * 128);
    result.type = kind;
    result.ndim = 3;
    result.dim[0] = 256;
    result.dim[1] = 2;
    result.dim[2] = 128;
    return result;
}
pub const Fixture = struct {
    graph: c.ds4_glm_gpu_graph = std.mem.zeroes(c.ds4_glm_gpu_graph),
    model: c.ds4_model = std.mem.zeroes(c.ds4_model),
    weights: c.ds4_weights = std.mem.zeroes(c.ds4_weights),
    policy: c.ds4_streaming_policy = std.mem.zeroes(c.ds4_streaming_policy),
    token: c.ds4_tensor = weight(100, 16),
    output: c.ds4_tensor = weight(900000, 16),
    norms: [3]c.ds4_tensor = .{ weight(200, 16), weight(400, 16), weight(600, 16) },
    experts: [3][3]c.ds4_tensor = undefined,

    pub fn init(self: *Fixture, kind: u32, used: u32, outlier: bool) void {
        self.* = .{};
        self.graph.ssd_streaming = true;
        self.graph.has_token_embd = true;
        self.graph.config.shape.family = c.DS4_MODEL_FAMILY_GLM_DSA;
        self.graph.config.shape.n_layer = 3;
        self.graph.config.shape.n_leading_dense = 1;
        self.graph.config.shape.n_expert = 128;
        self.graph.config.shape.n_expert_used = used;
        self.weights.token_embd = &self.token;
        self.weights.output = &self.output;
        self.model.size = 1000000;
        self.graph.weights = &self.weights;
        for (&self.experts, 0..) |*layer, index| {
            for (layer, 0..) |*tensor, projection| tensor.* = expert(if (outlier and index == 2) c.DS4_TENSOR_Q4_K else kind, 1000 + index * 200000 + projection * 50000);
            self.weights.layer[index].attn_norm = &self.norms[index];
            if (index == 0) continue;
            self.weights.layer[index].ffn_gate_exps = &layer[0];
            self.weights.layer[index].ffn_up_exps = &layer[1];
            self.weights.layer[index].ffn_down_exps = &layer[2];
        }
        self.policy.static_decode_map = true;
    }
};
pub const variables = [_][*:0]const u8{
    "DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP",
    "DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE",
    "DS4_METAL_GLM_STREAMING_DECODE_FULL_LAYER_MAP",
    "DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP",
    "DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE",
    "DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION",
    "DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
    "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE",
    "DS4_METAL_MOE_WRITE_CLAMPED_ACT",
    "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
    "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
    "DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE",
    "DS4_METAL_STREAMING_MAP_TRACE",
};
pub fn clear() void {
    for (variables) |name| _ = c.unsetenv(name);
    budget = 128;
    succeeds = true;
    installed = .{};
}

test "static map publication survives reuse, hidden input and failed replacement" {
    clear();
    defer clear();
    const fixture = try std.heap.c_allocator.create(Fixture);
    defer std.heap.c_allocator.destroy(fixture);
    fixture.init(c.DS4_TENSOR_Q2_K, 6, false);
    const graph = &fixture.graph;
    var static_map = false;
    try std.testing.expect(Storage.mapBegin(graph, &fixture.model, &fixture.weights, false, &static_map, &fixture.policy));
    try std.testing.expect(static_map and graph.streaming_static_decode_map_current and installed.count != 0);
    installed = .{};
    try std.testing.expect(Storage.mapBegin(graph, &fixture.model, &fixture.weights, false, &static_map, &fixture.policy));
    try std.testing.expectEqual(@as(u32, 0), installed.count);
    try std.testing.expect(Storage.mapBegin(graph, &fixture.model, &fixture.weights, true, &static_map, &fixture.policy));
    try std.testing.expect(!static_map and graph.streaming_static_decode_map_current);
    _ = c.setenv(variables[0], "", 1);
    succeeds = false;
    try std.testing.expect(!Storage.mapBegin(graph, &fixture.model, &fixture.weights, false, &static_map, &fixture.policy));
    try std.testing.expect(!static_map and !graph.streaming_static_decode_map_current);
    clear();
    graph.streaming_static_decode_map_current = true;
    _ = c.setenv(variables[1], "", 1);
    succeeds = false;
    try std.testing.expect(!Storage.mapBegin(graph, &fixture.model, &fixture.weights, false, &static_map, &fixture.policy));
    // A failed all-static replacement retains the previous current-map flag.
    try std.testing.expect(static_map and graph.streaming_static_decode_map_current);
}

test "tiny IQ2 prefill, live cache budget, resident and MTP choices retain boundaries" {
    clear();
    defer clear();
    const fixture = try std.heap.c_allocator.create(Fixture);
    defer std.heap.c_allocator.destroy(fixture);
    fixture.init(c.DS4_TENSOR_IQ2_XXS, 8, false);
    const graph = &fixture.graph;
    const layer = &fixture.weights.layer[1];
    try std.testing.expect(Storage.prefillCache(graph, &fixture.weights, layer, 1, 255));
    try std.testing.expect(!Storage.prefillCache(graph, &fixture.weights, layer, 1, 256));
    budget = 127;
    try std.testing.expect(!Storage.prefillCache(graph, &fixture.weights, layer, 1, 255));
    try std.testing.expect(Storage.selectedCache(graph, layer, 1, &fixture.policy));
    fixture.policy.resident_start = 1;
    fixture.policy.resident_layers = 1;
    try std.testing.expect(!Storage.selectedCache(graph, layer, 1, &fixture.policy));
    try std.testing.expect(Storage.mtpCache(graph, layer, 1));
    try std.testing.expect(!Storage.seedExpected(graph, &fixture.weights, layer, 1, &fixture.policy));
    budget = 7;
    try std.testing.expect(!Storage.mtpCache(graph, layer, 1));
    fixture.policy.resident_layers = 0;
    try std.testing.expect(Storage.seedExpected(graph, &fixture.weights, layer, 1, &fixture.policy));
}
