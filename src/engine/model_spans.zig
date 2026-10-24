//! Model tensor/view policy with explicit shape and resident-layer inputs.
//! These CPU-only spans serve admission, initial mapping and bounded page-in.
//! Their tensor pointers borrow Weights; only the span vector owns memory.
const std = @import("std");
const c = @import("gguf_c");
const Cache = @import("engine_cache_sizes.zig");
const allocator = std.heap.c_allocator;
const Shape = c.ds4_shape;
const Policy = c.ds4_streaming_policy;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Weight = c.ds4_tensor;
pub const Span = c.ds4_model_map_span;
pub const Vec = c.ds4_model_map_span_vec;
const maximum = std.math.maxInt(u64);

fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn fatal() noreturn {
    c.ds4_engine_model_die("out of memory");
}
fn groups() u32 {
    const value = c.getenv("DS4_METAL_Q4_PRO_MAP_GROUPS");
    if (value == null or value[0] == 0) return 1;
    var end: [*c]u8 = null;
    const count = c.strtoul(value, &end, 10);
    return if (end != value and end[0] == 0 and count > 0 and count <= 384 and 384 % count == 0) @intCast(count) else 1;
}
fn before(_: void, a: Span, b: Span) bool {
    return a.off < b.off or (a.off == b.off and a.end < b.end);
}

pub const Spans = struct {
    items: std.ArrayList(Span) = .empty,
    max_tensor_bytes: u64 = 0,

    pub fn deinit(self: *Spans) void {
        self.items.deinit(allocator);
    }
    pub fn append(self: *Spans, first: u64, end: u64, isolated: bool) void {
        if (first == maximum or end <= first) return;
        self.items.append(allocator, .{ .off = first, .end = end, .isolate = isolated }) catch fatal();
    }
    pub fn include(self: *Spans, pointer: [*c]const Weight) void {
        if (pointer == null or pointer.*.bytes == 0) return;
        const weight = pointer.*;
        const threshold = 2 * 1024 * 1024 * 1024;
        const count = groups();
        if (weight.type == c.DS4_TENSOR_Q4_K and weight.ndim == 3 and weight.dim[2] == 384 and
            weight.bytes >= threshold and weight.bytes % count == 0)
        {
            // Large PRO Q4 expert blobs must not hide in broad layer views.
            // Optional stable groups avoid stacking giant overlapping aliases
            // when the selected-expert path binds its own borrowed ranges.
            const bytes = weight.bytes / count;
            self.max_tensor_bytes = @max(self.max_tensor_bytes, bytes);
            for (0..count) |index| {
                const first = weight.abs_offset +% (@as(u64, index) *% bytes);
                self.append(first, first +% bytes, true);
            }
            return;
        }
        self.max_tensor_bytes = @max(self.max_tensor_bytes, weight.bytes);
        const isolate = (weight.type == c.DS4_TENSOR_Q4_K or weight.type == c.DS4_TENSOR_MXFP4) and weight.bytes >= threshold;
        self.append(weight.abs_offset, weight.abs_offset +% weight.bytes, isolate);
    }
    pub fn includeExperts(self: *Spans, layer: *const Layer) void {
        self.include(layer.ffn_gate_exps);
        self.include(layer.ffn_up_exps);
        self.include(layer.ffn_down_exps);
    }
    /// A null size class excludes every routed tensor for guard-only static
    /// accounting. A concrete class also maps outliers that cannot use its slab.
    pub fn includeLayer(self: *Spans, layer: *const Layer, decode: bool, size_class: ?u64) void {
        inline for (comptime std.meta.fieldNames(Layer)) |name| {
            const routed = comptime std.mem.eql(u8, name, "ffn_gate_exps") or
                std.mem.eql(u8, name, "ffn_up_exps") or std.mem.eql(u8, name, "ffn_down_exps");
            if (!decode or !routed) self.include(@field(layer, name));
        }
        if (decode) if (size_class) |slab| {
            if (Cache.layerBytes(layer)) |bytes| if (bytes != slab) self.includeExperts(layer);
        };
    }
    pub fn includeOutput(self: *Spans, weights: *const Weights) void {
        for ([_][*c]const Weight{
            weights.output_hc_base, weights.output_hc_fn, weights.output_hc_scale,
            weights.output_norm,    weights.output,       weights.output_hc_norm,
            weights.output_hc_down, weights.output_hc_up,
        }) |weight| self.include(weight);
        // N-gram/PLE rows remain disk-only and never enter GPU model views.
    }
    pub fn finish(self: *Spans) bool {
        if (self.items.items.len == 0 or self.max_tensor_bytes == 0) return false;
        std.mem.sortUnstable(Span, self.items.items, {}, before);
        var count: usize = 0;
        for (self.items.items) |span| {
            if (count == 0 or span.off > self.items.items[count - 1].end or
                span.isolate or self.items.items[count - 1].isolate)
            {
                self.items.items[count] = span;
                count += 1;
            } else self.items.items[count - 1].end = @max(self.items.items[count - 1].end, span.end);
        }
        self.items.items.len = count;
        return count != 0;
    }
    fn take(self: *Spans, out: *Vec) bool {
        if (!self.finish()) return false;
        out.* = .{
            .v = self.items.items.ptr,
            .len = @intCast(self.items.items.len),
            .cap = @intCast(self.items.capacity),
            .max_tensor_bytes = self.max_tensor_bytes,
        };
        self.items = .empty;
        self.max_tensor_bytes = 0;
        return true;
    }
};

fn hasExperts(layer: *const Layer) bool {
    return layer.ffn_gate_exps != null and layer.ffn_up_exps != null and layer.ffn_down_exps != null;
}
fn matches(layer: *const Layer, size_class: ?u64) bool {
    const bytes = Cache.layerBytes(layer) orelse return true;
    const slab = size_class orelse return true;
    return bytes == slab;
}
pub fn uniform(weights: *const Weights, shape: *const Shape, index: u32) bool {
    if (index >= shape.n_layer) return true;
    return matches(&weights.layer[index], Cache.sizeClass(shape, weights));
}
pub fn residentSupported(shape: *const Shape, layer: *const Layer, index: u32) bool {
    if (shape.family != c.DS4_MODEL_FAMILY_GLM_DSA or index < shape.n_leading_dense or !hasExperts(layer)) return false;
    const gate = layer.ffn_gate_exps.*.type;
    const up = layer.ffn_up_exps.*.type;
    const down = layer.ffn_down_exps.*.type;
    if (gate == c.DS4_TENSOR_IQ2_XXS and up == c.DS4_TENSOR_IQ2_XXS and
        (down == c.DS4_TENSOR_IQ2_XXS or down == c.DS4_TENSOR_Q2_K)) return true;
    return gate == up and gate == down and (gate == c.DS4_TENSOR_Q2_K or gate == c.DS4_TENSOR_Q4_K);
}
pub fn residentLayer(shape: *const Shape, policy: *const Policy, layer: *const Layer, index: u32) bool {
    return residentSupported(shape, layer, index) and policy.resident_layers != 0 and
        index >= policy.resident_start and index - policy.resident_start < policy.resident_layers;
}
fn addrEligible(shape: *const Shape, layer: *const Layer, index: u32) bool {
    if (shape.family != c.DS4_MODEL_FAMILY_GLM_DSA or index >= shape.n_layer or index < shape.n_leading_dense or !hasExperts(layer) or
        shape.n_expert_used == 0 or shape.n_expert_used > 8 or shape.n_expert < 128 or
        present("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) return false;
    return true;
}
fn addrLayout(shape: *const Shape, layer: *const Layer, index: u32, is_uniform: bool) bool {
    if (!addrEligible(shape, layer, index) or !is_uniform) return false;
    const gate = layer.ffn_gate_exps.*.type;
    return gate == layer.ffn_up_exps.*.type and gate == layer.ffn_down_exps.*.type and
        (gate == c.DS4_TENSOR_Q2_K or gate == c.DS4_TENSOR_Q4_K);
}
pub fn glmAddrLayout(weights: *const Weights, shape: *const Shape, layer: *const Layer, index: u32) bool {
    if (!addrEligible(shape, layer, index)) return false;
    return addrLayout(shape, layer, index, uniform(weights, shape, index));
}
pub fn glmSelectedLayout(shape: *const Shape, layer: *const Layer, index: u32) bool {
    if (shape.family != c.DS4_MODEL_FAMILY_GLM_DSA or index < shape.n_leading_dense or !hasExperts(layer) or
        shape.n_expert_used == 0 or shape.n_expert_used > 8 or shape.n_expert < 128 or
        present("DS4_METAL_MOE_WRITE_CLAMPED_ACT") or present("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") or
        present("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) return false;
    if (layer.ffn_gate_exps.*.type != c.DS4_TENSOR_IQ2_XXS or layer.ffn_up_exps.*.type != c.DS4_TENSOR_IQ2_XXS) return false;
    return switch (layer.ffn_down_exps.*.type) {
        // Metal's IQ2/Q2 selected-slot down kernel sums six experts. The
        // separate IQ2/IQ2 address-table path below supports up to eight.
        c.DS4_TENSOR_Q2_K => shape.n_expert_used == 6 and !present("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS"),
        c.DS4_TENSOR_IQ2_XXS => !present("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE"),
        else => false,
    };
}
pub fn glmStreamed(weights: *const Weights, shape: *const Shape, layer: *const Layer, index: u32) bool {
    if (shape.family != c.DS4_MODEL_FAMILY_GLM_DSA) return false;
    return glmAddrLayout(weights, shape, layer, index) or glmSelectedLayout(shape, layer, index);
}
pub fn staticSupported(weights: *const Weights, shape: *const Shape, policy: *const Policy) bool {
    if (shape.family == c.DS4_MODEL_FAMILY_GLM_DSA) return policy.static_decode_map;
    const slab = Cache.sizeClass(shape, weights);
    for (weights.layer[0..shape.n_layer]) |*layer| if (hasExperts(layer) and !matches(layer, slab)) return false;
    return true;
}

pub const Mode = enum { full, static, runtime };
const Bounds = struct { first: u32, last: u32 };
fn bounds(shape: *const Shape, first: u32, last: u32) ?Bounds {
    if (first >= shape.n_layer or shape.n_layer > c.DS4_MAX_LAYER) return null;
    const end = if (last == std.math.maxInt(u32)) shape.n_layer - 1 else last;
    return if (end >= shape.n_layer or end < first) null else .{ .first = first, .last = end };
}
pub fn build(weights: *const Weights, shape: *const Shape, policy: *const Policy, first: u32, last: u32, token: bool, output: bool, mode: Mode, out: *Vec) bool {
    const slice = bounds(shape, first, last) orelse return false;
    out.* = std.mem.zeroes(Vec);
    var spans = Spans{};
    defer spans.deinit();
    if (token) spans.include(weights.token_embd);
    const slab = if (mode == .runtime) Cache.sizeClass(shape, weights) else null;
    for (slice.first..slice.last + 1) |index| {
        const layer = &weights.layer[index];
        spans.includeLayer(layer, mode != .full, null);
        if (mode == .runtime) {
            const is_uniform = matches(layer, slab);
            if (!is_uniform or
                (shape.family == c.DS4_MODEL_FAMILY_GLM_DSA and !(addrLayout(shape, layer, @intCast(index), is_uniform) or glmSelectedLayout(shape, layer, @intCast(index)))) or
                residentLayer(shape, policy, layer, @intCast(index))) spans.includeExperts(layer);
        }
    }
    if (output) spans.includeOutput(weights);
    return spans.take(out);
}
pub fn tokenOnly(weights: *const Weights, out: *Vec) bool {
    out.* = std.mem.zeroes(Vec);
    var spans = Spans{};
    defer spans.deinit();
    spans.include(weights.token_embd);
    return spans.take(out);
}
pub fn outputOnly(weights: *const Weights, out: *Vec) bool {
    out.* = std.mem.zeroes(Vec);
    var spans = Spans{};
    defer spans.deinit();
    spans.includeOutput(weights);
    return spans.take(out);
}
pub fn total(vector: *const Vec) u64 {
    var bytes: u64 = 0;
    for (vector.v[0..vector.len]) |span| bytes +|= span.end - span.off;
    return bytes;
}
pub fn staticBytes(weights: *const Weights, shape: *const Shape, token: bool, output: bool, result: *u64) bool {
    result.* = 0;
    var vector = std.mem.zeroes(Vec);
    defer c.free(vector.v);
    const policy = std.mem.zeroes(Policy);
    if (!build(weights, shape, &policy, 0, std.math.maxInt(u32), token, output, .static, &vector)) return false;
    result.* = total(&vector);
    return true;
}
