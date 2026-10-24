//! Metal GLM admission and context-aware cache fitting. Mutable accounting is
//! the actual Core/Pool/streaming-policy storage. Weight-span selection owns
//! which tensors remain mapped; this owner measures those spans and budgets
//! model, live/future graphs, prefill reserve and complete expert-cache slots.
const std = @import("std");
const c = @import("gguf_c");
const sizes = @import("engine_cache_sizes.zig");
const accounting = @import("session_resource_accounting.zig");
const guard_math = @import("engine_admission_math.zig");
const Core = c.ds4_engine_core;
const Shape = c.ds4_shape;
const Policy = c.ds4_streaming_policy;
const Placement = c.ds4_model_residency_inputs;
const Vec = c.ds4_model_map_span_vec;
const gib: u64 = 1024 * 1024 * 1024;

fn asGiB(bytes: u64) f64 {
    @setFloatMode(.optimized);
    return @as(f64, @floatFromInt(bytes)) / gib;
}
fn hasText(pointer: [*c]const u8) bool {
    return pointer != null and pointer[0] != 0;
}
fn environment(name: [*:0]const u8, fallback: f64, minimum: f64, maximum: f64) f64 {
    const value = c.getenv(name);
    if (!hasText(value)) return fallback;
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const number = c.strtod(value, &end);
    if (end == value or c.__error().* != 0 or !std.math.isFinite(number)) return fallback;
    return @max(minimum, @min(maximum, number));
}
pub fn guardDisabled() bool {
    const value = c.getenv("DS4_GLM_MEMORY_GUARD");
    if (!hasText(value)) return false;
    const text = std.mem.span(value);
    return std.mem.eql(u8, text, "0") or std.ascii.eqlIgnoreCase(text, "false") or
        std.ascii.eqlIgnoreCase(text, "off") or std.ascii.eqlIgnoreCase(text, "no");
}
pub const defaultReserve = guard_math.defaultReserve;
pub const budgetBytes = guard_math.budgetBytes;
const Budget = struct { base: u64, bytes: u64, fraction: f64, reserve: f64 };
fn budget(model_bytes: u64, shape: *const Shape) ?Budget {
    var base: u64 = 0;
    var length: usize = @sizeOf(u64);
    if (c.sysctlbyname("hw.memsize", &base, &length, null, 0) != 0) base = 0;
    if (base == 0) base = c.ds4_gpu_recommended_working_set_size();
    if (base == 0) return null;
    var wired_mb: i64 = 0;
    length = @sizeOf(i64);
    if (c.sysctlbyname("iogpu.wired_limit_mb", &wired_mb, &length, null, 0) != 0) wired_mb = 0;
    const wired = if (wired_mb > 0) @as(u64, @intCast(wired_mb)) * (1024 * 1024) else 0;
    const fraction = environment("DS4_GLM_MEMORY_GUARD_FRACTION", 0.99, 0.50, 1);
    const reserve = environment("DS4_GLM_MEMORY_GUARD_RESERVE_GB", defaultReserve(base, model_bytes, shape.variant == c.DS4_VARIANT_GLM53), 0, 1024);
    return .{ .base = base, .bytes = budgetBytes(base, wired, fraction, reserve), .fraction = fraction, .reserve = reserve };
}
fn total(vector: *const Vec) u64 {
    var bytes: u64 = 0;
    for (vector.v[0..vector.len]) |span| bytes +|= span.end - span.off;
    return bytes;
}
fn release(vector: *Vec) void {
    c.free(vector.v);
    vector.* = std.mem.zeroes(Vec);
}
pub fn activeModelBytes(core: *const Core, shape: *const Shape) u64 {
    var maximum: u64 = 0;
    var group: [*:0]const u8 = "none";
    var maximum_layer: u32 = std.math.maxInt(u32);
    var vector: Vec = std.mem.zeroes(Vec);
    if (c.ds4_native_model_spans_token(&core.weights, &vector)) {
        maximum = total(&vector);
        group = "token";
        release(&vector);
    }
    if (c.ds4_native_model_spans_output(&core.weights, &vector)) {
        const bytes = total(&vector);
        if (bytes > maximum) {
            maximum = bytes;
            group = "output";
        }
        release(&vector);
    }
    for (0..shape.n_layer) |layer| {
        if (!c.ds4_native_model_spans_full(&core.weights, shape, @intCast(layer), @intCast(layer), false, &vector)) continue;
        const bytes = total(&vector);
        if (bytes > maximum) {
            maximum = bytes;
            group = "layer";
            maximum_layer = @intCast(layer);
        }
        release(&vector);
    }
    if (hasText(c.getenv("DS4_GLM_MEMORY_GUARD_REPORT"))) {
        if (maximum_layer == std.math.maxInt(u32)) {
            _ = c.fprintf(c.__stderrp, "ds4: GLM streaming largest active model group %s: %.2f GiB\n", group, asGiB(maximum));
        } else {
            _ = c.fprintf(c.__stderrp, "ds4: GLM streaming largest active model group layer %u: %.2f GiB\n", maximum_layer, asGiB(maximum));
        }
    }
    return maximum;
}
pub fn modelBytes(core: *const Core, shape: *const Shape, placement: *const Placement) u64 {
    // Ordinary single-Metal inputs exclude TP. Its shard override stays with
    // that separate compatibility route, rather than another global here.
    if (placement.load_slice) {
        var vector: Vec = std.mem.zeroes(Vec);
        const valid = if (core.ssd_streaming)
            c.ds4_native_model_spans_static(&core.weights, shape, placement.layer_start, placement.layer_end, placement.layer_start == 0, placement.load_output, &vector)
        else
            c.ds4_native_model_spans_full(&core.weights, shape, placement.layer_start, placement.layer_end, placement.load_output, &vector);
        if (valid) {
            const bytes = total(&vector);
            release(&vector);
            if (bytes != 0) return bytes;
        }
    }
    if (!core.ssd_streaming) return core.model.size;
    const active = activeModelBytes(core, shape);
    return if (active != 0) active else core.model.size;
}
fn geometry(core: *const Core, shape: *const Shape, placement: *const Placement, compact: u32) c.ds4_context_memory {
    const context: u32 = @intCast(placement.context);
    const work = c.ds4_native_glm_full_attention_cap(context, core.ssd_streaming);
    const first: u32 = if (placement.load_slice) placement.layer_start else 0;
    const last = if (placement.load_slice) placement.layer_end else shape.n_layer - shape.n_nextn_predict - 1;
    return c.ds4_native_glm_memory(shape, context, work, compact, first, last);
}
pub fn admit(core: *const Core, shape: *const Shape, placement: *const Placement, compact: u32, extra: u64, phase: [*c]const u8) bool {
    if (guardDisabled()) return true;
    const memory = geometry(core, shape, placement, compact);
    const model = modelBytes(core, shape, placement);
    const required = model +| memory.total_bytes +| extra;
    const limit = budget(model, shape) orelse return true;
    if (required <= limit.bytes) {
        if (hasText(c.getenv("DS4_GLM_MEMORY_GUARD_REPORT"))) {
            _ = c.fprintf(c.__stderrp, "ds4: GLM memory guard ctx=%u compact_cap=%u required=%.2f GiB budget=%.2f GiB (model %.2f GiB, graph %.2f GiB, transient %.2f GiB)\n", @as(u32, @intCast(placement.context)), memory.comp_cap, asGiB(required), asGiB(limit.bytes), asGiB(model), asGiB(memory.total_bytes), asGiB(extra));
            if ((core.ssd_streaming or placement.load_slice) and model != core.model.size)
                _ = c.fprintf(c.__stderrp, "ds4:   GLM %s guard uses %s model span %.2f GiB (full GGUF %.2f GiB)\n", @as([*:0]const u8, if (core.ssd_streaming) "streaming" else "memory"), @as([*:0]const u8, if (core.ssd_streaming) "active" else "sliced"), asGiB(model), asGiB(core.model.size));
        }
        return true;
    }
    _ = c.fprintf(c.__stderrp, "ds4: GLM memory guard refused ctx=%u compact_cap=%u %s\n", @as(u32, @intCast(placement.context)), memory.comp_cap, if (phase != null) phase else @as([*:0]const u8, "before Metal graph allocation"));
    if ((core.ssd_streaming or placement.load_slice) and model != core.model.size) {
        _ = c.fprintf(c.__stderrp, "ds4:   %s model map: %.2f GiB (full GGUF %.2f GiB)\n", @as([*:0]const u8, if (core.ssd_streaming) "streamed active" else "sliced"), asGiB(model), asGiB(core.model.size));
    } else _ = c.fprintf(c.__stderrp, "ds4:   model map: %.2f GiB\n", asGiB(model));
    _ = c.fprintf(c.__stderrp, "ds4:   graph cache/scratch: %.2f GiB (full KV %.2f GiB, compact DSA %.2f GiB, scratch %.2f GiB)\n", asGiB(memory.total_bytes), asGiB(memory.raw_bytes), asGiB(memory.compressed_bytes), asGiB(memory.scratch_bytes));
    _ = c.fprintf(c.__stderrp, "ds4:   required model+graph: %.2f GiB; guard budget: %.2f GiB (base %.2f GiB, fraction %.2f, reserve %.2f GiB, transient %.2f GiB)\n", asGiB(required), asGiB(limit.bytes), asGiB(limit.base), limit.fraction, limit.reserve, asGiB(extra));
    _ = c.fputs("ds4:   set DS4_GLM_MEMORY_GUARD=0 to bypass, use a smaller --ctx, tensor parallelism, or SSD streaming\n", c.__stderrp);
    return false;
}

fn runtimeMapBytes(core: *const Core, shape: *const Shape, policy: *const Policy) ?u64 {
    var vector: Vec = std.mem.zeroes(Vec);
    if (!c.ds4_native_model_spans_runtime(&core.weights, shape, policy, 0, shape.n_layer - 1, true, true, &vector)) return null;
    defer release(&vector);
    const bytes = total(&vector);
    return if (bytes != 0) bytes else null;
}
fn prefixBytes(core: *const Core, shape: *const Shape, policy: *const Policy, layers: u32) ?u64 {
    if (shape.n_expert == 0) return null;
    var bytes: u64 = 0;
    for (0..layers) |index| {
        const layer = policy.resident_start + @as(u32, @intCast(index));
        if (!c.ds4_native_model_spans_resident_layer(shape, policy, &core.weights.layer[layer], layer)) return null;
        const expert = sizes.layerBytes(&core.weights.layer[layer]) orelse return null;
        const full = std.math.mul(u64, expert, shape.n_expert) catch return null;
        bytes = std.math.add(u64, bytes, full) catch return null;
    }
    return bytes;
}
pub fn fit(core: *Core, pool: *const c.ds4_session_resource_pool, shape: *const Shape, policy: *Policy, placement: *const Placement, session_hint: c_int) void {
    if (!core.ssd_streaming or placement.context <= 0) return;
    core.ssd_streaming_static_decode_map = false;
    core.memory.ssd_streaming_decode_map_bytes = 0;
    core.memory.ssd_streaming_decode_map_extra_bytes = 0;
    policy.static_decode_map = false;
    var context: u32 = undefined;
    if (!c.ds4_native_glm_context_request(shape, placement.context, &context)) return;
    const memory = geometry(core, shape, placement, context);
    const model = modelBytes(core, shape, placement);
    const limit = budget(model, shape) orelse return;
    const fixed = model +| accounting.graphBudget(pool.glm_session_graph_bytes, pool.glm_session_count, session_hint, memory.total_bytes);
    const available = limit.bytes -| fixed;
    const slab = sizes.sizeClass(shape, &core.weights);
    const old_layers = core.memory.ssd_streaming_full_layers;
    const old_experts = core.memory.ssd_streaming_cache_experts;
    const old_headroom = core.memory.ssd_streaming_prefill_headroom_bytes;
    const cache_serves_decode = slab == null or (shape.n_expert_used != 0 and old_experts >= shape.n_expert_used);
    const minimum_cache = if (slab != null and cache_serves_decode) slab.? * shape.n_expert_used else 0;
    var full_layers: u32 = 0;
    var full_bytes: u64 = 0;
    var decode_bytes: u64 = 0;
    var decode_extra: u64 = 0;
    var static = false;
    if (c.getenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP") == null and cache_serves_decode) {
        var candidate = old_layers;
        while (true) {
            policy.resident_layers = candidate;
            if (runtimeMapBytes(core, shape, policy)) |map_bytes| {
                const extra = map_bytes -| model;
                if (extra +| old_headroom +| minimum_cache <= available) {
                    static = true;
                    full_layers = candidate;
                    decode_bytes = map_bytes;
                    decode_extra = extra;
                    break;
                }
            }
            if (candidate == 0) break;
            candidate -= 1;
        }
    }
    if (!static) policy.resident_layers = 0;
    if (full_layers != 0) {
        if (prefixBytes(core, shape, policy, full_layers)) |bytes| {
            full_bytes = bytes;
        } else {
            full_layers = 0;
            static = false;
            decode_bytes = 0;
            decode_extra = 0;
            policy.resident_layers = 0;
        }
    }
    var option_bytes = available -| decode_extra;
    const headroom = @min(old_headroom, option_bytes);
    option_bytes -= headroom;
    var experts = old_experts;
    var dynamic = sizes.dynamicBytes(&core.memory, true, shape, &core.weights);
    if (slab) |per_expert| if (dynamic > option_bytes) {
        experts = @intCast(@min(option_bytes / per_expert, old_experts, std.math.maxInt(u32)));
        dynamic = @as(u64, experts) * per_expert;
    };
    core.memory.ssd_streaming_full_layers = full_layers;
    core.memory.ssd_streaming_full_layer_bytes = full_bytes;
    core.memory.ssd_streaming_prefill_headroom_bytes = headroom;
    core.memory.ssd_streaming_cache_experts = experts;
    if (core.memory.ssd_streaming_cache_bytes != 0) core.memory.ssd_streaming_cache_bytes = dynamic;
    core.ssd_streaming_static_decode_map = static;
    core.memory.ssd_streaming_decode_map_bytes = decode_bytes;
    core.memory.ssd_streaming_decode_map_extra_bytes = decode_extra;
    policy.resident_layers = full_layers;
    policy.static_decode_map = static;
    if (old_layers != full_layers or old_experts != experts or old_headroom != headroom)
        _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming request adjusted to fit memory: full layers %u -> %u, cache %u -> %u experts, prefill reserve %.2f -> %.2f GiB\n", old_layers, full_layers, old_experts, experts, asGiB(old_headroom), asGiB(headroom));
    if (static) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming decode map: global %.2f GiB (%.2f GiB beyond the active layer window)\n", asGiB(decode_bytes), asGiB(decode_extra));
    } else _ = c.fputs("ds4: GLM SSD streaming decode map: per-layer fallback\n", c.__stderrp);
}
