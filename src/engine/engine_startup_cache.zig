//! Metal startup cache policy. Automatic planning keeps fixed weights inside
//! the working-set target; configured byte budgets first reserve prefill and
//! explicit resident layers, then round down to complete dynamic expert slots.
//! DS41's model owner retains its existing policy; these are the ordinary
//! shared/GLM startup algorithms, with no Engine layout or process globals.
const std = @import("std");
const c = @import("gguf_c");
const sizes = @import("engine_cache_sizes.zig");
const Core = c.ds4_engine_core;
const Shape = c.ds4_shape;
const Policy = c.ds4_streaming_policy;
const gib: u64 = 1024 * 1024 * 1024;
fn asGiB(bytes: u64) f64 {
    @setFloatMode(.optimized);
    return @as(f64, @floatFromInt(bytes)) / gib;
}
fn fail(message: [*:0]const u8) bool {
    _ = c.fputs(message, c.__stderrp);
    return false;
}
pub fn safeBytes(core: *const Core, shape: *const Shape, context: c_int, _: u32, recommended: u64) u64 {
    if (recommended == 0) return 0;
    // GLM's prefill/workspace geometry already owns its fixed chunk policy.
    // Qwen rejects SSD streaming; DS41 uses its model's existing Memory owner.
    const rows: u32 = if (context > 0) @intCast(context) else 1;
    const work = c.ds4_native_glm_full_attention_cap(rows, core.ssd_streaming);
    const last = shape.n_layer - shape.n_nextn_predict - 1;
    const memory = c.ds4_native_glm_memory(shape, rows, work, rows, 0, last);
    const target = if (recommended > std.math.maxInt(u64) / 7) std.math.maxInt(u64) else recommended * 7 / 8;
    const rounded = (target -| memory.total_bytes) / gib * gib;
    return @max(rounded, gib);
}
pub fn manualSafe(core: *const Core, shape: *const Shape, context: c_int, chunk: u32) u64 {
    return safeBytes(core, shape, context, chunk, c.ds4_gpu_recommended_working_set_size());
}
pub fn automaticWithBudget(core: *Core, shape: *const Shape, recommended: u64, model_limit: u64) bool {
    if (!core.ssd_streaming or core.memory.ssd_streaming_cache_experts != 0 or core.memory.ssd_streaming_cache_bytes != 0) return true;
    if (recommended == 0) return fail("ds4: SSD streaming auto cache: recommended working set unavailable; set --ssd-streaming-cache-experts N or NGB explicitly\n");
    var non_routed: u64 = 0;
    if (!c.ds4_native_model_spans_static_bytes(&core.weights, shape, true, true, &non_routed))
        return fail("ds4: SSD streaming auto cache could not measure non-routed model weights\n");
    const slab = sizes.sizeClass(shape, &core.weights) orelse
        return fail("ds4: SSD streaming auto cache could not measure routed expert size\n");
    const count = sizes.cacheable(shape, &core.weights) orelse
        return fail("ds4: SSD streaming auto cache could not count local cacheable experts\n");
    const glm = shape.family == c.DS4_MODEL_FAMILY_GLM_DSA;
    var plan: c.ds4_ssd_cache_plan = undefined;
    if (!c.ds4_ssd_auto_cache_plan(recommended, if (glm) 80 else 86, model_limit, non_routed, slab, count.experts, &plan))
        return fail("ds4: SSD streaming auto cache could not compute a valid cache budget\n");
    core.memory.ssd_streaming_cache_experts = plan.cache_experts;
    core.memory.ssd_streaming_cache_bytes = plan.effective_cache_bytes;
    _ = c.fputs("ds4: SSD streaming auto cache budget\n", c.__stderrp);
    _ = c.fprintf(c.__stderrp, "ds4:   metal recommends %.2f GiB working set\n", asGiB(recommended));
    _ = c.fprintf(c.__stderrp, "ds4:   using %.0f%% total for model + cached experts: %.2f GiB\n", 100 * @as(f64, @floatFromInt(plan.model_target_bytes)) / @as(f64, @floatFromInt(recommended)), asGiB(plan.model_target_bytes));
    _ = c.fprintf(c.__stderrp, "ds4:   non-routed weights: %.2f GiB\n", asGiB(non_routed));
    _ = c.fprintf(c.__stderrp, "ds4:   routed expert size: %.2f MiB\n", @as(f64, @floatFromInt(slab)) / (1024 * 1024));
    _ = c.fprintf(c.__stderrp, "ds4:   expert budget before %sreserve: %u (%.2f GiB)\n", @as([*:0]const u8, if (glm) "prefill/full-layer " else "prefill "), plan.cache_experts, asGiB(plan.effective_cache_bytes));
    if (plan.model_target_bytes <= non_routed)
        _ = c.fputs("ds4:   note: non-routed weights already fill the target; keeping a one-expert cache\n", c.__stderrp);
    return true;
}
pub fn automatic(core: *Core, shape: *const Shape, context: c_int, chunk: u32) bool {
    if (!core.ssd_streaming or core.memory.ssd_streaming_cache_experts != 0 or core.memory.ssd_streaming_cache_bytes != 0) return true;
    const recommended = c.ds4_gpu_recommended_working_set_size();
    if (recommended == 0) return automaticWithBudget(core, shape, 0, 0);
    const model_limit = manualSafe(core, shape, if (context > 0) context else 4096, chunk);
    return automaticWithBudget(core, shape, recommended, model_limit);
}

const Prefix = struct { start: u32, count: u32 };
fn supportedPrefix(core: *const Core, shape: *const Shape) Prefix {
    const normal = if (shape.n_nextn_predict != 0 and shape.n_layer > shape.n_nextn_predict)
        shape.n_layer - shape.n_nextn_predict
    else
        shape.n_layer;
    if (normal <= shape.n_leading_dense) return .{ .start = 0, .count = 0 };
    var first = shape.n_leading_dense;
    while (first < normal and !c.ds4_native_model_spans_resident_supported(shape, &core.weights.layer[first], first)) first += 1;
    if (first == normal) return .{ .start = 0, .count = 0 };
    var count: u32 = 0;
    while (first + count < normal and c.ds4_native_model_spans_resident_supported(shape, &core.weights.layer[first + count], first + count)) count += 1;
    return .{ .start = first, .count = count };
}
fn prefixBytes(core: *const Core, shape: *const Shape, first: u32, count: u32) ?u64 {
    if (shape.n_expert == 0) return null;
    var bytes: u64 = 0;
    for (0..count) |index| {
        const layer = first + @as(u32, @intCast(index));
        if (!c.ds4_native_model_spans_resident_supported(shape, &core.weights.layer[layer], layer)) return null;
        const expert = sizes.layerBytes(&core.weights.layer[layer]) orelse return null;
        const full = std.math.mul(u64, expert, shape.n_expert) catch return null;
        bytes = std.math.add(u64, bytes, full) catch return null;
    }
    return bytes;
}
pub fn reserve(core: *Core, shape: *const Shape, policy: *Policy) bool {
    policy.* = std.mem.zeroes(Policy);
    core.ssd_streaming_static_decode_map = false;
    core.memory.ssd_streaming_decode_map_bytes = 0;
    core.memory.ssd_streaming_decode_map_extra_bytes = 0;
    if (!core.ssd_streaming) return true;
    const glm = shape.family == c.DS4_MODEL_FAMILY_GLM_DSA;
    const memory = &core.memory;
    if (core.ssd_streaming_full_layers_set and memory.ssd_streaming_full_layers != 0 and !glm) {
        _ = c.fputs("ds4: --ssd-streaming-full-layers is currently supported only for GLM graph SSD streaming; ignoring\n", c.__stderrp);
        memory.ssd_streaming_full_layers = 0;
    }
    const need_expert = memory.ssd_streaming_cache_experts != 0 or memory.ssd_streaming_cache_bytes != 0 or
        (glm and !core.ssd_streaming_full_layers_set) or memory.ssd_streaming_full_layers != 0;
    const slab = if (need_expert) sizes.sizeClass(shape, &core.weights) orelse
        return fail("ds4: SSD streaming could not measure routed expert size\n") else 0;
    const count = if (need_expert) sizes.cacheable(shape, &core.weights) orelse
        return fail("ds4: SSD streaming could not count local cacheable experts\n") else sizes.Count{ .experts = 0, .layers = 0 };
    const maximum: u32 = @intCast(@min(count.experts, std.math.maxInt(u32)));
    if (memory.ssd_streaming_cache_bytes == 0 and maximum != 0 and memory.ssd_streaming_cache_experts > maximum) {
        _ = c.fprintf(c.__stderrp, "ds4: SSD streaming expert cache capped from %u to %u experts cacheable by this layer slice\n", memory.ssd_streaming_cache_experts, maximum);
        memory.ssd_streaming_cache_experts = maximum;
    }
    const target = memory.ssd_streaming_cache_bytes;
    var headroom: u64 = 0;
    if (target != 0) {
        headroom = sizes.prefillHeadroom(shape, &core.weights) orelse
            return fail("ds4: SSD streaming prefill headroom byte accounting failed\n");
        if (headroom >= target) {
            const fitted = target -| slab;
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming prefill reserve reduced from %.2f to %.2f GiB to honor the %.2f GiB cache target\n", asGiB(headroom), asGiB(fitted), asGiB(target));
            headroom = fitted;
        }
    }
    memory.ssd_streaming_prefill_headroom_bytes = headroom;
    const after_prefill = target - headroom;
    const supported = if (glm) supportedPrefix(core, shape) else Prefix{ .start = 0, .count = 0 };
    const automatic_layers = glm and !core.ssd_streaming_full_layers_set and target != 0;
    if (automatic_layers) memory.ssd_streaming_full_layers = 0;
    var full_layers: u32 = 0;
    var full_bytes: u64 = 0;
    if (memory.ssd_streaming_full_layers != 0) {
        const requested = memory.ssd_streaming_full_layers;
        full_layers = @min(requested, supported.count);
        if (full_layers != requested)
            _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers capped from %u to %u supported routed prefix layers\n", requested, full_layers);
        if (target != 0 and slab != 0) {
            const minimum = std.math.mul(u64, @max(shape.n_expert, 1), slab) catch
                return fail("ds4: SSD streaming full-layer budget overflow\n");
            const before = full_layers;
            while (full_layers != 0) {
                const bytes = prefixBytes(core, shape, supported.start, full_layers) orelse
                    return fail("ds4: SSD streaming full-layer byte accounting failed\n");
                if (after_prefill > minimum and bytes <= after_prefill - minimum) {
                    full_bytes = bytes;
                    break;
                }
                full_layers -= 1;
            }
            if (full_layers != before)
                _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers capped from %u to %u by %.2f GiB total expert budget\n", before, full_layers, asGiB(target));
        } else if (full_layers != 0) {
            full_bytes = prefixBytes(core, shape, supported.start, full_layers) orelse
                return fail("ds4: SSD streaming full-layer byte accounting failed\n");
        }
    }
    if (target != 0) {
        if (full_bytes != 0 and full_bytes >= after_prefill)
            return fail("ds4: SSD streaming full-layer budget leaves no dynamic expert cache\n");
        const dynamic = after_prefill - full_bytes;
        var experts = c.ds4_ssd_cache_experts_for_byte_budget(dynamic, slab);
        if (maximum != 0) experts = @min(experts, maximum);
        if (experts == 0 or slab == 0) {
            _ = c.fputs("ds4: SSD streaming cache target leaves no complete expert slot; using direct per-layer reads\n", c.__stderrp);
            experts = 0;
        }
        memory.ssd_streaming_cache_experts = experts;
        memory.ssd_streaming_cache_bytes = @as(u64, experts) * slab;
        const slot_bytes = if (experts != 0) slab else 0;
        const effective = headroom +| full_bytes +| memory.ssd_streaming_cache_bytes;
        if (full_layers != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers: %u (%.2f GiB, %s)\n", full_layers, asGiB(full_bytes), @as([*:0]const u8, if (automatic_layers) "auto" else "explicit"));
            _ = c.fprintf(c.__stderrp, "ds4: metal SSD streaming cache target %.2f GiB; effective %.2f GiB = %.2f GiB prefill headroom + %.2f GiB full layers + %.2f GiB dynamic cache (%u experts, %.2f MiB each)\n", asGiB(target), asGiB(effective), asGiB(headroom), asGiB(full_bytes), asGiB(memory.ssd_streaming_cache_bytes), experts, @as(f64, @floatFromInt(slot_bytes)) / (1024 * 1024));
        } else {
            if (glm) _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers: 0 (%s)\n", @as([*:0]const u8, if (core.ssd_streaming_full_layers_set) "disabled by --ssd-streaming-full-layers 0" else "auto selected 0 layers"));
            _ = c.fprintf(c.__stderrp, "ds4: metal SSD streaming cache target %.2f GiB; effective %.2f GiB = %.2f GiB prefill headroom + %.2f GiB dynamic cache (%u experts, %.2f MiB each)\n", asGiB(target), asGiB(effective), asGiB(headroom), asGiB(memory.ssd_streaming_cache_bytes), experts, @as(f64, @floatFromInt(slot_bytes)) / (1024 * 1024));
        }
    } else if (full_layers != 0) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers: %u (%.2f GiB, %s), dynamic cache budget: %u experts\n", full_layers, asGiB(full_bytes), @as([*:0]const u8, if (automatic_layers) "auto" else "explicit"), memory.ssd_streaming_cache_experts);
    } else if (glm) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers: 0 (%s)\n", @as([*:0]const u8, if (core.ssd_streaming_full_layers_set) "disabled by --ssd-streaming-full-layers 0" else "not auto-tuned for expert-count cache budgets"));
    }
    memory.ssd_streaming_full_layers = full_layers;
    memory.ssd_streaming_full_layer_bytes = full_bytes;
    policy.resident_start = supported.start;
    policy.resident_layers = full_layers;
    return true;
}
