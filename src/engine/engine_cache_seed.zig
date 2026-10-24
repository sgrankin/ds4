//! Startup/repeated-session hotlist seeding for native model families. Their
//! shipped profiles have no legacy built-in ranked list; an explicit hotlist
//! retains the existing three-column FILE format and cache priority semantics.
const std = @import("std");
const c = @import("gguf_c");
const sizes = @import("engine_cache_sizes.zig");
const Shape = c.ds4_shape;
const Core = c.ds4_engine_core;
const Layer = c.ds4_layer_weights;
const max_layers = c.DS4_MAX_LAYER;
const max_experts = c.DS4_MAX_EXPERT;

fn text(pointer: [*c]const u8) bool {
    return pointer != null and pointer[0] != 0;
}
fn enabled(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn eligible(core: *const Core, shape: *const Shape, layer: *const Layer) bool {
    if (!core.ssd_streaming or core.quality or layer.ffn_gate_exps == null or
        layer.ffn_up_exps == null or layer.ffn_down_exps == null) return false;
    const gate = layer.ffn_gate_exps.*.type;
    const up = layer.ffn_up_exps.*.type;
    const down = layer.ffn_down_exps.*.type;
    if (shape.n_expert_used == 6 and shape.n_expert >= 128 and
        !enabled("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and !enabled("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION"))
    {
        if (gate == c.DS4_TENSOR_IQ2_XXS and up == gate and down == c.DS4_TENSOR_Q2_K and
            !enabled("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS")) return true;
        if (gate == c.DS4_TENSOR_MXFP4 and up == gate and down == gate and
            !enabled("DS4_METAL_DISABLE_MXFP4_SELECTED_EXPERT_VIEWS")) return true;
    }
    return shape.family == c.DS4_MODEL_FAMILY_GLM_DSA and shape.n_expert_used != 0 and
        shape.n_expert_used <= 8 and shape.n_expert >= 128 and
        !enabled("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE") and gate == up and up == down and
        (gate == c.DS4_TENSOR_IQ2_XXS or gate == c.DS4_TENSOR_Q2_K or gate == c.DS4_TENSOR_Q4_K);
}
fn preload(core: *const Core, shape: *const Shape, budget: u32) u32 {
    var count = core.memory.ssd_streaming_preload_experts;
    if (count == 0) {
        count = budget;
        const value = c.getenv("DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP");
        var cap: u32 = 4096;
        if (text(value)) {
            var end: [*c]u8 = null;
            const requested = c.strtoul(value, &end, 10);
            if (end != value and end[0] == 0) cap = @intCast(@min(requested, std.math.maxInt(u32)));
        }
        if (cap != 0) count = @min(count, cap);
    }
    return @intCast(@min(count, budget, @as(u64, shape.n_layer) * shape.n_expert));
}

pub const Hotlist = struct {
    experts: [max_layers][max_experts]i32,
    priorities: [max_layers][max_experts]u32,
    counts: [max_layers]u32,
    seen: [max_layers][max_experts]bool,
    loaded: u32,

    fn add(self: *Hotlist, shape: *const Shape, layer: u32, expert: u32, priority: u32) bool {
        if (layer >= shape.n_layer or expert >= shape.n_expert or layer >= max_layers or expert >= max_experts) return true;
        if (self.seen[layer][expert]) return true;
        const count = self.counts[layer];
        if (count >= max_experts) return false;
        self.seen[layer][expert] = true;
        self.experts[layer][count] = @intCast(expert);
        self.priorities[layer][count] = @max(priority, 1);
        self.counts[layer] += 1;
        self.loaded += 1;
        return true;
    }
};
fn skipSpace(pointer: *[*c]u8) void {
    while (pointer.*[0] != 0 and c.isspace(pointer.*[0]) != 0) pointer.* += 1;
}
fn number(pointer: *[*c]u8) ?c_ulonglong {
    skipSpace(pointer);
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const value = c.strtoull(pointer.*, &end, 10);
    if (end == pointer.* or c.__error().* != 0) return null;
    pointer.* = end;
    return value;
}
pub fn loadFile(list: *Hotlist, shape: *const Shape, path: [*:0]const u8, maximum: u32) bool {
    const stream = c.fopen(path, "rb") orelse {
        _ = c.fprintf(c.__stderrp, "ds4: failed to open streaming expert hotlist %s: %s\n", path, c.strerror(c.__error().*));
        return false;
    };
    defer _ = c.fclose(stream);
    var line: [256]u8 = undefined;
    var line_number: u64 = 0;
    while (c.fgets(&line, line.len, stream) != null) {
        if (list.loaded >= maximum) break;
        line_number += 1;
        var pointer: [*c]u8 = &line;
        skipSpace(&pointer);
        if (pointer[0] == 0 or pointer[0] == '#') continue;
        const layer = number(&pointer);
        const expert = if (layer != null) number(&pointer) else null;
        const hits = if (expert != null) number(&pointer) else null;
        if (hits != null) {
            if (hits.? == 0) continue;
            if (list.add(shape, @truncate(layer.?), @truncate(expert.?), @intCast(@min(hits.?, std.math.maxInt(u32))))) continue;
        }
        _ = c.fprintf(c.__stderrp, "ds4: invalid streaming expert hotlist line %llu in %s\n", @as(c_ulonglong, line_number), path);
        return false;
    }
    if (c.ferror(stream) != 0) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to read streaming expert hotlist %s: %s\n", path, c.strerror(c.__error().*));
        return false;
    }
    if (list.loaded == 0) _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist %s had no usable nonzero entries\n", path);
    return true;
}
fn time() f64 {
    var value: c.struct_timeval = undefined;
    _ = c.gettimeofday(&value, null);
    return @as(f64, @floatFromInt(value.tv_sec)) + @as(f64, @floatFromInt(value.tv_usec)) / 1.0e6;
}
pub fn seed(core: *const Core, shape: *const Shape) bool {
    if (!core.ssd_streaming or core.ssd_streaming_cold or enabled("DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST")) return true;
    var cache_budget: u32 = 0;
    for (core.weights.layer[0..shape.n_layer], 0..) |*layer, index| {
        if (!eligible(core, shape, layer)) continue;
        const gate_row = sizes.rowBytes(layer.ffn_gate_exps);
        const down_row = sizes.rowBytes(layer.ffn_down_exps);
        const gate = std.math.mul(u64, layer.ffn_gate_exps.*.dim[1], gate_row) catch {
            _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist budget byte size overflow at layer %u\n", @as(u32, @intCast(index)));
            return false;
        };
        const down = std.math.mul(u64, layer.ffn_down_exps.*.dim[1], down_row) catch {
            _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist budget byte size overflow at layer %u\n", @as(u32, @intCast(index)));
            return false;
        };
        cache_budget = c.ds4_gpu_stream_expert_cache_budget_for_expert_size(gate, down);
        break;
    }
    if (cache_budget == 0) return true;
    const count = preload(core, shape, cache_budget);
    if (count == 0) return true;
    const path = c.getenv("DS4_METAL_STREAMING_EXPERT_HOTLIST");
    const profile = enabled("DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE");
    const current = c.ds4_gpu_stream_expert_cache_current_count();
    if (!text(path) and current >= count) {
        if (profile) _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist seed skipped preload=%u current=%u\n", count, current);
        return true;
    }
    // Pro/Flash/GLM52 built-ins belong to legacy model routes. None of the
    // native four model profiles selected here has one of those ranked lists.
    if (!text(path)) return true;
    var list = std.mem.zeroes(Hotlist);
    if (!loadFile(&list, shape, @ptrCast(path), count)) return false;
    if (list.loaded == 0) return true;
    const started = if (profile) time() else 0;
    var seeded_layers: u32 = 0;
    var seeded_experts: u32 = 0;
    for (core.weights.layer[0..shape.n_layer], 0..) |*layer, index| {
        const n = list.counts[index];
        if (n == 0 or !eligible(core, shape, layer)) continue;
        const gate = std.math.mul(u64, layer.ffn_gate_exps.*.dim[1], sizes.rowBytes(layer.ffn_gate_exps)) catch {
            _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist seed byte size overflow at layer %u\n", @as(u32, @intCast(index)));
            return false;
        };
        const down = std.math.mul(u64, layer.ffn_down_exps.*.dim[1], sizes.rowBytes(layer.ffn_down_exps)) catch {
            _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist seed byte size overflow at layer %u\n", @as(u32, @intCast(index)));
            return false;
        };
        const table = c.ds4_gpu_stream_expert_table{
            .model_map = core.model.map,
            .model_size = core.model.size,
            .layer = @intCast(index),
            .n_total_expert = shape.n_expert,
            .gate_offset = layer.ffn_gate_exps.*.abs_offset,
            .up_offset = layer.ffn_up_exps.*.abs_offset,
            .down_offset = layer.ffn_down_exps.*.abs_offset,
            .gate_expert_bytes = gate,
            .down_expert_bytes = down,
        };
        if (c.ds4_gpu_stream_expert_cache_seed_experts(&table, &list.experts[index], &list.priorities[index], n) == 0) return false;
        seeded_layers += 1;
        seeded_experts += n;
    }
    if (profile)
        _ = c.fprintf(c.__stderrp, "ds4: streaming expert hotlist seed source=%s preload=%u loaded=%u layers=%u experts=%u time=%.3f ms\n", path, count, list.loaded, seeded_layers, seeded_experts, (time() - started) * 1000);
    return true;
}
