//! Recurrent rollback snapshots. Attention KV is append-only and masked by pos;
//! the mutable GDN/conv/PLE state must be copied or swapped on draft rejection.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");
const resources = @import("resources.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Tensor = c.ds4_gpu_tensor;

// Names match the historical snapshot indices: first is after verify row0,
// second after row1, boundary is the pre-verify exact-sampling rewind point.
pub const Set = enum(u32) { boundary = 0, first = 1, second = 2 };

const View = struct {
    state: *[c.DS4_MAX_LAYER]?*Tensor,
    history: *[c.DS4_MAX_LAYER]?*Tensor,
    ple: *?*Tensor,
    previous: *[c.DS4_MAX_PLE_NGRAM]c_int,
    position: *u32,
    mrope: *i32,
    valid: *bool,
};

fn view(graph: *Graph, which: Set) View {
    return switch (which) {
        .first => .{ .state = &graph.snap_lin_state, .history = &graph.snap_lin_hist, .ple = &graph.snap_ple_hist, .previous = &graph.snap_ple_prev, .position = &graph.snap_pos, .mrope = &graph.snap_mrope_delta, .valid = &graph.snap_valid },
        .second => .{ .state = &graph.snap2_lin_state, .history = &graph.snap2_lin_hist, .ple = &graph.snap2_ple_hist, .previous = &graph.snap2_ple_prev, .position = &graph.snap2_pos, .mrope = &graph.snap2_mrope_delta, .valid = &graph.snap2_valid },
        .boundary => .{ .state = &graph.snap0_lin_state, .history = &graph.snap0_lin_hist, .ple = &graph.snap0_ple_hist, .previous = &graph.snap0_ple_prev, .position = &graph.snap0_pos, .mrope = &graph.snap0_mrope_delta, .valid = &graph.snap0_valid },
    };
}

pub fn copy(graph: *Graph, save: bool, which: Set) bool {
    const target = view(graph, which);
    if (target.ple.* == null) return false;
    const shape = &graph.config.shape;
    const state_bytes = dimensions.stateFloats(shape) * @sizeOf(f32);
    const history_bytes = dimensions.historyFloats(shape) * @sizeOf(f32);
    const ple_bytes = dimensions.pleHistoryFloats(shape) * @sizeOf(f32);
    if (c.ds4_gpu_commands_active() == 0 and c.ds4_gpu_begin_commands() == 0) return false;
    var ok = true;
    for (0..shape.n_layer) |layer| {
        if (!ok) break;
        const snapshot_state = target.state[layer] orelse continue;
        const snapshot_history = target.history[layer];
        const live_state = graph.layer_lin_state[layer];
        const live_history = graph.layer_lin_hist[layer];
        ok = c.ds4_gpu_tensor_copy(if (save) snapshot_state else live_state, 0, if (save) live_state else snapshot_state, 0, state_bytes) != 0 and
            c.ds4_gpu_tensor_copy(if (save) snapshot_history else live_history, 0, if (save) live_history else snapshot_history, 0, history_bytes) != 0;
    }
    if (ok) ok = c.ds4_gpu_tensor_copy(if (save) target.ple.* else graph.ple_hist, 0, if (save) graph.ple_hist else target.ple.*, 0, ple_bytes) != 0;
    if (c.ds4_gpu_end_commands() == 0) ok = false;
    if (save) {
        target.previous.* = graph.ple_prev;
        target.position.* = graph.pos;
        target.mrope.* = graph.mrope_delta;
    } else {
        graph.ple_prev = target.previous.*;
        graph.pos = target.position.*;
        graph.mrope_delta = target.mrope.*;
    }
    return ok;
}

/// The stale live buffers become the next snapshot targets. Kernels overwrite
/// them fully before publication, so swap restores avoid copying36 GDN layers.
pub fn swap(graph: *Graph, which: Set) bool {
    const target = view(graph, which);
    if (target.ple.* == null) return false;
    if (which == .first) {
        const raw = c.getenv("DS4_QWEN4_MTP_SWAP_RESTORE");
        if (raw != null and raw[0] != 0 and c.strcmp(raw, "0") == 0) return copy(graph, false, which);
    }
    for (0..graph.config.shape.n_layer) |layer| {
        if (target.state[layer] == null) continue;
        std.mem.swap(?*Tensor, &graph.layer_lin_state[layer], &target.state[layer]);
        std.mem.swap(?*Tensor, &graph.layer_lin_hist[layer], &target.history[layer]);
    }
    std.mem.swap(?*Tensor, &graph.ple_hist, target.ple);
    graph.ple_prev = target.previous.*;
    graph.pos = target.position.*;
    graph.mrope_delta = target.mrope.*;
    target.valid.* = false;
    return true;
}

/// Snapshots0/2 are lazy. Publish the PLE handle only after every recurrent
/// allocation succeeds, so allocation failure cannot masquerade as a ready set.
pub fn ensure(graph: *Graph, which: Set) bool {
    const target = view(graph, which);
    if (target.ple.* != null) return true;
    if (graph.snap_ple_hist == null) return false;
    const shape = &graph.config.shape;
    const ple = resources.allocateFloats(dimensions.pleHistoryFloats(shape)) catch return false;
    var complete = false;
    defer if (!complete) {
        c.ds4_gpu_tensor_free(ple);
        for (target.state, target.history) |*state, *history| {
            c.ds4_gpu_tensor_free(state.*);
            c.ds4_gpu_tensor_free(history.*);
            state.* = null;
            history.* = null;
        }
    };
    for (0..shape.n_layer) |layer| {
        if (graph.snap_lin_state[layer] == null) continue;
        target.state[layer] = resources.allocateFloats(dimensions.stateFloats(shape)) catch return false;
        target.history[layer] = resources.allocateFloats(dimensions.historyFloats(shape)) catch return false;
    }
    target.ple.* = ple;
    complete = true;
    return true;
}
