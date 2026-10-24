//! Qwen graph tensors and shared-scratch lifetime. View handles retain their
//! backing buffers, while raw scratch handles are borrowed from one engine arena.
//! Only serialized forwards may use that arena. No Engine/Session layout is needed.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");
const Config = dimensions.Config;
const Graph = c.ds4_qwen4_gpu_graph;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;
const Weights = c.ds4_weights;
const AllocationError = error{ AllocationFailed, UnsupportedWeights };

pub const scratch_fields = .{
    "R",          "xn",         "lo",            "inj",       "inj_alt",    "mixed",    "blk",
    "qkv",        "z",          "ga",            "gb",        "lin_o",      "ple_emb",  "ple_key",
    "ple_val",    "ple_gated",  "ple_normed",    "qg",        "kp",         "vp",       "iq",
    "ik",         "q",          "gate",          "iqn",       "attn_o",     "score",    "tile_max",
    "sel_blocks", "sel_tokens", "n_sel",         "attn_part", "router",     "selected", "weights",
    "mid",        "part",       "sh_gate_logit", "moe_lists", "moe_counts", "sh_gate",  "sh_up",
    "sh_mid",     "sh_out",     "hc_u",          "hc_lo_act",
};

const layer_fields = .{
    "layer_lin_state", "layer_lin_hist", "layer_k_cache",   "layer_v_cache",  "layer_ik_cache",  "layer_block_key",
    "snap_lin_state",  "snap_lin_hist",  "snap2_lin_state", "snap2_lin_hist", "snap0_lin_state", "snap0_lin_hist",
};

pub fn denseSupported(weight: [*c]const Weight) bool {
    if (weight == null) return false;
    return switch (weight.*.type) {
        c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_F16, c.DS4_TENSOR_F32, c.DS4_TENSOR_BF16, c.DS4_TENSOR_Q4_0 => true,
        else => false,
    };
}

pub fn expertHasMatrixKernel(kind: u32) bool {
    return switch (kind) {
        c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_MXFP4, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_IQ2_XXS => true,
        else => false,
    };
}

fn expertSupported(weight: [*c]const Weight) bool {
    if (weight == null) return false;
    return switch (weight.*.type) {
        c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_MXFP4, c.DS4_TENSOR_Q4_0, c.DS4_TENSOR_F16, c.DS4_TENSOR_BF16, c.DS4_TENSOR_F32 => true,
        c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_IQ2_XXS => weight.*.dim[0] % 256 == 0,
        else => false,
    };
}

pub fn weightsSupported(weights: *const Weights, config: *const Config) bool {
    if (!denseSupported(weights.token_embd) or !denseSupported(weights.output) or
        !denseSupported(weights.output_hc_down) or !denseSupported(weights.output_hc_up))
    {
        _ = c.fputs("ds4: Qwen3.8 GPU graph needs Q8_0/Q4_0/F16/BF16/F32 dense weights\n", c.__stderrp);
        return false;
    }
    if (weights.ple_embd == null or weights.ple_embd.*.type != c.DS4_TENSOR_BF16) {
        _ = c.fputs("ds4: Qwen3.8 requires original BF16 n-grams in the model GGUF\n", c.__stderrp);
        return false;
    }
    if (config.shape.n_nextn_predict != 0) {
        const layer = &weights.layer[config.shape.n_layer - 1];
        const head_ok = denseSupported(layer.nextn_hc_head_down) and layer.nextn_hc_head_up != null and
            smallProjectionType(layer.nextn_hc_head_up.*.type);
        if (!denseSupported(layer.nextn_eh_proj) or !head_ok) {
            _ = c.fputs("ds4: Qwen3.8 GPU graph: unsupported nextn weight types\n", c.__stderrp);
            return false;
        }
    }
    for (weights.layer[0..config.shape.n_layer], 0..) |*layer, index| {
        const hc = [_][*c]const Weight{ layer.hc_attn_up, layer.hc_attn_inject, layer.hc_ffn_up, layer.hc_ffn_inject };
        for (hc) |weight| {
            if (weight == null or !smallProjectionType(weight.*.type)) {
                _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 GPU graph needs F16/F32/Q8_0 hc up/inject weights (layer %u)\n", @as(c_uint, @intCast(index)));
                return false;
            }
        }
        if (layer.ple_conv != null and layer.ple_conv.*.type != c.DS4_TENSOR_F32 and layer.ple_conv.*.type != c.DS4_TENSOR_F16) {
            _ = c.fputs("ds4: Qwen3.8 GPU graph needs F32/F16 PLE conv taps\n", c.__stderrp);
            return false;
        }
        if (!denseSupported(layer.hc_attn_down) or !denseSupported(layer.hc_ffn_down)) {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 GPU graph: unsupported hc down weight type (layer %u)\n", @as(c_uint, @intCast(index)));
            return false;
        }
        if (dimensions.isLinear(config, @intCast(index)) and layer.lin_beta.*.type != layer.lin_alpha.*.type) {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 GPU graph needs matching alpha/beta projection types (layer %u)\n", @as(c_uint, @intCast(index)));
            return false;
        }
        const projections = [_][*c]const Weight{
            layer.lin_qkv,        layer.lin_gate,       layer.lin_beta,     layer.lin_alpha,      layer.lin_out,
            layer.attn_q,         layer.attn_k,         layer.attn_v,       layer.attn_output,    layer.indexer_q_proj,
            layer.indexer_k_proj, layer.ffn_gate_shexp, layer.ffn_up_shexp, layer.ffn_down_shexp, layer.ple_key,
            layer.ple_value,
        };
        for (projections) |weight| {
            if (weight != null and !denseSupported(weight)) {
                _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 GPU graph: unsupported dense weight type %u in layer %u\n", weight.*.type, @as(c_uint, @intCast(index)));
                return false;
            }
        }
        if (!expertSupported(layer.ffn_gate_exps) or !expertSupported(layer.ffn_up_exps) or !expertSupported(layer.ffn_down_exps)) {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 GPU graph: unsupported routed expert type in layer %u (q8_0, mxfp4, q4_K, q2_K, iq2_xxs, f16, f32)\n", @as(c_uint, @intCast(index)));
            return false;
        }
    }
    return true;
}

fn smallProjectionType(kind: u32) bool {
    return kind == c.DS4_TENSOR_F16 or kind == c.DS4_TENSOR_F32 or kind == c.DS4_TENSOR_Q8_0;
}

pub fn allocateFloats(count: u64) AllocationError!*Tensor {
    const tensor = c.ds4_gpu_tensor_alloc(count * @sizeOf(f32)) orelse return error.AllocationFailed;
    _ = c.ds4_gpu_tensor_fill_f32(tensor, 0, count);
    return tensor;
}

fn allocateBytes(count: u64) AllocationError!*Tensor {
    return c.ds4_gpu_tensor_alloc(count) orelse error.AllocationFailed;
}

fn hostArray(comptime T: type, count: u64) [*]T {
    const allocation = c.malloc(@as(usize, @intCast(count)) * @sizeOf(T)) orelse c.ds4_engine_model_die("out of memory");
    return @ptrCast(@alignCast(allocation));
}

pub fn free(graph: *Graph) void {
    if (graph.owns_scratch) {
        inline for (scratch_fields) |name| c.ds4_gpu_tensor_free(@field(graph.*, name));
        inline for (.{ "batch_logits", "batch_attn_rows", "batch_attn_part", "batch_head_x", "batch_gdn_rows" }) |name|
            c.ds4_gpu_tensor_free(@field(graph.*, name));
        c.free(graph.host_row);
    }
    c.free(graph.host_pos3);
    c.free(graph.draft_ids);
    c.free(graph.host_logits);
    inline for (.{
        "ple_hist",      "logits",         "mtp_e",          "mtp_cat", "mtp_proj",   "mtp_R",      "mtp_argmax", "mtp_argmax_tmp",
        "snap_ple_hist", "snap2_ple_hist", "snap0_ple_hist", "pos3",    "draft_head", "steer_dirs",
    }) |name| c.ds4_gpu_tensor_free(@field(graph.*, name));
    inline for (layer_fields) |name| for (@field(graph.*, name)) |tensor| c.ds4_gpu_tensor_free(tensor);
    graph.* = std.mem.zeroes(Graph);
}

pub fn transferScratch(destination: *Graph, source: *Graph) void {
    destination.* = std.mem.zeroes(Graph);
    destination.config = source.config;
    destination.cap_tokens = source.cap_tokens;
    destination.n_block_cap = source.n_block_cap;
    destination.k_blocks = source.k_blocks;
    destination.sel_stride = source.sel_stride;
    inline for (scratch_fields) |name| @field(destination.*, name) = @field(source.*, name);
    destination.host_row = source.host_row;
    destination.owns_scratch = true;
    source.owns_scratch = false;
}

pub fn scratchBytes(graph: *const Graph) u64 {
    var bytes: u64 = 0;
    inline for (scratch_fields) |name| bytes += c.ds4_gpu_tensor_bytes(@field(graph.*, name));
    return bytes;
}

var warned_context = false;

pub fn allocate(graph: *Graph, weights: *const Weights, context: u32, capacity: u32, requested_mtp: bool, shared: ?*const Graph, state_pool: [*c]const ?*Tensor, history_pool: [*c]const ?*Tensor, slot: u32, config: *const Config) AllocationError!void {
    graph.* = std.mem.zeroes(Graph);
    graph.config = config.*;
    errdefer free(graph);
    if (!weightsSupported(weights, config)) return error.UnsupportedWeights;
    const shape = &config.shape;
    const mtp = requested_mtp and shape.n_nextn_predict != 0;
    const embd: u64 = shape.n_embd;
    const hc: u64 = shape.n_hc;
    const hc_width = embd * hc;
    const rows: u64 = capacity;
    const conv = dimensions.convWidth(shape);
    const values = dimensions.valueWidth(shape);
    const queries = @as(u64, shape.n_head) * shape.n_head_dim;
    const keys = @as(u64, shape.n_head_kv) * shape.n_head_dim;
    const index_queries = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim;
    graph.ctx_cap = context;
    graph.cap_tokens = capacity;
    graph.k_blocks = shape.n_indexer_top_k / 4;
    graph.n_block_cap = context / 4 + 1;
    graph.n_logit_rows = if (mtp) 3 else 1;
    graph.sel_stride = graph.k_blocks * 4 + 4;
    if (!warned_context and config.native_ctx != 0 and context > config.native_ctx and !config.rope_yarn) {
        warned_context = true;
        _ = c.fprintf(c.__stderrp, "ds4: context %u exceeds the native %u tokens; prompts past that need DS4_QWEN4_YARN_FACTOR (see README)\n", context, config.native_ctx);
    }
    graph.owns_scratch = shared == null;
    if (shared) |arena| {
        inline for (scratch_fields) |name| @field(graph.*, name) = @field(arena.*, name);
        graph.host_row = arena.host_row;
    } else {
        graph.R = try allocateFloats(rows * hc_width);
        graph.xn = try allocateFloats(rows * hc_width);
        graph.lo = try allocateFloats(rows * shape.n_hc_lowrank);
        graph.inj = try allocateFloats(rows * hc * c.DS4_QWEN4_HC_CHUNKS * hc);
        graph.inj_alt = try allocateFloats(rows * hc * c.DS4_QWEN4_HC_CHUNKS * hc);
        graph.mixed = try allocateFloats(rows * embd);
        graph.blk = try allocateFloats(rows * embd);
        graph.qkv = try allocateFloats(rows * conv);
        graph.z = try allocateFloats(rows * values);
        graph.ga = try allocateFloats(rows * shape.n_lin_v_head);
        graph.gb = try allocateFloats(rows * shape.n_lin_v_head);
        graph.lin_o = try allocateFloats(rows * values);
        graph.ple_emb = try allocateFloats(rows * embd);
        graph.ple_key = try allocateFloats(rows * hc_width);
        graph.ple_val = try allocateFloats(rows * embd);
        graph.ple_gated = try allocateFloats(rows * hc_width);
        graph.ple_normed = try allocateFloats(rows * hc_width);
        graph.qg = try allocateFloats(rows * 2 * queries);
        graph.kp = try allocateFloats(rows * keys);
        graph.vp = try allocateFloats(rows * keys);
        graph.iq = try allocateFloats(rows * index_queries);
        graph.ik = try allocateFloats(rows * shape.n_indexer_head_dim);
        graph.attn_part = try allocateFloats(c.ds4_gpu_qwen4_attn_part_floats(3, shape.n_head, shape.n_head_dim));
        graph.q = try allocateFloats(rows * queries);
        graph.gate = try allocateFloats(rows * queries);
        graph.iqn = try allocateFloats(rows * index_queries);
        graph.attn_o = try allocateFloats(rows * queries);
        graph.score = try allocateFloats(rows * graph.n_block_cap);
        graph.tile_max = try allocateFloats(rows * (graph.n_block_cap / 8 + 1));
        graph.sel_blocks = try allocateFloats(rows * graph.k_blocks);
        graph.sel_tokens = try allocateFloats(rows * graph.sel_stride);
        graph.n_sel = try allocateFloats(rows);
        graph.router = try allocateFloats(rows * shape.n_expert);
        graph.selected = try allocateFloats(rows * shape.n_expert_used);
        graph.weights = try allocateFloats(rows * shape.n_expert_used);
        graph.mid = try allocateFloats(rows * (shape.n_expert_used + 1) * shape.n_ff_exp);
        graph.part = try allocateFloats(rows * (shape.n_expert_used + 1) * embd);
        graph.sh_gate_logit = try allocateFloats(rows);
        graph.moe_lists = try allocateFloats(@as(u64, shape.n_expert) * rows);
        graph.moe_counts = try allocateFloats(shape.n_expert);
        graph.sh_gate = try allocateFloats(rows * shape.n_ff_exp);
        graph.sh_up = try allocateFloats(rows * shape.n_ff_exp);
        graph.sh_mid = try allocateFloats(rows * shape.n_ff_exp);
        graph.sh_out = try allocateFloats(rows * embd);
        graph.hc_u = try allocateFloats(rows * hc_width);
        graph.hc_lo_act = try allocateFloats(rows * shape.n_hc_lowrank);
        graph.host_row = hostArray(f32, rows * hc_width);
    }
    // PLE history and logit rows belong to this session even with shared scratch.
    graph.ple_hist = try allocateFloats(dimensions.pleHistoryFloats(shape));
    graph.logits = try allocateFloats(@as(u64, graph.n_logit_rows) * shape.n_vocab);
    if (mtp) {
        graph.mtp_e = try allocateFloats(3 * embd);
        graph.mtp_cat = try allocateFloats(3 * (hc + 1) * 2 * embd);
        graph.mtp_proj = try allocateFloats(3 * (hc + 1) * embd);
        graph.mtp_R = try allocateFloats(3 * hc_width);
        graph.mtp_argmax = try allocateFloats(1);
        graph.mtp_argmax_tmp = try allocateFloats(@as(u64, (shape.n_vocab + 4095) / 4096) * 2);
        graph.snap_ple_hist = try allocateFloats(dimensions.pleHistoryFloats(shape));
    }
    const state_bytes = dimensions.stateFloats(shape) * @sizeOf(f32);
    const history_bytes = dimensions.historyFloats(shape) * @sizeOf(f32);
    for (0..shape.n_layer) |layer| {
        if (dimensions.isLinear(config, @intCast(layer))) {
            if (state_pool != null and history_pool != null) {
                graph.layer_lin_state[layer] = c.ds4_gpu_tensor_view(state_pool[layer], @as(u64, slot) * state_bytes, state_bytes) orelse return error.AllocationFailed;
                graph.layer_lin_hist[layer] = c.ds4_gpu_tensor_view(history_pool[layer], @as(u64, slot) * history_bytes, history_bytes) orelse return error.AllocationFailed;
            } else {
                graph.layer_lin_state[layer] = try allocateFloats(dimensions.stateFloats(shape));
                graph.layer_lin_hist[layer] = try allocateFloats(dimensions.historyFloats(shape));
            }
            if (mtp) {
                graph.snap_lin_state[layer] = try allocateFloats(dimensions.stateFloats(shape));
                graph.snap_lin_hist[layer] = try allocateFloats(dimensions.historyFloats(shape));
            }
        } else {
            graph.layer_k_cache[layer] = try allocateBytes(@as(u64, context) * keys * 2);
            graph.layer_v_cache[layer] = try allocateBytes(@as(u64, context) * keys * 2);
            graph.layer_ik_cache[layer] = try allocateFloats(@as(u64, context) * shape.n_indexer_head_dim);
            graph.layer_block_key[layer] = try allocateBytes(@as(u64, graph.n_block_cap) * shape.n_indexer_head_dim * 2);
        }
    }
    graph.pos3 = try allocateFloats(@as(u64, context) * 4);
    graph.host_pos3 = hostArray(u32, rows * 4);
    graph.host_logits = hostArray(f32, @as(u64, graph.n_logit_rows) * shape.n_vocab);
}

pub fn reset(graph: *Graph) void {
    _ = c.ds4_gpu_synchronize();
    const shape = &graph.config.shape;
    for (0..shape.n_layer) |layer| {
        if (graph.layer_lin_state[layer]) |state| _ = c.ds4_gpu_tensor_fill_f32(state, 0, dimensions.stateFloats(shape));
        if (graph.layer_lin_hist[layer]) |history| _ = c.ds4_gpu_tensor_fill_f32(history, 0, dimensions.historyFloats(shape));
    }
    _ = c.ds4_gpu_tensor_fill_f32(graph.ple_hist, 0, dimensions.pleHistoryFloats(shape));
    @memset(&graph.ple_prev, @intCast(shape.ple_eos_id));
    graph.pos = 0;
    graph.mtp_pos = 0;
    graph.mtp_last_rows = 0;
    graph.mrope_delta = 0;
    graph.snap_valid = false;
    graph.snap2_valid = false;
    graph.snap0_valid = false;
    graph.snap_after_first = false;
    graph.snap_after_second = false;
}

pub fn loadSteering(graph: *Graph, path: [*c]const u8, attention_scale: f32, ffn_scale: f32) bool {
    if (attention_scale == 0 and ffn_scale == 0) return true;
    if (path == null or path[0] == 0) {
        _ = c.fputs("ds4: directional steering needs --dir-steering-file\n", c.__stderrp);
        return false;
    }
    const layers = dimensions.trunkLayers(&graph.config);
    if (layers == 0) return false;
    const count = @as(u64, layers) * graph.config.shape.n_embd;
    const vectors = hostArray(f32, count);
    defer c.free(vectors);
    var status: c.struct_stat = undefined;
    var ok = c.stat(path, &status) == 0 and status.st_size >= 0 and @as(u64, @intCast(status.st_size)) == count * @sizeOf(f32);
    if (ok) {
        const stream = c.fopen(path, "rb");
        if (stream) |file| {
            ok = c.fread(vectors, @sizeOf(f32), @intCast(count), file) == count;
            if (c.fclose(file) != 0) ok = false;
        } else ok = false;
    }
    if (ok) {
        graph.steer_dirs = c.ds4_gpu_tensor_alloc(count * @sizeOf(f32));
        ok = graph.steer_dirs != null and c.ds4_gpu_tensor_write(graph.steer_dirs, 0, vectors, count * @sizeOf(f32)) != 0;
    }
    if (!ok) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    graph.steer_attn_scale = attention_scale;
    graph.steer_ffn_scale = ffn_scale;
    _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 directional steering enabled: %s attn=%g ffn=%g\n", path, @as(f64, attention_scale), @as(f64, ffn_scale));
    return true;
}

pub fn steerAttention(graph: *Graph, layer: u32, rows: u32) bool {
    if (graph.steer_dirs == null or graph.steer_attn_scale == 0) return true;
    return c.ds4_gpu_directional_steering_project_tensor(graph.blk, graph.steer_dirs, layer, graph.config.shape.n_embd, rows, graph.steer_attn_scale) != 0;
}

pub fn steerFfn(graph: *Graph, layer: u32, rows: u32) bool {
    if (graph.steer_dirs == null or graph.steer_ffn_scale == 0) return true;
    return c.ds4_gpu_directional_steering_project_tensor(graph.R, graph.steer_dirs, layer, graph.config.shape.n_embd, rows * graph.config.shape.n_hc, graph.steer_ffn_scale) != 0;
}
