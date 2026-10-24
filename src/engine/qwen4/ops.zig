//! Weight projections and hyper-connections. Preserve the released matrix
//! choices: FP32 operands, small Q8 batch tiles and exact2/1-row verify splits.
const c = @import("gguf_c");
const Graph = c.ds4_qwen4_gpu_graph;
const Model = c.ds4_model;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;

pub fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}

pub fn beginCommands() bool {
    return c.ds4_gpu_commands_active() != 0 or c.ds4_gpu_begin_commands() != 0;
}

pub fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}

pub fn dense(output: ?*Tensor, model: *const Model, weight: [*c]const Weight, input: ?*const Tensor, tokens: u32) bool {
    return denseRows(output, model, weight, input, tokens, 0);
}

pub fn denseRows(output: ?*Tensor, model: *const Model, weight: [*c]const Weight, input: ?*const Tensor, tokens: u32, leading_rows: u64) bool {
    const width = weight.*.dim[0];
    const full_rows = if (weight.*.ndim >= 2) weight.*.dim[1] else 1;
    const rows = if (leading_rows != 0 and leading_rows < full_rows) leading_rows else full_rows;
    const legacy = present("DS4_QWEN4_DENSE_MM_LEGACY");
    const half_batch = !legacy and tokens > 3 and weight.*.type == c.DS4_TENSOR_F16 and tokens <= 64 and (rows <= 512 or tokens > 8);
    if (((tokens > 8 and weight.*.type == c.DS4_TENSOR_F32) or half_batch) and width % 32 == 0) {
        if (c.ds4_gpu_qwen4_dense_mm_tensor(output, input, model.map, model.size, weight.*.abs_offset, weight.*.type, tokens, @intCast(width), @intCast(rows)) != 0) return true;
    }
    // These buffers hold32 rows. Round the first tile to8/16 and ignore
    // padded scratch rows; a second16-row product handles tokens17..32.
    if (!legacy and weight.*.type == c.DS4_TENSOR_Q8_0 and tokens >= 5 and tokens <= 32 and
        width % 32 == 0 and rows % 32 == 0 and !present("DS4_QWEN4_NO_BATCH_MM") and
        c.ds4_gpu_tensor_bytes(input) >= 32 * width * @sizeOf(f32) and c.ds4_gpu_tensor_bytes(output) >= 32 * rows * @sizeOf(f32))
    {
        const first: u32 = if (tokens <= 8) 8 else 16;
        var ok = c.ds4_gpu_qwen4_batch_mm_q8_tensor(output, input, model.map, model.size, weight.*.abs_offset, first, @intCast(width), @intCast(rows)) != 0;
        if (ok and tokens > 16) {
            const input2 = c.ds4_gpu_tensor_view(input, 16 * width * @sizeOf(f32), 16 * width * @sizeOf(f32));
            defer c.ds4_gpu_tensor_free(input2);
            const output2 = c.ds4_gpu_tensor_view(output, 16 * rows * @sizeOf(f32), 16 * rows * @sizeOf(f32));
            defer c.ds4_gpu_tensor_free(output2);
            ok = input2 != null and output2 != null and c.ds4_gpu_qwen4_batch_mm_q8_tensor(output2, input2, model.map, model.size, weight.*.abs_offset, 16, @intCast(width), @intCast(rows)) != 0;
        }
        if (ok) return true;
    }
    const ok = switch (weight.*.type) {
        c.DS4_TENSOR_Q8_0 => c.ds4_gpu_qwen4_matmul_q8_0_tensor(output, model.map, model.size, weight.*.abs_offset, width, rows, input, tokens) != 0,
        c.DS4_TENSOR_F16 => c.ds4_gpu_matmul_f16_tensor(output, model.map, model.size, weight.*.abs_offset, width, rows, input, tokens) != 0,
        c.DS4_TENSOR_F32 => c.ds4_gpu_matmul_f32_tensor(output, model.map, model.size, weight.*.abs_offset, width, rows, input, tokens) != 0,
        c.DS4_TENSOR_Q4_0 => c.ds4_gpu_matmul_quant_tensor(output, model.map, model.size, weight.*.abs_offset, weight.*.type, width, rows, input, tokens) != 0,
        c.DS4_TENSOR_BF16 => blk: {
            var outputs = [_]?*Tensor{output};
            const offsets = [_]u64{weight.*.abs_offset};
            const kinds = [_]u32{weight.*.type};
            const counts = [_]u32{@intCast(rows)};
            break :blk c.ds4_gpu_qwen4_multi_gemv_tensor(input, tokens, @intCast(width), 1, &outputs, model.map, model.size, &offsets, &kinds, &counts) != 0;
        },
        else => false,
    };
    if (!ok) _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 matmul failed for %.*s (type %u, %llu x %llu, %u tokens)\n", @as(c_int, @intCast(weight.*.name.len)), weight.*.name.ptr, weight.*.type, @as(c_ulonglong, width), @as(c_ulonglong, rows), tokens);
    return ok;
}

var no_fuse: ?bool = null;
var no_index_select: ?bool = null;

pub fn fused(graph: *const Graph, tokens: u32) bool {
    if (no_fuse == null) no_fuse = present("DS4_QWEN4_NO_FUSE");
    if (no_fuse.? or tokens <= 2) return tokens <= 2 and !no_fuse.?;
    // A three-token prefill tail keeps its normal arithmetic. Only an exact
    // speculative verify uses the fused2/1-row paths at this width.
    return tokens == 3 and graph.verify_rows_exact;
}

pub fn selectBlocks(selected: ?*Tensor, scores: ?*const Tensor, tile_max: ?*const Tensor, blocks: u32, tokens: u32, top_k: u32) bool {
    if (no_index_select == null) no_index_select = present("DS4_QWEN4_NO_IDX_SELECT");
    return if (no_index_select.?)
        c.ds4_gpu_indexer_topk_tensor(selected, scores, blocks, tokens, top_k) != 0
    else
        c.ds4_gpu_qwen4_idx_select_tensor(selected, scores, tile_max, blocks, tokens, top_k) != 0;
}

pub fn rowsView(tensor: ?*const Tensor, first: u32, count: u32, width: u64) ?*Tensor {
    return c.ds4_gpu_tensor_view(tensor, @as(u64, first) * width * @sizeOf(f32), @as(u64, count) * width * @sizeOf(f32));
}

pub fn hcMix(graph: *Graph, model: *const Model, norm: [*c]const Weight, down: [*c]const Weight, up: [*c]const Weight, injection: [*c]const Weight, tokens: u32) bool {
    const shape = &graph.config.shape;
    const has_injection = injection != null;
    if (c.ds4_gpu_qwen4_hc_norm_tensor(graph.xn, graph.inj, graph.R, model.map, model.size, norm.*.abs_offset, if (has_injection) injection.*.abs_offset else 0, if (has_injection) injection.*.type else c.DS4_TENSOR_F32, tokens, shape.n_embd, shape.n_hc, if (has_injection) shape.n_hc else 0, shape.rms_eps) == 0 or
        !dense(graph.lo, model, down, graph.xn, tokens)) return false;
    if (tokens > 8 and up.*.type != c.DS4_TENSOR_Q8_0) {
        return c.ds4_gpu_qwen4_hc_lo_act_tensor(graph.hc_lo_act, graph.lo, tokens, shape.n_hc, shape.n_hc_lowrank) != 0 and
            dense(graph.hc_u, model, up, graph.hc_lo_act, tokens) and
            c.ds4_gpu_qwen4_hc_mix_rows_tensor(graph.mixed, graph.hc_u, graph.xn, tokens, shape.n_embd, shape.n_hc) != 0;
    }
    if (tokens == 3 and graph.verify_rows_exact) {
        // Keep the exact pair kernel followed by the one-row generic kernel.
        // Views are released after encoding; the command owner retains storage.
        const width = @as(u64, shape.n_embd) * shape.n_hc;
        var first: u32 = 0;
        while (first < tokens) : (first += 2) {
            const count = @min(2, tokens - first);
            const xn = rowsView(graph.xn, first, count, width) orelse return false;
            defer c.ds4_gpu_tensor_free(xn);
            const low = rowsView(graph.lo, first, count, shape.n_hc_lowrank) orelse return false;
            defer c.ds4_gpu_tensor_free(low);
            const mixed = rowsView(graph.mixed, first, count, shape.n_embd) orelse return false;
            defer c.ds4_gpu_tensor_free(mixed);
            if (c.ds4_gpu_qwen4_hc_gate_mix_tensor(mixed, xn, low, model.map, model.size, up.*.abs_offset, up.*.type, count, shape.n_embd, shape.n_hc, shape.n_hc_lowrank) == 0) return false;
        }
        return true;
    }
    return c.ds4_gpu_qwen4_hc_gate_mix_tensor(graph.mixed, graph.xn, graph.lo, model.map, model.size, up.*.abs_offset, up.*.type, tokens, shape.n_embd, shape.n_hc, shape.n_hc_lowrank) != 0;
}
