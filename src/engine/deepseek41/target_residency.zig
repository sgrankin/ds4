//! Initial V4.1 Metal target residency. Model bytes/descriptors remain borrowed;
//! this owner installs cache/map policy and transfers only its span arrays to
//! the caller's optional-cache phase. ARC/Metal view lifetime stays in the
//! existing public GPU API; Model close releases static page locks via munmap.
const std = @import("std");
const c = @import("gguf_c");
const Memory = @import("memory.zig").Memory;
const experts = @import("expert_cache.zig");
const Spans = @import("streaming.zig").Spans;
const allocator = std.heap.c_allocator;
const gib: f64 = 1073741824.0;

fn asGiB(value: u64) f64 {
    @setFloatMode(.optimized);
    return @as(f64, @floatFromInt(value)) / gib;
}
fn oom() noreturn {
    c.ds4_engine_model_die("out of memory");
}
pub fn freeSelection(selection: *c.ds4_model_map_selection) void {
    c.free(@ptrCast(selection.offsets));
    c.free(@ptrCast(selection.sizes));
    selection.* = std.mem.zeroes(c.ds4_model_map_selection);
}

pub const TargetResidency = struct {
    memory: Memory,
    model: *const c.ds4_model,
    placement: c.ds4_model_residency_inputs,

    /// Fixed weights should not compete with streamed experts in file cache.
    /// Bound every VM request, stop attempting locks after the first failure,
    /// and leave the remaining spans pageable. No extra resident byte charge:
    /// these exact bytes are already included by startup memory planning.
    fn lockStatic(self: *const TargetResidency) void {
        @setFloatMode(.optimized);
        const p = &self.placement;
        if (!self.memory.inputs.ssd_streaming or p.load_slice or p.tp_shard or
            c.getenv("DS4_METAL_DISABLE_STREAMING_STATIC_LOCK") != null) return;
        const budget = self.memory.safeCacheBytes(p.context, p.chunk, c.ds4_gpu_recommended_working_set_size());
        const fixed = @import("streaming.zig").staticBytes(self.memory.config, self.memory.weights, self.memory.inputs.include_token, self.memory.inputs.include_output);
        const state = self.memory.state;
        const dynamic = self.dynamicBytes();
        const cached = dynamic +| (state.ssd_streaming_prefill_headroom_bytes +| state.ssd_streaming_full_layer_bytes);
        if (fixed == null or fixed.? > budget or cached > budget - fixed.?) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal SSD static weights remain pageable to preserve runtime headroom\n");
            return;
        }
        var spans = Spans{};
        defer spans.deinit();
        self.includeStatic(&spans);
        if (!spans.finish()) return;
        const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
        var locked: u64 = 0;
        var failed: u64 = 0;
        var can_lock = true;
        for (spans.items.items) |span| {
            const first = span.first / page * page;
            if (span.end > self.model.size or span.end <= first) continue;
            var offset = first;
            while (offset < span.end) {
                const length = @min(span.end - offset, @as(u64, 256) << 20);
                const pointer: *anyopaque = @ptrCast(@constCast(self.model.map + offset));
                if (can_lock and c.mlock(pointer, @intCast(length)) == 0) locked +%= length else {
                    can_lock = false;
                    failed +%= length;
                }
                offset += length;
            }
        }
        _ = c.fprintf(c.__stderrp, "ds4: Metal SSD static weights locked %.2f GiB; pageable %.2f GiB\n", asGiB(locked), asGiB(failed));
    }

    fn dynamicBytes(self: *const TargetResidency) u64 {
        const state = self.memory.state;
        if (state.ssd_streaming_cache_bytes != 0) return state.ssd_streaming_cache_bytes;
        if (state.ssd_streaming_cache_experts == 0) return 0;
        const per_expert = experts.sizeClass(self.memory.config, self.memory.weights) orelse return 0;
        return @as(u64, state.ssd_streaming_cache_experts) *| per_expert;
    }

    fn includeStatic(self: *const TargetResidency, spans: *Spans) void {
        const weights = self.memory.weights;
        spans.include(weights.token_embd);
        for (weights.layer[0..self.memory.config.shape.n_layer]) |*layer| spans.includeLayer(layer, true, null);
        spans.includeOutput(weights);
    }

    /// One slab class serves uniform layers only. Outliers keep their mapped
    /// experts; a small cache is legal (identical weight bytes) but can thrash.
    fn cacheFormat(self: *const TargetResidency) void {
        @setFloatMode(.optimized);
        if (!self.memory.inputs.ssd_streaming) return;
        const config = self.memory.config;
        const class = experts.sizeClass(config, self.memory.weights) orelse return;
        c.ds4_gpu_set_streaming_expert_cache_expert_bytes(class);
        var routed: u32 = 0;
        var boosted: u32 = 0;
        for (self.memory.weights.layer[0..config.shape.n_layer]) |*layer| {
            if (layer.ffn_gate_exps == null or layer.ffn_up_exps == null or layer.ffn_down_exps == null) continue;
            routed += 1;
            // Failure to measure a layer is "uniform" in the old helper.
            if (experts.layerBytes(layer)) |bytes| if (bytes != class) {
                boosted += 1;
            };
        }
        if (boosted != 0)
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming mixed-precision model: %u/%u routed layers off the slab size class will bypass the expert cache and read experts via mapped model views\n", boosted, routed);
        if (boosted * 2 > routed)
            _ = c.fprintf(c.__stderrp, "ds4: WARNING: the majority of routed layers (%u/%u) are off the slab size class (is the FIRST routed layer itself boosted?); expert-cache hit rate will be catastrophic\n", boosted, routed);
        const per_token = @as(u64, routed - boosted) * config.shape.n_expert_used;
        const budget = self.memory.state.ssd_streaming_cache_experts;
        if (per_token != 0 and budget != 0 and budget < 2 * per_token)
            _ = c.fprintf(c.__stderrp, "ds4: WARNING: SSD streaming expert cache (%u experts) is under twice the per-token routed working set (%u layers x %u experts = %llu); expect heavy thrashing below %.2f GiB\n", budget, routed - boosted, config.shape.n_expert_used, @as(c_ulonglong, per_token), asGiB(2 *% per_token *% class));
    }

    fn includeShard(self: *const TargetResidency, spans: *Spans) bool {
        const rank = self.placement.tp_rank;
        if (rank != 0 and rank != 1) return false;
        const weights = self.memory.weights;
        spans.include(weights.token_embd);
        for (weights.layer[0..self.memory.config.shape.n_layer]) |*layer| {
            spans.includeLayer(layer, true, null);
            const tensors = [_][*c]const c.ds4_tensor{ layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps };
            for (tensors) |tensor| {
                if (tensor == null or tensor.*.ndim != 3 or tensor.*.dim[2] < 2) continue;
                const weight: *const c.ds4_tensor = @ptrCast(tensor);
                // TP owns a contiguous expert-ID half; rank1 gets a remainder.
                // Metal never installs the CUDA MXFP4 compact split here.
                const per_expert = self.shardExpertBytes(weight);
                const low = tensor.*.dim[2] / 2;
                const first = if (rank == 1) low else 0;
                const count = if (rank == 1) tensor.*.dim[2] - low else low;
                const size = count *% per_expert;
                const offset = tensor.*.abs_offset +% (first *% per_expert);
                spans.append(offset, offset +% size, true);
                spans.max_tensor_bytes = @max(spans.max_tensor_bytes, size);
            }
        }
        spans.includeOutput(weights);
        return spans.finish();
    }

    /// Replicated dense + this rank's routed half, before memory admission.
    /// A failed candidate leaves the compatibility override zero, as in C.
    pub fn shardBytes(self: *const TargetResidency) u64 {
        var spans = Spans{};
        defer spans.deinit();
        if (!self.includeShard(&spans)) return 0;
        var total: u64 = 0;
        for (spans.items.items) |span| total +|= span.end - span.first;
        return total;
    }

    fn shardExpertBytes(self: *const TargetResidency, weight: *const c.ds4_tensor) u64 {
        // Preserve tensor_expert_bytes' general rounded-block contract here,
        // rather than the stricter aligned-row cache format helper.
        const info = c.ds4_gguf_tensor_type(weight.type);
        if (info == null or info.*.block_elems == 0) c.ds4_engine_model_die("unsupported expert tensor type");
        const blocks = (weight.dim[0] +% info.*.block_elems -% 1) / info.*.block_elems;
        const row = blocks *% info.*.block_bytes;
        if (weight.abs_offset > self.model.size or weight.bytes > self.model.size - weight.abs_offset)
            c.ds4_engine_model_die("tensor is outside the mapped weights; Engram requires disk row reads");
        return weight.dim[1] *% row;
    }

    fn sliceEnd(self: *const TargetResidency, output: []u8) void {
        const p = &self.placement;
        if (p.load_output and p.layer_end == std.math.maxInt(u32)) {
            _ = c.snprintf(output.ptr, output.len, "output");
        } else if (p.load_output) {
            _ = c.snprintf(output.ptr, output.len, "%u+output", p.layer_end);
        } else _ = c.snprintf(output.ptr, output.len, "%u", p.layer_end);
    }

    pub fn install(self: *const TargetResidency, output: *c.ds4_model_map_selection) bool {
        @setFloatMode(.optimized);
        c.ds4_gpu_set_streaming_expert_cache_budget(self.memory.state.ssd_streaming_cache_experts);
        self.lockStatic();
        self.cacheFormat();
        _ = c.ds4_gpu_set_model_fd(self.model.fd);
        var selection = std.mem.zeroes(c.ds4_model_map_selection);
        defer freeSelection(&selection);
        var spans = Spans{};
        defer spans.deinit();
        const p = &self.placement;
        const streaming = self.memory.inputs.ssd_streaming;
        if (streaming) {
            self.includeStatic(&spans);
            if (!spans.finish()) {
                _ = c.fprintf(c.__stderrp, "ds4: invalid SSD streaming initial token embedding map\n");
                return false;
            }
        } else if (p.load_slice) {
            const layers = self.memory.config.shape.n_layer;
            const end = if (p.layer_end == std.math.maxInt(u32)) layers - 1 else p.layer_end;
            if (p.layer_start < layers and end < layers and end >= p.layer_start) {
                if (p.layer_start == 0) spans.include(self.memory.weights.token_embd);
                for (self.memory.weights.layer[p.layer_start .. end + 1]) |*layer| spans.includeLayer(layer, false, null);
                if (p.load_output) spans.includeOutput(self.memory.weights);
            }
            if (!spans.finish()) {
                var end_text: [32]u8 = undefined;
                self.sliceEnd(&end_text);
                _ = c.fprintf(c.__stderrp, "ds4: invalid model load layer slice %u:%s\n", p.layer_start, &end_text);
                return false;
            }
        } else if (p.tp_shard and !self.includeShard(&spans)) {
            _ = c.fprintf(c.__stderrp, "ds4: sharded model span build failed\n");
            return false;
        }
        var mapped: c_int = 0;
        if (streaming or p.load_slice or p.tp_shard) {
            const count = spans.items.items.len;
            const offsets = allocator.alloc(u64, count) catch oom();
            selection.offsets = offsets.ptr;
            const sizes = allocator.alloc(u64, count) catch oom();
            selection.sizes = sizes.ptr;
            selection.count = @intCast(count);
            var total: u64 = 0;
            for (spans.items.items, offsets, sizes) |span, *offset, *size| {
                offset.* = span.first;
                size.* = span.end - span.first;
                total +%= size.*;
            }
            if (p.tp_shard and !streaming and !p.load_slice) self.memory.state.startup_model_span_bytes +|= total else self.memory.state.startup_model_span_bytes = total;
            if (p.load_slice) {
                var end_text: [32]u8 = undefined;
                self.sliceEnd(&end_text);
                if (streaming) {
                    _ = c.fprintf(c.__stderrp, "ds4: SSD streaming initial metal model map restricted to %snon-routed layers %u:%s (%u spans, %.2f GiB tensor span)\n", @as([*:0]const u8, if (p.layer_start == 0) "token + " else ""), p.layer_start, &end_text, selection.count, asGiB(total));
                } else _ = c.fprintf(c.__stderrp, "ds4: restricting metal model map to layers %u:%s (%u spans, %.2f GiB tensor span)\n", p.layer_start, &end_text, selection.count, asGiB(total));
            } else if (streaming) {
                _ = c.fprintf(c.__stderrp, "ds4: SSD streaming initial metal model map (%u spans, %.2f GiB tensor span)\n", selection.count, asGiB(total));
            } else _ = c.fprintf(c.__stderrp, "ds4: TP expert shard (rank %d): mapping %u spans, %.2f GiB of %.2f GiB\n", p.tp_rank, selection.count, asGiB(total), asGiB(self.model.size -% self.model.tensor_data_pos));
            mapped = c.ds4_gpu_set_model_map_spans(self.model.map, self.model.size, offsets.ptr, sizes.ptr, selection.count, spans.max_tensor_bytes);
        } else {
            self.memory.state.startup_model_span_bytes = if (self.model.size > self.model.tensor_data_pos) self.model.size - self.model.tensor_data_pos else 0;
            mapped = c.ds4_gpu_set_model_map_range(self.model.map, self.model.size, self.model.tensor_data_pos, self.model.size -% self.model.tensor_data_pos, self.model.max_tensor_bytes);
        }
        if (mapped == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: metal failed to map model views; aborting startup. This is commonly caused by insufficient memory or accelerator VM budget.\n");
            return false;
        }
        output.* = selection;
        selection = std.mem.zeroes(c.ds4_model_map_selection);
        return true;
    }
};
