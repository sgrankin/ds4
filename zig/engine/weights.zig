//! Bind names once into a typed V4.1 table, then validate every tensor consumed
//! by its specialized pipeline. All pointers borrow the owning Model mapping.
const std = @import("std");
const c = @import("gguf_c");
const ModelConfig = @import("config.zig").ModelConfig;
const Table = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_tensor;
const InvalidWeights = error{InvalidWeights};

pub const LoadOptions = struct {
    slice: bool = false,
    start: u32 = 0,
    end: u32 = std.math.maxInt(u32),
    require_output: bool = true,
    optional_output: bool = false,
};

// Own the pointer table and explicit configuration by value, not the mapped
// tensors. Model must outlive this value and every graph that borrows it.
pub const Weights = struct {
    table: Table = std.mem.zeroes(Table),
    config: ModelConfig,

    pub fn bind(self: *Weights, model: *const c.ds4_model, options: LoadOptions, error_text: []u8) InvalidWeights!void {
        self.table = std.mem.zeroes(Table);
        var binding = Binding{ .table = &self.table, .model = model, .config = &self.config, .error_text = error_text };
        try binding.bind(options);
    }
};

fn bytes(text: c.ds4_str) []const u8 {
    return if (text.len == 0) &.{} else text.ptr[0..@intCast(text.len)];
}

const Binding = struct {
    table: *Table,
    model: *const c.ds4_model,
    config: *const ModelConfig,
    error_text: []u8,

    fn fail(self: *Binding, comptime format: []const u8, args: anytype) InvalidWeights {
        if (self.error_text.len != 0) {
            _ = std.mem.printSentinel(self.error_text, format, args, 0) catch {
                self.error_text[self.error_text.len - 1] = 0;
                return error.InvalidWeights;
            };
        }
        return error.InvalidWeights;
    }

    // Keep directory order: duplicate tensor names resolve to the first match.
    fn find(self: *const Binding, name: []const u8) ?*Tensor {
        if (self.model.n_tensors == 0) return null;
        const tensors: [*]Tensor = self.model.tensors;
        for (tensors[0..@intCast(self.model.n_tensors)]) |*tensor| {
            if (std.mem.eql(u8, bytes(tensor.name), name)) return tensor;
        }
        return null;
    }

    fn required(self: *Binding, name: []const u8) InvalidWeights!*Tensor {
        return self.find(name) orelse return self.fail("required tensor is missing: {s}", .{name});
    }

    fn layerTensor(self: *Binding, index: u32, suffix: []const u8, must_exist: bool) InvalidWeights!?*Tensor {
        var storage: [128]u8 = undefined;
        const name = std.mem.printSentinel(&storage, "blk.{d}.{s}", .{ index, suffix }, 0) catch
            return self.fail("tensor name is too long", .{});
        return if (must_exist) try self.required(name) else self.find(name);
    }

    fn typeName(kind: u32) []const u8 {
        const info = c.ds4_gguf_tensor_type(kind);
        return if (info == null) "unknown" else std.mem.span(info.*.name);
    }

    fn layout(self: *Binding, maybe_tensor: ?*const Tensor, kind: u32, dimensions: []const u64) InvalidWeights!void {
        const tensor = maybe_tensor orelse return self.fail("internal error: missing tensor while validating layout", .{});
        if (tensor.type != kind)
            return self.fail("tensor {s} has type {s}, expected {s}", .{ bytes(tensor.name), typeName(tensor.type), typeName(kind) });
        if (tensor.ndim != dimensions.len)
            return self.fail("tensor {s} has {d} dimensions, expected {d}", .{ bytes(tensor.name), tensor.ndim, dimensions.len });
        for (dimensions, 0..) |wanted, index| {
            if (tensor.dim[index] != wanted)
                return self.fail("tensor {s} has dim[{d}]={d}, expected {d}", .{ bytes(tensor.name), index, tensor.dim[index], wanted });
        }
    }

    fn dense(self: *Binding, maybe_tensor: ?*const Tensor, dimensions: []const u64) InvalidWeights!void {
        const tensor = maybe_tensor orelse return self.fail("internal error: missing tensor while validating dense quant layout", .{});
        switch (tensor.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q4_0 => {},
            else => return self.fail("tensor {s} has type {s}, expected q8_0, q4_K, or q4_0", .{ bytes(tensor.name), typeName(tensor.type) }),
        }
        try self.layout(tensor, tensor.type, dimensions);
    }

    fn expert(self: *Binding, maybe_tensor: ?*const Tensor, dimensions: []const u64) InvalidWeights!void {
        const tensor = maybe_tensor orelse return self.fail("internal error: missing routed expert tensor while validating layout", .{});
        switch (tensor.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_Q6_K, c.DS4_TENSOR_MXFP4 => {},
            else => return self.fail("tensor {s} has type {d} ({s}), expected a routed expert quant type", .{ bytes(tensor.name), tensor.type, typeName(tensor.type) }),
        }
        try self.layout(tensor, tensor.type, dimensions);
    }

    fn bind(self: *Binding, options: LoadOptions) InvalidWeights!void {
        const shape = &self.config.shape;
        if (shape.family != c.DS4_MODEL_FAMILY_DEEPSEEK41)
            return self.fail("native weight binding requires DeepSeek V4.1", .{});
        var start: u32 = 0;
        var end = shape.n_layer - 1;
        var require_embedding = true;
        var require_output = options.require_output;
        var optional_output = options.optional_output;
        if (options.slice) {
            if (options.start >= shape.n_layer) return self.fail("invalid model load layer slice", .{});
            start = options.start;
            end = if (options.end == std.math.maxInt(u32)) shape.n_layer - 1 else options.end;
            if (end >= shape.n_layer or end < start) return self.fail("invalid model load layer slice", .{});
            require_embedding = start == 0;
        } else {
            require_output = true;
            optional_output = false;
        }
        self.table.token_embd = if (require_embedding) try self.required("token_embd.weight") else self.find("token_embd.weight");
        if (require_output) {
            self.table.output_norm = try self.required("output_norm.weight");
            self.table.output = try self.required("output.weight");
        } else if (optional_output) {
            self.table.output_norm = self.find("output_norm.weight");
            self.table.output = self.find("output.weight");
        }
        if (optional_output and self.partialOutput() and !self.haveOutput())
            return self.fail("partial output head in GGUF", .{});
        for (start..end + 1) |index| try self.bindLayer(@intCast(index));
        try self.validate(start, end, require_embedding, require_output);
    }

    fn haveOutput(self: *const Binding) bool {
        return self.table.output_norm != null and self.table.output != null;
    }

    fn partialOutput(self: *const Binding) bool {
        return self.table.output_norm != null or self.table.output != null;
    }

    fn bindLayer(self: *Binding, index: u32) InvalidWeights!void {
        const layer = &self.table.layer[index];
        const policy = self.config.layerPolicy(index).?;
        inline for (.{ "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv", "attn_kv_a_norm", "attn_sinks", "attn_output_a", "attn_output_b" }) |field| {
            @field(layer, field) = try self.layerTensor(index, field ++ ".weight", true);
        }
        layer.ffn_exp_probs_vl = try self.layerTensor(index, "exp_probs_b_vl.bias", false);
        if (layer.ffn_exp_probs_vl != null)
            try self.layout(layer.ffn_exp_probs_vl, c.DS4_TENSOR_F32, &.{self.config.shape.n_expert});
        if (policy.publishes_kv) {
            layer.attn_compressor_kv = try self.layerTensor(index, "attn_compressor_kv.weight", true);
            layer.attn_compressor_norm = try self.layerTensor(index, "attn_compressor_norm.weight", true);
            if (policy.compression_ratio == 2)
                layer.attn_compressor_gate = try self.layerTensor(index, "attn_compressor_gate.weight", true);
            layer.indexer_attn_k = try self.layerTensor(index, "indexer.attn_k.weight", true);
            layer.indexer_k_norm = try self.layerTensor(index, "indexer.k_norm.weight", true);
        }
        if (policy.publishes_index) {
            layer.indexer_attn_q_b = try self.layerTensor(index, "indexer.attn_q_b.weight", true);
            layer.indexer_proj = try self.layerTensor(index, "indexer.proj.weight", true);
        }
        if (policy.uses_engram) {
            layer.engram_kv = try self.layerTensor(index, "engram_kv.weight", true);
            layer.engram_q_norm = try self.layerTensor(index, "engram_q_norm.weight", true);
            layer.engram_k_norm = try self.layerTensor(index, "engram_k_norm.weight", true);
        }
        inline for (.{ "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm", "ffn_gate_inp" }) |field| {
            @field(layer, field) = try self.layerTensor(index, field ++ ".weight", true);
        }
        // Optional name lookup is intentional. The final layer-completeness
        // check requires this bias, preserving the original failure ordering.
        layer.ffn_exp_probs_b = try self.layerTensor(index, "exp_probs_b.bias", false);
        inline for (.{ "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" }) |field| {
            @field(layer, field) = try self.layerTensor(index, field ++ ".weight", true);
        }
        if (index < self.config.shape.n_hash_layer)
            layer.ffn_gate_tid2eid = try self.layerTensor(index, "ffn_gate_tid2eid.weight", true);
    }

    fn layerComplete(self: *const Binding, layer: *const Layer, index: u32) bool {
        inline for (.{ "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv", "attn_kv_a_norm", "attn_sinks", "attn_output_a", "attn_output_b", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_exp_probs_b" }) |field| {
            if (@field(layer, field) == null) return false;
        }
        const policy = self.config.layerPolicy(index).?;
        if (policy.publishes_kv and (layer.attn_compressor_kv == null or layer.attn_compressor_norm == null or
            layer.indexer_attn_k == null or layer.indexer_k_norm == null or
            (policy.compression_ratio == 2 and layer.attn_compressor_gate == null))) return false;
        if (policy.publishes_index and (layer.indexer_attn_q_b == null or layer.indexer_proj == null)) return false;
        if (policy.uses_engram and (layer.engram_kv == null or layer.engram_q_norm == null or layer.engram_k_norm == null)) return false;
        return true;
    }

    fn validate(self: *Binding, start: u32, end: u32, require_embedding: bool, require_output: bool) InvalidWeights!void {
        const shape = &self.config.shape;
        const hc_dim = @as(u64, shape.n_embd) * shape.n_hc;
        const mix_dim = 2 * @as(u64, shape.n_hc) + @as(u64, shape.n_hc) * shape.n_hc;
        const q_dim = @as(u64, shape.n_head) * shape.n_head_dim;
        const output_low = @as(u64, shape.n_out_group) * shape.n_lora_o;
        if (require_embedding and self.table.token_embd == null)
            return self.fail("required token embedding tensor is missing", .{});
        if (self.table.token_embd != null)
            try self.layout(self.table.token_embd, c.DS4_TENSOR_F16, &.{ shape.n_embd, shape.n_vocab });
        if (require_output and !self.haveOutput()) return self.fail("required output head tensors are missing", .{});
        if (self.partialOutput() and !self.haveOutput()) return self.fail("partial output head in GGUF", .{});
        if (self.haveOutput()) {
            try self.layout(self.table.output_norm, c.DS4_TENSOR_F32, &.{shape.n_embd});
            try self.dense(self.table.output, &.{ shape.n_embd, shape.n_vocab });
        }
        for (start..end + 1) |index| {
            const layer = &self.table.layer[index];
            const policy = self.config.layerPolicy(@intCast(index)).?;
            if (!self.layerComplete(layer, @intCast(index)))
                return self.fail("required tensors for layer {d} are missing", .{index});
            try self.layout(layer.hc_attn_fn, c.DS4_TENSOR_F16, &.{ hc_dim, mix_dim });
            try self.layout(layer.hc_attn_scale, c.DS4_TENSOR_F32, &.{3});
            try self.layout(layer.hc_attn_base, c.DS4_TENSOR_F32, &.{mix_dim});
            try self.layout(layer.attn_norm, c.DS4_TENSOR_F32, &.{shape.n_embd});
            try self.dense(layer.attn_q_a, &.{ shape.n_embd, shape.n_lora_q });
            try self.layout(layer.attn_q_a_norm, c.DS4_TENSOR_F32, &.{shape.n_lora_q});
            try self.dense(layer.attn_q_b, &.{ shape.n_lora_q, q_dim });
            try self.dense(layer.attn_kv, &.{ shape.n_embd, shape.n_head_dim });
            try self.layout(layer.attn_kv_a_norm, c.DS4_TENSOR_F32, &.{shape.n_head_dim});
            try self.layout(layer.attn_sinks, c.DS4_TENSOR_F32, &.{shape.n_head});
            const output_group = @as(u64, shape.n_head_dim) * (shape.n_head / shape.n_out_group);
            try self.dense(layer.attn_output_a, &.{ output_group, output_low });
            try self.dense(layer.attn_output_b, &.{ output_low, shape.n_embd });
            // V4.1 grouped-output kernels consume Q8 blocks directly, even
            // though the general dense projections admit other quantizations.
            try self.layout(layer.attn_output_a, c.DS4_TENSOR_Q8_0, &.{ output_group, output_low });
            try self.layout(layer.attn_output_b, c.DS4_TENSOR_Q8_0, &.{ output_low, shape.n_embd });
            if (policy.publishes_kv) {
                try self.layout(layer.attn_compressor_kv, c.DS4_TENSOR_F16, &.{ shape.n_embd, shape.n_head_dim });
                try self.layout(layer.attn_compressor_norm, c.DS4_TENSOR_F32, &.{shape.n_head_dim});
                if (policy.compression_ratio == 2)
                    try self.layout(layer.attn_compressor_gate, c.DS4_TENSOR_F16, &.{ shape.n_embd, shape.n_head_dim });
                try self.layout(layer.indexer_attn_k, c.DS4_TENSOR_F16, &.{ shape.n_head_dim, shape.n_indexer_head_dim });
                try self.layout(layer.indexer_k_norm, c.DS4_TENSOR_F32, &.{shape.n_indexer_head_dim});
            }
            if (policy.publishes_index) {
                try self.layout(layer.indexer_attn_q_b, c.DS4_TENSOR_F16, &.{ shape.n_lora_q, @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim });
                try self.layout(layer.indexer_proj, c.DS4_TENSOR_F16, &.{ shape.n_embd, shape.n_indexer_head });
            }
            if (policy.uses_engram) {
                try self.layout(layer.engram_kv, c.DS4_TENSOR_F16, &.{ 24 * 256, (@as(u64, shape.n_hc) + 1) * shape.n_embd });
                try self.layout(layer.engram_q_norm, c.DS4_TENSOR_F32, &.{ shape.n_embd, shape.n_hc });
                try self.layout(layer.engram_k_norm, c.DS4_TENSOR_F32, &.{ shape.n_embd, shape.n_hc });
            }
            try self.layout(layer.hc_ffn_fn, c.DS4_TENSOR_F16, &.{ hc_dim, mix_dim });
            try self.layout(layer.hc_ffn_scale, c.DS4_TENSOR_F32, &.{3});
            try self.layout(layer.hc_ffn_base, c.DS4_TENSOR_F32, &.{mix_dim});
            try self.layout(layer.ffn_norm, c.DS4_TENSOR_F32, &.{shape.n_embd});
            try self.layout(layer.ffn_gate_inp, c.DS4_TENSOR_F32, &.{ shape.n_embd, shape.n_expert });
            try self.layout(layer.ffn_exp_probs_b, c.DS4_TENSOR_F32, &.{shape.n_expert});
            try self.expert(layer.ffn_gate_exps, &.{ shape.n_embd, shape.n_ff_exp, shape.n_expert });
            try self.expert(layer.ffn_up_exps, &.{ shape.n_embd, shape.n_ff_exp, shape.n_expert });
            try self.expert(layer.ffn_down_exps, &.{ shape.n_ff_exp, shape.n_embd, shape.n_expert });
            if (layer.ffn_gate_exps.*.type != layer.ffn_up_exps.*.type)
                return self.fail("routed gate/up experts use different quant types in layer {d}", .{index});
            try self.dense(layer.ffn_gate_shexp, &.{ shape.n_embd, shape.n_ff_exp });
            try self.dense(layer.ffn_up_shexp, &.{ shape.n_embd, shape.n_ff_exp });
            try self.dense(layer.ffn_down_shexp, &.{ shape.n_ff_exp, shape.n_embd });
            if (index < shape.n_hash_layer)
                try self.layout(layer.ffn_gate_tid2eid, c.DS4_TENSOR_I32, &.{ shape.n_expert_used, shape.n_vocab });
        }
    }
};

pub fn bindRecord(output: *Table, model: *const c.ds4_model, config: *const c.ds4_model_config, load_slice: bool, start: u32, end: u32, require_output: bool, optional_output: bool, error_pointer: [*c]u8, error_capacity: usize) bool {
    const error_text: []u8 = if (error_capacity == 0) &.{} else error_pointer[0..error_capacity];
    if (error_text.len != 0) error_text[0] = 0;
    var weights = Weights{ .config = .{ .shape = config.shape, .compress_ratios = config.compress_ratios } };
    defer output.* = weights.table;
    weights.bind(model, .{ .slice = load_slice, .start = start, .end = end, .require_output = require_output, .optional_output = optional_output }, error_text) catch return false;
    return true;
}
