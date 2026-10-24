//! Typed GLM weight binding. Flash adds mHC and recurrent KDA layers; the legacy
//! DSA variant keeps its original branches. Quantization acceptance is explicit
//! and separate for dense projections and routed experts.
const std = @import("std");
const c = @import("gguf_c");
const policy = @import("config.zig");
const Config = c.ds4_glm_model_config;
const Tensor = c.ds4_tensor;
const Table = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const InvalidWeights = error{InvalidWeights};

pub const Options = struct { slice: bool = false, start: u32 = 0, end: u32 = std.math.maxInt(u32), require_output: bool = true, optional_output: bool = false };
const hc_fields = .{ "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base" };
const kda_fields = .{ "kda_q", "kda_k", "kda_v", "kda_q_conv", "kda_k_conv", "kda_v_conv", "kda_f_a", "kda_f_b", "kda_dt_bias", "kda_a_log", "kda_beta", "kda_g_a", "kda_g_b", "kda_o_norm", "kda_output" };
const attention_fields = .{ "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv_a_mqa", "attn_kv_a_norm", "attn_k_b", "attn_v_b", "attn_output", "indexer_attn_q_b", "indexer_attn_k", "indexer_k_norm", "indexer_k_norm_b", "indexer_proj" };
const expert_fields = .{ "ffn_gate_inp", "ffn_exp_probs_b", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
const nextn_fields = .{ "nextn_eh_proj", "nextn_enorm", "nextn_hnorm", "nextn_shared_head_norm" };

fn flash(config: *const Config) bool {
    return config.shape.variant == c.DS4_VARIANT_GLM53;
}

fn nextn(config: *const Config, layer: u32) bool {
    return config.shape.n_nextn_predict != 0 and layer >= policy.trunkLayers(config);
}

pub fn layerComplete(layer: *const Layer, index: u32, config: *const Config) bool {
    if (layer.attn_norm == null or layer.ffn_norm == null) return false;
    if (flash(config) and index < policy.trunkLayers(config)) inline for (hc_fields) |field| {
        if (@field(layer, field) == null) return false;
    };
    if (policy.isKda(config, index)) {
        inline for (kda_fields) |field| if (@field(layer, field) == null) return false;
    } else {
        inline for (attention_fields) |field| if (@field(layer, field) == null) return false;
        if (flash(config) and (layer.indexer_compressor_ape == null or layer.indexer_compressor_gate == null)) return false;
    }
    if (index < config.shape.n_leading_dense) {
        if (layer.ffn_gate == null or layer.ffn_up == null or layer.ffn_down == null) return false;
    } else {
        inline for (expert_fields) |field| if (@field(layer, field) == null) return false;
    }
    if (nextn(config, index)) inline for (nextn_fields) |field| {
        if (@field(layer, field) == null) return false;
    };
    return true;
}

fn bytes(text: c.ds4_str) []const u8 {
    return if (text.len == 0) &.{} else text.ptr[0..@intCast(text.len)];
}

pub const Binding = struct {
    table: *Table,
    model: ?*const c.ds4_model = null,
    config: *const Config,
    errors: []u8,

    fn fail(self: *const Binding, comptime format: []const u8, args: anytype) InvalidWeights {
        if (self.errors.len != 0) {
            _ = std.mem.printSentinel(self.errors, format, args, 0) catch {
                self.errors[self.errors.len - 1] = 0;
                return error.InvalidWeights;
            };
        }
        return error.InvalidWeights;
    }

    fn find(self: *const Binding, name: []const u8) ?*Tensor {
        const model = self.model.?;
        if (model.n_tensors == 0) return null;
        for (model.tensors[0..@intCast(model.n_tensors)]) |*descriptor|
            if (std.mem.eql(u8, bytes(descriptor.name), name)) return descriptor;
        return null;
    }

    fn tensor(self: *const Binding, name: []const u8, required: bool) InvalidWeights!?*Tensor {
        if (self.find(name)) |value| return value;
        if (required) return self.fail("required tensor is missing: {s}", .{name});
        return null;
    }

    fn layerTensor(self: *const Binding, index: u32, suffix: []const u8) InvalidWeights!*Tensor {
        var storage: [128]u8 = undefined;
        const name = std.mem.printSentinel(&storage, "blk.{d}.{s}", .{ index, suffix }, 0) catch return self.fail("tensor name is too long", .{});
        return (try self.tensor(name, true)).?;
    }

    fn layout(self: *const Binding, maybe_tensor: ?*const Tensor, kind: u32, dimensions: []const u64) InvalidWeights!void {
        const t = maybe_tensor orelse return self.fail("internal error: missing tensor while validating GLM layout", .{});
        if (t.type != kind) return self.fail("tensor {s} has type {d}, expected {d}", .{ bytes(t.name), t.type, kind });
        if (t.ndim != dimensions.len) return self.fail("tensor {s} has {d} dimensions, expected {d}", .{ bytes(t.name), t.ndim, dimensions.len });
        for (dimensions, 0..) |dimension, i|
            if (t.dim[i] != dimension) return self.fail("tensor {s} has dim[{d}]={d}, expected {d}", .{ bytes(t.name), i, t.dim[i], dimension });
    }

    fn dense(self: *const Binding, maybe_tensor: ?*const Tensor, dimensions: []const u64) InvalidWeights!void {
        const t = maybe_tensor orelse return self.fail("internal error: missing tensor while validating GLM dense layout", .{});
        switch (t.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q4_0, c.DS4_TENSOR_BF16 => {},
            else => return self.fail("tensor {s} has type {d}, expected bf16, q8_0, q4_K, or q4_0", .{ bytes(t.name), t.type }),
        }
        try self.layout(t, t.type, dimensions);
    }

    fn expert(self: *const Binding, maybe_tensor: ?*const Tensor, dimensions: []const u64) InvalidWeights!void {
        const t = maybe_tensor orelse return self.fail("internal error: missing routed expert tensor while validating layout", .{});
        switch (t.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_Q6_K, c.DS4_TENSOR_MXFP4 => {},
            else => return self.fail("tensor {s} has type {d}, expected a routed expert quant type", .{ bytes(t.name), t.type }),
        }
        try self.layout(t, t.type, dimensions);
    }

    fn haveOutput(self: *const Binding) bool {
        return self.table.output_norm != null and self.table.output != null;
    }

    fn partialOutput(self: *const Binding) bool {
        return self.table.output_norm != null or self.table.output != null;
    }

    fn bindLayer(self: *Binding, index: u32) InvalidWeights!void {
        const layer = &self.table.layer[index];
        layer.attn_norm = try self.layerTensor(index, "attn_norm.weight");
        layer.ffn_norm = try self.layerTensor(index, "ffn_norm.weight");
        if (flash(self.config) and index < policy.trunkLayers(self.config)) inline for (hc_fields) |field| {
            @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
        };
        if (policy.isKda(self.config, index)) {
            inline for (kda_fields) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
        } else {
            inline for (.{ "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv_a_mqa", "attn_kv_a_norm", "attn_k_b", "attn_v_b", "attn_output" }) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
            inline for (.{ .{ "indexer_attn_q_b", "indexer.attn_q_b.weight" }, .{ "indexer_attn_k", "indexer.attn_k.weight" }, .{ "indexer_k_norm", "indexer.k_norm.weight" }, .{ "indexer_k_norm_b", "indexer.k_norm.bias" }, .{ "indexer_proj", "indexer.proj.weight" } }) |field| @field(layer, field[0]) = try self.layerTensor(index, field[1]);
            if (flash(self.config)) {
                layer.indexer_compressor_ape = try self.layerTensor(index, "indexer.pool_ape.weight");
                layer.indexer_compressor_gate = try self.layerTensor(index, "indexer.pool_gate.weight");
            }
        }
        if (index < self.config.shape.n_leading_dense) {
            inline for (.{ "ffn_gate", "ffn_up", "ffn_down" }) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
        } else {
            layer.ffn_gate_inp = try self.layerTensor(index, "ffn_gate_inp.weight");
            layer.ffn_exp_probs_b = try self.layerTensor(index, "exp_probs_b.bias");
            inline for (.{ "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" }) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
        }
        if (nextn(self.config, index)) {
            inline for (.{ .{ "nextn_eh_proj", "nextn.eh_proj.weight" }, .{ "nextn_enorm", "nextn.enorm.weight" }, .{ "nextn_hnorm", "nextn.hnorm.weight" }, .{ "nextn_shared_head_norm", "nextn.shared_head_norm.weight" } }) |field| @field(layer, field[0]) = try self.layerTensor(index, field[1]);
        }
    }

    pub fn bind(self: *Binding, options: Options) InvalidWeights!void {
        self.table.* = std.mem.zeroes(Table);
        const trunk = policy.trunkLayers(self.config);
        var start: u32 = 0;
        var end = trunk - 1;
        var require_embedding = true;
        var require_output = options.require_output;
        var optional_output = options.optional_output;
        if (options.slice) {
            start = options.start;
            end = if (options.end == std.math.maxInt(u32)) trunk - 1 else options.end;
            if (start >= trunk or end >= trunk or end < start) return self.fail("invalid model load layer slice", .{});
            require_embedding = start == 0;
        } else {
            require_output = true;
            optional_output = false;
        }
        self.table.token_embd = try self.tensor("token_embd.weight", require_embedding);
        if (require_output or optional_output) {
            self.table.output_norm = try self.tensor("output_norm.weight", require_output);
            self.table.output = try self.tensor("output.weight", require_output);
        }
        if (optional_output and self.partialOutput() and !self.haveOutput()) return self.fail("partial output head in GGUF", .{});
        for (start..end + 1) |index| try self.bindLayer(@intCast(index));
        // Draft blocks are not trunk execution layers, but full loads retain
        // their bindings for MTP. Sliced inference must not require those blocks.
        if (!options.slice) for (trunk..self.config.shape.n_layer) |index| try self.bindLayer(@intCast(index));
        try self.validate(start, end, require_embedding, require_output);
    }

    pub fn validate(self: *const Binding, start: u32, last: u32, require_embedding: bool, require_output: bool) InvalidWeights!void {
        const s = &self.config.shape;
        if (start >= s.n_layer) return self.fail("invalid first layer in GLM weight layout validation", .{});
        const end = if (last == std.math.maxInt(u32)) s.n_layer - 1 else last;
        if (end >= s.n_layer or end < start) return self.fail("invalid layer range in GLM weight layout validation", .{});
        const q = @as(u64, s.n_head) * s.n_key_mla;
        const q_nope = s.n_key_mla - s.n_rot;
        const hc = @as(u64, s.n_embd) * s.n_hc;
        const mix = 2 * @as(u64, s.n_hc) + @as(u64, s.n_hc) * s.n_hc;
        const recurrent = @as(u64, s.n_kda_head) * s.n_kda_head_dim;
        if (require_embedding and self.table.token_embd == null) return self.fail("required token embedding tensor is missing", .{});
        if (self.table.token_embd != null) try self.dense(self.table.token_embd, &.{ s.n_embd, s.n_vocab });
        if (require_output and !self.haveOutput()) return self.fail("required output head tensors are missing", .{});
        if (self.partialOutput() and !self.haveOutput()) return self.fail("partial output head in GGUF", .{});
        if (self.haveOutput()) {
            try self.layout(self.table.output_norm, c.DS4_TENSOR_F32, &.{s.n_embd});
            try self.dense(self.table.output, &.{ s.n_embd, s.n_vocab });
        }
        for (start..end + 1) |index| {
            const layer = &self.table.layer[index];
            const il: u32 = @intCast(index);
            if (!layerComplete(layer, il, self.config)) return self.fail("required GLM tensors for layer {d} are missing", .{index});
            try self.layout(layer.attn_norm, c.DS4_TENSOR_F32, &.{s.n_embd});
            if (flash(self.config) and index < policy.trunkLayers(self.config)) {
                inline for (.{ "hc_attn_fn", "hc_ffn_fn" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_BF16, &.{ hc, mix });
                inline for (.{ "hc_attn_scale", "hc_ffn_scale" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{3});
                inline for (.{ "hc_attn_base", "hc_ffn_base" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{mix});
            }
            if (policy.isKda(self.config, il)) {
                inline for (.{ "kda_q", "kda_k", "kda_v" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, recurrent });
                inline for (.{ "kda_q_conv", "kda_k_conv", "kda_v_conv" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{ s.n_kda_conv, 1, recurrent });
                inline for (.{ "kda_f_a", "kda_g_a" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, s.n_kda_head_dim });
                inline for (.{ "kda_f_b", "kda_g_b" }) |field| try self.dense(@field(layer, field), &.{ s.n_kda_head_dim, recurrent });
                try self.layout(layer.kda_dt_bias, c.DS4_TENSOR_F32, &.{recurrent});
                try self.layout(layer.kda_a_log, c.DS4_TENSOR_F32, &.{s.n_kda_head});
                try self.dense(layer.kda_beta, &.{ s.n_embd, s.n_kda_head });
                try self.layout(layer.kda_o_norm, c.DS4_TENSOR_F32, &.{s.n_kda_head_dim});
                try self.dense(layer.kda_output, &.{ recurrent, s.n_embd });
            } else {
                try self.dense(layer.attn_q_a, &.{ s.n_embd, s.n_lora_q });
                try self.layout(layer.attn_q_a_norm, c.DS4_TENSOR_F32, &.{s.n_lora_q});
                try self.dense(layer.attn_q_b, &.{ s.n_lora_q, q });
                try self.dense(layer.attn_kv_a_mqa, &.{ s.n_embd, s.n_kv_lora + s.n_rot });
                try self.layout(layer.attn_kv_a_norm, c.DS4_TENSOR_F32, &.{s.n_kv_lora});
                try self.dense(layer.attn_k_b, &.{ q_nope, s.n_kv_lora, s.n_head });
                try self.dense(layer.attn_v_b, &.{ s.n_kv_lora, s.n_value_mla, s.n_head });
                try self.dense(layer.attn_output, &.{ @as(u64, s.n_head) * s.n_value_mla, s.n_embd });
                try self.dense(layer.indexer_attn_k, &.{ s.n_embd, s.n_indexer_head_dim });
                try self.dense(layer.indexer_attn_q_b, &.{ s.n_lora_q, @as(u64, s.n_indexer_head) * s.n_indexer_head_dim });
                inline for (.{ "indexer_k_norm", "indexer_k_norm_b" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{s.n_indexer_head_dim});
                if (flash(self.config)) {
                    try self.dense(layer.indexer_proj, &.{ s.n_embd, s.n_indexer_head });
                    try self.layout(layer.indexer_compressor_ape, c.DS4_TENSOR_BF16, &.{ s.n_indexer_head_dim, 4 });
                    try self.layout(layer.indexer_compressor_gate, c.DS4_TENSOR_BF16, &.{ s.n_embd, s.n_indexer_head_dim });
                } else try self.layout(layer.indexer_proj, c.DS4_TENSOR_F32, &.{ s.n_embd, s.n_indexer_head });
            }
            try self.layout(layer.ffn_norm, c.DS4_TENSOR_F32, &.{s.n_embd});
            if (index < s.n_leading_dense) {
                inline for (.{ "ffn_gate", "ffn_up" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, s.n_ff_dense });
                try self.dense(layer.ffn_down, &.{ s.n_ff_dense, s.n_embd });
            } else {
                try self.layout(layer.ffn_gate_inp, c.DS4_TENSOR_F32, &.{ s.n_embd, s.n_expert });
                try self.layout(layer.ffn_exp_probs_b, c.DS4_TENSOR_F32, &.{s.n_expert});
                inline for (.{ "ffn_gate_exps", "ffn_up_exps" }) |field| try self.expert(@field(layer, field), &.{ s.n_embd, s.n_ff_exp, s.n_expert });
                try self.expert(layer.ffn_down_exps, &.{ s.n_ff_exp, s.n_embd, s.n_expert });
                if (layer.ffn_gate_exps.*.type != layer.ffn_up_exps.*.type) return self.fail("GLM routed gate/up experts use different quant types in layer {d}", .{index});
                inline for (.{ "ffn_gate_shexp", "ffn_up_shexp" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, s.n_ff_exp });
                try self.dense(layer.ffn_down_shexp, &.{ s.n_ff_exp, s.n_embd });
            }
            if (nextn(self.config, il)) {
                try self.dense(layer.nextn_eh_proj, &.{ @as(u64, 2) * s.n_embd, s.n_embd });
                inline for (.{ "nextn_enorm", "nextn_hnorm", "nextn_shared_head_norm" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{s.n_embd});
            }
        }
    }
};
