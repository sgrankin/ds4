//! Bind the trained Qwen tensor names once. Shapes describe logical activations;
//! the padded Q2_K down row is the one intentional physical-stride exception.
//! Tables borrow descriptors and bytes from the original GGUF model.
const std = @import("std");
const c = @import("gguf_c");
const policy = @import("config.zig");
const Config = c.ds4_qwen4_config;
const Tensor = c.ds4_tensor;
const Table = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const InvalidWeights = error{InvalidWeights};

pub const Options = struct {
    slice: bool = false,
    start: u32 = 0,
    end: u32 = std.math.maxInt(u32),
    require_output: bool = true,
    optional_output: bool = false,
};

const hc_fields = .{ "hc_attn_norm", "hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_norm", "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject" };
const linear_fields = .{ "lin_qkv", "lin_gate", "lin_conv", "lin_dt_bias", "lin_a", "lin_beta", "lin_alpha", "lin_norm", "lin_out" };
const attention_fields = .{ "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "indexer_q_proj", "indexer_k_proj", "indexer_q_norm", "indexer_k_norm" };
const ple_fields = .{ "ple_key", "ple_value", "ple_norm_key", "ple_norm_query", "ple_norm_conv", "ple_conv" };
const expert_fields = .{ "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
const nextn_fields = .{ "nextn_eh_proj", "nextn_enorm", "nextn_hnorm", "nextn_hc_head_norm", "nextn_hc_head_down", "nextn_hc_head_up" };
const output_fields = .{ "output_hc_norm", "output_hc_down", "output_hc_up", "output" };

pub fn layerComplete(layer: *const Layer, index: u32, config: *const Config) bool {
    inline for (hc_fields ++ expert_fields) |field| if (@field(layer, field) == null) return false;
    if (policy.isLinear(config, index)) {
        inline for (linear_fields) |field| if (@field(layer, field) == null) return false;
    } else {
        inline for (attention_fields) |field| if (@field(layer, field) == null) return false;
    }
    if (policy.isPle(config, index)) inline for (ple_fields) |field| {
        if (@field(layer, field) == null) return false;
    };
    if (policy.isNextn(config, index)) inline for (nextn_fields) |field| {
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
        // As in the GGUF accessors, the first duplicate directory entry wins.
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
        const t = maybe_tensor orelse return self.fail("internal error: missing tensor while validating layout", .{});
        if (t.type != kind) return self.fail("tensor {s} has type {d}, expected {d}", .{ bytes(t.name), t.type, kind });
        if (t.ndim != dimensions.len) return self.fail("tensor {s} has {d} dimensions, expected {d}", .{ bytes(t.name), t.ndim, dimensions.len });
        for (dimensions, 0..) |dimension, i|
            if (t.dim[i] != dimension) return self.fail("tensor {s} has dim[{d}]={d}, expected {d}", .{ bytes(t.name), i, t.dim[i], dimension });
    }

    fn dense(self: *const Binding, maybe_tensor: ?*const Tensor, dimensions: []const u64) InvalidWeights!void {
        const t = maybe_tensor orelse return self.fail("internal error: missing tensor while validating layout", .{});
        switch (t.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_Q4_0, c.DS4_TENSOR_F16, c.DS4_TENSOR_BF16, c.DS4_TENSOR_F32 => {},
            else => return self.fail("tensor {s} has type {d}, expected Q8_0, Q4_0, F16, BF16 or F32", .{ bytes(t.name), t.type }),
        }
        try self.layout(t, t.type, dimensions);
    }

    fn expert(self: *const Binding, maybe_tensor: ?*const Tensor, dimensions: []const u64) InvalidWeights!void {
        const t = maybe_tensor orelse return self.fail("internal error: missing tensor while validating layout", .{});
        switch (t.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_Q6_K, c.DS4_TENSOR_MXFP4, c.DS4_TENSOR_F16, c.DS4_TENSOR_F32, c.DS4_TENSOR_Q4_0 => {},
            else => return self.fail("routed expert tensor {s} has unsupported type {d}", .{ bytes(t.name), t.type }),
        }
        try self.layout(t, t.type, dimensions);
    }

    fn haveOutput(self: *const Binding) bool {
        inline for (output_fields) |field| if (@field(self.table, field) == null) return false;
        return true;
    }

    fn partialOutput(self: *const Binding) bool {
        inline for (output_fields) |field| if (@field(self.table, field) != null) return true;
        return false;
    }

    fn bindLayer(self: *Binding, index: u32) InvalidWeights!void {
        const layer = &self.table.layer[index];
        inline for (hc_fields) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
        if (policy.isLinear(self.config, index)) {
            inline for (.{ .{ "lin_qkv", "attn_qkv.weight" }, .{ "lin_gate", "attn_gate.weight" }, .{ "lin_conv", "ssm_conv1d.weight" }, .{ "lin_dt_bias", "ssm_dt.bias" }, .{ "lin_a", "ssm_a" }, .{ "lin_beta", "ssm_beta.weight" }, .{ "lin_alpha", "ssm_alpha.weight" }, .{ "lin_norm", "ssm_norm.weight" }, .{ "lin_out", "ssm_out.weight" } }) |field|
                @field(layer, field[0]) = try self.layerTensor(index, field[1]);
        } else {
            inline for (.{ "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm" }) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
            inline for (.{ .{ "indexer_q_proj", "indexer.q_proj.weight" }, .{ "indexer_k_proj", "indexer.k_proj.weight" }, .{ "indexer_q_norm", "indexer.q_norm.weight" }, .{ "indexer_k_norm", "indexer.k_norm.weight" } }) |field|
                @field(layer, field[0]) = try self.layerTensor(index, field[1]);
        }
        if (policy.isPle(self.config, index)) {
            inline for (.{ "ple_key", "ple_value", "ple_norm_key", "ple_norm_query", "ple_norm_conv" }) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
            layer.ple_conv = try self.layerTensor(index, "ple_conv1d.weight");
        }
        inline for (expert_fields) |field| @field(layer, field) = try self.layerTensor(index, field ++ ".weight");
        if (policy.isNextn(self.config, index)) {
            inline for (.{ .{ "nextn_eh_proj", "nextn.eh_proj.weight" }, .{ "nextn_enorm", "nextn.enorm.weight" }, .{ "nextn_hnorm", "nextn.hnorm.weight" }, .{ "nextn_hc_head_norm", "nextn.hc_head_norm.weight" }, .{ "nextn_hc_head_down", "nextn.hc_head_down.weight" }, .{ "nextn_hc_head_up", "nextn.hc_head_up.weight" } }) |field|
                @field(layer, field[0]) = try self.layerTensor(index, field[1]);
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
        self.table.ple_embd = self.find("per_layer_token_embd.weight");
        if (require_output or optional_output) inline for (output_fields) |field| {
            @field(self.table, field) = try self.tensor(field ++ ".weight", require_output);
        };
        if (optional_output and self.partialOutput() and !self.haveOutput()) return self.fail("partial output head in GGUF", .{});
        for (start..end + 1) |index| try self.bindLayer(@intCast(index));
        // MTP is outside the trunk execution range, but a full load must bind
        // it so the drafter can use the same borrowed table.
        if (!options.slice) for (trunk..self.config.shape.n_layer) |index| try self.bindLayer(@intCast(index));
        try self.validate(start, end, require_embedding, require_output);
    }

    pub fn validate(self: *const Binding, start: u32, last: u32, require_embedding: bool, require_output: bool) InvalidWeights!void {
        const s = &self.config.shape;
        if (start >= s.n_layer) return self.fail("invalid first layer in Qwen weight layout validation", .{});
        const end = if (last == std.math.maxInt(u32)) s.n_layer - 1 else last;
        if (end >= s.n_layer or end < start) return self.fail("invalid layer range in Qwen weight layout validation", .{});
        const hc = @as(u64, s.n_embd) * s.n_hc;
        const q = @as(u64, s.n_head) * s.n_head_dim;
        const kv = @as(u64, s.n_head_kv) * s.n_head_dim;
        const linear_k = @as(u64, s.n_lin_k_head) * s.n_lin_head_dim;
        const linear_v = @as(u64, s.n_lin_v_head) * s.n_lin_head_dim;
        if (require_embedding and self.table.token_embd == null) return self.fail("required token embedding tensor is missing", .{});
        if (self.table.token_embd != null) try self.dense(self.table.token_embd, &.{ s.n_embd, s.n_vocab });
        if (require_embedding and self.table.ple_embd == null) return self.fail("Qwen GGUF lacks its n-grams; repack with gguf-tools/qwen4_native_ngrams.py", .{});
        if (self.table.ple_embd != null) {
            const t = self.table.ple_embd;
            if (t.*.type != c.DS4_TENSOR_BF16) return self.fail("n-gram embeddings must use original BF16, got type {d}", .{t.*.type});
            if (t.*.ndim != 2 or t.*.dim[0] != s.n_ple_head_dim or t.*.dim[1] < self.config.ple.n_rows)
                return self.fail("per_layer_token_embd layout [{d}, {d}] does not cover {d} x {d} hash rows", .{ t.*.dim[0], t.*.dim[1], s.n_ple_head_dim, self.config.ple.n_rows });
        }
        if (require_output and !self.haveOutput()) return self.fail("required output head tensors are missing", .{});
        if (self.partialOutput() and !self.haveOutput()) return self.fail("partial output head in GGUF", .{});
        if (self.haveOutput()) {
            try self.layout(self.table.output_hc_norm, c.DS4_TENSOR_F32, &.{hc});
            try self.dense(self.table.output_hc_down, &.{ hc, s.n_hc_lowrank });
            try self.dense(self.table.output_hc_up, &.{ s.n_hc_lowrank, hc });
            try self.dense(self.table.output, &.{ s.n_embd, s.n_vocab });
        }
        for (start..end + 1) |index| {
            const layer = &self.table.layer[index];
            const il: u32 = @intCast(index);
            if (!layerComplete(layer, il, self.config)) return self.fail("required Qwen tensors for layer {d} are missing", .{index});
            inline for (.{ "hc_attn_norm", "hc_ffn_norm" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{hc});
            inline for (.{ "hc_attn_down", "hc_ffn_down" }) |field| try self.dense(@field(layer, field), &.{ hc, s.n_hc_lowrank });
            inline for (.{ "hc_attn_up", "hc_ffn_up" }) |field| try self.dense(@field(layer, field), &.{ s.n_hc_lowrank, hc });
            inline for (.{ "hc_attn_inject", "hc_ffn_inject" }) |field| try self.dense(@field(layer, field), &.{ hc, s.n_hc });
            if (policy.isLinear(self.config, il)) {
                try self.dense(layer.lin_qkv, &.{ s.n_embd, 2 * linear_k + linear_v });
                try self.dense(layer.lin_gate, &.{ s.n_embd, linear_v });
                try self.layout(layer.lin_conv, c.DS4_TENSOR_F32, &.{ s.n_lin_conv, 2 * linear_k + linear_v });
                inline for (.{ "lin_dt_bias", "lin_a" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{s.n_lin_v_head});
                inline for (.{ "lin_beta", "lin_alpha" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, s.n_lin_v_head });
                try self.layout(layer.lin_norm, c.DS4_TENSOR_F32, &.{s.n_lin_head_dim});
                try self.dense(layer.lin_out, &.{ linear_v, s.n_embd });
            } else {
                try self.dense(layer.attn_q, &.{ s.n_embd, 2 * q });
                inline for (.{ "attn_k", "attn_v" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, kv });
                try self.dense(layer.attn_output, &.{ q, s.n_embd });
                inline for (.{ "attn_q_norm", "attn_k_norm" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{s.n_head_dim});
                try self.dense(layer.indexer_q_proj, &.{ s.n_embd, @as(u64, s.n_indexer_head) * s.n_indexer_head_dim });
                try self.dense(layer.indexer_k_proj, &.{ s.n_embd, s.n_indexer_head_dim });
                inline for (.{ "indexer_q_norm", "indexer_k_norm" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{s.n_indexer_head_dim});
            }
            if (policy.isPle(self.config, il)) {
                try self.dense(layer.ple_key, &.{ s.n_embd, hc });
                try self.dense(layer.ple_value, &.{ s.n_embd, s.n_embd });
                inline for (.{ "ple_norm_key", "ple_norm_query", "ple_norm_conv" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{hc});
                try self.layout(layer.ple_conv, if (layer.ple_conv.*.type == c.DS4_TENSOR_F16) c.DS4_TENSOR_F16 else c.DS4_TENSOR_F32, &.{ s.n_ple_conv, hc });
            }
            try self.layout(layer.ffn_gate_inp, c.DS4_TENSOR_F32, &.{ s.n_embd, s.n_expert });
            inline for (.{ "ffn_gate_exps", "ffn_up_exps" }) |field| try self.expert(@field(layer, field), &.{ s.n_embd, s.n_ff_exp, s.n_expert });
            // Q2_K stores 640 logical inputs in three 256-value blocks. Keep
            // activations logical-width; only the weight row stride is padded.
            const down: u64 = if (layer.ffn_down_exps.*.type == c.DS4_TENSOR_Q2_K) (s.n_ff_exp + 255) / 256 * 256 else s.n_ff_exp;
            try self.expert(layer.ffn_down_exps, &.{ down, s.n_embd, s.n_expert });
            if (layer.ffn_gate_exps.*.type != layer.ffn_up_exps.*.type) return self.fail("routed gate/up experts use different quant types in layer {d}", .{index});
            if (layer.ffn_gate_inp_shexp.*.ndim == 1) try self.layout(layer.ffn_gate_inp_shexp, c.DS4_TENSOR_F32, &.{s.n_embd}) else try self.layout(layer.ffn_gate_inp_shexp, c.DS4_TENSOR_F32, &.{ s.n_embd, 1 });
            inline for (.{ "ffn_gate_shexp", "ffn_up_shexp" }) |field| try self.dense(@field(layer, field), &.{ s.n_embd, s.n_ff_exp });
            try self.dense(layer.ffn_down_shexp, &.{ s.n_ff_exp, s.n_embd });
            if (policy.isNextn(self.config, il)) {
                try self.dense(layer.nextn_eh_proj, &.{ @as(u64, 2) * s.n_embd, s.n_embd });
                try self.layout(layer.nextn_enorm, c.DS4_TENSOR_F32, &.{s.n_embd});
                inline for (.{ "nextn_hnorm", "nextn_hc_head_norm" }) |field| try self.layout(@field(layer, field), c.DS4_TENSOR_F32, &.{hc});
                try self.dense(layer.nextn_hc_head_down, &.{ hc, s.n_hc_lowrank });
                try self.dense(layer.nextn_hc_head_up, &.{ s.n_hc_lowrank, hc });
            }
        }
    }
};
