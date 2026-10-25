//! The actual six-layer learned joint head on Metal. The CPU oracle remains in
//! head.zig; only the final small softmax and caller-order mapping run on CPU.
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const records = @import("record.zig");
const types = @import("types.zig");
const a = std.heap.c_allocator;
const View = b.View;
const Matrix = struct { weight: View, bias: ?View = null, n: u32, k: u32 };
const Norm = struct { weight: View, bias: View };
const Attention = struct {
    qkv: Matrix,
    out: Matrix,
    fn query(self: Attention) Matrix {
        return .{ .weight = self.qkv.weight, .bias = self.qkv.bias, .n = self.qkv.k, .k = self.qkv.k };
    }
    fn valueBias(self: Attention) View {
        return self.qkv.bias.?.at(@as(usize, self.qkv.k) * 2 * 4);
    }
};
const Evidence = struct { norm: Norm, ffnorm: Norm, attention: Attention, ff1: Matrix, ff2: Matrix };
const Decoder = struct { norm1: Norm, norm2: Norm, norm3: Norm, self_attention: Attention, cross: Attention, ff1: Matrix, ff2: Matrix };
const Rows = extern struct { rows: u32, width: u32, mode: u32 = 0, pad: u32 = 0 };
const Gemm = extern struct { rows: u32, columns: u32, inner: u32 };
const Gemv = extern struct { rows: u32, columns: u32, inner: u32, pad: u32 = 0 };
const NormArgs = extern struct { width: u32, epsilon: f32 = 1e-5 };
const AttnArgs = extern struct { queries: u32, keys: u32, width: u32 = 1024, heads: u32 = 16, q_stride: u32, kv_stride: u32, value_bias: u32, pad: u32 = 0 };
const Question = extern struct { offset: u32, count: u32, kind: u32, pad: u32 = 0 };
const Score = extern struct { rows: u32, width: u32 = 1024, hidden: u32, pad: u32 = 0, prior: f32, joint: f32, gate: f32, bias: f32 };
const Post = enum(u32) { bias = 0, gelu = 1, residual = 2 };

pub const State = struct {
    context: Matrix,
    lexical: Matrix,
    option_question: Matrix,
    question: Matrix,
    global: Matrix,
    type_embedding: View,
    summary_norm: Norm,
    field_norm: Norm,
    option_norm: Norm,
    evidence: []Evidence,
    decoder: []Decoder,
    scorer1: Matrix,
    scorer2: Matrix,
    prior: f32,
    joint: f32,
    gate: f32,
    score_bias: f32,
    scratch: ?Scratch = null,
    pub fn open(engine: *b.Engine) !State {
        const cfg = engine.config;
        if (cfg.W != 1024 or cfg.head_heads != 16) return error.UnsupportedHeadShape;
        try engine.compile(@embedFile("../../metal/clef/head_full.metal"));
        for ([_][:0]const u8{ "hf_pool", "hf_gemv", "hf_post", "hf_options", "hf_attend", "hf_summary", "hf_fields", "hf_features", "hf_logits" }) |kernel| {
            if (!c.flint_gpu_prepare(engine.gpu, kernel)) return error.GpuPipeline;
        }
        const load = Loader{ .engine = engine };
        const evidence = try a.alloc(Evidence, cfg.routing_layers);
        errdefer a.free(evidence);
        const decoder = try a.alloc(Decoder, cfg.head_layers);
        errdefer a.free(decoder);
        const type_tensor = try engine.tensor("head.type_embedding.weight");
        const embedding = try engine.buffer(type_tensor.bytes.len);
        @memcpy(b.contents(u8, embedding, type_tensor.bytes.len), type_tensor.bytes);
        var out = State{
            .context = try load.linear("head.option_context_projection", 1024, cfg.H, false),
            .lexical = try load.linear("head.option_lexical_projection", 1024, cfg.H, false),
            .option_question = try load.linear("head.option_question_projection", 1024, cfg.H, false),
            .question = try load.linear("head.question_projection", 1024, cfg.H, false),
            .global = try load.linear("head.global_projection", 1024, cfg.H, false),
            .type_embedding = embedding,
            .summary_norm = try load.norm("head.option_summary_norm"),
            .field_norm = try load.norm("head.field_norm"),
            .option_norm = try load.norm("head.option_norm"),
            .evidence = evidence,
            .decoder = decoder,
            .scorer1 = try load.linear("head.residual_scorer.0", 1024, 4096, true),
            .scorer2 = try load.linear("head.residual_scorer.3", 1, 1024, true),
            .prior = @exp(@min(try load.scalar("head.prior_logit_scale"), @log(@as(f32, 100)))),
            .joint = @exp(@min(try load.scalar("head.joint_logit_scale"), @log(@as(f32, 100)))),
            .gate = 1 / (1 + @exp(-try load.scalar("head.residual_gate"))),
            .score_bias = try load.scalar("head.residual_scorer.3.bias"),
        };
        for (evidence, 0..) |*layer, i| {
            var storage: [192]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&storage, "head.evidence_layers.{d}", .{i});
            const names = Named{ .load = load, .prefix = prefix };
            layer.* = .{ .norm = try names.norm("query_norm"), .ffnorm = try names.norm("feedforward_norm"), .attention = try names.attention("attention"), .ff1 = try names.linear("feedforward.0", cfg.head_ff, 1024), .ff2 = try names.linear("feedforward.3", 1024, cfg.head_ff) };
        }
        for (decoder, 0..) |*layer, i| {
            var storage: [192]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&storage, "head.layers.{d}", .{i});
            const names = Named{ .load = load, .prefix = prefix };
            layer.* = .{ .norm1 = try names.norm("norm1"), .norm2 = try names.norm("norm2"), .norm3 = try names.norm("norm3"), .self_attention = try names.attention("self_attn"), .cross = try names.attention("multihead_attn"), .ff1 = try names.linear("linear1", cfg.head_ff, 1024), .ff2 = try names.linear("linear2", 1024, cfg.head_ff) };
        }
        // The last scalar bias is applied by the fused logit kernel.
        out.scorer2.bias = null;
        return out;
    }
    pub fn deinit(self: *State) void {
        if (self.scratch) |*scratch| scratch.deinit();
        a.free(self.evidence);
        a.free(self.decoder);
    }
    pub fn run(self: *State, engine: *b.Engine, allocator: std.mem.Allocator, record: *const records.Record, embeddings: []const f32, cancel: b.Cancel, context: ?*anyopaque) !types.Result {
        const cfg = engine.config;
        const Q: u32 = @intCast(record.questions.len);
        var count: usize = 0;
        for (record.questions) |question| count += question.options.len;
        const C: u32 = @intCast(count);
        const L: u32 = @intCast(record.embedding_ids.len);
        const T: u32 = @intCast(record.ids.len);
        if (Q == 0 or C == 0 or embeddings.len != @as(usize, L) * cfg.H) return error.InvalidHeadInput;
        if (self.scratch == null or self.scratch.?.q_capacity < Q or self.scratch.?.c_capacity < C or self.scratch.?.lex_capacity < L) {
            if (self.scratch) |*scratch| scratch.deinit();
            self.scratch = null;
            self.scratch = try Scratch.init(engine.gpu, cfg, Q, C, L);
        }
        const s = &self.scratch.?;
        @memcpy(b.contents(f32, s.lexical_input, embeddings.len), embeddings);
        const qs = b.contents([2]u32, s.question_spans, Q);
        const os = b.contents([2]u32, s.option_spans, C);
        const ls = b.contents([2]u32, s.lexical_spans, C);
        const map = b.contents(u32, s.option_map, C);
        const metadata = b.contents(Question, s.questions, Q);
        var offset: usize = 0;
        for (record.questions, 0..) |question, i| {
            qs[i] = .{ @intCast(question.span.start), @intCast(question.span.end) };
            metadata[i] = .{ .offset = @intCast(offset), .count = @intCast(question.options.len), .kind = @backingInt(question.kind) };
            for (question.options) |option| {
                os[offset] = .{ @intCast(option.span.start), @intCast(option.span.end) };
                ls[offset] = .{ @intCast(option.lexical_span.start), @intCast(option.lexical_span.end) };
                map[offset] = @intCast(i);
                offset += 1;
            }
        }
        if (cancel) |check| if (check(context)) return error.Cancelled;
        const prepared = &engine.head_gpu.?.scratch.?;
        const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(engine.gpu, command, true);
        };
        const e = b.Encoder{ .gpu = engine.gpu, .command = command };
        try pool(e, prepared.normalized_hidden, s.question_spans, s.question_vectors, Q, cfg.H);
        try pool(e, prepared.normalized_hidden, s.option_spans, s.contexts, C, cfg.H);
        try pool(e, s.lexical_input, s.lexical_spans, s.lexical_vectors, C, cfg.H);
        try project(e, self.context, s.contexts, s.routed, C);
        try project(e, self.lexical, s.lexical_vectors, s.tmp, C);
        try project(e, self.option_question, s.question_vectors, s.option_question, Q);
        try e.dispatch("hf_options", Rows{ .rows = C, .width = 1024 }, &.{ s.routed.bind(1), s.tmp.bind(2), s.option_question.bind(3), s.option_map.bind(4) }, .{ ceil(C * 1024, 256), 1, 1 }, 256);
        for (self.evidence, 0..) |layer, i| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            try normalize(e, layer.norm, s.routed, s.normal, C);
            try project(e, layer.attention.query(), s.normal, s.query, C);
            try post(e, layer.attention.query(), s.query, s.query, C, .bias);
            try attend(e, s.query, 1024, prepared.kv[i], prepared.kv[i].at(1024 * 4), 2048, T, layer.attention.valueBias(), s.attended, C, true);
            try project(e, layer.attention.out, s.attended, s.tmp, C);
            try post(e, layer.attention.out, s.tmp, s.routed, C, .residual);
            try normalize(e, layer.ffnorm, s.routed, s.normal, C);
            try project(e, layer.ff1, s.normal, s.ff, C);
            try post(e, layer.ff1, s.ff, s.ff, C, .gelu);
            try project(e, layer.ff2, s.ff, s.tmp, C);
            try post(e, layer.ff2, s.tmp, s.routed, C, .residual);
        }
        try project(e, self.question, s.question_vectors, s.base, Q);
        const global_hidden = prepared.normalized_hidden.at(@as(usize, T - 1) * cfg.H * 4);
        try project(e, self.global, global_hidden, s.global, 1);
        try e.dispatch("hf_summary", Rows{ .rows = Q, .width = 1024 }, &.{ s.base.bind(1), s.routed.bind(2), s.questions.bind(3), s.summary.bind(4) }, .{ Q, 1, 1 }, 256);
        try normalize(e, self.summary_norm, s.summary, s.normal, Q);
        try e.dispatch("hf_fields", Rows{ .rows = Q, .width = 1024 }, &.{ s.base.bind(1), s.normal.bind(2), s.global.bind(3), self.type_embedding.bind(4), s.questions.bind(5), s.fields.bind(6) }, .{ ceil(Q * 1024, 256), 1, 1 }, 256);
        for (self.decoder, 0..) |layer, i| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            try normalize(e, layer.norm1, s.fields, s.normal, Q);
            try project(e, layer.self_attention.qkv, s.normal, s.query, Q);
            try post(e, layer.self_attention.qkv, s.query, s.query, Q, .bias);
            try attend(e, s.query, 3072, s.query.at(1024 * 4), s.query.at(2048 * 4), 3072, Q, s.zero, s.attended, Q, false);
            try project(e, layer.self_attention.out, s.attended, s.tmp, Q);
            try post(e, layer.self_attention.out, s.tmp, s.fields, Q, .residual);
            try normalize(e, layer.norm2, s.fields, s.normal, Q);
            try project(e, layer.cross.query(), s.normal, s.query, Q);
            try post(e, layer.cross.query(), s.query, s.query, Q, .bias);
            const kv = prepared.kv[self.evidence.len + i];
            try attend(e, s.query, 1024, kv, kv.at(1024 * 4), 2048, T, layer.cross.valueBias(), s.attended, Q, true);
            try project(e, layer.cross.out, s.attended, s.tmp, Q);
            try post(e, layer.cross.out, s.tmp, s.fields, Q, .residual);
            try normalize(e, layer.norm3, s.fields, s.normal, Q);
            try project(e, layer.ff1, s.normal, s.ff, Q);
            try post(e, layer.ff1, s.ff, s.ff, Q, .gelu);
            try project(e, layer.ff2, s.ff, s.tmp, Q);
            try post(e, layer.ff2, s.tmp, s.fields, Q, .residual);
        }
        try normalize(e, self.field_norm, s.fields, s.normal, Q);
        try normalize(e, self.option_norm, s.routed, s.option_normal, C);
        try e.dispatch("hf_features", Rows{ .rows = C, .width = 1024 }, &.{ s.normal.bind(1), s.option_normal.bind(2), s.option_map.bind(3), s.features.bind(4) }, .{ ceil(C * 1024, 256), 1, 1 }, 256);
        try project(e, self.scorer1, s.features, s.scorer, C);
        try post(e, self.scorer1, s.scorer, s.scorer, C, .gelu);
        try project(e, self.scorer2, s.scorer, s.residual, C);
        try e.dispatch("hf_logits", Score{ .rows = C, .hidden = cfg.H, .prior = self.prior, .joint = self.joint, .gate = self.gate, .bias = self.score_bias }, &.{ s.question_vectors.bind(1), global_hidden.bind(2), s.lexical_vectors.bind(3), s.normal.bind(4), s.option_normal.bind(5), s.option_map.bind(6), s.residual.bind(7), s.logits.bind(8) }, .{ C, 1, 1 }, 256);
        pending = false;
        if (!c.flint_gpu_command_end(engine.gpu, command, true)) return error.GpuExecution;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        const logits = b.contents(f32, s.logits, C);
        const answers = try allocator.alloc(types.Answer, Q);
        offset = 0;
        for (record.questions, answers) |question, *answer| {
            const probabilities = try allocator.alloc(f32, question.options.len);
            const row = logits[offset..][0..question.options.len];
            var maximum: f32 = -std.math.inf(f32);
            for (row) |v| maximum = @max(maximum, v);
            var total: f64 = 0;
            for (row, question.options) |v, option| {
                const p = @exp(v - maximum);
                probabilities[option.source_index] = p;
                total += p;
            }
            const inv: f32 = @floatCast(1 / total);
            for (probabilities) |*p| p.* *= inv;
            answer.* = .{ .probabilities = probabilities };
            offset += question.options.len;
        }
        return .{ .answers = answers, .input_tokens = T };
    }
};
const Loader = struct {
    engine: *b.Engine,
    fn scalar(self: Loader, name: []const u8) !f32 {
        return b.scalar(try self.engine.tensor(name), 0);
    }
    fn norm(self: Loader, prefix: []const u8) !Norm {
        var name: [256]u8 = undefined;
        const weight = try self.engine.vector(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}), 0, false);
        const bias = try self.engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false);
        return .{ .weight = weight, .bias = bias };
    }
    fn linear(self: Loader, prefix: []const u8, n: u32, k: u32, bias: bool) !Matrix {
        var name: [256]u8 = undefined;
        const t = try self.engine.tensor(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}));
        if (t.dtype != .bf16 or t.bytes.len != @as(usize, n) * k * 2) return error.InvalidHeadShape;
        const weight = try self.engine.buffer(t.bytes.len);
        @memcpy(b.contents(u8, weight, t.bytes.len), t.bytes);
        return .{ .weight = weight, .bias = if (bias) try self.engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false) else null, .n = n, .k = k };
    }
    fn attention(self: Loader, prefix: []const u8) !Attention {
        var name: [256]u8 = undefined;
        const t = try self.engine.tensor(try std.fmt.bufPrint(&name, "{s}.in_proj_weight", .{prefix}));
        const weight = try self.engine.buffer(t.bytes.len);
        @memcpy(b.contents(u8, weight, t.bytes.len), t.bytes);
        const bias = try self.engine.vector(try std.fmt.bufPrint(&name, "{s}.in_proj_bias", .{prefix}), 0, false);
        const out = try self.linear(try std.fmt.bufPrint(&name, "{s}.out_proj", .{prefix}), 1024, 1024, true);
        return .{ .qkv = .{ .weight = weight, .bias = bias, .n = 3072, .k = 1024 }, .out = out };
    }
};
const Named = struct {
    load: Loader,
    prefix: []const u8,
    fn norm(self: Named, suffix: []const u8) !Norm {
        var s: [256]u8 = undefined;
        return self.load.norm(try std.fmt.bufPrint(&s, "{s}.{s}", .{ self.prefix, suffix }));
    }
    fn linear(self: Named, suffix: []const u8, n: u32, k: u32) !Matrix {
        var s: [256]u8 = undefined;
        return self.load.linear(try std.fmt.bufPrint(&s, "{s}.{s}", .{ self.prefix, suffix }), n, k, true);
    }
    fn attention(self: Named, suffix: []const u8) !Attention {
        var s: [256]u8 = undefined;
        return self.load.attention(try std.fmt.bufPrint(&s, "{s}.{s}", .{ self.prefix, suffix }));
    }
};
fn ceil(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}
fn pool(e: b.Encoder, input: View, spans: View, out: View, rows: u32, width: u32) !void {
    try e.dispatch("hf_pool", Rows{ .rows = rows, .width = width }, &.{ input.bind(1), spans.bind(2), out.bind(3) }, .{ ceil(width, 256), rows, 1 }, 256);
}
fn normalize(e: b.Encoder, norm: Norm, input: View, out: View, rows: u32) !void {
    try e.dispatch("clef_head_norm", NormArgs{ .width = 1024 }, &.{ input.bind(1), norm.weight.bind(2), norm.bias.bind(3), out.bind(4) }, .{ rows, 1, 1 }, 256);
}
fn project(e: b.Encoder, m: Matrix, input: View, out: View, rows: u32) !void {
    const narrow = rows <= 8 or m.n == 1;
    if (narrow) try e.dispatch("hf_gemv", Gemv{ .rows = rows, .columns = m.n, .inner = m.k }, &.{ input.bind(1), m.weight.bind(2), out.bind(3) }, .{ ceil(m.n, 16), rows, 1 }, 128) else try e.dispatch("clef_head_gemm", Gemm{ .rows = rows, .columns = m.n, .inner = m.k }, &.{ input.bind(1), m.weight.bind(2), out.bind(3) }, .{ ceil(m.n, 128), ceil(rows, 16), 1 }, 128);
}
fn post(e: b.Encoder, m: Matrix, input: View, residual: View, rows: u32, mode: Post) !void {
    try e.dispatch("hf_post", Rows{ .rows = rows, .width = m.n, .mode = @backingInt(mode) }, &.{ input.bind(1), m.bias.?.bind(2), residual.bind(3) }, .{ ceil(rows * m.n, 256), 1, 1 }, 256);
}
fn attend(e: b.Encoder, q: View, q_stride: u32, k: View, v: View, kv_stride: u32, keys: u32, bias: View, out: View, queries: u32, value_bias: bool) !void {
    try e.dispatch("hf_attend", AttnArgs{ .queries = queries, .keys = keys, .q_stride = q_stride, .kv_stride = kv_stride, .value_bias = @intFromBool(value_bias) }, &.{ q.bind(1), k.bind(2), v.bind(3), bias.bind(4), out.bind(5) }, .{ queries, 16, 1 }, 32);
}
const Scratch = struct {
    gpu: *c.flint_gpu,
    q_capacity: u32,
    c_capacity: u32,
    lex_capacity: u32,
    buffers: std.ArrayList(b.Buffer) = .empty,
    lexical_input: View = undefined,
    question_spans: View = undefined,
    option_spans: View = undefined,
    lexical_spans: View = undefined,
    option_map: View = undefined,
    questions: View = undefined,
    question_vectors: View = undefined,
    contexts: View = undefined,
    lexical_vectors: View = undefined,
    routed: View = undefined,
    tmp: View = undefined,
    option_question: View = undefined,
    normal: View = undefined,
    query: View = undefined,
    attended: View = undefined,
    ff: View = undefined,
    base: View = undefined,
    global: View = undefined,
    summary: View = undefined,
    fields: View = undefined,
    option_normal: View = undefined,
    features: View = undefined,
    scorer: View = undefined,
    residual: View = undefined,
    logits: View = undefined,
    zero: View = undefined,
    fn alloc(self: *Scratch, size: usize) !View {
        const handle = c.flint_gpu_buffer(self.gpu, size) orelse return error.GpuAllocation;
        self.buffers.append(a, handle) catch |err| {
            c.flint_gpu_buffer_free(handle);
            return err;
        };
        return .{ .buffer = handle };
    }
    fn init(gpu: *c.flint_gpu, cfg: b.Config, Q: u32, C: u32, L: u32) !Scratch {
        var s = Scratch{ .gpu = gpu, .q_capacity = Q, .c_capacity = C, .lex_capacity = L };
        errdefer s.deinit();
        const q: usize = Q;
        const count: usize = C;
        const rows = @max(q, count);
        const H = cfg.H;
        s.lexical_input = try s.alloc(@as(usize, L) * H * 4);
        s.question_spans = try s.alloc(q * 8);
        s.option_spans = try s.alloc(count * 8);
        s.lexical_spans = try s.alloc(count * 8);
        s.option_map = try s.alloc(count * 4);
        s.questions = try s.alloc(q * @sizeOf(Question));
        s.question_vectors = try s.alloc(q * H * 4);
        s.contexts = try s.alloc(count * H * 4);
        s.lexical_vectors = try s.alloc(count * H * 4);
        s.routed = try s.alloc(count * 1024 * 4);
        s.tmp = try s.alloc(rows * 1024 * 4);
        s.option_question = try s.alloc(q * 1024 * 4);
        s.normal = try s.alloc(rows * 1024 * 4);
        s.query = try s.alloc(rows * 3072 * 4);
        s.attended = try s.alloc(rows * 1024 * 4);
        s.ff = try s.alloc(rows * cfg.head_ff * 4);
        s.base = try s.alloc(q * 1024 * 4);
        s.global = try s.alloc(1024 * 4);
        s.summary = try s.alloc(q * 1024 * 4);
        s.fields = try s.alloc(q * 1024 * 4);
        s.option_normal = try s.alloc(count * 1024 * 4);
        s.features = try s.alloc(count * 4096 * 4);
        s.scorer = try s.alloc(count * 1024 * 4);
        s.residual = try s.alloc(count * 4);
        s.logits = try s.alloc(count * 4);
        s.zero = try s.alloc(4096);
        @memset(b.contents(u8, s.zero, 4096), 0);
        return s;
    }
    fn deinit(self: *Scratch) void {
        for (self.buffers.items) |handle| c.flint_gpu_buffer_free(handle);
        self.buffers.deinit(a);
    }
};
