//! Native FP32 joint schema head, adapted from Cloudflare's Apache-2.0
//! joint_schema_model.py and Cleffa's MIT CPU formulation. Matrix products use
//! Accelerate; weights and model policy are owned here, independently of serving.
const std = @import("std");
const backend = @import("backend.zig");
const records = @import("record.zig");
const types = @import("types.zig");
const Allocator = std.mem.Allocator;

extern "c" fn cblas_sgemm(order: c_int, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, x: [*]const f32, ldx: c_int, y: [*]const f32, ldy: c_int, beta: f32, out: [*]f32, ldout: c_int) void;
extern "c" fn erff(x: f32) f32;

const Linear = struct {
    weight: []const f32,
    bias: []const f32 = &.{},
    input: usize,
    output: usize,

    fn run(self: Linear, x: []const f32, rows: usize, y: []f32) void {
        gemm(false, true, rows, self.output, self.input, 1, x, self.input, self.weight, self.input, y, self.output);
        if (self.bias.len > 0) for (0..rows) |r| {
            for (y[r * self.output ..][0..self.output], self.bias) |*v, b| v.* += b;
        };
    }
};
const Norm = struct {
    weight: []const f32,
    bias: []const f32,
    fn run(self: Norm, x: []const f32, rows: usize, y: []f32) void {
        const width = self.weight.len;
        for (0..rows) |r| {
            const row = x[r * width ..][0..width];
            var mean: f64 = 0;
            for (row) |v| mean += v;
            mean /= @floatFromInt(width);
            var variance: f64 = 0;
            for (row) |v| {
                const d = v - mean;
                variance += d * d;
            }
            variance /= @floatFromInt(width);
            const inv: f32 = @floatCast(1 / @sqrt(variance + 1e-5));
            for (row, self.weight, self.bias, y[r * width ..][0..width]) |v, w, b, *out| out.* = @as(f32, @floatCast(v - mean)) * inv * w + b;
        }
    }
};
const Attention = struct {
    qkv: Linear,
    out: Linear,
    heads: usize,
    fn q(self: Attention) Linear {
        const w = self.qkv.input;
        return .{ .weight = self.qkv.weight[0 .. w * w], .bias = self.qkv.bias[0..w], .input = w, .output = w };
    }
    fn memory(self: Attention, x: []const f32, rows: usize, kv: []f32) void {
        const w = self.qkv.input;
        // Key bias is constant over keys and cancels in softmax. Value bias is
        // added after attention, following Cleffa's equivalent formulation.
        const projection = Linear{ .weight = self.qkv.weight[w * w ..], .input = w, .output = 2 * w };
        projection.run(x, rows, kv);
    }
};
const Evidence = struct { query_norm: Norm, memory_norm: Norm, ff_norm: Norm, attention: Attention, ff1: Linear, ff2: Linear };
const Decoder = struct { norm1: Norm, norm2: Norm, norm3: Norm, self_attention: Attention, cross: Attention, ff1: Linear, ff2: Linear };

pub const Head = struct {
    arena: std.heap.ArenaAllocator,
    hidden: usize,
    width: usize,
    feedforward: usize,
    hidden_norm: Norm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    option_summary_norm: Norm,
    field_norm: Norm,
    option_norm: Norm,
    type_embedding: []const f32,
    evidence: []Evidence,
    decoder: []Decoder,
    scorer0: Linear,
    scorer3: Linear,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,

    pub fn open(allocator: Allocator, engine: *const backend.Engine) !Head {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        var loader = Loader{ .a = arena.allocator(), .engine = engine };
        const cfg = engine.config;
        const h: usize = cfg.H;
        const w: usize = cfg.W;
        const ff: usize = cfg.head_ff;
        var self: Head = .{
            .arena = arena,
            .hidden = h,
            .width = w,
            .feedforward = ff,
            .hidden_norm = try loader.norm("head.hidden_norm", h),
            .memory_projection = try loader.linear("head.memory_projection", w, h, false),
            .question_projection = try loader.linear("head.question_projection", w, h, false),
            .option_question_projection = try loader.linear("head.option_question_projection", w, h, false),
            .global_projection = try loader.linear("head.global_projection", w, h, false),
            .option_context_projection = try loader.linear("head.option_context_projection", w, h, false),
            .option_lexical_projection = try loader.linear("head.option_lexical_projection", w, h, false),
            .option_summary_norm = try loader.norm("head.option_summary_norm", w),
            .field_norm = try loader.norm("head.field_norm", w),
            .option_norm = try loader.norm("head.option_norm", w),
            .type_embedding = try loader.values("head.type_embedding.weight", 3 * w),
            .evidence = try loader.a.alloc(Evidence, cfg.routing_layers),
            .decoder = try loader.a.alloc(Decoder, cfg.head_layers),
            .scorer0 = try loader.linear("head.residual_scorer.0", w, 4 * w, true),
            .scorer3 = try loader.linear("head.residual_scorer.3", 1, w, true),
            .prior_logit_scale = (try loader.values("head.prior_logit_scale", 1))[0],
            .joint_logit_scale = (try loader.values("head.joint_logit_scale", 1))[0],
            .residual_gate = (try loader.values("head.residual_gate", 1))[0],
        };
        for (self.evidence, 0..) |*layer, i| {
            const prefix = try std.fmt.allocPrint(loader.a, "head.evidence_layers.{d}", .{i});
            layer.* = .{
                .query_norm = try loader.norm(try loader.name(prefix, "query_norm"), w),
                .memory_norm = try loader.norm(try loader.name(prefix, "memory_norm"), w),
                .ff_norm = try loader.norm(try loader.name(prefix, "feedforward_norm"), w),
                .attention = try loader.attention(try loader.name(prefix, "attention"), w, cfg.head_heads),
                .ff1 = try loader.linear(try loader.name(prefix, "feedforward.0"), ff, w, true),
                .ff2 = try loader.linear(try loader.name(prefix, "feedforward.3"), w, ff, true),
            };
        }
        for (self.decoder, 0..) |*layer, i| {
            const prefix = try std.fmt.allocPrint(loader.a, "head.layers.{d}", .{i});
            layer.* = .{
                .norm1 = try loader.norm(try loader.name(prefix, "norm1"), w),
                .norm2 = try loader.norm(try loader.name(prefix, "norm2"), w),
                .norm3 = try loader.norm(try loader.name(prefix, "norm3"), w),
                .self_attention = try loader.attention(try loader.name(prefix, "self_attn"), w, cfg.head_heads),
                .cross = try loader.attention(try loader.name(prefix, "multihead_attn"), w, cfg.head_heads),
                .ff1 = try loader.linear(try loader.name(prefix, "linear1"), ff, w, true),
                .ff2 = try loader.linear(try loader.name(prefix, "linear2"), w, ff, true),
            };
        }
        // ArenaAllocator state is copied only after its final allocation.
        self.arena = arena;
        return self;
    }

    pub fn deinit(self: *Head) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// hidden is [input_tokens,H]; embeddings is [record.embedding_ids.len,H].
    /// Results allocate from the caller's request arena and preserve option order.
    pub fn run(self: *const Head, allocator: Allocator, record: *const records.Record, hidden: []const f32, embeddings: []const f32) !types.Result {
        return self.runImpl(allocator, record, hidden, embeddings, null);
    }

    /// GPU-projected memory avoids repeating the wide CPU GEMMs. The remaining
    /// learned head and its result semantics are shared with the CPU oracle.
    pub fn runPrepared(self: *const Head, allocator: Allocator, record: *const records.Record, prepared: types.HeadFeatures, embeddings: []const f32) !types.Result {
        return self.runImpl(allocator, record, prepared.normalized_hidden, embeddings, prepared);
    }

    fn runImpl(self: *const Head, allocator: Allocator, record: *const records.Record, hidden: []const f32, embeddings: []const f32, prepared: ?types.HeadFeatures) !types.Result {
        const h = self.hidden;
        const w = self.width;
        const t = record.ids.len;
        const nq = record.questions.len;
        if (hidden.len != t * h or embeddings.len != record.embedding_ids.len * h) return error.InvalidHeadInput;
        var count: usize = 0;
        for (record.questions) |question| count += question.options.len;
        var scratch = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer scratch.deinit();
        const a = scratch.allocator();
        const nh: []const f32 = if (prepared) |features| features.normalized_hidden else normalized: {
            const rows = try a.alloc(f32, t * h);
            self.hidden_norm.run(hidden, t, rows);
            break :normalized rows;
        };
        const memory: []const f32 = if (prepared) |features| features.memory else projected: {
            const rows = try a.alloc(f32, t * w);
            self.memory_projection.run(nh, t, rows);
            break :projected rows;
        };
        if (prepared) |features| {
            if (memory.len != t * w or features.kv.len != self.evidence.len + self.decoder.len) return error.InvalidHeadInput;
            for (features.kv) |rows| if (rows.len != t * 2 * w) return error.InvalidHeadInput;
        }
        const memory_norm = try a.alloc(f32, if (prepared == null) t * w else 0);
        const kv = try a.alloc(f32, if (prepared == null) t * 2 * w else 0);
        const qv = try zeros(a, nq * h);
        const ctx = try zeros(a, count * h);
        const lex = try zeros(a, count * h);
        var oi: usize = 0;
        for (record.questions, 0..) |question, qi| {
            meanRows(nh, h, question.span, qv[qi * h ..][0..h]);
            for (question.options) |option| {
                meanRows(nh, h, option.span, ctx[oi * h ..][0..h]);
                meanRows(embeddings, h, option.lexical_span, lex[oi * h ..][0..h]);
                oi += 1;
            }
        }
        const global = nh[(t - 1) * h ..][0..h];
        const routed = try a.alloc(f32, count * w);
        const tmp = try a.alloc(f32, @max(count, nq) * w);
        self.option_context_projection.run(ctx, count, routed);
        self.option_lexical_projection.run(lex, count, tmp);
        add(routed, tmp[0 .. count * w]);
        const qq = try a.alloc(f32, nq * w);
        self.option_question_projection.run(qv, nq, qq);
        oi = 0;
        for (record.questions, 0..) |question, qi| for (question.options) |_| {
            add(routed[oi * w ..][0..w], qq[qi * w ..][0..w]);
            oi += 1;
        };
        const normal = try a.alloc(f32, @max(count, nq) * w);
        const query = try a.alloc(f32, @max(count, nq) * 3 * w);
        const attended = try a.alloc(f32, @max(count, nq) * w);
        const ff = try a.alloc(f32, @max(count, nq) * self.feedforward);
        for (self.evidence, 0..) |layer, layer_index| {
            layer.query_norm.run(routed, count, normal);
            layer.attention.q().run(normal, count, query);
            const memory_kv = if (prepared) |features| features.kv[layer_index] else projected: {
                layer.memory_norm.run(memory, t, memory_norm);
                layer.attention.memory(memory_norm, t, kv);
                break :projected kv;
            };
            try attend(a, query, w, count, memory_kv, memory_kv[w..], 2 * w, t, layer.attention.qkv.bias[2 * w ..], w, layer.attention.heads, attended);
            layer.attention.out.run(attended, count, tmp);
            add(routed, tmp[0 .. count * w]);
            layer.ff_norm.run(routed, count, normal);
            layer.ff1.run(normal, count, ff);
            geluRows(ff[0 .. count * self.feedforward]);
            layer.ff2.run(ff, count, tmp);
            add(routed, tmp[0 .. count * w]);
        }
        const base = try a.alloc(f32, nq * w);
        self.question_projection.run(qv, nq, base);
        const gp = try a.alloc(f32, w);
        self.global_projection.run(global, 1, gp);
        const summary = try zeros(a, nq * w);
        oi = 0;
        for (record.questions, 0..) |question, qi| {
            const probabilities = try a.alloc(f32, question.options.len);
            const field = base[qi * w ..][0..w];
            for (probabilities, 0..) |*p, j| p.* = @floatCast(dot(routed[(oi + j) * w ..][0..w], field) / @sqrt(@as(f64, @floatFromInt(w))));
            softmax(probabilities);
            for (probabilities, 0..) |p, j| for (summary[qi * w ..][0..w], routed[(oi + j) * w ..][0..w]) |*out, v| {
                out.* += p * v;
            };
            oi += question.options.len;
        }
        self.option_summary_norm.run(summary, nq, normal);
        const fields = try a.alloc(f32, nq * w);
        for (record.questions, 0..) |question, qi| {
            const te = self.type_embedding[@as(usize, @backingInt(question.kind)) * w ..][0..w];
            for (fields[qi * w ..][0..w], base[qi * w ..][0..w], normal[qi * w ..][0..w], gp, te) |*out, b, n, g, type_value| out.* = b + n + g + type_value;
        }
        for (self.decoder, 0..) |layer, layer_index| {
            layer.norm1.run(fields, nq, normal);
            layer.self_attention.qkv.run(normal, nq, query);
            try attend(a, query, 3 * w, nq, query[w..], query[2 * w ..], 3 * w, nq, &.{}, w, layer.self_attention.heads, attended);
            layer.self_attention.out.run(attended, nq, tmp);
            add(fields, tmp[0 .. nq * w]);
            layer.norm2.run(fields, nq, normal);
            layer.cross.q().run(normal, nq, query);
            const memory_kv = if (prepared) |features| features.kv[self.evidence.len + layer_index] else projected: {
                layer.cross.memory(memory, t, kv);
                break :projected kv;
            };
            try attend(a, query, w, nq, memory_kv, memory_kv[w..], 2 * w, t, layer.cross.qkv.bias[2 * w ..], w, layer.cross.heads, attended);
            layer.cross.out.run(attended, nq, tmp);
            add(fields, tmp[0 .. nq * w]);
            layer.norm3.run(fields, nq, normal);
            layer.ff1.run(normal, nq, ff);
            geluRows(ff[0 .. nq * self.feedforward]);
            layer.ff2.run(ff, nq, tmp);
            add(fields, tmp[0 .. nq * w]);
        }
        self.field_norm.run(fields, nq, normal);
        const option_norm = try a.alloc(f32, count * w);
        self.option_norm.run(routed, count, option_norm);
        const features = try a.alloc(f32, count * 4 * w);
        oi = 0;
        for (record.questions, 0..) |question, qi| for (question.options) |_| {
            const field = normal[qi * w ..][0..w];
            const option = option_norm[oi * w ..][0..w];
            const feature = features[oi * 4 * w ..][0 .. 4 * w];
            for (field, option, 0..) |f, o, i| {
                feature[i] = f;
                feature[w + i] = o;
                feature[2 * w + i] = f * o;
                feature[3 * w + i] = @abs(f - o);
            }
            oi += 1;
        };
        const scorer = try a.alloc(f32, count * w);
        const residuals = try a.alloc(f32, count);
        self.scorer0.run(features, count, scorer);
        geluRows(scorer);
        self.scorer3.run(scorer, count, residuals);
        const prior_scale = @exp(@min(self.prior_logit_scale, @log(@as(f32, 100))));
        const joint_scale = @exp(@min(self.joint_logit_scale, @log(@as(f32, 100))));
        const gate = 1 / (1 + @exp(-self.residual_gate));
        const answers = try allocator.alloc(types.Answer, nq);
        const anchor = try a.alloc(f32, h);
        const lexical_unit = try a.alloc(f32, h);
        oi = 0;
        for (record.questions, answers, 0..) |question, *answer, qi| {
            for (anchor, qv[qi * h ..][0..h], global) |*out, q, g| out.* = q + g;
            normalize(anchor, anchor);
            const field = normal[qi * w ..][0..w];
            const field_norm = @sqrt(dot(field, field));
            const logits = try a.alloc(f32, question.options.len);
            for (logits) |*logit| {
                normalize(lex[oi * h ..][0..h], lexical_unit);
                const option = option_norm[oi * w ..][0..w];
                const cosine = dot(field, option) / @max(field_norm * @sqrt(dot(option, option)), 1e-8);
                const prior = dot(lexical_unit, anchor);
                logit.* = @floatCast(prior_scale * prior + gate * (joint_scale * cosine + residuals[oi]));
                oi += 1;
            }
            softmax(logits);
            const probabilities = try allocator.alloc(f32, logits.len);
            for (logits, question.options) |p, option| probabilities[option.source_index] = p;
            answer.* = .{ .probabilities = probabilities };
        }
        return .{ .answers = answers, .input_tokens = t };
    }
};

const Loader = struct {
    a: Allocator,
    engine: *const backend.Engine,
    fn name(self: Loader, prefix: []const u8, suffix: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.a, "{s}.{s}", .{ prefix, suffix });
    }
    fn values(self: Loader, name_: []const u8, count: usize) ![]const f32 {
        const tensor = try self.engine.tensor(name_);
        var size: usize = 1;
        for (tensor.shape) |n| size *= n;
        if (size != count) {
            std.debug.print("clef head: wrong shape for {s}: {d} != {d}\n", .{ name_, size, count });
            return error.InvalidHeadShape;
        }
        const out = try self.a.alloc(f32, count);
        switch (tensor.dtype) {
            .bf16 => for (out, 0..) |*v, i| {
                const bits = std.mem.readInt(u16, tensor.bytes[i * 2 ..][0..2], .little);
                v.* = @bitCast(@as(u32, bits) << 16);
            },
            .f32 => @memcpy(std.mem.sliceAsBytes(out), tensor.bytes),
            else => return error.InvalidHeadDtype,
        }
        return out;
    }
    fn linear(self: Loader, prefix: []const u8, output: usize, input: usize, bias: bool) !Linear {
        return .{ .weight = try self.values(try self.name(prefix, "weight"), output * input), .bias = if (bias) try self.values(try self.name(prefix, "bias"), output) else &.{}, .input = input, .output = output };
    }
    fn norm(self: Loader, prefix: []const u8, width: usize) !Norm {
        return .{ .weight = try self.values(try self.name(prefix, "weight"), width), .bias = try self.values(try self.name(prefix, "bias"), width) };
    }
    fn attention(self: Loader, prefix: []const u8, width: usize, heads: usize) !Attention {
        if (width % heads != 0) return error.InvalidHeadShape;
        return .{ .qkv = .{ .weight = try self.values(try self.name(prefix, "in_proj_weight"), 3 * width * width), .bias = try self.values(try self.name(prefix, "in_proj_bias"), 3 * width), .input = width, .output = 3 * width }, .out = try self.linear(try self.name(prefix, "out_proj"), width, width, true), .heads = heads };
    }
};

fn gemm(ta: bool, tb: bool, m: usize, n: usize, k: usize, scale: f32, x: []const f32, ldx: usize, y: []const f32, ldy: usize, out: []f32, ldo: usize) void {
    cblas_sgemm(101, if (ta) 112 else 111, if (tb) 112 else 111, @intCast(m), @intCast(n), @intCast(k), scale, x.ptr, @intCast(ldx), y.ptr, @intCast(ldy), 0, out.ptr, @intCast(ldo));
}
fn attend(a: Allocator, q: []const f32, ldq: usize, nq: usize, k: []const f32, v: []const f32, ldk: usize, length: usize, bias_v: []const f32, width: usize, heads: usize, out: []f32) !void {
    const scores = try a.alloc(f32, nq * length);
    defer a.free(scores);
    const hd = width / heads;
    for (0..heads) |head| {
        const offset = head * hd;
        gemm(false, true, nq, length, hd, 1 / @sqrt(@as(f32, @floatFromInt(hd))), q[offset..], ldq, k[offset..], ldk, scores, length);
        for (0..nq) |r| softmax(scores[r * length ..][0..length]);
        gemm(false, false, nq, hd, length, 1, scores, length, v[offset..], ldk, out[offset..], width);
    }
    if (bias_v.len > 0) for (0..nq) |r| {
        add(out[r * width ..][0..width], bias_v);
    };
}
fn zeros(a: Allocator, n: usize) ![]f32 {
    const out = try a.alloc(f32, n);
    @memset(out, 0);
    return out;
}
fn add(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| d.* += s;
}
fn dot(x: []const f32, y: []const f32) f64 {
    var sum: f64 = 0;
    for (x, y) |a, b| sum += @as(f64, a) * b;
    return sum;
}
fn normalize(x: []const f32, out: []f32) void {
    const inv: f32 = @floatCast(1 / @max(@sqrt(dot(x, x)), 1e-12));
    for (x, out) |v, *o| o.* = v * inv;
}
fn meanRows(x: []const f32, width: usize, span: types.Span, out: []f32) void {
    @memset(out, 0);
    for (span.start..span.end) |r| add(out, x[r * width ..][0..width]);
    const inv = 1 / @as(f32, @floatFromInt(span.len()));
    for (out) |*v| v.* *= inv;
}
fn softmax(row: []f32) void {
    var maximum: f32 = -std.math.inf(f32);
    for (row) |v| maximum = @max(maximum, v);
    var total: f64 = 0;
    for (row) |*v| {
        v.* = @exp(v.* - maximum);
        total += v.*;
    }
    const inv: f32 = @floatCast(1 / total);
    for (row) |*v| v.* *= inv;
}
fn geluRows(row: []f32) void {
    for (row) |*v| v.* = 0.5 * v.* * (1 + erff(v.* * 0.70710678118654752));
}

test "stable decision probabilities and normalized means" {
    var p = [_]f32{ 1001, 1000, 999 };
    softmax(&p);
    try std.testing.expectApproxEqAbs(@as(f32, 1), p[0] + p[1] + p[2], 1e-6);
    try std.testing.expect(p[0] > p[1] and p[1] > p[2]);
    var out: [2]f32 = undefined;
    meanRows(&.{ 1, 2, 3, 4 }, 2, .{ .start = 0, .end = 2 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &out);
}
