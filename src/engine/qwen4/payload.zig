//! Canonical Qwen state, preserving the existing DSV4v2/QW8v2 wire layout.
//! Common bounded I/O leaves handle files/tensor transfers; this owner decides
//! model spans, metadata and when a restored frontier becomes reusable.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");
const resources = @import("resources.zig");
const Config = dimensions.Config;
const Graph = c.ds4_qwen4_gpu_graph;
const Frontier = c.ds4_session_frontier;
const Tensor = c.ds4_gpu_tensor;
pub const tag: u32 = 0x51573802;
const chunk = 8 * 1024 * 1024;
const Header = [c.DS4_SESSION_PAYLOAD_U32_FIELDS]u32;

fn fail(errors: []u8, message: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return 1;
}

const Span = struct { tensor: ?*Tensor, bytes: u64 };
const LayerSpans = struct { items: [4]Span, count: usize };

fn cacheBytes(config: *const Config, rows: u32) [3]u64 {
    const shape = &config.shape;
    return .{
        @as(u64, rows) * shape.n_head_kv * shape.n_head_dim * 2,
        @as(u64, rows) * shape.n_indexer_head_dim * @sizeOf(f32),
        @as(u64, rows / 4) * shape.n_indexer_head_dim * 2,
    };
}

fn layerSpans(graph: *Graph, layer: u32, rows: u32, mtp_rows: u32) LayerSpans {
    if (dimensions.isLinear(&graph.config, layer)) return .{
        .items = .{
            .{ .tensor = graph.layer_lin_state[layer], .bytes = dimensions.stateFloats(&graph.config.shape) * @sizeOf(f32) },
            .{ .tensor = graph.layer_lin_hist[layer], .bytes = dimensions.historyFloats(&graph.config.shape) * @sizeOf(f32) },
            undefined,
            undefined,
        },
        .count = 2,
    };
    const bytes = cacheBytes(&graph.config, if (dimensions.isNextn(&graph.config, layer)) mtp_rows else rows);
    return .{
        .items = .{
            .{ .tensor = graph.layer_k_cache[layer], .bytes = bytes[0] },
            .{ .tensor = graph.layer_v_cache[layer], .bytes = bytes[0] },
            .{ .tensor = graph.layer_ik_cache[layer], .bytes = bytes[1] },
            .{ .tensor = graph.layer_block_key[layer], .bytes = bytes[2] },
        },
        .count = 4,
    };
}

pub fn tensorBytes(config: *const Config, rows: u32, mtp_rows: u32) u64 {
    const shape = &config.shape;
    var bytes: u64 = @sizeOf(u32); // Predictor's valid row count precedes tensors.
    for (0..shape.n_layer) |layer| {
        if (dimensions.isLinear(config, @intCast(layer))) {
            bytes += (dimensions.stateFloats(shape) + dimensions.historyFloats(shape)) * @sizeOf(f32);
        } else {
            const counts = cacheBytes(config, if (dimensions.isNextn(config, @intCast(layer))) mtp_rows else rows);
            bytes += 2 * counts[0] + counts[1] + counts[2];
        }
    }
    return bytes + dimensions.pleHistoryFloats(shape) * @sizeOf(f32) +
        c.DS4_MAX_PLE_NGRAM * @sizeOf(u32) + @sizeOf(u32) + @as(u64, rows) * 16;
}

const Stream = struct {
    file: *c.FILE,
    errors: []u8,

    fn errorPointer(self: Stream) [*c]u8 {
        return if (self.errors.len == 0) null else self.errors.ptr;
    }
    fn writeU32(self: Stream, value: u32) bool {
        return c.ds4_native_payload_write_u32(self.file, value, self.errorPointer(), self.errors.len) == 0;
    }
    fn readU32(self: Stream, value: *u32, remaining: *u64) bool {
        return c.ds4_native_payload_read_u32(self.file, value, remaining, self.errorPointer(), self.errors.len) == 0;
    }
    fn writeSpan(self: Stream, span: Span, buffer: []u8) bool {
        return c.ds4_native_payload_write_tensor_span(self.file, span.tensor, 0, span.bytes, buffer.ptr, buffer.len, self.errorPointer(), self.errors.len) == 0;
    }
    fn readSpan(self: Stream, span: Span, buffer: []u8, remaining: *u64) bool {
        return c.ds4_native_payload_read_tensor_span(self.file, span.tensor, 0, span.bytes, buffer.ptr, buffer.len, remaining, self.errorPointer(), self.errors.len) == 0;
    }
};

pub fn save(graph: *Graph, state: *const Frontier, ready: bool, rewound: bool, file: *c.FILE, errors: []u8) c_int {
    if (!ready or rewound or graph.pos != state.checkpoint.len) return fail(errors, "Qwen3.8 snapshot requires sync or eval after rewind");
    if (c.ds4_gpu_synchronize() == 0) return fail(errors, "failed to synchronize accelerator before Qwen3.8 snapshot");
    const shape = &graph.config.shape;
    const rows: u32 = @intCast(state.checkpoint.len);
    const header = Header{
        c.DS4_SESSION_PAYLOAD_MAGIC, c.DS4_SESSION_PAYLOAD_VERSION,
        @intCast(state.ctx_size),    state.prefill_cap,
        graph.ctx_cap,               graph.ctx_cap,
        shape.n_embd * shape.n_hc,   rows,
        shape.n_layer,               shape.n_head_dim,
        shape.n_indexer_head_dim,    shape.n_vocab,
        tag,
    };
    const io = Stream{ .file = file, .errors = errors };
    for (header) |value| if (!io.writeU32(value)) return 1;
    for (state.checkpoint.v[0..rows]) |token| if (!io.writeU32(@intCast(token))) return 1;
    if (c.ds4_native_payload_write_bytes(file, state.logits, @as(u64, shape.n_vocab) * @sizeOf(f32), io.errorPointer(), errors.len) != 0) return 1;
    const buffer = std.heap.c_allocator.alloc(u8, chunk) catch c.ds4_engine_model_die("out of memory");
    defer std.heap.c_allocator.free(buffer);
    const mtp_rows = @min(graph.mtp_pos, rows);
    if (!io.writeU32(mtp_rows)) return 1;
    for (0..shape.n_layer) |layer| {
        const spans = layerSpans(graph, @intCast(layer), rows, mtp_rows);
        for (spans.items[0..spans.count]) |span| if (!io.writeSpan(span, buffer)) return 1;
    }
    if (!io.writeSpan(.{ .tensor = graph.ple_hist, .bytes = dimensions.pleHistoryFloats(shape) * @sizeOf(f32) }, buffer)) return 1;
    for (graph.ple_prev) |token| if (!io.writeU32(@intCast(token))) return 1;
    if (!io.writeU32(@bitCast(graph.mrope_delta))) return 1;
    return if (io.writeSpan(.{ .tensor = graph.pos3, .bytes = @as(u64, rows) * 16 }, buffer)) 0 else 1;
}

pub fn load(graph: *Graph, state: *Frontier, ready: bool, file: *c.FILE, header: *const Header, remaining: *u64, mutated: *bool, errors: []u8) c_int {
    mutated.* = false;
    if (!ready) return fail(errors, "Qwen3.8 graph is not ready for restore");
    const shape = &graph.config.shape;
    const rows = header[7];
    if (header[12] != tag or header[8] != shape.n_layer or header[9] != shape.n_head_dim or
        header[10] != shape.n_indexer_head_dim or header[11] != shape.n_vocab or header[6] != shape.n_embd * shape.n_hc)
        return fail(errors, "KV checkpoint was written by a different model family or shape");
    if (rows > graph.ctx_cap or rows > state.ctx_size) return fail(errors, "KV checkpoint is longer than this session's context");
    const io = Stream{ .file = file, .errors = errors };
    var checkpoint = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&checkpoint);
    for (0..rows) |_| {
        var token: u32 = undefined;
        if (!io.readU32(&token, remaining)) return 1;
        if (token >= shape.n_vocab) return fail(errors, "KV checkpoint token id is outside the vocabulary");
        c.ds4_tokens_push(&checkpoint, @intCast(token));
    }
    // Before the first destination write, invalidate every transcript-dependent
    // snapshot. The outer session clears its pending draft counters via mutated.
    mutated.* = true;
    state.checkpoint_valid = false;
    state.mtp_draft_valid = false;
    graph.snap_valid = false;
    graph.snap2_valid = false;
    graph.snap0_valid = false;
    if (c.ds4_native_payload_read_bytes(file, state.logits, @as(u64, shape.n_vocab) * @sizeOf(f32), remaining, io.errorPointer(), errors.len) != 0) return 1;
    if (c.ds4_gpu_synchronize() == 0) return fail(errors, "failed to synchronize accelerator before Qwen3.8 restore");
    // Preserve the established distinction: failures while loading the tensor
    // body reset recurrence and token length; earlier parse failures only
    // invalidate the now-partially-written frontier.
    var complete = false;
    defer if (!complete) {
        resources.reset(graph);
        state.checkpoint.len = 0;
    };
    var mtp_rows: u32 = undefined;
    if (!io.readU32(&mtp_rows, remaining)) return 1;
    if (mtp_rows > rows) return fail(errors, "KV checkpoint MTP rows exceed the token count");
    const buffer = std.heap.c_allocator.alloc(u8, chunk) catch c.ds4_engine_model_die("out of memory");
    defer std.heap.c_allocator.free(buffer);
    for (0..shape.n_layer) |layer| {
        const spans = layerSpans(graph, @intCast(layer), rows, mtp_rows);
        for (spans.items[0..spans.count]) |span| if (!io.readSpan(span, buffer, remaining)) return 1;
    }
    if (!io.readSpan(.{ .tensor = graph.ple_hist, .bytes = dimensions.pleHistoryFloats(shape) * @sizeOf(f32) }, buffer, remaining)) return 1;
    for (&graph.ple_prev) |*previous| {
        var token: u32 = undefined;
        if (!io.readU32(&token, remaining)) return 1;
        if (token >= shape.n_vocab) return fail(errors, "KV checkpoint n-gram history is outside the vocabulary");
        previous.* = @intCast(token);
    }
    var delta: u32 = undefined;
    if (!io.readU32(&delta, remaining)) return 1;
    graph.mrope_delta = @bitCast(delta);
    if (!io.readSpan(.{ .tensor = graph.pos3, .bytes = @as(u64, rows) * 16 }, buffer, remaining)) return 1;
    graph.pos = rows;
    graph.mtp_pos = mtp_rows;
    c.ds4_tokens_free(&state.checkpoint);
    state.checkpoint = checkpoint;
    checkpoint = std.mem.zeroes(c.ds4_tokens);
    state.checkpoint_valid = true;
    complete = true;
    return 0;
}
