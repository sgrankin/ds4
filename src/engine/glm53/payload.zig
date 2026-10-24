//! GLM's canonical DSV4 v2 payload. Recurrence and unfinished indexer pools are
//! FP32 state. KV/index storage is FP16 on Metal but retains FP32 wire bytes,
//! preserving existing SSD/session checkpoints across the ownership move.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Layout = @import("layout.zig");
const Graph = c.ds4_glm_gpu_graph;
const Frontier = c.ds4_session_frontier;
const Tensor = c.ds4_gpu_tensor;
const allocator = std.heap.c_allocator;
const io_chunk = 8 * 1024 * 1024;
const Failure = error{Io};
pub const Header = [c.DS4_SESSION_PAYLOAD_U32_FIELDS]u32;

const File = struct {
    stream: *c.FILE,
    error_text: []u8,

    fn fail(self: File, message: []const u8) Failure {
        if (self.error_text.len != 0) {
            const length = @min(message.len, self.error_text.len - 1);
            @memcpy(self.error_text[0..length], message[0..length]);
            self.error_text[length] = 0;
        }
        return error.Io;
    }
    fn check(result: c_int) Failure!void {
        if (result != 0) return error.Io;
    }
    fn writeU32(self: File, value: u32) Failure!void {
        try check(c.ds4_native_payload_write_u32(self.stream, value, self.error_text.ptr, self.error_text.len));
    }
    fn readU32(self: File, remaining: *u64) Failure!u32 {
        var value: u32 = 0;
        try check(c.ds4_native_payload_read_u32(self.stream, &value, remaining, self.error_text.ptr, self.error_text.len));
        return value;
    }
    fn write(self: File, data: []const u8) Failure!void {
        try check(c.ds4_native_payload_write_bytes(self.stream, data.ptr, data.len, self.error_text.ptr, self.error_text.len));
    }
    fn read(self: File, data: []u8, remaining: *u64) Failure!void {
        try check(c.ds4_native_payload_read_bytes(self.stream, data.ptr, data.len, remaining, self.error_text.ptr, self.error_text.len));
    }
};

const Span = struct {
    tensor: ?*Tensor,
    count: u64,
    is_f16: bool,
    allow_missing: bool = false,
};

fn spans(graph: *Graph, layer: u32, full_live: u32, compact_live: u32, storage: *[7]Span) []const Span {
    const shape = &graph.config.shape;
    if (Config.isKda(&graph.config, layer)) {
        storage[0] = .{ .tensor = graph.layer_kda_conv_state[layer], .count = Config.convStateBytes(&graph.config), .is_f16 = false };
        storage[1] = .{ .tensor = graph.layer_kda_recurrent_state[layer], .count = Config.recurrentStateBytes(&graph.config), .is_f16 = false };
        return storage[0..2];
    }
    storage[0] = .{ .tensor = graph.layer_key_cache[layer], .count = @as(u64, full_live) * graph.q_dim, .is_f16 = true, .allow_missing = true };
    storage[1] = .{ .tensor = graph.layer_value_cache[layer], .count = @as(u64, full_live) * graph.heads_dim, .is_f16 = true, .allow_missing = true };
    if (compact_live == 0) return storage[0..2];
    storage[2] = .{ .tensor = graph.layer_kv_lora_cache[layer], .count = @as(u64, compact_live) * shape.n_kv_lora, .is_f16 = true };
    storage[3] = .{ .tensor = graph.layer_k_rope_cache[layer], .count = @as(u64, compact_live) * shape.n_rot, .is_f16 = true };
    if (!Config.hasIndexer(&graph.config, layer)) return storage[0..4];
    const index_live = Config.indexLiveRows(&graph.config, layer, compact_live);
    storage[4] = .{ .tensor = graph.layer_indexer_key_cache[layer], .count = @as(u64, index_live) * shape.n_indexer_head_dim, .is_f16 = true };
    const tail_bytes = @as(u64, graph.config.index_pool_size) * shape.n_indexer_head_dim * @sizeOf(f32);
    storage[5] = .{ .tensor = graph.layer_indexer_tail_k[layer], .count = tail_bytes, .is_f16 = false };
    storage[6] = .{ .tensor = graph.layer_indexer_tail_gate[layer], .count = tail_bytes, .is_f16 = false };
    return storage;
}

pub fn bytes(graph: *const Graph, frontier: *const Frontier, dense_cache_len: u32) u64 {
    const checkpoint: u32 = @intCast(frontier.checkpoint.len);
    const full_live = Layout.fullLiveRows(graph, checkpoint, dense_cache_len);
    const compact_live = Layout.compactLiveRows(graph, checkpoint);
    return (@as(u64, c.DS4_SESSION_PAYLOAD_U32_FIELDS) + checkpoint + graph.config.shape.n_vocab + 2 * graph.normal_layers) * @sizeOf(u32) +
        Layout.payloadLiveBytes(graph, full_live, compact_live);
}

pub fn save(graph: *Graph, frontier: *const Frontier, dense_cache_len: u32, stream: *c.FILE, error_text: []u8) Failure!void {
    const file = File{ .stream = stream, .error_text = error_text };
    if (!frontier.checkpoint_valid) return file.fail("session has no valid checkpoint to save");
    if (c.ds4_gpu_synchronize() == 0) return file.fail("failed to synchronize accelerator before GLM snapshot");
    const shape = &graph.config.shape;
    const checkpoint: u32 = @intCast(frontier.checkpoint.len);
    const full_live = Layout.fullLiveRows(graph, checkpoint, dense_cache_len);
    const compact_live = Layout.compactLiveRows(graph, checkpoint);
    const header = Header{
        c.DS4_SESSION_PAYLOAD_MAGIC, c.DS4_SESSION_PAYLOAD_VERSION,
        @intCast(frontier.ctx_size), frontier.prefill_cap,
        graph.ctx_cap,               graph.ctx_cap,
        graph.compact_cache_cap,     checkpoint,
        graph.normal_layers,         shape.n_key_mla,
        shape.n_value_mla,           shape.n_vocab,
        full_live,
    };
    for (header) |value| try file.writeU32(value);
    for (frontier.checkpoint.v[0..checkpoint]) |token| try file.writeU32(@bitCast(token));
    try file.write(std.mem.sliceAsBytes(frontier.logits[0..shape.n_vocab]));
    // The established row-count arrays include KDA layers too. Those layers
    // serialize only recurrence below; changing this tape would break v2.
    for (0..graph.normal_layers) |_| try file.writeU32(compact_live);
    for (0..graph.normal_layers) |layer| try file.writeU32(Config.indexLiveRows(&graph.config, @intCast(layer), compact_live));

    const scratch = allocator.alignedAlloc(u8, .@"8", io_chunk) catch return file.fail("failed to allocate session I/O scratch");
    defer allocator.free(scratch);
    var storage: [7]Span = undefined;
    for (0..graph.normal_layers) |layer| {
        for (spans(graph, @intCast(layer), full_live, compact_live, &storage)) |span| {
            if (span.count == 0) continue;
            if (span.is_f16) {
                try File.check(c.ds4_native_payload_write_f16_as_f32(stream, span.tensor, 0, span.count, scratch.ptr, scratch.len, error_text.ptr, error_text.len));
            } else {
                try File.check(c.ds4_native_payload_write_tensor_span(stream, span.tensor, 0, span.count, scratch.ptr, scratch.len, error_text.ptr, error_text.len));
            }
        }
    }
}

fn validate(file: File, graph: *const Graph, frontier: *const Frontier, header: *const Header) Failure!u32 {
    const shape = &graph.config.shape;
    const checkpoint = header[7];
    if (header[2] > frontier.ctx_size or checkpoint >= frontier.ctx_size) return file.fail("KV checkpoint does not fit current context");
    if (header[8] != graph.normal_layers or header[9] != shape.n_key_mla or header[10] != shape.n_value_mla or header[11] != shape.n_vocab)
        return file.fail("KV checkpoint was written for a different GLM layout");
    const full_cap = header[4];
    if (full_cap == 0 or full_cap > graph.ctx_cap or header[5] != full_cap)
        return file.fail("KV checkpoint GLM full-cache layout does not match current runtime");
    if (header[12] > @min(checkpoint, full_cap) or header[12] > graph.ctx_cap)
        return file.fail("KV checkpoint GLM full-cache row count is invalid");
    const compact_cap = header[6];
    if (compact_cap > graph.ctx_size) return file.fail("KV checkpoint GLM compact cache is larger than current context");
    if (graph.compact_cache_cap != 0 and checkpoint != 0 and compact_cap == 0)
        return file.fail("KV checkpoint lacks the GLM compact cache required by this context");
    const compact_live = if (compact_cap != 0) @min(checkpoint, compact_cap) else 0;
    // The existing compact cache is fixed at allocation; this mirrors its
    // capacity check without silently allocating a larger device graph.
    if (compact_live > graph.compact_cache_cap)
        return file.fail("KV checkpoint GLM compact cache could not be allocated");
    return compact_live;
}

pub fn load(graph: *Graph, frontier: *Frontier, dense_cache_len: *u32, stream: *c.FILE, header: *const Header, payload_remaining: u64, error_text: []u8) Failure!void {
    const file = File{ .stream = stream, .error_text = error_text };
    const compact_live = try validate(file, graph, frontier, header);
    var remaining = payload_remaining;
    var checkpoint = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&checkpoint);
    for (0..header[7]) |_| c.ds4_tokens_push(&checkpoint, @bitCast(try file.readU32(&remaining)));
    // Header and token failures preserve the old frontier. From the first
    // live-logit write onward, a short/corrupt payload must invalidate it.
    frontier.checkpoint_valid = false;
    frontier.mtp_draft_valid = false;
    dense_cache_len.* = 0;
    try file.read(std.mem.sliceAsBytes(frontier.logits[0..graph.config.shape.n_vocab]), &remaining);
    for (0..graph.normal_layers) |_| {
        if (try file.readU32(&remaining) != compact_live)
            return file.fail("KV checkpoint GLM compact row count does not match token count");
    }
    for (0..graph.normal_layers) |layer| {
        if (try file.readU32(&remaining) != Config.indexLiveRows(&graph.config, @intCast(layer), compact_live))
            return file.fail("KV checkpoint GLM indexer row count does not match token count");
    }
    if (c.ds4_gpu_synchronize() == 0) return file.fail("failed to synchronize accelerator before GLM KV restore");

    const scratch = allocator.alignedAlloc(u8, .@"8", io_chunk) catch return file.fail("failed to allocate session I/O scratch");
    defer allocator.free(scratch);
    var storage: [7]Span = undefined;
    for (0..graph.normal_layers) |layer| {
        for (spans(graph, @intCast(layer), header[12], compact_live, &storage)) |span| {
            if (span.count == 0) continue;
            if (span.allow_missing and span.tensor == null) {
                // An absorbed graph can read an older expanded-KV checkpoint:
                // consume its redundant full keys/values without storing them.
                var skip = span.count * @sizeOf(f32);
                while (skip != 0) {
                    const size = @min(skip, scratch.len);
                    try file.read(scratch[0..@intCast(size)], &remaining);
                    skip -= size;
                }
            } else if (span.is_f16) {
                try File.check(c.ds4_native_payload_read_f32_as_f16(stream, span.tensor, 0, span.count, scratch.ptr, scratch.len, &remaining, error_text.ptr, error_text.len));
            } else {
                try File.check(c.ds4_native_payload_read_tensor_span(stream, span.tensor, 0, span.count, scratch.ptr, scratch.len, &remaining, error_text.ptr, error_text.len));
            }
        }
    }
    if (remaining != 0) return file.fail("KV checkpoint has trailing payload bytes");
    if (c.ds4_gpu_synchronize() == 0) return file.fail("failed to synchronize accelerator after GLM KV restore");
    c.ds4_tokens_free(&frontier.checkpoint);
    frontier.checkpoint = checkpoint;
    checkpoint = std.mem.zeroes(c.ds4_tokens);
    frontier.checkpoint_valid = true;
    frontier.mtp_draft_valid = false;
    dense_cache_len.* = if (graph.full_kv_cache) header[12] else 0;
}
