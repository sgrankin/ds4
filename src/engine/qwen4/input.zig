//! Host staging of embeddings, image positions and disk-only PLE rows before
//! command encoding. Recurrent history advances in token order, including the
//! state captured after rows0/1 during speculative verification.
const std = @import("std");
const c = @import("gguf_c");
const embedding = @import("embedding.zig");
const ngrams = @import("ngrams.zig");
const Config = @import("config.zig").Config;
const Graph = c.ds4_qwen4_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Span = c.ds4_vision_span;

pub fn ropePosition(spans: []const Span, position: u32, delta: *i32) [3]u32 {
    for (spans) |*span| {
        const count = span.embedding.token_count;
        if (position < span.token_start or position - span.token_start >= count) continue;
        const width = if (span.embedding.grid_width > 0) span.embedding.grid_width else count;
        const height = if (span.embedding.grid_height > 0) span.embedding.grid_height else 1;
        const offset = position - span.token_start;
        const base = span.token_start +% @as(u32, @bitCast(delta.*));
        const result = [3]u32{ base, base +% offset / width, base +% offset % width };
        if (offset + 1 == count) delta.* -= @as(i32, @intCast(count)) - @as(i32, @intCast(@max(height, width)));
        return result;
    }
    const value = position +% @as(u32, @bitCast(delta.*));
    return @splat(value);
}

pub fn imageRow(spans: []const Span, position: u32, width: u32) ?[]const f32 {
    for (spans) |*span| {
        if (position >= span.token_start and position - span.token_start < span.embedding.token_count and span.embedding.data != null) {
            const offset = @as(usize, position - span.token_start) * width;
            return span.embedding.data[offset..][0..width];
        }
    }
    return null;
}

var fake_spans: std.ArrayList(Span) = .empty;
var fake_parsed = false;

fn fakeSpans(config: *const Config) []const Span {
    if (!fake_parsed) {
        fake_parsed = true;
        var cursor = c.getenv("DS4_QWEN4_FAKE_IMAGE");
        while (cursor != null and cursor[0] != 0) {
            var start: c_uint = undefined;
            var count: c_uint = undefined;
            var height: c_uint = undefined;
            var width: c_uint = undefined;
            if (c.sscanf(cursor, "%u:%u:%u:%u", &start, &count, &height, &width) != 4 or count == 0) break;
            const values = std.heap.c_allocator.alloc(f32, @as(usize, count) * config.shape.n_embd) catch c.ds4_engine_model_die("out of memory");
            for (values, 0..) |*value, index| {
                var hash = (@as(u32, @truncate(index)) +% 1) *% 2654435761 ^ (start *% 40503);
                hash ^= hash >> 15;
                hash *%= 2246822519;
                hash ^= hash >> 13;
                value.* = (@as(f32, @floatFromInt(hash & 0xffff)) / 65535 - 0.5) * 0.05;
            }
            var span = std.mem.zeroes(Span);
            span.token_start = start;
            span.embedding.token_count = count;
            span.embedding.grid_height = height;
            span.embedding.grid_width = width;
            span.embedding.data = values.ptr;
            fake_spans.append(std.heap.c_allocator, span) catch c.ds4_engine_model_die("out of memory");
            const next = c.strchr(cursor, ',');
            cursor = if (next != null) next + 1 else null;
        }
    }
    return fake_spans.items;
}

pub fn stage(graph: *Graph, model: *const Model, weights: *const Weights, tokens: []const c_int) bool {
    const config = &graph.config;
    const shape = &config.shape;
    const width: usize = shape.n_embd;
    const hc_width = width * shape.n_hc;
    const spans = if (graph.vis_spans != null) graph.vis_spans[0..graph.vis_span_count] else fakeSpans(config);
    const rows = graph.host_row[0 .. tokens.len * hc_width];
    for (tokens, 0..) |token, index| {
        const destination = rows[index * hc_width ..][0..hc_width];
        const position = graph.pos + @as(u32, @intCast(index));
        if (imageRow(spans, position, shape.n_embd)) |image| {
            @memcpy(destination[0..width], image);
        } else if (!embedding.row(model, weights.token_embd, @intCast(token), destination[0..width])) return false;
        for (1..shape.n_hc) |stream| @memcpy(destination[stream * width ..][0..width], destination[0..width]);
        const position3 = ropePosition(spans, position, &graph.mrope_delta);
        @memcpy(graph.host_pos3[index * 4 ..][0..3], &position3);
        graph.host_pos3[index * 4 + 3] = 0;
        if (index == 0 and graph.snap_after_first) graph.snap_mrope_delta = graph.mrope_delta;
        if (index == 1 and graph.snap_after_second) graph.snap2_mrope_delta = graph.mrope_delta;
    }
    if (c.ds4_gpu_tensor_write(graph.R, 0, rows.ptr, tokens.len * hc_width * @sizeOf(f32)) == 0 or
        c.ds4_gpu_tensor_write(graph.pos3, @as(u64, graph.pos) * 16, graph.host_pos3, tokens.len * 16) == 0) return false;
    var ids: [256 * c.DS4_MAX_PLE_HEADS]u32 = undefined;
    const heads = config.ple.n_heads;
    for (tokens, 0..) |token, index| {
        ngrams.step(config, token, &graph.ple_prev, ids[index % 256 * heads ..][0..heads]);
        if (index == 0 and graph.snap_after_first) {
            graph.snap_ple_prev = graph.ple_prev;
            graph.snap_pos = graph.pos + 1;
        }
        if (index == 1 and graph.snap_after_second) {
            graph.snap2_ple_prev = graph.ple_prev;
            graph.snap2_pos = graph.pos + 2;
        }
        if (index % 256 == 255 or index + 1 == tokens.len) {
            const first = index / 256 * 256;
            const count = index - first + 1;
            if (!ngrams.read(model, ids[0 .. count * heads], rows[first * width ..][0 .. count * width])) {
                _ = c.fprintf(c.__stderrp, "ds4: n-gram read failed: %s\n", c.strerror(c.__error().*));
                return false;
            }
        }
    }
    return c.ds4_gpu_tensor_write(graph.ple_emb, 0, rows.ptr, tokens.len * width * @sizeOf(f32)) != 0;
}
