//! Complete token/HC embedding, repeat and sparse KV row-gather transactions.
//! Exact model-view policy and every shader packet/grid are native. ARC leaves
//! retain real buffers, query PSOs and encode only the named resource bindings.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const Command = @import("routed_batch.zig").Command;
const View = c.ds4_metal_attention_view;
const Pipeline = c.ds4_metal_embedding_pipeline;
const Launch = c.ds4_metal_embedding_launch;
const max = std.math.maxInt(u64);
fn narrow(value: u32) i32 {
    return @bitCast(value);
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn report(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
fn plan(kind: u32, pipeline: Pipeline, source: View, ids: View, out: View) Launch {
    return .{
        .kind = kind,
        .pipeline = pipeline.pipeline,
        .args = std.mem.zeroes(c.ds4_metal_embedding_packet),
        .source = source,
        .ids = ids,
        .out = out,
        .inline_token = false,
        .dispatch_threads = false,
        .token = 0,
        .groups_x = 1,
        .groups_y = 1,
        .threads = 1,
    };
}
fn encode(command: *const Command, launch: *const Launch) bool {
    return c.ds4_metal_embedding_encode(command.token, launch) != 0;
}
fn rowsArgs(vocab: u32, tokens: u32, width: u32, half: bool) c.ds4_metal_embedding_rows_args {
    const source = @as(u64, width) * (if (half) @as(u64, 2) else 4);
    const row = @as(u64, width) * 4;
    const token = @as(u64, tokens) * 4;
    return .{
        .ne00t = narrow(width),
        .ne00 = narrow(width),
        .nb01 = source,
        .nb02 = @as(u64, vocab) *% source,
        .nb03 = @as(u64, vocab) *% source,
        .ne10 = narrow(tokens),
        .nb10 = 4,
        .nb11 = token,
        .nb12 = token,
        .nb1 = row,
        .nb2 = @as(u64, tokens) *% row,
        .nb3 = @as(u64, tokens) *% row,
    };
}
/// Shared current quant row grammar, also available to the following ordinary
/// quant owner. These are the supported three layouts, not a new type registry.
pub fn rowBytes(format: u32, width: u32) ?u64 {
    if (width == 0) return null;
    return switch (format) {
        c.DS4_METAL_TENSOR_Q8_0 => ((@as(u64, width) + 31) / 32) * 34,
        c.DS4_METAL_TENSOR_Q4_0 => ((@as(u64, width) + 31) / 32) * 18,
        c.DS4_METAL_TENSOR_Q4_K => if (width % 256 == 0) (@as(u64, width) / 256) * 144 else null,
        else => null,
    };
}
fn tableBytes(vocab: u32, row: u64) ?u64 {
    if (vocab == 0 or (row != 0 and vocab > max / row)) return null;
    return @as(u64, vocab) * row;
}
fn modelView(model: ?*const anyopaque, size: u64, offset: u64, bytes: u64, exact: bool) ?Buffer {
    var inner: u64 = 0;
    const retained = if (exact) c.ds4_metal_embedding_exact_view(model, size, offset, bytes, &inner) else api.retainModelRange(model, size, offset, bytes, &inner);
    const buffer = retained orelse return null;
    return .{ .view = .{ .buffer = buffer, .offset = inner, .bytes = bytes } };
}
fn getRows(
    command: *const Command,
    kind: u32,
    source: View,
    ids: View,
    out: View,
    vocab: u32,
    tokens: u32,
    width: u32,
    single: ?u32,
) bool {
    const pipeline = c.ds4_metal_embedding_pipeline_get(kind);
    if (pipeline.pipeline == null or source.buffer == null or out.buffer == null or (ids.buffer == null and (single == null or tokens != 1))) return false;
    var launch = plan(kind, pipeline, source, ids, out);
    launch.inline_token = ids.buffer == null;
    launch.token = if (single) |token| narrow(token) else 0;
    if (kind == c.DS4_EMBED_F16) {
        launch.args.rows = rowsArgs(vocab, tokens, width, true);
        launch.threads = @max(@min(@as(u64, width), pipeline.max_threads), 1);
        launch.groups_x = ((@as(u64, width) + launch.threads - 1) / launch.threads) * tokens;
    } else {
        const format: u32 = if (kind == c.DS4_EMBED_Q8) c.DS4_METAL_TENSOR_Q8_0 else if (kind == c.DS4_EMBED_Q4_0) c.DS4_METAL_TENSOR_Q4_0 else c.DS4_METAL_TENSOR_Q4_K;
        launch.args.quant = .{
            .n_embd = narrow(width),
            .n_vocab = narrow(vocab),
            .n_tokens = narrow(tokens),
            .src_row_bytes = rowBytes(format, width) orelse return false,
            .dst_row_bytes = @as(u64, width) * 4,
            .token_stride = 4,
        };
        const block: u64 = if (kind == c.DS4_EMBED_Q4_K) 256 else 32;
        launch.threads = @max(@min(block, pipeline.max_threads), 1);
        launch.groups_x = (@as(u64, width) + block - 1) / block;
        launch.groups_y = tokens;
    }
    return encode(command, &launch);
}
fn repeat(command: *const Command, source: View, out: View, tokens: u32, width: u32, copies: u32) bool {
    const pipeline = c.ds4_metal_embedding_pipeline_get(c.DS4_EMBED_REPEAT);
    if (pipeline.pipeline == null or source.buffer == null or out.buffer == null) return false;
    const row = @as(u64, width) * 4;
    var launch = plan(c.DS4_EMBED_REPEAT, pipeline, source, std.mem.zeroes(View), out);
    launch.args.repeat = .{
        .ne00 = narrow(width),
        .ne01 = 1,
        .ne02 = narrow(tokens),
        .ne03 = 1,
        .nb00 = 4,
        .nb01 = row,
        .nb02 = row,
        .nb03 = @as(u64, tokens) *% row,
        .ne0 = narrow(width),
        .ne1 = narrow(copies),
        .ne2 = narrow(tokens),
        .ne3 = 1,
        .nb0 = 4,
        .nb1 = row,
        .nb2 = @as(u64, copies) *% row,
        .nb3 = @as(u64, tokens) *% copies *% row,
    };
    launch.groups_x = copies;
    launch.groups_y = tokens;
    launch.threads = @max(@min(@as(u64, width), pipeline.max_threads), 1);
    return encode(command, &launch);
}
pub const Embedding = struct {
    out: ?*api.Tensor,
    ids: ?*const api.Tensor = null,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    vocab: u32,
    width: u32,
    tokens: u32 = 1,
    single: ?u32 = null,
    pub fn quant(self: Embedding, format: u32) bool {
        if (api.init() == 0 or self.out == null or self.model == null or self.vocab == 0 or self.width == 0 or self.tokens == 0 or (self.single == null and self.ids == null) or (self.single != null and self.single.? >= self.vocab)) return false;
        const q8 = format == c.DS4_METAL_TENSOR_Q8_0;
        const noun: [*:0]const u8 = if (q8) (if (self.single != null) "Q8_0 embedding" else "Q8_0 batched embedding") else if (self.single != null) "quant embedding" else "quant batched embedding";
        const output = Buffer.tensor(self.out);
        defer output.deinit();
        const ids = Buffer.tensor(self.ids);
        defer ids.deinit();
        if (!output.fits(@as(u64, self.tokens) *% self.width *% 4) or (self.single == null and !ids.fits(@as(u64, self.tokens) * 4))) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal %s received undersized %s\n", noun, @as([*:0]const u8, if (self.single != null) "output buffer" else "buffers"));
            return false;
        }
        const row = rowBytes(format, self.width) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: Metal %s range is outside the mapped model\n", noun);
            return false;
        };
        const bytes = tableBytes(self.vocab, row) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: Metal %s range is outside the mapped model\n", noun);
            return false;
        };
        if (self.offset > self.size or bytes > self.size - self.offset) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal %s range is outside the mapped model\n", noun);
            return false;
        }
        const exact = self.single != null and (!q8 or !present("DS4_METAL_DISABLE_TOKEN_EMBED_EXACT_VIEW"));
        var offset = self.offset;
        var bound_bytes = bytes;
        var kernel_token = self.single;
        if (exact) {
            const skip = @as(u64, self.single.?) * row;
            if (skip > bytes or row > bytes - skip) {
                _ = c.fprintf(c.__stderrp, "ds4: Metal %s token row is outside the mapped table\n", noun);
                return false;
            }
            offset +%= skip;
            bound_bytes = row;
            kernel_token = 0;
        }
        const weights = modelView(self.model, self.size, offset, bound_bytes, exact) orelse return false;
        defer weights.deinit();
        const kind: u32 = if (q8) c.DS4_EMBED_Q8 else if (format == c.DS4_METAL_TENSOR_Q4_0) c.DS4_EMBED_Q4_0 else c.DS4_EMBED_Q4_K;
        var command: Command = .{ .label = if (q8) (if (self.single != null) "q8_0 embed token" else "q8_0 embed tokens") else if (self.single != null) "quant embed token" else "quant embed tokens" };
        defer command.drop();
        return command.acquire() and getRows(&command, kind, weights.view, ids.view, output.view, self.vocab, self.tokens, self.width, kernel_token) and command.finish();
    }
    pub fn hc(self: Embedding, copies: u32) bool {
        if (api.init() == 0 or self.out == null or self.model == null or self.vocab == 0 or self.width == 0 or self.tokens == 0 or copies == 0 or (self.single == null and self.ids == null) or (self.single != null and self.single.? >= self.vocab)) return false;
        const output = Buffer.tensor(self.out);
        defer output.deinit();
        const ids = Buffer.tensor(self.ids);
        defer ids.deinit();
        if (!output.fits(@as(u64, self.tokens) *% self.width *% copies *% 4) or (self.single == null and !ids.fits(@as(u64, self.tokens) * 4))) return report(if (self.single != null) "ds4: Metal graph embedding received undersized HC output buffer" else "ds4: Metal graph batched embedding received undersized buffers");
        const bytes = @as(u64, self.vocab) *% self.width *% 2;
        if (self.offset > self.size or bytes > self.size - self.offset) return report(if (self.single != null) "ds4: Metal graph embedding range is outside the mapped model" else "ds4: Metal graph batched embedding range is outside the mapped model");
        const weights = modelView(self.model, self.size, self.offset, bytes, false) orelse return false;
        defer weights.deinit();
        const scratch = Buffer{ .view = c.ds4_metal_embedding_scratch(@as(u64, self.tokens) *% self.width *% 4) };
        defer scratch.deinit();
        if (!scratch.valid()) return false;
        var command: Command = .{ .label = if (self.single != null) "graph embed token" else "graph embed tokens" };
        defer command.drop();
        return command.acquire() and getRows(&command, c.DS4_EMBED_F16, weights.view, ids.view, scratch.view, self.vocab, self.tokens, self.width, self.single) and repeat(&command, scratch.view, output.view, self.tokens, self.width, copies) and command.finish();
    }
    pub fn bf16(self: Embedding) bool {
        if (api.init() == 0) return false;
        const output = Buffer.tensor(self.out);
        defer output.deinit();
        const ids = Buffer.tensor(self.ids);
        defer ids.deinit();
        const weights = @as(u64, self.width) * self.vocab;
        const elements = @as(u64, self.width) * self.tokens;
        if (self.tokens == 0 or self.width == 0 or self.vocab == 0 or !ids.fits(@as(u64, self.tokens) * 4) or elements > max / 4 or !output.fits(elements * 4)) return report("ds4: GLM-5.3 BF16 embedding received invalid buffers");
        const bytes = weights *% 2;
        if (self.model == null or self.offset > self.size or bytes > self.size - self.offset) return report("ds4: GLM-5.3 BF16 embedding range is outside the mapped model");
        const weight = modelView(self.model, self.size, self.offset, bytes, false) orelse return false;
        defer weight.deinit();
        const pipeline = c.ds4_metal_embedding_pipeline_get(c.DS4_EMBED_BF16);
        if (pipeline.pipeline == null) return false;
        var launch = plan(c.DS4_EMBED_BF16, pipeline, weight.view, ids.view, output.view);
        launch.args.bf16 = .{ .in_dim = self.width, .out_dim = self.vocab, .n_rows = self.tokens };
        launch.dispatch_threads = true;
        launch.groups_x = self.width;
        launch.groups_y = self.tokens;
        launch.threads = @min(self.width, 256);
        var command: Command = .{ .label = "GLM-5.3 BF16 embedding" };
        defer command.drop();
        return command.acquire() and encode(&command, &launch) and command.finish();
    }
};
pub fn repeatRows(
    out: ?*api.Tensor,
    rows: ?*const api.Tensor,
    tokens: u32,
    width: u32,
    copies: u32,
    single: bool,
) bool {
    if (api.init() == 0 or out == null or rows == null or tokens == 0 or width == 0 or copies == 0) return false;
    const source = Buffer.tensor(rows);
    defer source.deinit();
    const output = Buffer.tensor(out);
    defer output.deinit();
    const bytes = @as(u64, tokens) *% width *% 4;
    if (!source.fits(bytes) or !output.fits(bytes *% copies)) return report(if (single) "ds4: Metal HC repeat received undersized buffers" else "ds4: Metal HC row repeat received undersized buffers");
    var command: Command = .{ .label = if (single) "HC repeat" else "HC row repeat" };
    defer command.drop();
    return command.acquire() and repeat(&command, source.view, output.view, tokens, width, copies) and command.finish();
}
pub fn gather(
    out: ?*api.Tensor,
    source: ?*const api.Tensor,
    ids: ?*const api.Tensor,
    source_rows: u32,
    selected_rows: u32,
    kind: u32,
) bool {
    if (source_rows == 0 or selected_rows == 0 or selected_rows > 512 or selected_rows > source_rows) return false;
    const input = Buffer.tensor(source);
    defer input.deinit();
    const tokens = Buffer.tensor(ids);
    defer tokens.deinit();
    const output = Buffer.tensor(out);
    defer output.deinit();
    const half = kind != c.DS4_EMBED_F32_GATHER;
    const is_packed = kind == c.DS4_EMBED_PACKED_GATHER;
    if (!input.fits(@as(u64, source_rows) * 512 * (if (half) @as(u64, 2) else 4)) or !tokens.fits(@as(u64, selected_rows) * 4) or !output.fits(@as(u64, selected_rows) * 512 * (if (is_packed) @as(u64, 2) else 4))) return false;
    if (api.init() == 0) return false;
    var command: Command = .{ .label = if (is_packed) "V4.1 packed selected F16 KV gather" else if (half) "V4.1 packed F16 sparse KV gather" else "V4.1 sparse KV gather" };
    defer command.drop();
    if (!is_packed and half) return command.acquire() and getRows(&command, c.DS4_EMBED_F16, input.view, tokens.view, output.view, source_rows, selected_rows, 512, null) and command.finish();
    const pipeline = c.ds4_metal_embedding_pipeline_get(kind);
    if (pipeline.pipeline == null) return false;
    var launch = plan(kind, pipeline, input.view, tokens.view, output.view);
    if (is_packed) {
        const legacy = c.ds4_metal_embedding_pipeline_get(c.DS4_EMBED_F16);
        // Keep launch arithmetic wide: @min may infer a narrow bounded integer.
        const threads: u64 = @min(legacy.max_threads, 512);
        if (threads == 0 or threads > pipeline.max_threads) return false;
        launch.args.packed_rows = .{ source_rows, selected_rows };
        launch.threads = threads;
        launch.groups_x = ((512 + threads - 1) / threads) * selected_rows;
    } else {
        launch.args.rows = rowsArgs(source_rows, selected_rows, 512, false);
        launch.threads = 256;
        launch.groups_x = 2 * selected_rows;
    }
    return command.acquire() and encode(&command, &launch) and command.finish();
}
