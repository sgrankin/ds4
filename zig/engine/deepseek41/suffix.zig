//! Private stale-suffix graph state. Session provenance, checkpoint ownership,
//! approximate-state policy and fresh-tail evaluation belong to the C facade.
//! Sources borrow immutable payload/original bytes throughout each operation.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const Payload = @import("state_payload.zig").Payload;
const Graph = c.ds41_gpu_graph;
const Archive = c.ds41_prerope_archive;
const Tensor = c.ds4_gpu_tensor;
const Cache = c.ds4_suffix_cache;
const raw_rows = 128;
const tape_width = 128; // Last64 BF16 rotary lanes, not a full F32 key.
const archive_magic: u32 = 0x31525044;
const archive_header_bytes = 12 + 40 * raw_rows * @sizeOf(u32);
const tile_rows = 128;

pub const Failure = error{ InvalidSource, InvalidGeometry, Interrupted, Archive, Device, Allocation };
pub const Cancel = struct {
    callback: c.ds4_session_cancel_fn = null,
    userdata: ?*anyopaque = null,

    fn requested(self: Cancel) bool {
        return if (self.callback) |check| check(self.userdata) else false;
    }
};
pub const Result = struct {
    reuse: c.ds4_suffix_reuse_stats,
    relocation: c.ds4_suffix_relocation_stats,
    end: u32,
};

fn fail(text: []u8, reason: Failure, message: []const u8) Failure {
    if (text.len != 0) {
        const count = @min(message.len, text.len - 1);
        @memcpy(text[0..count], message[0..count]);
        text[count] = 0;
    }
    return reason;
}
fn readU32(bytes: []const u8) u32 {
    return std.mem.readInt(u32, bytes[0..4], .little);
}
fn ratio(owner: usize) u32 {
    return if (owner < 3) 2 else 1;
}
pub fn archiveBytes(end: u32) u64 {
    var bytes: u64 = archive_header_bytes + 40 * raw_rows * tape_width;
    for (0..4) |owner| bytes += @as(u64, end / ratio(owner)) * 257;
    return bytes;
}
// Graph resources retain ownership of the live tape on reset/free as well as
// allocation failure. Share those operations with the suffix controller.
pub const clearArchive = @import("resources.zig").clearPrerope;
pub const freeArchive = @import("resources.zig").freePrerope;

const Originals = struct {
    epochs: []const u8,
    raw: [40][]const u8,
    published: [4][]const u8,
    compressed: [4][]const u8,
    index: [4][]const u8,

    fn view(cache: *const Cache, context: u32, end: u32) ?Originals {
        if (cache.prerope == null or cache.prerope_bytes != archiveBytes(end)) return null;
        const memory: [*]const u8 = @ptrCast(cache.prerope);
        const blob = memory[0..@intCast(cache.prerope_bytes)];
        if (readU32(blob) != archive_magic or readU32(blob[4..]) != context or readU32(blob[8..]) != end) return null;
        var result: Originals = undefined;
        result.epochs = blob[12..archive_header_bytes];
        var offset: usize = archive_header_bytes;
        for (&result.raw) |*raw| {
            raw.* = blob[offset..][0 .. raw_rows * tape_width];
            offset += raw_rows * tape_width;
        }
        for (0..4) |owner| {
            const rows: usize = end / ratio(owner);
            result.published[owner] = blob[offset..][0..rows];
            offset += rows;
            result.compressed[owner] = blob[offset..][0 .. rows * tape_width];
            offset += rows * tape_width;
            result.index[owner] = blob[offset..][0 .. rows * tape_width];
            offset += rows * tape_width;
        }
        return result;
    }
    fn epoch(self: *const Originals, layer: usize, slot: usize) u32 {
        return readU32(self.epochs[(layer * raw_rows + slot) * 4 ..]);
    }
};

const Source = struct {
    end: u32,
    tokens: []const u8,
    window: [40][]const u8,
    compressed: [4][]const u8,
    index: [4][]const u8,
    previous_keys: [3]?[]const u8 = @splat(null),
    previous_scores: [3]?[]const u8 = @splat(null),

    fn view(graph: *Graph, config: *const Config, cache: *const Cache) ?Source {
        const state = &cache.state;
        const header_bytes = c.DS4_SESSION_PAYLOAD_U32_FIELDS * 4;
        if (state.ptr == null or state.len < header_bytes or state.len > state.cap) return null;
        const blob = state.ptr[0..@as(usize, @intCast(state.len))];
        var header: [c.DS4_SESSION_PAYLOAD_U32_FIELDS]u32 = undefined;
        for (&header, 0..) |*value, index| value.* = readU32(blob[index * 4 ..]);
        const shape = &config.shape;
        const end = header[7];
        const magic: u32 = if (cache.approximate and cache.prerope != null) c.DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC else c.DS4_SESSION_PAYLOAD_MAGIC;
        const payload = Payload{ .graph = graph, .config = config };
        if (header[0] != magic or header[1] != c.DS4_SESSION_PAYLOAD_VERSION or
            header[2] != graph.ctx or header[3] != 1 or header[4] != 128 or header[5] != 128 or
            header[6] != header[2] +% 1 or end == 0 or end >= header[2] or header[8] != shape.n_layer or
            header[9] != shape.n_head_dim or header[10] != shape.n_indexer_head_dim or
            header[11] != shape.n_vocab or header[12] != 0x413431 or state.len != header_bytes + payload.bodyBytes(end)) return null;
        var result = Source{ .end = end, .tokens = blob[header_bytes..][0 .. @as(usize, end) * 4], .window = undefined, .compressed = undefined, .index = undefined };
        for (0..end) |index| if (readU32(result.tokens[index * 4 ..]) >= shape.n_vocab) return null;
        var offset: usize = header_bytes + (@as(usize, end) + shape.n_vocab) * 4; // Source logits are not reused.
        const raw: usize = @min(end, raw_rows);
        for (&result.window) |*keys| {
            const bytes = raw * shape.n_head_dim * 4;
            keys.* = blob[offset..][0..bytes];
            offset += bytes;
        }
        for (0..4) |owner| {
            const rows: usize = end / ratio(owner);
            const key_bytes = rows * shape.n_head_dim * 4;
            const index_bytes = rows * shape.n_indexer_head_dim * 4;
            result.compressed[owner] = blob[offset..][0..key_bytes];
            offset += key_bytes;
            result.index[owner] = blob[offset..][0..index_bytes];
            offset += index_bytes;
            if (owner < 3 and end & 1 != 0) {
                result.previous_keys[owner] = blob[offset..][0 .. shape.n_head_dim * 4];
                offset += shape.n_head_dim * 4;
                result.previous_scores[owner] = blob[offset..][0 .. shape.n_head_dim * 4];
                offset += shape.n_head_dim * 4;
            }
        }
        return if (offset == blob.len) result else null;
    }
};

pub const Suffix = struct {
    graph: *Graph,
    config: *const Config,

    pub fn enableArchive(self: *const Suffix) bool {
        const graph = self.graph;
        if (graph.prerope != null) return true;
        const memory = c.calloc(1, @sizeOf(Archive)) orelse return false;
        const archive: *Archive = @ptrCast(@alignCast(memory));
        var complete = false;
        defer if (!complete) freeArchive(archive);
        archive.bytes = @sizeOf(Archive);
        for (&archive.raw) |*tensor| {
            tensor.* = c.ds4_gpu_tensor_alloc(raw_rows * tape_width) orelse return false;
            if (c.ds4_gpu_tensor_fill_f32(tensor.*, 0, raw_rows * 32) == 0) return false;
            archive.bytes += raw_rows * tape_width;
        }
        for (0..4) |owner| {
            const rows = graph.ctx / ratio(owner) + 1;
            const bytes = @as(u64, rows) * tape_width;
            archive.published[owner] = @ptrCast(c.calloc(rows, 1) orelse return false);
            archive.compressed[owner] = c.ds4_gpu_tensor_alloc(bytes) orelse return false;
            archive.index[owner] = c.ds4_gpu_tensor_alloc(bytes) orelse return false;
            if (c.ds4_gpu_tensor_fill_f32(archive.compressed[owner], 0, @as(u64, rows) * 32) == 0 or
                c.ds4_gpu_tensor_fill_f32(archive.index[owner], 0, @as(u64, rows) * 32) == 0) return false;
            archive.bytes += rows + 2 * bytes;
        }
        clearArchive(archive, graph.ctx);
        graph.prerope = archive;
        complete = true;
        return true;
    }

    // Pointer-free tape stays separate from canonical payloads. Publish the
    // new owned blob only after every GPU read succeeds; the old blob survives
    // a failed capture unchanged.
    pub fn captureArchive(self: *const Suffix, cache: *Cache) bool {
        const graph = self.graph;
        if (graph.prerope == null) return false;
        const archive: *const Archive = @ptrCast(graph.prerope);
        const bytes = archiveBytes(graph.pos);
        if (bytes > std.math.maxInt(usize)) return false;
        const memory = c.calloc(1, @intCast(bytes)) orelse return false;
        const data: [*]u8 = @ptrCast(memory);
        const blob = data[0..@as(usize, @intCast(bytes))];
        var complete = false;
        defer if (!complete) c.free(memory);
        std.mem.writeInt(u32, blob[0..4], archive_magic, .little);
        std.mem.writeInt(u32, blob[4..8], graph.ctx, .little);
        std.mem.writeInt(u32, blob[8..12], graph.pos, .little);
        @memcpy(blob[12..archive_header_bytes], std.mem.asBytes(&archive.raw_epoch));
        if (c.ds4_gpu_synchronize() == 0) return false;
        var offset: usize = archive_header_bytes;
        for (archive.raw) |tensor| {
            if (c.ds4_gpu_tensor_read(tensor, 0, blob.ptr + offset, raw_rows * tape_width) == 0) return false;
            offset += raw_rows * tape_width;
        }
        for (0..4) |owner| {
            const rows: usize = graph.pos / ratio(owner);
            @memcpy(blob[offset..][0..rows], archive.published[owner][0..rows]);
            offset += rows;
            const count = rows * tape_width;
            if (count != 0 and c.ds4_gpu_tensor_read(archive.compressed[owner], 0, blob.ptr + offset, count) == 0) return false;
            offset += count;
            if (count != 0 and c.ds4_gpu_tensor_read(archive.index[owner], 0, blob.ptr + offset, count) == 0) return false;
            offset += count;
        }
        c.free(cache.prerope);
        cache.prerope = memory;
        cache.prerope_bytes = bytes;
        complete = true;
        return true;
    }

    // Prefix restore permits sparse coverage, but never an epoch from outside
    // the saved live ring or unknown publication bits. Run before payload load.
    pub fn validatePrefix(self: *const Suffix, cache: *const Cache, error_text: []u8) Failure!u32 {
        const source = Source.view(self.graph, self.config, cache) orelse
            return fail(error_text, error.InvalidSource, "suffix prefix restore requires exact canonical DS4.1 state");
        if (cache.prerope != null) {
            const originals = Originals.view(cache, self.graph.ctx, source.end) orelse
                return fail(error_text, error.Archive, "suffix prefix pre-RoPE archive layout mismatch");
            const first = source.end -| raw_rows;
            for (0..40) |layer| {
                for (0..raw_rows) |slot| {
                    const position = originals.epoch(layer, slot);
                    if (position != std.math.maxInt(u32) and (position < first or position >= source.end or position % raw_rows != slot))
                        return fail(error_text, error.Archive, "suffix prefix raw pre-RoPE coverage mismatch");
                }
            }
            for (originals.published) |published| for (published) |bits| {
                if (bits & ~@as(u8, 3) != 0) return fail(error_text, error.Archive, "suffix prefix published pre-RoPE coverage mismatch");
            };
        }
        return source.end;
    }

    // The facade first loads the canonical graph payload. Original coverage is
    // then published only after all tape bytes have been restored successfully.
    pub fn restoreArchive(self: *const Suffix, cache: *const Cache, end: u32, cancel: Cancel) Failure!void {
        const originals = Originals.view(cache, self.graph.ctx, end) orelse return error.Archive;
        if (self.graph.prerope == null) return error.Archive;
        const archive: *Archive = @ptrCast(self.graph.prerope);
        for (archive.raw, originals.raw) |tensor, bytes| {
            if (cancel.requested() or c.ds4_gpu_tensor_write(tensor, 0, bytes.ptr, bytes.len) == 0) return error.Device;
        }
        for (0..4) |owner| {
            const bytes = originals.compressed[owner].len;
            if (cancel.requested() or (bytes != 0 and
                (c.ds4_gpu_tensor_write(archive.compressed[owner], 0, originals.compressed[owner].ptr, bytes) == 0 or
                    c.ds4_gpu_tensor_write(archive.index[owner], 0, originals.index[owner].ptr, bytes) == 0))) return error.Device;
        }
        @memcpy(std.mem.asBytes(&archive.raw_epoch), originals.epochs);
        for (0..4) |owner| @memcpy(archive.published[owner][0..originals.published[owner].len], originals.published[owner]);
    }

    pub fn graft(self: *const Suffix, cache: *const Cache, prompt: []const c_int, source_start: u32, start: u32, cancel: Cancel, mutated: *bool, error_text: []u8) Failure!Result {
        mutated.* = false;
        const graph = self.graph;
        const shape = &self.config.shape;
        const source = Source.view(graph, self.config, cache) orelse
            return fail(error_text, error.InvalidGeometry, "suffix reuse requires even-aligned source and destination within the context");
        if ((source_start | start) & 1 != 0 or source_start >= source.end or prompt.len >= graph.ctx)
            return fail(error_text, error.InvalidGeometry, "suffix reuse requires even-aligned source and destination within the context");
        const displacement = @as(i64, source_start) - start;
        const end64 = @as(u64, start) + source.end - source_start;
        if (end64 >= prompt.len or end64 >= graph.ctx)
            return fail(error_text, error.InvalidGeometry, "suffix reuse requires a fresh tail after the relocated endpoint");
        const end: u32 = @intCast(end64);
        for (prompt, 0..) |token, index| {
            if (token < 0 or @as(u32, @bitCast(token)) >= shape.n_vocab or
                (index >= start and index < end and @as(u32, @bitCast(token)) != readU32(source.tokens[(@as(usize, source_start) + index - start) * 4 ..])))
                return fail(error_text, error.InvalidSource, "suffix tokens do not match the saved endpoint");
        }
        const use_originals = cache.prerope != null;
        if (start > source_start and !use_originals)
            return fail(error_text, error.Archive, "right-shift suffix reuse requires immutable pre-RoPE originals");
        var originals: Originals = undefined;
        if (use_originals) {
            if (graph.prerope == null)
                return fail(error_text, error.Archive, "suffix pre-RoPE archive layout or live mode mismatch");
            originals = Originals.view(cache, graph.ctx, source.end) orelse
                return fail(error_text, error.Archive, "suffix pre-RoPE archive layout or live mode mismatch");
        } else if (cache.approximate or graph.prerope != null)
            return fail(error_text, error.Archive, "suffix originals are unavailable; rebuild the edited context");
        const raw: u32 = @min(end - start, raw_rows);
        if (use_originals) {
            for (0..40) |layer| {
                for (source.end - raw..source.end) |position| {
                    if (originals.epoch(layer, position % raw_rows) != position)
                        return fail(error_text, error.Archive, "suffix raw pre-RoPE coverage is incomplete");
                }
            }
            for (0..4) |owner| {
                for (source_start / ratio(owner)..source.end / ratio(owner)) |row| {
                    if (originals.published[owner][row] != 3)
                        return fail(error_text, error.Archive, "suffix compressed/index pre-RoPE coverage is incomplete");
                }
            }
        }
        if (cancel.requested() or c.ds4_gpu_synchronize() == 0)
            return fail(error_text, error.Interrupted, "suffix reuse was interrupted before grafting");
        // Allocate the 256KiB F32 repair tile before invalidating the frontier.
        // Source canonical bytes and originals remain immutable in both modes.
        var tile: ?*Tensor = null;
        defer c.ds4_gpu_tensor_free(tile);
        if (graph.compressed_kv_f16) tile = c.ds4_gpu_tensor_alloc(@as(u64, tile_rows) * shape.n_head_dim * 4) orelse
            return fail(error_text, error.Allocation, "cannot allocate packed suffix repair tile");
        var result = Result{
            .reuse = std.mem.zeroes(c.ds4_suffix_reuse_stats),
            .relocation = std.mem.zeroes(c.ds4_suffix_relocation_stats),
            .end = end,
        };
        result.reuse.reused_tokens = end - start;
        result.reuse.raw_rows_per_layer = raw;
        result.reuse.fresh_tail_tokens = @intCast(prompt.len - end);
        result.relocation.source_start = source_start;
        result.relocation.destination_start = start;
        result.relocation.source_end = source.end;
        result.relocation.destination_end = end;
        graph.valid = false;
        mutated.* = true;
        // Split at both physical wraps. All source bytes come from an owned
        // immutable snapshot; no in-place memmove assumptions are required.
        for (0..40) |layer| {
            var position = source.end - raw;
            while (position < source.end) {
                const source_slot = position % raw_rows;
                const destination: u32 = @intCast(@as(i64, position) - displacement);
                const destination_slot = destination % raw_rows;
                const rows: u32 = @min(source.end - position, @min(raw_rows - source_slot, raw_rows - destination_slot));
                const bytes = @as(u64, rows) * shape.n_head_dim * 4;
                const offset = @as(u64, destination_slot) * shape.n_head_dim * 4;
                if (c.ds4_gpu_tensor_write(graph.window[layer], offset, source.window[layer].ptr + @as(u64, source_slot) * shape.n_head_dim * 4, bytes) == 0) return error.Device;
                if (use_originals) {
                    const archive: *Archive = @ptrCast(graph.prerope);
                    if (c.ds4_gpu_tensor_write(archive.raw[layer], @as(u64, destination_slot) * tape_width, originals.raw[layer].ptr + @as(u64, source_slot) * tape_width, @as(u64, rows) * tape_width) == 0) return error.Device;
                    for (0..rows) |row| archive.raw_epoch[layer][destination_slot + row] = destination + @as(u32, @intCast(row));
                    if (displacement != 0 and !originalKeys(graph.window[layer], offset, archive.raw[layer], destination_slot, shape.n_head_dim, rows, destination, 1, self.config.layerPolicy(@intCast(layer)).?.compression_ratio != 0, c.DS4_V41_FP8_E8M0)) return error.Device;
                } else if (displacement != 0 and !repairKeys(graph.window[layer], offset, shape.n_head_dim, rows, position, destination, 1, self.config.layerPolicy(@intCast(layer)).?.compression_ratio != 0, c.DS4_V41_FP8_E8M0)) return error.Device;
                if (displacement != 0) result.relocation.raw_keys_repaired += rows;
                result.reuse.copied_bytes += bytes;
                position += rows;
            }
        }
        for (0..4) |owner| {
            const compression = ratio(owner);
            const first = start / compression;
            const source_first = source_start / compression;
            const rows = source.end / compression - source_first;
            const key_offset = @as(u64, first) * shape.n_head_dim * 4;
            const index_offset = @as(u64, first) * shape.n_indexer_head_dim * 4;
            const source_key_offset = @as(u64, source_first) * shape.n_head_dim * 4;
            const source_index_offset = @as(u64, source_first) * shape.n_indexer_head_dim * 4;
            if (rows != 0 and ((!graph.compressed_kv_f16 and c.ds4_gpu_tensor_write(graph.compressed[owner], key_offset, source.compressed[owner].ptr + source_key_offset, @as(u64, rows) * shape.n_head_dim * 4) == 0) or
                c.ds4_gpu_tensor_write(graph.index_cache[owner], index_offset, source.index[owner].ptr + source_index_offset, @as(u64, rows) * shape.n_indexer_head_dim * 4) == 0)) return error.Device;
            if (use_originals and rows != 0) {
                const archive: *Archive = @ptrCast(graph.prerope);
                if (c.ds4_gpu_tensor_write(archive.compressed[owner], @as(u64, first) * tape_width, originals.compressed[owner].ptr + @as(u64, source_first) * tape_width, @as(u64, rows) * tape_width) == 0 or
                    c.ds4_gpu_tensor_write(archive.index[owner], @as(u64, first) * tape_width, originals.index[owner].ptr + @as(u64, source_first) * tape_width, @as(u64, rows) * tape_width) == 0) return error.Device;
                @memset(archive.published[owner][first .. first + rows], 3);
                if (displacement != 0 and ((!graph.compressed_kv_f16 and !originalKeys(graph.compressed[owner], key_offset, archive.compressed[owner], first, shape.n_head_dim, rows, first * compression, compression, true, c.DS4_V41_FP4_E4M3)) or
                    !originalKeys(graph.index_cache[owner], index_offset, archive.index[owner], first, shape.n_indexer_head_dim, rows, first * compression, compression, true, c.DS4_V41_FP4_E8M0))) return error.Device;
            } else if (displacement != 0 and ((!graph.compressed_kv_f16 and !repairKeys(graph.compressed[owner], key_offset, shape.n_head_dim, rows, source_first * compression, first * compression, compression, true, c.DS4_V41_FP4_E4M3)) or
                !repairKeys(graph.index_cache[owner], index_offset, shape.n_indexer_head_dim, rows, source_first * compression, first * compression, compression, true, c.DS4_V41_FP4_E8M0))) return error.Device;
            if (graph.compressed_kv_f16) try self.storePacked(tile, owner, source.compressed[owner][@intCast(source_key_offset)..], source_first, first, rows, compression, use_originals, cancel);
            if (displacement != 0) {
                result.relocation.compressed_keys_repaired += rows;
                result.relocation.index_keys_repaired += rows;
            }
            result.reuse.compressed_rows += rows;
            result.reuse.index_rows += rows;
            result.reuse.copied_bytes += @as(u64, rows) * (shape.n_head_dim + shape.n_indexer_head_dim) * 4;
            if (owner < 3 and source.end & 1 != 0) {
                if (c.ds4_gpu_tensor_write(graph.previous_kv[owner], 0, source.previous_keys[owner].?.ptr, @as(u64, shape.n_head_dim) * 4) == 0 or
                    c.ds4_gpu_tensor_write(graph.previous_score[owner], 0, source.previous_scores[owner].?.ptr, @as(u64, shape.n_head_dim) * 4) == 0) return error.Device;
                result.reuse.pending_pairs += 1;
                result.reuse.copied_bytes += 2 * @as(u64, shape.n_head_dim) * 4;
            }
        }
        // Old logits/residual are intentionally unused. The facade appends the
        // retained prompt IDs and evaluates the fresh tail to regenerate them.
        c.ds4_engram_history_reset(&graph.history);
        for (0..@min(@as(u32, 3), end)) |index| graph.history.tail[index] = @intCast(graph.token_map[@intCast(prompt[end - 1 - index])]);
        graph.pos = end;
        graph.valid = true;
        result.relocation.reuse = result.reuse;
        return result;
    }

    fn storePacked(self: *const Suffix, tile: ?*Tensor, owner: usize, source: []const u8, source_first: u32, first: u32, rows: u32, compression: u32, originals: bool, cancel: Cancel) Failure!void {
        const graph = self.graph;
        const width = self.config.shape.n_head_dim;
        var offset: u32 = 0;
        while (offset < rows) : (offset += tile_rows) {
            if (cancel.requested()) return error.Interrupted;
            const count: u32 = @min(rows - offset, tile_rows);
            if (c.ds4_gpu_tensor_write(tile, 0, source.ptr + @as(u64, offset) * width * 4, @as(u64, count) * width * 4) == 0) return error.Device;
            if (source_first != first) {
                const repaired = if (originals)
                    originalKeys(tile, 0, graph.prerope.*.compressed[owner], first + offset, width, count, (first + offset) * compression, compression, true, c.DS4_V41_FP4_E4M3)
                else
                    repairKeys(tile, 0, width, count, (source_first + offset) * compression, (first + offset) * compression, compression, true, c.DS4_V41_FP4_E4M3);
                if (!repaired) return error.Device;
            }
            if (c.ds4_gpu_tensor_copy_f32_to_f16(graph.compressed[owner], @as(u64, first + offset) * width * 2, tile, 0, @as(u64, count) * width) == 0) return error.Device;
        }
    }
};

fn repairKeys(base: ?*Tensor, offset: u64, width: u32, rows: u32, old_position: u32, new_position: u32, stride: u32, compressed: bool, format: c.ds4_v41_activation_format) bool {
    if (rows == 0 or old_position == new_position) return true;
    const view = c.ds4_gpu_tensor_view(base, offset, @as(u64, width) * rows * 4) orelse return false;
    defer c.ds4_gpu_tensor_free(view);
    // Legacy keys were already rounded. Both rotary passes round to BF16;
    // cache requantization makes this deliberately approximate for nonzero shifts.
    return c.ds4_gpu_dsv41_rope_stride(view, width, 1, rows, old_position, stride, compressed, true) != 0 and
        c.ds4_gpu_dsv41_rope_stride(view, width, 1, rows, new_position, stride, compressed, false) != 0 and
        c.ds4_gpu_dsv41_quantize(view, width, rows, format) != 0;
}
fn originalKeys(base: ?*Tensor, offset: u64, original: ?*Tensor, original_row: u32, width: u32, rows: u32, new_position: u32, stride: u32, compressed: bool, format: c.ds4_v41_activation_format) bool {
    if (rows == 0) return true;
    const view = c.ds4_gpu_tensor_view(base, offset, @as(u64, width) * rows * 4) orelse return false;
    defer c.ds4_gpu_tensor_free(view);
    return c.ds4_gpu_dsv41_prerope_copy(original, original_row, view, width, rows, false) != 0 and
        c.ds4_gpu_dsv41_rope_stride(view, width, 1, rows, new_position, stride, compressed, false) != 0 and
        c.ds4_gpu_dsv41_prerope_quantize(view, width, rows, format) != 0;
}
