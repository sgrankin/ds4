//! Canonical V4.1 graph payloads: fixed little-endian DSV4 header/tokens,
//! F32 logits and typed live cache spans. Versions 2/3 store F32 main rows,
//! versions 4/5 pack main FP4; odd versions also pack the index. All layouts
//! remain readable through bounded conversion. DRP1/DSSD wrappers
//! and session provenance remain with their callers. In particular, this
//! numerical state owner does not decide whether an approximate session may
//! use a canonical API.
//!
//! Restore copies into the graph's existing tensors; it never mmaps a cache
//! file or allocates a second graph-sized blob. F16 compressed storage uses
//! F32 wire bytes; main FP4 can copy native rows without a rounded F32 owner.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const resources = @import("resources.zig");
const draft_checkpoint = @import("draft_checkpoint.zig");
const format = @import("state_format.zig");
const Graph = c.ds41_gpu_graph;
const Tensor = c.ds4_gpu_tensor;
const allocator = std.heap.c_allocator;
const io_chunk: usize = 8 * 1024 * 1024;
pub const Header = [c.DS4_SESSION_PAYLOAD_U32_FIELDS]u32;
pub const Failure = error{ InvalidFrontier, InvalidDimensions, InvalidToken, Truncated, Read, Write, Device, ConversionBuffer };

pub const Span = struct {
    tensor: ?*Tensor,
    wire_bytes: u64,
    packed_f16: bool = false,
    main_fp4: bool = false,
    main_wire_fp4: bool = false,
    index_fp4: bool = false,
    wire_fp4: bool = false,
};

// The existing public token vector retains libc allocation/guard semantics.
// A successful load lends no mapped/file bytes: the caller owns this vector,
// commits its session checkpoint, then frees or takes the real C record.
pub const OwnedTokens = struct {
    record: c.ds4_tokens = .{ .v = null, .len = 0, .cap = 0 },

    pub fn deinit(self: *OwnedTokens) void {
        c.ds4_tokens_free(&self.record);
    }
    pub fn items(self: *const OwnedTokens) []const c_int {
        return if (self.record.len == 0) &.{} else self.record.v[0..@as(usize, @intCast(self.record.len))];
    }
    pub fn take(self: *OwnedTokens) c.ds4_tokens {
        const result = self.record;
        self.record = .{ .v = null, .len = 0, .cap = 0 };
        return result;
    }
};

const File = struct {
    stream: *c.FILE,
    error_text: []u8,

    fn fail(self: File, reason: Failure, message: []const u8) Failure {
        if (self.error_text.len != 0) {
            const count = @min(message.len, self.error_text.len - 1);
            @memcpy(self.error_text[0..count], message[0..count]);
            self.error_text[count] = 0;
        }
        return reason;
    }
    fn write(self: File, bytes: []const u8) Failure!void {
        if (bytes.len != 0 and c.fwrite(bytes.ptr, 1, bytes.len, self.stream) != bytes.len)
            return self.fail(error.Write, "failed to write session payload");
    }
    fn read(self: File, bytes: []u8, remaining: *u64) Failure!void {
        if (remaining.* < bytes.len) return self.fail(error.Truncated, "truncated session payload");
        if (bytes.len != 0 and c.fread(bytes.ptr, 1, bytes.len, self.stream) != bytes.len)
            return self.fail(error.Read, "failed to read session payload");
        remaining.* -= bytes.len;
    }
    fn writeU32(self: File, value: u32) Failure!void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        try self.write(&bytes);
    }
    fn readU32(self: File, remaining: *u64) Failure!u32 {
        var bytes: [4]u8 = undefined;
        try self.read(&bytes, remaining);
        return std.mem.readInt(u32, &bytes, .little);
    }
};

// All raw cache slots are persisted once the ring wraps; before that only its
// written prefix exists. Selection masks are rebuilt by index-source layers.
// Persist all live owner keys and only unfinished paired-compression inputs.
pub const Payload = struct {
    graph: *Graph,
    config: *const Config,
    // Only canonical checkpoints opt into a draft-feature appendix.
    include_draft: bool = false,
    allow_draft: bool = false,
    // Canonical and DSSD owners select their actual physical layouts. Private
    // DRP1/SWB1 callers leave these false and retain version2 conversion.
    index_wire_fp4: bool = false,
    main_wire_fp4: bool = false,

    pub fn spans(self: *const Payload, position: u32, storage: *[54]Span) []Span {
        const graph = self.graph;
        const shape = &self.config.shape;
        var count: usize = 0;
        const raw: u32 = @min(position, 128);
        for (graph.window) |tensor| {
            storage[count] = .{ .tensor = tensor, .wire_bytes = @as(u64, raw) * shape.n_head_dim * 4 };
            count += 1;
        }
        for (0..4) |owner| {
            const live = position / @as(u32, if (owner < 3) 2 else 1);
            storage[count] = .{
                .tensor = graph.compressed[owner],
                .wire_bytes = @as(u64, live) * format.mainRowBytes(self.main_wire_fp4),
                .packed_f16 = graph.compressed_kv_f16,
                .main_fp4 = graph.compressed_kv_fp4,
                .main_wire_fp4 = self.main_wire_fp4,
            };
            storage[count + 1] = .{
                .tensor = graph.index_cache[owner],
                .wire_bytes = @as(u64, live) * format.indexRowBytes(self.index_wire_fp4),
                .index_fp4 = graph.index_fp4,
                .wire_fp4 = self.index_wire_fp4,
            };
            count += 2;
            if (owner < 3 and position & 1 != 0) {
                storage[count] = .{ .tensor = graph.previous_kv[owner], .wire_bytes = @as(u64, shape.n_head_dim) * 4 };
                storage[count + 1] = .{ .tensor = graph.previous_score[owner], .wire_bytes = @as(u64, shape.n_head_dim) * 4 };
                count += 2;
            }
        }
        return storage[0..count];
    }

    pub fn baseBodyBytes(self: *const Payload, position: u32) u64 {
        var storage: [54]Span = undefined;
        var bytes = (@as(u64, position) + self.config.shape.n_vocab) * 4;
        for (self.spans(position, &storage)) |span| bytes += span.wire_bytes;
        return bytes;
    }

    pub fn bodyBytes(self: *const Payload, position: u32) u64 {
        const extra = if (self.draftReady(position)) draft_checkpoint.wireBytes(position, self.config.shape.n_embd) else 0;
        return self.baseBodyBytes(position) + extra;
    }

    fn draftReady(self: *const Payload, position: u32) bool {
        if (!self.include_draft or position == 0) return false;
        const state: *c.ds41_dspark = @ptrCast(self.graph.dspark orelse return false);
        const weights: *const c.ds4_dspark_weights = @ptrCast(state.weights);
        if (weights.target_layer_count != 3) return false;
        for (weights.target_layers[0..3]) |layer| if (layer >= self.config.shape.n_layer) return false;
        const start = position - draft_checkpoint.rows(position);
        for (state.positions, state.target) |tags, target| {
            if (target == null) return false;
            for (start..position) |p| if (tags[p % 128] != p) return false;
        }
        return true;
    }

    pub fn blockBytes(self: *const Payload, position: u32) u64 {
        var storage: [54]Span = undefined;
        var bytes = @as(u64, position) * 4;
        const spans_ = self.spans(position, &storage);
        for (spans_[self.config.shape.n_layer..]) |span| bytes += span.wire_bytes;
        return bytes;
    }

    pub fn header(self: *const Payload) Header {
        const graph = self.graph;
        const shape = &self.config.shape;
        return .{
            c.DS4_SESSION_PAYLOAD_MAGIC, format.version(self.main_wire_fp4, self.index_wire_fp4),
            graph.ctx,                   1,
            128,                         128,
            graph.ctx + 1,               graph.pos,
            shape.n_layer,               shape.n_head_dim,
            shape.n_indexer_head_dim,    shape.n_vocab,
            0x413431,
        };
    }

    pub fn save(self: *const Payload, stream: *c.FILE, tokens: []const c_int, logits: []const f32, error_text: []u8) Failure!void {
        const file = File{ .stream = stream, .error_text = error_text };
        if (!self.graph.valid or self.graph.pos != tokens.len or logits.len != self.config.shape.n_vocab or c.ds4_gpu_synchronize() == 0)
            return file.fail(error.InvalidFrontier, "V4.1 graph has no complete frontier to save");
        for (self.header()) |value| try file.writeU32(value);
        for (tokens) |token| try file.writeU32(@bitCast(token));
        try file.write(std.mem.sliceAsBytes(logits));
        const scratch = allocator.alignedAlloc(u8, .@"8", io_chunk) catch c.ds4_engine_model_die("out of memory");
        defer allocator.free(scratch);
        var storage: [54]Span = undefined;
        for (self.spans(self.graph.pos, &storage)) |span| try writeSpan(file, span, scratch);
        if (self.draftReady(self.graph.pos)) try self.writeDraft(file, scratch);
    }

    pub fn load(self: *const Payload, stream: *c.FILE, saved: *const Header, body_bytes: u64, logits: []f32, mutated: *bool, error_text: []u8) Failure!OwnedTokens {
        return self.loadMode(false, stream, saved, body_bytes, logits, mutated, error_text);
    }
    pub fn loadBlocks(self: *const Payload, stream: *c.FILE, saved: *const Header, body_bytes: u64, logits: []f32, mutated: *bool, error_text: []u8) Failure!OwnedTokens {
        return self.loadMode(true, stream, saved, body_bytes, logits, mutated, error_text);
    }
    fn loadMode(self: *const Payload, comptime blocks_only: bool, stream: *c.FILE, saved: *const Header, body_bytes: u64, logits: []f32, mutated: *bool, error_text: []u8) Failure!OwnedTokens {
        const file = File{ .stream = stream, .error_text = error_text };
        const graph = self.graph;
        const shape = &self.config.shape;
        const position = saved[7];
        var source = self.*;
        source.index_wire_fp4 = format.indexFp4(saved[1]);
        source.main_wire_fp4 = format.mainFp4(saved[1]);
        const base_bytes = if (blocks_only) source.blockBytes(position) else source.baseBodyBytes(position);
        const draft_bytes = draft_checkpoint.wireBytes(position, shape.n_embd);
        const has_draft = !blocks_only and self.allow_draft and body_bytes == base_bytes + draft_bytes;
        mutated.* = false;
        // The generic facade has checked magic/version before family dispatch.
        // A smaller current context is allowed when all live rows still fit.
        if (!format.validVersion(saved[1]) or (blocks_only and (source.index_wire_fp4 or source.main_wire_fp4)) or
            position == 0 or position >= graph.ctx or position >= saved[2] or saved[2] > 1048576 or
            saved[3] != 1 or saved[4] != 128 or saved[5] != 128 or saved[6] != saved[2] + 1 or
            saved[8] != shape.n_layer or saved[9] != shape.n_head_dim or saved[10] != shape.n_indexer_head_dim or
            saved[11] != shape.n_vocab or saved[12] != 0x413431 or (body_bytes != base_bytes and !has_draft) or logits.len != shape.n_vocab)
            return file.fail(error.InvalidDimensions, "invalid V4.1 snapshot dimensions or size");
        var tokens = OwnedTokens{};
        errdefer tokens.deinit();
        var remaining = body_bytes;
        for (0..position) |_| {
            const token = file.readU32(&remaining) catch return file.fail(error.InvalidToken, "invalid V4.1 snapshot token");
            if (token >= shape.n_vocab) return file.fail(error.InvalidToken, "invalid V4.1 snapshot token");
            c.ds4_tokens_push(&tokens.record, @intCast(token));
        }
        // Invalid headers/tokens leave the previous frontier intact. Once any
        // logits/KV write may occur, both graph and facade checkpoint must be
        // invalid until the entire load succeeds. The caller observes mutated
        // even on error, without this module importing the Session record.
        if (c.ds4_gpu_synchronize() == 0) return error.Device;
        mutated.* = true;
        graph.valid = false;
        errdefer graph.valid = false;
        if (!blocks_only) {
            try file.read(std.mem.sliceAsBytes(logits), &remaining);
            if (c.ds4_gpu_tensor_write(graph.logits, 0, logits.ptr, @as(u64, shape.n_vocab) * 4) == 0)
                return file.fail(error.Device, "failed to restore V4.1 device logits");
        }
        const scratch = allocator.alignedAlloc(u8, .@"8", io_chunk) catch c.ds4_engine_model_die("out of memory");
        defer allocator.free(scratch);
        var storage: [54]Span = undefined;
        const spans_ = source.spans(position, &storage);
        for (spans_[(if (blocks_only) shape.n_layer else 0)..]) |span| try readSpan(file, span, scratch, &remaining);
        resources.resetRecord(graph);
        // resetRecord clears drafter tags. Restore them only after every
        // accepted target row has been installed, never the transient draft KV.
        if (has_draft) try self.readDraft(file, position, scratch, &remaining);
        const values = tokens.items();
        for (0..@min(@as(u32, 3), position)) |index|
            graph.history.tail[index] = @intCast(graph.token_map[@intCast(values[position - 1 - index])]);
        graph.pos = position;
        return tokens;
    }

    fn writeDraft(self: *const Payload, file: File, scratch: []align(8) u8) Failure!void {
        const state: *c.ds41_dspark = @ptrCast(self.graph.dspark.?);
        const weights: *const c.ds4_dspark_weights = @ptrCast(state.weights);
        const position = self.graph.pos;
        const width = self.config.shape.n_embd;
        const rows = draft_checkpoint.rows(position);
        const layers: [3]u32 = weights.target_layers[0..3].*;
        for (draft_checkpoint.header(position, width, layers)) |word| try file.writeU32(word);
        const bytes = @as(u64, rows) * width * 2;
        const encoded = c.ds4_gpu_tensor_alloc(bytes) orelse return file.fail(error.Device, "cannot allocate draft feature packing tensor");
        defer c.ds4_gpu_tensor_free(encoded);
        for (state.target) |target| {
            try transferDraft(file, encoded, target, position, width, true);
            if (bytes > scratch.len or c.ds4_gpu_tensor_read(encoded, 0, scratch.ptr, bytes) == 0)
                return file.fail(error.Device, "failed to read packed draft features");
            try file.write(scratch[0..@intCast(bytes)]);
        }
    }

    fn readDraft(self: *const Payload, file: File, position: u32, scratch: []align(8) u8, remaining: *u64) Failure!void {
        const width = self.config.shape.n_embd;
        var header_: draft_checkpoint.Header = undefined;
        for (&header_) |*word| word.* = try file.readU32(remaining);
        if (!draft_checkpoint.validHeader(header_, position, width))
            return file.fail(error.InvalidDimensions, "invalid V4.1 draft-feature appendix");
        const state: ?*c.ds41_dspark = if (self.graph.dspark) |draft| @ptrCast(draft) else null;
        var restore = self.include_draft and state != null;
        if (state) |draft| {
            const weights: *const c.ds4_dspark_weights = @ptrCast(draft.weights);
            restore = restore and weights.target_layer_count == 3 and std.mem.eql(u32, header_[5..8], weights.target_layers[0..3]);
        }
        const bytes = @as(u64, draft_checkpoint.rows(position)) * width * 2;
        if (bytes > scratch.len) return file.fail(error.ConversionBuffer, "draft-feature conversion buffer is too small");
        const encoded = if (restore) c.ds4_gpu_tensor_alloc(bytes) else null;
        if (restore and encoded == null) return file.fail(error.Device, "cannot allocate draft feature restore tensor");
        defer c.ds4_gpu_tensor_free(encoded);
        for (0..3) |stage| {
            try file.read(scratch[0..@intCast(bytes)], remaining);
            if (restore) {
                if (c.ds4_gpu_tensor_write(encoded, 0, scratch.ptr, bytes) == 0)
                    return file.fail(error.Device, "failed to upload packed draft features");
                try transferDraft(file, encoded, state.?.target[stage], position, width, false);
            }
        }
        if (restore) {
            const start = position - draft_checkpoint.rows(position);
            for (&state.?.positions) |*tags| {
                for (start..position) |p| tags[p % 128] = @intCast(p);
            }
            // Projected keys are cheap to rebuild from these exact BF16
            // features on the very next draft; stale verify/rewind state stays reset.
            state.?.kv_valid = false;
            state.?.kv_end = 0;
        }
    }
};

// Convert the two ring segments into one chronological BF16 stage (or back).
// capture() already rounds target features to BF16; bit packing is lossless.
fn transferDraft(file: File, encoded: ?*Tensor, target: ?*Tensor, position: u32, width: u32, pack: bool) Failure!void {
    const rows = draft_checkpoint.rows(position);
    const start = position - rows;
    const slot = start % 128;
    const first = @min(rows, 128 - slot);
    const row_bytes = @as(u64, width) * 4;
    const head = c.ds4_gpu_tensor_view(target, @as(u64, slot) * row_bytes, @as(u64, first) * row_bytes);
    defer c.ds4_gpu_tensor_free(head);
    if (head == null or c.ds4_gpu_dsv41_carry_copy(encoded, 0, head, width, first, c.DS4_V41_CARRY_BF16, pack) == 0)
        return file.fail(error.Device, "failed to convert draft feature ring");
    if (first != rows) {
        const tail = c.ds4_gpu_tensor_view(target, 0, @as(u64, rows - first) * row_bytes);
        defer c.ds4_gpu_tensor_free(tail);
        if (tail == null or c.ds4_gpu_dsv41_carry_copy(encoded, first, tail, width, rows - first, c.DS4_V41_CARRY_BF16, pack) == 0)
            return file.fail(error.Device, "failed to convert wrapped draft feature ring");
    }
}

fn checkSpan(file: File, span: Span) Failure!void {
    const tensor = span.tensor orelse return file.fail(error.InvalidDimensions, if (span.packed_f16)
        "session tensor is smaller than the F16 payload"
    else
        "session tensor is smaller than the payload");
    const wire_row = format.indexRowBytes(span.wire_fp4);
    const storage_bytes = if (span.main_fp4 or span.main_wire_fp4)
        span.wire_bytes / format.mainRowBytes(span.main_wire_fp4) * @as(u64, if (span.main_fp4) 288 else if (span.packed_f16) 1024 else 2048)
    else if (span.index_fp4 or span.wire_fp4)
        span.wire_bytes / wire_row * format.indexRowBytes(span.index_fp4)
    else if (span.packed_f16) span.wire_bytes / 2 else span.wire_bytes;
    if ((span.index_fp4 or span.wire_fp4) and span.wire_bytes % wire_row != 0)
        return file.fail(error.InvalidDimensions, "invalid packed index row size");
    if ((span.main_fp4 or span.main_wire_fp4) and span.wire_bytes % format.mainRowBytes(span.main_wire_fp4) != 0)
        return file.fail(error.InvalidDimensions, "invalid packed main row size");
    if (span.wire_bytes % 4 != 0 or storage_bytes > c.ds4_gpu_tensor_bytes(tensor))
        return file.fail(error.InvalidDimensions, if (span.packed_f16)
            "session tensor is smaller than the F16 payload"
        else
            "session tensor is smaller than the payload");
}

fn writeSpan(file: File, span: Span, scratch: []align(8) u8) Failure!void {
    @setFloatMode(.strict);
    try checkSpan(file, span);
    if (span.main_fp4 != span.main_wire_fp4 or (span.packed_f16 and span.main_wire_fp4)) return transferMain(true, file, span, scratch, null);
    if (span.index_fp4 != span.wire_fp4) return transferIndex(true, file, span, scratch, null);
    if (!span.packed_f16) {
        var done: u64 = 0;
        while (done < span.wire_bytes) {
            const count: usize = @intCast(@min(span.wire_bytes - done, scratch.len));
            if (c.ds4_gpu_tensor_read(span.tensor, done, scratch.ptr, count) == 0)
                return file.fail(error.Device, "failed to read accelerator session tensor");
            try file.write(scratch[0..count]);
            done += count;
        }
        return;
    }
    // Even capacity leaves the F32 region naturally aligned after the F16
    // region. Both fit inside the same fixed scratch, without a float mirror.
    const capacity = scratch.len / 6 & ~@as(usize, 1);
    if (capacity == 0) return file.fail(error.ConversionBuffer, "session tensor conversion buffer is too small");
    const half: [*]u16 = @ptrCast(scratch.ptr);
    const full: [*]f32 = @ptrCast(@alignCast(scratch.ptr + capacity * 2));
    const values = span.wire_bytes / 4;
    var done: u64 = 0;
    while (done < values) {
        const count: usize = @intCast(@min(values - done, capacity));
        if (c.ds4_gpu_tensor_read(span.tensor, done * 2, half, count * 2) == 0)
            return file.fail(error.Device, "failed to read Metal F16 session tensor");
        for (0..count) |index| full[index] = @floatCast(@as(f16, @bitCast(half[index])));
        try file.write(std.mem.sliceAsBytes(full[0..count]));
        done += count;
    }
}

fn readSpan(file: File, span: Span, scratch: []align(8) u8, remaining: *u64) Failure!void {
    @setFloatMode(.strict);
    try checkSpan(file, span);
    if (span.main_fp4 != span.main_wire_fp4 or (span.packed_f16 and span.main_wire_fp4)) return transferMain(false, file, span, scratch, remaining);
    if (span.index_fp4 != span.wire_fp4) return transferIndex(false, file, span, scratch, remaining);
    if (!span.packed_f16) {
        var done: u64 = 0;
        while (done < span.wire_bytes) {
            const count: usize = @intCast(@min(span.wire_bytes - done, scratch.len));
            try file.read(scratch[0..count], remaining);
            if (c.ds4_gpu_tensor_write(span.tensor, done, scratch.ptr, count) == 0)
                return file.fail(error.Device, "failed to restore accelerator session tensor");
            done += count;
        }
        return;
    }
    const capacity = scratch.len / 6 & ~@as(usize, 1);
    if (capacity == 0) return file.fail(error.ConversionBuffer, "session tensor conversion buffer is too small");
    const half: [*]u16 = @ptrCast(scratch.ptr);
    const full: [*]f32 = @ptrCast(@alignCast(scratch.ptr + capacity * 2));
    const values = span.wire_bytes / 4;
    var done: u64 = 0;
    while (done < values) {
        const count: usize = @intCast(@min(values - done, capacity));
        try file.read(std.mem.sliceAsBytes(full[0..count]), remaining);
        for (0..count) |index| half[index] = @bitCast(@as(f16, @floatCast(full[index])));
        if (c.ds4_gpu_tensor_write(span.tensor, done * 2, half, count * 2) == 0)
            return file.fail(error.Device, "failed to restore Metal F16 session tensor");
        done += count;
    }
}

// Old F32/F16 payloads and new packed rows convert through bounded tiles.
// Repacking decoded mode3 values is separate from fresh quantization: coarse
// subnormal E4M3 scale rounding can leave a maximum of 4*scale, not 6*scale.
fn transferMain(comptime writing: bool, file: File, span: Span, scratch: []align(8) u8, remaining: ?*u64) Failure!void {
    if (span.wire_bytes == 0) return;
    const capacity = scratch.len / 2048;
    if (capacity == 0) return file.fail(error.ConversionBuffer, "main conversion buffer is too small");
    const wire_row = format.mainRowBytes(span.main_wire_fp4);
    const stored_row: u64 = if (span.main_fp4) 288 else if (span.packed_f16) 1024 else 2048;
    const full = c.ds4_gpu_tensor_alloc(capacity * 2048) orelse return file.fail(error.Device, "cannot allocate main conversion tile");
    defer c.ds4_gpu_tensor_free(full);
    const packed_tile = c.ds4_gpu_tensor_alloc(capacity * 288) orelse return file.fail(error.Device, "cannot allocate packed main conversion tile");
    defer c.ds4_gpu_tensor_free(packed_tile);
    const rows = span.wire_bytes / wire_row;
    var done: u64 = 0;
    while (done < rows) {
        const count: u32 = @intCast(@min(rows - done, capacity));
        const bytes: usize = @intCast(@as(u64, count) * wire_row);
        const view = c.ds4_gpu_tensor_view(span.tensor, done * stored_row, @as(u64, count) * stored_row) orelse return file.fail(error.Device, "cannot view main conversion rows");
        defer c.ds4_gpu_tensor_free(view);
        if (writing) {
            var converted: c_int = 0;
            if (span.main_wire_fp4) {
                var source = view;
                if (span.packed_f16) {
                    const values = @as(usize, count) * 512;
                    if (c.ds4_gpu_tensor_read(view, 0, scratch.ptr, values * 2) == 0) return file.fail(error.Device, "cannot read F16 main conversion rows");
                    const halves: [*]u16 = @ptrCast(scratch.ptr);
                    const floats: [*]f32 = @ptrCast(@alignCast(scratch.ptr));
                    var index = values;
                    while (index != 0) {
                        index -= 1;
                        floats[index] = @floatCast(@as(f16, @bitCast(halves[index])));
                    }
                    if (c.ds4_gpu_tensor_write(full, 0, scratch.ptr, values * 4) == 0) return file.fail(error.Device, "cannot upload F16 main conversion rows");
                    source = full;
                }
                converted = c.ds4_gpu_dsv41_main_repack_fp4(packed_tile, source, count);
            } else converted = c.ds4_gpu_dsv41_main_unpack_fp4(full, view, count);
            if (converted == 0 or c.ds4_gpu_tensor_read(if (span.main_wire_fp4) packed_tile else full, 0, scratch.ptr, bytes) == 0)
                return file.fail(error.Device, "failed to encode session main rows");
            try file.write(scratch[0..bytes]);
        } else {
            try file.read(scratch[0..bytes], remaining.?);
            if (c.ds4_gpu_tensor_write(if (span.main_wire_fp4) packed_tile else full, 0, scratch.ptr, bytes) == 0)
                return file.fail(error.Device, "failed to upload session main rows");
            const converted = if (span.main_fp4)
                c.ds4_gpu_dsv41_main_repack_fp4(view, full, count) != 0
            else if (span.packed_f16)
                c.ds4_gpu_dsv41_main_unpack_fp4(full, packed_tile, count) != 0 and c.ds4_gpu_tensor_copy_f32_to_f16(view, 0, full, 0, @as(u64, count) * 512) != 0
            else
                c.ds4_gpu_dsv41_main_unpack_fp4(view, packed_tile, count) != 0;
            if (!converted or c.ds4_gpu_synchronize() == 0)
                return file.fail(error.Device, "failed to restore session main rows");
        }
        done += count;
    }
}

// The temporary holds only one wire-format tile. There is no persistent F32
// mirror of packed graph state, and ordinary version-3 saves copy bytes directly.
fn transferIndex(comptime writing: bool, file: File, span: Span, scratch: []align(8) u8, remaining: ?*u64) Failure!void {
    if (span.wire_bytes == 0) return;
    const capacity = scratch.len / format.f32_index_bytes;
    if (capacity == 0) return file.fail(error.ConversionBuffer, "index conversion buffer is too small");
    const wire_row = format.indexRowBytes(span.wire_fp4);
    const stored_row = format.indexRowBytes(span.index_fp4);
    const temporary = c.ds4_gpu_tensor_alloc(capacity * wire_row) orelse
        return file.fail(error.Device, "cannot allocate bounded index conversion tile");
    defer c.ds4_gpu_tensor_free(temporary);
    const rows = span.wire_bytes / wire_row;
    var done: u64 = 0;
    while (done < rows) {
        const count: u32 = @intCast(@min(rows - done, capacity));
        const bytes: usize = @intCast(@as(u64, count) * wire_row);
        const view = c.ds4_gpu_tensor_view(span.tensor, done * stored_row, @as(u64, count) * stored_row) orelse
            return file.fail(error.Device, "cannot view index conversion rows");
        defer c.ds4_gpu_tensor_free(view);
        if (writing) {
            const converted = if (span.wire_fp4)
                c.ds4_gpu_dsv41_indexer_pack_fp4(temporary, view, count)
            else
                c.ds4_gpu_dsv41_indexer_unpack_fp4(temporary, view, count);
            if (converted == 0 or c.ds4_gpu_tensor_read(temporary, 0, scratch.ptr, bytes) == 0)
                return file.fail(error.Device, "failed to encode session index rows");
            try file.write(scratch[0..bytes]);
        } else {
            try file.read(scratch[0..bytes], remaining.?);
            if (c.ds4_gpu_tensor_write(temporary, 0, scratch.ptr, bytes) == 0)
                return file.fail(error.Device, "failed to upload session index rows");
            const converted = if (span.index_fp4)
                c.ds4_gpu_dsv41_indexer_pack_fp4(view, temporary, count)
            else
                c.ds4_gpu_dsv41_indexer_unpack_fp4(view, temporary, count);
            if (converted == 0 or c.ds4_gpu_synchronize() == 0)
                return file.fail(error.Device, "failed to restore session index rows");
        }
        done += count;
    }
}
