//! V4.1 canonical checkpoints, memory snapshots and private replay envelopes.
//! The actual Frontier and Graph remain borrowed. This owner commits tokens
//! and provenance only at the same boundaries as the original public APIs;
//! Payload owns the numerical spans and canonical F32 wire representation.
//! Distributed/TP routing and the outer legacy invalidate action remain C.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const payload = @import("state_payload.zig");
const policy = @import("prefill_policy.zig");
const Snapshot = c.ds4_session_snapshot;
const buffers = @import("../snapshot_buffer.zig");
const header_bytes: u64 = c.DS4_SESSION_PAYLOAD_U32_FIELDS * 4;
const replay_header_bytes: u64 = c.DS4_SESSION_REPLAY_SNAPSHOT_HEADER_BYTES;

fn fail(errors: []u8, message: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return 1;
}
fn openMemory(pointer: [*c]u8, bytes: u64, mode: [*:0]const u8) ?*c.FILE {
    const stream = c.fmemopen(pointer, @intCast(bytes), mode);
    return if (stream == null) null else @ptrCast(stream);
}
fn reserve(snapshot: *Snapshot, bytes: u64, errors: []u8, message: []const u8) bool {
    if (snapshot.cap >= bytes) return true;
    const memory = c.realloc(snapshot.ptr, @intCast(bytes)) orelse {
        _ = fail(errors, message);
        return false;
    };
    snapshot.ptr = @ptrCast(memory);
    snapshot.cap = bytes;
    return true;
}
fn readHeader(stream: *c.FILE, remaining: *u64, header: *payload.Header, errors: []u8) bool {
    for (header) |*value| {
        if (remaining.* < 4) {
            _ = fail(errors, "truncated session payload");
            return false;
        }
        var bytes: [4]u8 = undefined;
        if (c.fread(&bytes, 1, bytes.len, stream) != bytes.len) {
            _ = fail(errors, "failed to read session payload");
            return false;
        }
        remaining.* -= 4;
        value.* = std.mem.readInt(u32, &bytes, .little);
    }
    if (header[0] != c.DS4_SESSION_PAYLOAD_MAGIC or header[1] != c.DS4_SESSION_PAYLOAD_VERSION) {
        _ = fail(errors, "unsupported session payload version");
        return false;
    }
    return true;
}
fn writeReplayHeader(stream: *c.FILE, header: [8]u32, errors: []u8) bool {
    for (header) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        if (c.fwrite(&bytes, 1, bytes.len, stream) != bytes.len) {
            _ = fail(errors, "failed to write session payload");
            return false;
        }
    }
    return true;
}

pub const Checkpoint = struct {
    frontier: *c.ds4_session_frontier,
    graph: *c.ds41_gpu_graph,
    config: *const Config,
    ready: bool,

    fn state(self: *const Checkpoint) payload.Payload {
        return .{ .graph = self.graph, .config = self.config };
    }
    fn tokens(self: *const Checkpoint) []const c_int {
        const checkpoint = &self.frontier.checkpoint;
        return if (checkpoint.len == 0) &.{} else checkpoint.v[0..@intCast(checkpoint.len)];
    }
    fn logits(self: *const Checkpoint) []f32 {
        return self.frontier.logits[0..self.config.shape.n_vocab];
    }

    pub fn canonicalBytes(self: *const Checkpoint) u64 {
        const frontier = self.frontier;
        if (frontier.approximate_suffix or !frontier.checkpoint_valid or !self.ready or !self.graph.valid or
            self.graph.pos != @as(u32, @bitCast(frontier.checkpoint.len))) return 0;
        return header_bytes + self.state().bodyBytes(@intCast(frontier.checkpoint.len));
    }
    fn saveWire(self: *const Checkpoint, stream: *c.FILE, errors: []u8) c_int {
        if (!self.ready) return fail(errors, "V4.1 graph has no complete frontier to save");
        self.state().save(stream, self.tokens(), self.logits(), errors) catch return 1;
        return 0;
    }
    pub fn saveCanonical(self: *const Checkpoint, maybe_stream: ?*c.FILE, errors: []u8) c_int {
        if (self.frontier.approximate_suffix)
            return fail(errors, "private approximate suffix state cannot be exported as canonical KV");
        const stream = maybe_stream orelse return fail(errors, "session has no valid checkpoint to save");
        if (!self.frontier.checkpoint_valid) return fail(errors, "session has no valid checkpoint to save");
        return self.saveWire(stream, errors);
    }
    fn loadWire(self: *const Checkpoint, stream: *c.FILE, bytes: u64, errors: []u8) c_int {
        var remaining = bytes;
        var header: payload.Header = undefined;
        if (!readHeader(stream, &remaining, &header, errors)) return 1;
        if (!self.ready) return fail(errors, "invalid V4.1 snapshot dimensions or size");
        var mutated = false;
        var loaded = self.state().load(stream, &header, remaining, self.logits(), &mutated, errors) catch {
            // Invalid headers/tokens preserve the old frontier. Once device
            // writes can occur, no old checkpoint may remain exportable.
            if (mutated) self.frontier.checkpoint_valid = false;
            return 1;
        };
        defer loaded.deinit();
        c.ds4_tokens_copy(&self.frontier.checkpoint, &loaded.record);
        self.frontier.checkpoint_valid = true;
        return 0;
    }
    pub fn loadCanonical(self: *const Checkpoint, maybe_stream: ?*c.FILE, bytes: u64, errors: []u8) c_int {
        const stream = maybe_stream orelse return fail(errors, "invalid session payload load");
        const result = self.loadWire(stream, bytes, errors);
        if (result == 0) {
            self.frontier.overthink_prompt_len = self.frontier.checkpoint.len;
            self.frontier.approximate_suffix = false;
            self.frontier.approximate_decoder_replay = false;
            self.frontier.decoder_replay_rows = 0;
            self.frontier.decoder_replay_selected = false;
        }
        return result;
    }

    fn replaySupported(self: *const Checkpoint) bool {
        const graph = self.graph;
        return self.ready and self.frontier.checkpoint_image_count == 0 and self.frontier.sync_image_count == 0 and
            !graph.workspace_failed and graph.tp_world == 1 and graph.dspark == null and
            !graph.quality and graph.imatrix == null and graph.image_count == 0 and
            !graph.predict_active and !graph.predict_load.active;
    }
    pub fn saveSnapshot(self: *const Checkpoint, snapshot: ?*Snapshot, errors: []u8) c_int {
        return buffers.save(c, self, snapshot, errors);
    }
    pub fn loadSnapshot(self: *const Checkpoint, snapshot: ?*const Snapshot, errors: []u8) c_int {
        return buffers.load(c, self, snapshot, errors);
    }

    pub fn replayBytes(self: *const Checkpoint) u64 {
        const frontier = self.frontier;
        if (!self.replaySupported() or !frontier.approximate_decoder_replay or !frontier.approximate_suffix or
            !frontier.checkpoint_valid or !self.graph.valid or frontier.checkpoint.len == 0 or
            self.graph.pos != @as(u32, @bitCast(frontier.checkpoint.len)) or
            (frontier.decoder_replay_rows != 128 and frontier.decoder_replay_rows != 256)) return 0;
        return replay_header_bytes + header_bytes + self.state().bodyBytes(self.graph.pos);
    }
    pub fn saveReplay(self: *const Checkpoint, maybe_snapshot: ?*Snapshot, errors: []u8) c_int {
        if (maybe_snapshot) |snapshot| snapshot.len = 0;
        const bytes = self.replayBytes();
        const snapshot = maybe_snapshot orelse return fail(errors, "no complete decoder replay checkpoint to save");
        if (bytes == 0 or bytes >= std.math.maxInt(usize)) return fail(errors, "no complete decoder replay checkpoint to save");
        if (!reserve(snapshot, bytes + 1, errors, "cannot allocate decoder replay checkpoint")) return 1;
        const body = bytes - replay_header_bytes;
        const header: [8]u32 = .{
            c.DS4_SESSION_REPLAY_SNAPSHOT_MAGIC,                 c.DS4_SESSION_REPLAY_SNAPSHOT_VERSION,
            c.DS4_SESSION_REPLAY_SNAPSHOT_KIND,                  self.frontier.decoder_replay_rows,
            @intFromBool(self.frontier.decoder_replay_selected), @truncate(body),
            @truncate(body >> 32),                               0,
        };
        const stream = openMemory(snapshot.ptr, bytes + 1, "wb") orelse
            return fail(errors, "cannot open decoder replay checkpoint");
        var result: c_int = if (writeReplayHeader(stream, header, errors)) self.saveWire(stream, errors) else 1;
        if (c.fclose(stream) != 0 and result == 0) result = fail(errors, "cannot finalize decoder replay checkpoint");
        if (result == 0) snapshot.len = bytes;
        return result;
    }
    /// invalidate is a whole outer cleanup request, not a field callback. The
    /// C facade still resets its legacy draft/capture state on private failure.
    /// Header/policy rejection before opening never mutates the old frontier.
    pub fn loadReplay(self: *const Checkpoint, maybe_snapshot: ?*const Snapshot, invalidate: *bool, errors: []u8) c_int {
        invalidate.* = false;
        if (!self.replaySupported()) return fail(errors, "invalid decoder replay checkpoint");
        const snapshot = maybe_snapshot orelse return fail(errors, "invalid decoder replay checkpoint");
        if (snapshot.ptr == null or snapshot.len > snapshot.cap or snapshot.len > std.math.maxInt(usize) or
            snapshot.len < replay_header_bytes + header_bytes) return fail(errors, "invalid decoder replay checkpoint");
        const bytes: [*]u8 = @ptrCast(snapshot.ptr);
        var header: [8]u32 = undefined;
        for (&header, 0..) |*value, index| value.* = std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
        const body = @as(u64, header[5]) | @as(u64, header[6]) << 32;
        const inner = bytes + replay_header_bytes;
        if (header[0] != c.DS4_SESSION_REPLAY_SNAPSHOT_MAGIC or header[1] != c.DS4_SESSION_REPLAY_SNAPSHOT_VERSION or
            header[2] != c.DS4_SESSION_REPLAY_SNAPSHOT_KIND or (header[3] != 128 and header[3] != 256) or
            header[4] & ~@as(u32, 1) != 0 or header[7] != 0 or body != snapshot.len - replay_header_bytes or
            header[3] != policy.replaySetting() or (header[4] != 0) != policy.replaySelected() or
            std.mem.readInt(u32, inner[0..4], .little) != c.DS4_SESSION_PAYLOAD_MAGIC or
            std.mem.readInt(u32, inner[4..8], .little) != c.DS4_SESSION_PAYLOAD_VERSION or
            std.mem.readInt(u32, inner[8..12], .little) != self.graph.ctx)
            return fail(errors, "decoder replay checkpoint policy or layout mismatch");
        const stream = openMemory(inner, body, "rb") orelse return fail(errors, "cannot open decoder replay checkpoint restore");
        // Keep canonical exports blocked DURING restoration, even on errors.
        self.frontier.approximate_suffix = true;
        self.frontier.approximate_decoder_replay = true;
        self.frontier.decoder_replay_rows = header[3];
        self.frontier.decoder_replay_selected = header[4] != 0;
        invalidate.* = true;
        var result = self.loadWire(stream, body, errors);
        if (c.fclose(stream) != 0 and result == 0) result = fail(errors, "cannot close decoder replay checkpoint restore");
        if (result != 0) return 1;
        invalidate.* = false;
        self.frontier.overthink_prompt_len = self.frontier.checkpoint.len;
        self.frontier.mtp_draft_valid = false;
        return 0;
    }
};

pub fn freeSnapshot(snapshot: ?*Snapshot) void {
    buffers.free(c, snapshot);
}
