//! Agent saved-file/live-session transaction. Worker owns sync, progress,
//! cancellation and final transcript/identity/status publication. This owner
//! holds only the actual open file, decoded metadata and borrowed Engine/Session.
const std = @import("std");
const c = @import("agent_sessions_c");
const sessions = @import("sessions.zig");
const allocator = std.heap.c_allocator;

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn bytes(pointer: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
fn message(errors: []u8, value: [*c]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", value);
}
fn systemError(errors: []u8) void {
    message(errors, c.strerror(c.__error().*));
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch allocationFailure("ds4-agent: malloc")).ptr;
}
fn allocationFailure(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn sameTokens(left: [*c]const c.ds4_tokens, right: *const c.ds4_tokens) bool {
    if (left == null or left.*.len != right.len) return false;
    var index: c_int = 0;
    while (index < right.len) : (index += 1) {
        if (left.*.v[@intCast(index)] != right.v[@intCast(index)]) return false;
    }
    return true;
}

pub fn requiresRebuild(payload_bytes: u64, tensor_parallel: bool) bool {
    return payload_bytes == 0 or tensor_parallel;
}

pub const Load = struct {
    engine: ?*c.ds4_engine,
    session: ?*c.ds4_session,
    stream: ?*c.FILE,
    header: c.ds4_kvstore_entry = std.mem.zeroes(c.ds4_kvstore_entry),
    rendered: [*c]u8 = null,
    title: [*c]u8 = null,
    text_bytes: u32 = 0,
    expected_tokens: u32 = 0,
    has_title: bool = false,
    rebuild: bool = false,

    pub fn deinit(self: *Load) void {
        if (self.stream) |stream| _ = c.fclose(stream);
        c.free(self.title);
        c.free(self.rendered);
        allocator.destroy(self);
    }

    /// sysprompt uses exact text because its path is fixed. Named sessions use
    /// title||creation-time identity, or rendered-text SHA for legacy files.
    /// Validation precedes any live payload write or rebuilt-token publication.
    pub fn prepare(engine: ?*c.ds4_engine, session: ?*c.ds4_session, path: [*c]const u8, expected_sha: [*c]const u8, expected_text: [*c]const u8, expected_text_length: usize, tensor_parallel: bool, rebuilt: *c.ds4_tokens, errors: []u8) ?*Load {
        const stream = c.fopen(path, "rb");
        if (stream == null) {
            systemError(errors);
            return null;
        }
        const self = allocator.create(Load) catch allocationFailure("ds4-agent: malloc");
        self.* = .{ .engine = engine, .session = session, .stream = @ptrCast(stream) };
        var accepted = false;
        defer if (!accepted) self.deinit();
        if (!c.ds4_kvstore_read_header(stream, &self.header, &self.text_bytes)) {
            message(errors, "invalid KV header");
            return null;
        }
        if (!sessions.readText(@ptrCast(stream), self.text_bytes, &self.rendered, errors)) return null;
        self.has_title = self.header.ext_flags & c.DS4_KVSTORE_EXT_SESSION_TITLE != 0;
        if (self.has_title and !sessions.readTitle(@ptrCast(stream), &self.header, &self.title, errors)) return null;
        self.expected_tokens = self.header.tokens;
        if (self.header.payload_bytes != 0 and self.header.model_id != @as(u8, @intCast(c.ds4_engine_model_id(engine)))) {
            message(errors, "KV checkpoint was written for a different model");
            return null;
        }
        if (self.header.payload_bytes != 0 and self.header.quant_bits != @as(u8, @intCast(c.ds4_engine_routed_quant_bits(engine)))) {
            message(errors, "KV checkpoint was written for a different quantization");
            return null;
        }
        if (expected_text != null and (self.text_bytes != expected_text_length or
            !std.mem.eql(u8, bytes(self.rendered, self.text_bytes), bytes(expected_text, expected_text_length))))
        {
            message(errors, "cached text does not match current system prompt");
            return null;
        }
        if (expected_sha != null) {
            var actual: [41]u8 = undefined;
            sessions.fileSha(&self.header, self.rendered, self.text_bytes, self.title, &actual);
            if (c.strcmp(&actual, expected_sha) != 0) {
                message(errors, "cached session identity does not match file name");
                return null;
            }
        }

        self.rebuild = requiresRebuild(self.header.payload_bytes, tensor_parallel);
        if (self.rebuild) {
            // A TP payload contains only the leader's graph. Rebuild through
            // ordinary sync so both ranks receive the same token prefix.
            c.ds4_tokenize_rendered_chat(engine, self.rendered, rebuilt);
            self.expected_tokens = @bitCast(rebuilt.len);
        } else {
            var load_error: [160]u8 = @splat(0);
            if (c.ds4_session_load_payload(session, stream, self.header.payload_bytes, &load_error, load_error.len) != 0) {
                message(errors, if (load_error[0] != 0) &load_error else "failed to load KV payload");
                c.ds4_session_invalidate(session);
                return null;
            }
        }
        accepted = true;
        return self;
    }

    /// Worker has completed ordinary sync when rebuild=true. Close before
    /// validating/publishing, just as the original file transaction did.
    pub fn finish(self: *Load, synchronized: bool, loaded: [*c]c.ds4_tokens, metadata: [*c]c.agent_kv_session_meta, errors: []u8) bool {
        if (self.stream) |stream| {
            _ = c.fclose(stream);
            self.stream = null;
        }
        if (!synchronized) return false;
        const live = c.ds4_session_tokens(self.session);
        const valid = live != null and live.*.len == @as(c_int, @bitCast(self.expected_tokens));
        if (!valid) {
            message(errors, "KV payload token count mismatch");
            c.ds4_session_invalidate(self.session);
        } else if (loaded != null) {
            c.ds4_tokens_free(loaded);
            c.ds4_tokens_copy(loaded, live);
        }
        // Preserve the existing publication boundary: after a successful load/
        // sync, metadata is supplied even if the live token-count check fails.
        if (metadata != null) {
            const result: *c.agent_kv_session_meta = @ptrCast(metadata);
            sessions.freeMeta(result);
            result.has_title_trailer = self.has_title;
            result.legacy_identity = !self.has_title;
            result.created_at = self.header.created_at;
            sessions.fileSha(&self.header, self.rendered, self.text_bytes, self.title, &result.sha);
            result.title = if (self.has_title) duplicate(text(self.title)) else sessions.titleText(self.rendered, self.text_bytes, 0);
        }
        return valid;
    }
};

/// Fixed sysprompt or SHA-named session is the caller's policy. Named full or
/// approximate transcripts use the existing zero-payload/stripped format: a
/// full transcript cannot sync because the backend needs generation room.
pub fn save(engine: ?*c.ds4_engine, session: ?*c.ds4_session, path: [*c]const u8, tokens: *const c.ds4_tokens, reason: [*c]const u8, sha_out: [*c]u8, title: [*c]const u8, session_created_at: u64, text_only: bool, errors: []u8) bool {
    const approximate = c.ds4_session_is_approximate(session);
    if (!text_only and !sameTokens(c.ds4_session_tokens(session), tokens)) {
        message(errors, "live KV state does not match session transcript");
        return false;
    }
    const quant_bits = c.ds4_engine_routed_quant_bits(engine);
    if (!c.ds4_kvstore_quant_bits_supported(quant_bits)) {
        message(errors, "unsupported routed quantization for KV save");
        return false;
    }
    const model_id = c.ds4_engine_model_id(engine);
    var text_length: usize = 0;
    const rendered = c.ds4_kvstore_render_tokens_text(engine, tokens, &text_length);
    if (rendered == null) {
        message(errors, "failed to render KV text key");
        return false;
    }
    var staged = std.mem.zeroes(c.ds4_session_payload_file);
    var temporary: [*c]u8 = null;
    defer {
        c.ds4_session_payload_file_free(&staged);
        c.free(temporary);
        c.free(rendered);
    }
    if (text_length > std.math.maxInt(u32)) {
        message(errors, "rendered KV text key is too large");
        return false;
    }
    if (approximate and text_only and !c.ds4_agent_context_text_safe(engine, tokens, rendered, text_length, errors.ptr, errors.len)) return false;
    const session_identity = title != null;
    const now: u64 = @bitCast(@as(i64, c.time(null)));
    const created_at = if (session_identity and session_created_at != 0) session_created_at else now;
    var sha: [41]u8 = undefined;
    if (session_identity) sessions.identitySha(title, created_at, &sha) else c.ds4_kvstore_sha1_bytes_hex(rendered, text_length, &sha);
    if (sha_out != null) @memcpy(sha_out[0..41], &sha);

    var save_error: [160]u8 = @splat(0);
    if (!text_only and c.ds4_session_stage_payload(session, &staged, &save_error, save_error.len) != 0) {
        message(errors, if (save_error[0] != 0) &save_error else "session has no valid KV payload");
        return false;
    }
    var template: std.ArrayList(u8) = .empty;
    defer template.deinit(allocator);
    template.appendSlice(allocator, text(path)) catch allocationFailure("ds4-agent: realloc");
    template.appendSlice(allocator, ".tmp.XXXXXX") catch allocationFailure("ds4-agent: realloc");
    temporary = (template.toOwnedSliceSentinel(allocator, 0) catch allocationFailure("ds4-agent: realloc")).ptr;
    const descriptor = c.mkstemp(temporary);
    if (descriptor < 0) {
        systemError(errors);
        return false;
    }
    const stream = c.fdopen(descriptor, "wb");
    if (stream == null) {
        systemError(errors);
        _ = c.close(descriptor);
        _ = c.unlink(temporary);
        return false;
    }
    var header: [c.DS4_KVSTORE_FIXED_HEADER]u8 = undefined;
    c.ds4_kvstore_fill_header(&header, @intCast(model_id), @intCast(quant_bits), c.ds4_kvstore_reason_code(reason), if (session_identity) c.DS4_KVSTORE_EXT_SESSION_TITLE else 0, @bitCast(tokens.len), 0, @bitCast(c.ds4_session_ctx(session)), created_at, now, staged.bytes);
    var encoded_length: [4]u8 = undefined;
    std.mem.writeInt(u32, &encoded_length, @intCast(text_length), .little);
    c.__error().* = 0;
    var ok = c.fwrite(&header, 1, header.len, stream) == header.len and
        c.fwrite(&encoded_length, 1, encoded_length.len, stream) == encoded_length.len and
        c.fwrite(rendered, 1, text_length, stream) == text_length and
        (text_only or c.ds4_session_write_staged_payload(&staged, stream, &save_error, save_error.len) == 0) and
        (!session_identity or sessions.writeTitle(@ptrCast(stream), title, &save_error)) and c.fflush(stream) == 0;
    var saved_errno = c.__error().*;
    if (c.fclose(stream) != 0) {
        if (saved_errno == 0) saved_errno = c.__error().*;
        ok = false;
    }
    if (ok and c.rename(temporary, path) != 0) {
        saved_errno = c.__error().*;
        ok = false;
    }
    if (!ok) {
        message(errors, if (saved_errno != 0) c.strerror(saved_errno) else if (save_error[0] != 0) &save_error else "failed to write KV file");
        _ = c.unlink(temporary);
    }
    return ok;
}
