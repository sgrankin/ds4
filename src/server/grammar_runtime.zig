//! Lazy vocabulary compiler owned by the serving engine, and one matcher per
//! generation. No-constraint requests allocate nothing and keep their decoder.
const std = @import("std");
const Native = @import("grammar");
const Format = @import("grammar_format.zig");
const allocator = std.heap.c_allocator;
const Registry = struct {
    compiler: Native.Compiler,
    ids: Format.Ids,
};
pub fn close(pointer: ?*anyopaque) void {
    const registry: *Registry = @ptrCast(@alignCast(pointer orelse return));
    registry.compiler.deinit();
    allocator.destroy(registry);
}
fn registryFor(comptime c: type, server: *c.server, diagnostic: *Native.Diagnostic) !*Registry {
    _ = c.pthread_mutex_lock(&server.tool_mu);
    defer _ = c.pthread_mutex_unlock(&server.tool_mu);
    if (server.grammar) |pointer| return @ptrCast(@alignCast(pointer));
    const count: usize = @intCast(c.ds4_engine_vocab_size(server.engine));
    if (count == 0) return error.EmptyVocabulary;
    const vocabulary = try allocator.alloc([]const u8, count);
    defer allocator.free(vocabulary);
    var initialized: usize = 0;
    defer for (vocabulary[0..initialized]) |piece| allocator.free(piece);
    for (vocabulary, 0..) |*piece, index| {
        var length: usize = 0;
        const raw = c.ds4_token_text(server.engine, @intCast(index), &length);
        defer c.free(raw);
        piece.* = try allocator.dupe(u8, if (raw == null) "" else raw[0..length]);
        initialized += 1;
    }
    var ids: Format.Ids = @splat(-1);
    var controls: std.ArrayList(i32) = .empty;
    defer controls.deinit(allocator);
    const flint = c.ds4_engine_is_flint(server.engine);
    for (Format.markers, 0..) |spelling, index| {
        const applicable = if (flint) (index <= 1 or (index >= 3 and index <= 8) or index >= 14) else (index <= 2 or (index >= 9 and index <= 13));
        if (!applicable) continue;
        const text = try allocator.dupeSentinel(u8, spelling, 0);
        defer allocator.free(text);
        var tokens = std.mem.zeroes(c.ds4_tokens);
        defer c.ds4_tokens_free(&tokens);
        c.ds4_tokenize_rendered_chat(server.engine, text.ptr, &tokens);
        if (tokens.len != 1) continue;
        ids[index] = tokens.v[0];
        if (std.mem.findScalar(i32, controls.items, tokens.v[0]) == null) try controls.append(allocator, tokens.v[0]);
    }
    const eos: i32 = c.ds4_token_eos(server.engine);
    var compiler = try Native.Compiler.initTokenAware(allocator, vocabulary, &.{eos}, controls.items, diagnostic);
    errdefer compiler.deinit();
    const result = try allocator.create(Registry);
    result.* = .{ .compiler = compiler, .ids = ids };
    server.grammar = result;
    return result;
}

pub const Session = struct {
    matcher: Native.Matcher,
    words: []u32,
    logits: []f32,
    diagnostic: Native.Diagnostic = .{},
    accepted: usize = 0,

    pub fn init(comptime c: type, server: *c.server, request: *const c.request, errors: []u8) !?Session {
        if (request.constraint_json == null) return null;
        var diagnostic = Native.Diagnostic{};
        errdefer {
            const reason = diagnostic.text();
            if (errors.len != 0) {
                const message = if (reason.len == 0) "Native output constraint preparation failed" else reason;
                const n = @min(message.len, errors.len - 1);
                @memcpy(errors[0..n], message[0..n]);
                errors[n] = 0;
            }
        }
        const registry = try registryFor(c, server, &diagnostic);
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const descriptor = try std.json.parseFromSliceLeaky(std.json.Value, a, std.mem.span(request.constraint_json), .{});
        var framed = descriptor;
        try framed.object.put(a, "parallel", .{ .bool = request.parallel_tool_calls });
        const source = try Format.build(a, framed, c.ds4_engine_is_flint(server.engine), registry.ids);
        var compiled = blk: {
            _ = c.pthread_mutex_lock(&server.tool_mu);
            defer _ = c.pthread_mutex_unlock(&server.tool_mu);
            // Non-strict schemas still need to be valid declarations, even
            // though their argument masks deliberately do not enforce them.
            for (descriptor.object.get("tools").?.array.items) |tool| {
                const function = tool.object.get("function").?;
                if (!function.object.get("strict").?.bool) {
                    const json = try std.json.Stringify.valueAlloc(a, function.object.get("parameters").?, .{});
                    var validation = try registry.compiler.compile(.json_schema, json, false, &diagnostic);
                    validation.deinit();
                }
            }
            break :blk try registry.compiler.compile(.structural, source, false, &diagnostic);
        };
        defer compiled.deinit();
        var matcher = try compiled.matcher(&diagnostic);
        errdefer matcher.deinit();
        const words = try allocator.alloc(u32, matcher.wordCount());
        errdefer allocator.free(words);
        const logits = try allocator.alloc(f32, compiled.vocab_size);
        return .{ .matcher = matcher, .words = words, .logits = logits };
    }
    pub fn deinit(self: *Session) void {
        self.matcher.deinit();
        allocator.free(self.words);
        allocator.free(self.logits);
        self.* = undefined;
    }

    /// Packets borrow the generation's matcher. The first input token has
    /// already been accepted by sample(); backends advance subsequent committed
    /// tokens and roll back any temporary draft walk before returning.
    pub fn callbacks(self: *Session, comptime c: type) c.ds4_token_constraint {
        const Bridge = struct {
            fn owner(context: ?*anyopaque) *Session {
                return @ptrCast(@alignCast(context.?));
            }
            fn mask(context: ?*anyopaque, words: [*c]u32, count: usize) callconv(.c) c_int {
                const session = owner(context);
                if (words == null or count != session.matcher.wordCount()) return -1;
                @memset(words[0..count], 0xffffffff);
                _ = session.matcher.fill(words[0..count], &session.diagnostic) catch return -1;
                return 0;
            }
            fn accept(context: ?*anyopaque, token: c_int) callconv(.c) c_int {
                owner(context).accept(token) catch return -1;
                return 0;
            }
            fn rollback(context: ?*anyopaque, count: usize) callconv(.c) c_int {
                owner(context).rollback(count) catch return -1;
                return 0;
            }
        };
        return .{ .context = self, .mask = Bridge.mask, .accept = Bridge.accept, .rollback = Bridge.rollback };
    }

    pub fn accept(self: *Session, token: i32) !void {
        if (!(try self.matcher.accept(token, &self.diagnostic))) return error.GrammarRejectedSample;
        self.accepted += 1;
    }

    pub fn rollback(self: *Session, count: usize) !void {
        if (count > self.accepted) return error.InvalidGrammarRollback;
        try self.matcher.rollback(count, &self.diagnostic);
        self.accepted -= count;
    }

    pub fn sample(self: *Session, comptime c: type, model: ?*c.ds4_session, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: *u64, forced: ?i32, ignore_eos: bool, engine: ?*c.ds4_engine, mode: c.ds4_think_mode) !i32 {
        if (model == null) return error.MissingSession;
        @memset(self.words, 0xffffffff);
        const constrained = try self.matcher.fill(self.words, &self.diagnostic);
        var selected: i32 = -2;
        if (forced == null and !ignore_eos and c.ds4_engine_is_flint(engine)) {
            selected = c.ds4_session_sample_masked(model, self.words.ptr, self.words.len, temperature, top_k, top_p, min_p, rng, &self.diagnostic.bytes, self.diagnostic.bytes.len);
            if (selected == -1) return error.NativeMaskedSampler;
        }
        if (selected == -2) {
            const copied = c.ds4_session_copy_sampling_logits(model, self.logits.ptr, @intCast(self.logits.len));
            if (copied < 0 or @as(usize, @intCast(copied)) != self.logits.len) return error.MissingLogits;
            var available = false;
            for (self.logits, 0..) |*value, index| {
                if ((constrained and !Native.allowed(self.words, index)) or (ignore_eos and c.ds4_token_is_stop_for_think_mode(engine, @intCast(index), mode))) value.* = -std.math.inf(f32);
                available = available or std.math.isFinite(value.*);
            }
            if (!available) return error.EmptyGrammarMask;
            selected = if (forced) |id| blk: {
                if (id < 0 or @as(usize, @intCast(id)) >= self.logits.len or !std.math.isFinite(self.logits[@intCast(id)])) return error.InvalidForcedToken;
                break :blk id;
            } else c.ds4_sample_logits(self.logits.ptr, @intCast(self.logits.len), temperature, top_k, top_p, min_p, rng);
        }
        if (selected < 0) return error.GrammarRejectedSample;
        try self.accept(selected);
        return selected;
    }
};

/// Validate before queueing/prefill; the immutable compiler cache serves the
/// worker's later matcher construction without recompiling the same request.
pub fn preflight(comptime c: type, server: *c.server, request: *const c.request, errors: []u8) bool {
    var session = Session.init(c, server, request, errors) catch return false;
    if (session) |*value| value.deinit();
    return true;
}
