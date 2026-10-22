//! Live-context text transactions. The worker owns graph/session decisions;
//! this cohesive owner stages/commits token text and stable turn metadata only.
const std = @import("std");
const c = @import("agent_context_c");
const allocator = std.heap.c_allocator;
const Record = c.ds4_agent_context;
const Edit = c.ds4_agent_context_edit;
const Turn = c.agent_live_context_turn;
const Invalid = error{InvalidContext};

fn tokenSlice(tokens: *const c.ds4_tokens) []const c_int {
    return if (tokens.len == 0) &.{} else @as([*]const c_int, @ptrCast(tokens.v))[0..@intCast(tokens.len)];
}
fn turnSlice(context: *const Record) []const Turn {
    return if (context.turn_count == 0) &.{} else @as([*]const Turn, @ptrCast(context.turns))[0..context.turn_count];
}
fn errorMessage(errors: []u8, text: [:0]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", text.ptr);
}
fn reject(errors: []u8, text: [:0]const u8) Invalid {
    errorMessage(errors, text);
    return error.InvalidContext;
}
fn fatal(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn allocateTurns(count: usize) []Turn {
    if (count == 0) return &.{};
    const pointer = c.malloc(count * @sizeOf(Turn)) orelse fatal("ds4-agent: malloc");
    return @as([*]Turn, @ptrCast(@alignCast(pointer)))[0..count];
}
fn usedMap(count: usize) []bool {
    if (count == 0) return &.{};
    const pointer = c.calloc(count, @sizeOf(bool)) orelse {
        _ = c.fprintf(c.__stderrp, "cannot allocate live context turn map\n");
        c.exit(1);
    };
    return @as([*]bool, @ptrCast(pointer))[0..count];
}
fn freeSlice(comptime T: type, values: []T) void {
    if (values.len != 0) c.free(values.ptr);
}

const Text = struct {
    list: std.ArrayList(u8) = .empty,

    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal("ds4-agent: realloc");
    }
    fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal("ds4-agent: malloc")).ptr;
    }
};
const Rendered = struct {
    pointer: [*c]u8,
    len: usize,

    fn init(engine: *c.ds4_engine, tokens: *const c.ds4_tokens) ?Rendered {
        var len: usize = 0;
        const text = c.ds4_kvstore_render_tokens_text(engine, tokens, &len);
        return if (text == null) null else .{ .pointer = text, .len = len };
    }
    fn deinit(self: Rendered) void {
        c.free(self.pointer);
    }
    fn bytes(self: Rendered) []const u8 {
        return self.pointer[0..self.len];
    }
};

pub fn clear(context: *Record) void {
    c.free(context.turns);
    c.ds4_tokens_free(&context.snapshot);
    context.* = std.mem.zeroes(Record);
}
fn setTurns(context: *Record, transcript: *const c.ds4_tokens, turns: [*c]Turn, count: usize) void {
    clear(context);
    context.turns = turns;
    context.turn_count = count;
    c.ds4_tokens_copy(&context.snapshot, transcript);
}

/// Quote only lines that could be display metadata. One extra backslash makes
/// an already quoted literal header line reversible as well.
pub fn bodyText(text: []const u8, quote: bool) [*c]u8 {
    var result = Text{};
    defer result.deinit();
    var at: usize = 0;
    while (at < text.len) {
        const end = std.mem.findScalarPos(u8, text, at, '\n') orelse text.len;
        var first = at;
        while (first < end and text[first] == '\\') first += 1;
        const reserved = std.mem.startsWith(u8, text[first..end], "[[DS4_TURN ");
        if (quote and reserved) result.append("\\");
        const start = at + @intFromBool(!quote and reserved and first > at);
        result.append(text[start..end]);
        if (end < text.len) result.append("\n");
        at = end + @intFromBool(end < text.len);
    }
    return result.take();
}

pub const Context = struct {
    engine: *c.ds4_engine,
    transcript: *c.ds4_tokens,
    state: *Record,

    /// Labels come only from real role token IDs. Retained complete turns keep
    /// IDs across edits/compaction; the last open turn keeps its ID on append.
    pub fn render(self: Context, system: *const c.ds4_tokens, out_len: ?*usize) ?[*c]u8 {
        if (!c.ds4_tokens_starts_with(self.transcript, system)) return null;
        const ids = tokenSlice(self.transcript);
        const old_ids = tokenSlice(&self.state.snapshot);
        const previous = turnSlice(self.state);
        const user = c.ds4_token_user(self.engine);
        const assistant = c.ds4_token_assistant(self.engine);
        var count: usize = 0;
        for (ids[@intCast(system.len)..]) |id| {
            if (id == user or id == assistant) count += 1;
        }
        const turns = allocateTurns(count);
        var retained = false;
        defer if (!retained) freeSlice(Turn, turns);
        const used = usedMap(previous.len);
        defer freeSlice(bool, used);
        var at: usize = 0;
        for (ids, 0..) |id, index| {
            if (index < @as(usize, @intCast(system.len)) or (id != user and id != assistant)) continue;
            if (at != 0) turns[at - 1].end = @intCast(index);
            turns[at] = .{ .id = 0, .start = @intCast(index), .end = self.transcript.len, .role_token = id };
            at += 1;
        }
        var common: usize = 0;
        while (common < old_ids.len and common < ids.len and old_ids[common] == ids[common]) common += 1;
        var tail: usize = 0;
        while (tail < old_ids.len - common and tail < ids.len - common and old_ids[old_ids.len - tail - 1] == ids[ids.len - tail - 1]) tail += 1;
        for (turns, 0..) |*turn, index| {
            const preferred: c_int = if (turn.end <= @as(c_int, @intCast(common))) turn.start else if (turn.start >= self.transcript.len - @as(c_int, @intCast(tail))) turn.start + self.state.snapshot.len - self.transcript.len else -1;
            // Positional prefix/tail matches take priority over identical
            // discarded twins elsewhere in the old snapshot.
            for (0..2) |pass| {
                if (turn.id != 0) break;
                for (previous, 0..) |old, old_index| {
                    const length = old.end - old.start;
                    if ((pass == 0 and old.start != preferred) or used[old_index] or old.role_token != turn.role_token or old.end > self.state.snapshot.len or length != turn.end - turn.start) continue;
                    if (!std.mem.eql(c_int, old_ids[@intCast(old.start)..@intCast(old.end)], ids[@intCast(turn.start)..@intCast(turn.end)])) continue;
                    turn.id = old.id;
                    used[old_index] = true;
                    break;
                }
            }
            if (turn.id == 0 and index + 1 == turns.len and previous.len != 0) {
                const last = previous.len - 1;
                const old = previous[last];
                const length = old.end - old.start;
                if (!used[last] and old.role_token == turn.role_token and old.start == turn.start and old.end <= self.state.snapshot.len and length <= turn.end - turn.start and std.mem.eql(c_int, old_ids[@intCast(old.start)..@intCast(old.end)], ids[@intCast(turn.start)..][0..@intCast(length)])) {
                    turn.id = old.id;
                    used[last] = true;
                }
            }
            while (turn.id == 0) {
                if (c.getentropy(&turn.id, @sizeOf(u64)) != 0) fatal("cannot generate live context turn ID");
                var duplicate = turn.id == 0;
                for (previous) |old| {
                    if (old.id == turn.id) duplicate = true;
                }
                for (turns[0..index]) |old| {
                    if (old.id == turn.id) duplicate = true;
                }
                if (duplicate) turn.id = 0;
            }
        }
        var result = Text{};
        defer result.deinit();
        var prefix = self.transcript.*;
        prefix.len = if (turns.len != 0) turns[0].start else self.transcript.len;
        const prefix_text = Rendered.init(self.engine, &prefix) orelse return null;
        defer prefix_text.deinit();
        result.append(prefix_text.bytes());
        for (turns) |turn| {
            var header: [128]u8 = undefined;
            _ = c.snprintf(&header, header.len, "\n[[DS4_TURN %016llx role=%s]]\n", @as(c_ulonglong, turn.id), @as([*:0]const u8, if (turn.role_token == user) "user" else "assistant"));
            result.append(std.mem.sliceTo(&header, 0));
            const body = c.ds4_tokens{ .v = self.transcript.v + @as(usize, @intCast(turn.start)), .len = turn.end - turn.start, .cap = 0 };
            const rendered = Rendered.init(self.engine, &body) orelse return null;
            defer rendered.deinit();
            const quoted = bodyText(rendered.bytes(), true);
            defer c.free(quoted);
            result.append(std.mem.span(quoted));
        }
        if (out_len) |length| length.* = result.list.items.len;
        setTurns(self.state, self.transcript, if (turns.len != 0) turns.ptr else null, turns.len);
        retained = true;
        return result.take();
    }

    pub fn prepare(self: Context, system: *const c.ds4_tokens, text: []const u8, image_count: usize, ctx_size: c_int, edit: *Edit, errors: []u8) Invalid!bool {
        edit.* = std.mem.zeroes(Edit);
        errdefer editFree(edit);
        const old = Rendered.init(self.engine, self.transcript) orelse return reject(errors, "failed to render live context");
        defer old.deinit();
        if (std.mem.eql(u8, old.bytes(), text)) return false;
        if (image_count != 0 or std.mem.findScalar(u8, text, 0) != null) return reject(errors, "live context edit must be text-only and contain no NUL bytes");
        const system_text = Rendered.init(self.engine, system) orelse return reject(errors, "live context edit changed the protected system/tool/prefix section");
        defer system_text.deinit();
        if (!std.mem.startsWith(u8, text, system_text.bytes())) return reject(errors, "live context edit changed the protected system/tool/prefix section");
        edit.structured = try self.spliceTurns(text, system, system_text.len, edit, errors);
        if (!edit.structured) try spliceTokens(self.engine, self.transcript, old.bytes(), text, @intCast(system.len), &edit.tokens, errors);
        edit.ready = true;
        if (std.mem.eql(c_int, tokenSlice(self.transcript), tokenSlice(&edit.tokens))) return false;
        // Leave room for a tool acknowledgment and the next assistant prefix.
        if (edit.tokens.len > ctx_size - 256) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "live context edit exceeds context budget (%d tokens, limit %d)", edit.tokens.len, ctx_size - 256);
            return error.InvalidContext;
        }
        return true;
    }

    fn spliceTurns(self: Context, text: []const u8, system: *const c.ds4_tokens, system_bytes: usize, edit: *Edit, errors: []u8) Invalid!bool {
        const delimiter = "\n[[DS4_TURN ";
        var header = std.mem.findPos(u8, text, system_bytes, delimiter) orelse return false;
        var prefix = system.*;
        const previous = turnSlice(self.state);
        if (previous.len != 0) {
            prefix = self.state.snapshot;
            prefix.len = previous[0].start;
        }
        // Session date metadata can follow the static system prompt. Preserve
        // the entire unlabelled span up to the first actual role token.
        const prefix_text = Rendered.init(self.engine, &prefix) orelse return reject(errors, "keep the protected prefix and complete labelled turn boundaries");
        defer prefix_text.deinit();
        if (header != prefix_text.len or !std.mem.eql(u8, text[0..header], prefix_text.bytes())) return reject(errors, "keep the protected prefix and complete labelled turn boundaries");
        const turns = allocateTurns(previous.len);
        var retained = false;
        defer if (!retained) freeSlice(Turn, turns);
        const used = usedMap(previous.len);
        defer freeSlice(bool, used);
        c.ds4_tokens_copy(&edit.tokens, &prefix);
        const user = c.ds4_token_user(self.engine);
        const assistant = c.ds4_token_assistant(self.engine);
        var count: usize = 0;
        while (true) {
            var id: c_ulonglong = 0;
            var role: [16]u8 = @splat(0);
            var consumed: c_int = 0;
            if (c.sscanf(text.ptr + header + 1, "[[DS4_TURN %16llx role=%15[a-z]]]%n", &id, &role, &consumed) != 2 or consumed == 0) return reject(errors, "invalid turn header; keep its ID and role unchanged");
            const newline = header + 1 + @as(usize, @intCast(consumed));
            // C sees the trailing NUL at EOF; the bounded slice excludes it.
            if (newline >= text.len or text[newline] != '\n') return reject(errors, "invalid turn header; keep its ID and role unchanged");
            const old_index = for (previous, 0..) |turn, index| {
                if (turn.id == id) break index;
            } else previous.len;
            if (old_index == previous.len or used[old_index]) return reject(errors, "turn ID is unknown or repeated");
            const old = previous[old_index];
            if (!std.mem.eql(u8, std.mem.sliceTo(&role, 0), if (old.role_token == user) "user" else "assistant")) return reject(errors, "turn role changed");
            const body_start = header + 2 + @as(usize, @intCast(consumed));
            const next = std.mem.findPos(u8, text, body_start, delimiter);
            const body = text[body_start..(next orelse text.len)];
            var role_len: usize = 0;
            const role_text = c.ds4_token_text(self.engine, old.role_token, &role_len);
            defer c.free(role_text);
            if (role_len == 0 or body.len < role_len or !std.mem.eql(u8, body[0..role_len], role_text[0..role_len])) return reject(errors, "keep the native opening role marker beneath each turn header");
            const body_text = bodyText(body, false);
            defer c.free(body_text);
            const original = c.ds4_tokens{ .v = self.state.snapshot.v + @as(usize, @intCast(old.start)), .len = old.end - old.start, .cap = old.end - old.start };
            const old_text = Rendered.init(self.engine, &original) orelse return error.InvalidContext;
            defer old_text.deinit();
            var part = std.mem.zeroes(c.ds4_tokens);
            defer c.ds4_tokens_free(&part);
            try spliceTokens(self.engine, &original, old_text.bytes(), std.mem.span(body_text), 0, &part, errors);
            const part_ids = tokenSlice(&part);
            if (part_ids.len == 0 or part_ids[0] != old.role_token) return reject(errors, "turn lost its opening role token");
            for (part_ids[1..]) |token| {
                if (token == user or token == assistant) return reject(errors, "keep one complete role span beneath each turn header");
            }
            turns[count] = .{ .id = id, .start = edit.tokens.len, .end = 0, .role_token = old.role_token };
            for (part_ids) |token| c.ds4_tokens_push(&edit.tokens, token);
            turns[count].end = edit.tokens.len;
            count += 1;
            used[old_index] = true;
            header = next orelse break;
        }
        edit.turns = if (turns.len != 0) turns.ptr else null;
        edit.turn_count = count;
        retained = true;
        return true;
    }
};

const markers = [_][:0]const u8{
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<｜System｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "｜DSML｜",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
};

fn ambiguousMiddle(engine: *c.ds4_engine, ids: []const c_int, old: []const u8, offsets: []const usize, first: usize, last: usize, middle: []const u8) bool {
    // Rendered bytes are not injective: ordinary BPE can spell chat controls.
    // A retained literal anywhere inside a reparsed middle is ambiguous, even
    // when it sits unchanged between two separated word edits.
    const changed = std.mem.sliceTo(middle, 0);
    const searchable = std.mem.sliceTo(old[offsets[first]..], 0);
    for (markers) |marker| {
        if (std.mem.find(u8, changed, marker) == null) continue;
        var one = std.mem.zeroes(c.ds4_tokens);
        defer c.ds4_tokens_free(&one);
        c.ds4_tokenize_rendered_chat(engine, marker.ptr, &one);
        if (one.len != 1 or one.v[0] < 0) continue;
        var search_from: usize = 0;
        var token_index = first;
        while (std.mem.findPos(u8, searchable, search_from, marker)) |relative| {
            const at = offsets[first] + relative;
            if (at >= offsets[last] or marker.len > offsets[last] - at) break;
            while (token_index < last and offsets[token_index] < at) token_index += 1;
            if (!(token_index < last and offsets[token_index] == at and offsets[token_index + 1] == at + marker.len and ids[token_index] == one.v[0])) return true;
            search_from = relative + marker.len;
        }
    }
    return false;
}
fn fillOffsets(engine: *c.ds4_engine, ids: []const c_int, text: []const u8, offsets: []usize) bool {
    offsets[0] = 0;
    for (ids, 0..) |id, index| {
        var len: usize = 0;
        const piece = c.ds4_token_text(engine, id, &len);
        defer c.free(piece);
        if (piece == null or offsets[index] > text.len or len > text.len - offsets[index] or !std.mem.eql(u8, text[offsets[index]..][0..len], piece[0..len])) return false;
        offsets[index + 1] = offsets[index] + len;
    }
    return true;
}
pub fn textSafe(engine: *c.ds4_engine, tokens: *const c.ds4_tokens, text: []const u8, errors: []u8) bool {
    const ids = tokenSlice(tokens);
    const offsets = allocator.alloc(usize, ids.len + 1) catch {
        errorMessage(errors, "cannot validate text-only session controls");
        return false;
    };
    defer allocator.free(offsets);
    const safe = fillOffsets(engine, ids, text, offsets) and offsets[ids.len] == text.len and !ambiguousMiddle(engine, ids, text, offsets, 0, ids.len, text);
    if (!safe) errorMessage(errors, "text-only approximate session cannot preserve literal control markup");
    return safe;
}
fn spliceTokens(engine: *c.ds4_engine, transcript: *const c.ds4_tokens, before: []const u8, after: []const u8, protected: usize, edited: *c.ds4_tokens, errors: []u8) Invalid!void {
    var common: usize = 0;
    while (common < before.len and common < after.len and before[common] == after[common]) common += 1;
    var suffix: usize = 0;
    while (suffix < before.len - common and suffix < after.len - common and before[before.len - suffix - 1] == after[after.len - suffix - 1]) suffix += 1;
    const ids = tokenSlice(transcript);
    const offsets = allocator.alloc(usize, ids.len + 1) catch return reject(errors, "failed to allocate live context token offsets");
    defer allocator.free(offsets);
    if (!fillOffsets(engine, ids, before, offsets)) return reject(errors, "live context token rendering changed");
    if (offsets[ids.len] != before.len or protected > ids.len or offsets[protected] > common) return reject(errors, "live context protected token boundary changed");
    var prefix = protected;
    while (prefix < ids.len and offsets[prefix + 1] <= common) prefix += 1;
    // Byte-level pieces can split UTF8. Give the tokenizer a whole character.
    while (prefix > protected and offsets[prefix] < before.len and before[offsets[prefix]] & 0xc0 == 0x80) prefix -= 1;
    var tail = prefix;
    while (tail < ids.len and offsets[tail] < before.len - suffix) tail += 1;
    while (tail < ids.len and offsets[tail] < before.len and before[offsets[tail]] & 0xc0 == 0x80) tail += 1;
    const middle_bytes = after[offsets[prefix] .. after.len - (before.len - offsets[tail])];
    const middle = allocator.dupeSentinel(u8, middle_bytes, 0) catch return reject(errors, "failed to allocate live context edit");
    defer allocator.free(middle);
    if (ambiguousMiddle(engine, ids, before, offsets, prefix, tail, middle)) return reject(errors, "live context edit has ambiguous literal control markup inside the changed span; edit smaller spans");
    for (ids[0..prefix]) |id| c.ds4_tokens_push(edited, id);
    c.ds4_tokenize_rendered_chat(engine, middle.ptr, edited);
    for (ids[tail..]) |id| c.ds4_tokens_push(edited, id);
    const rendered = Rendered.init(engine, edited) orelse return reject(errors, "live context edit did not round-trip exactly");
    defer rendered.deinit();
    if (!std.mem.eql(u8, rendered.bytes(), after)) return reject(errors, "live context edit did not round-trip exactly");
}

pub fn editFree(edit: *Edit) void {
    c.ds4_tokens_free(&edit.tokens);
    c.free(edit.turns);
    edit.* = std.mem.zeroes(Edit);
}
pub fn commit(transcript: *c.ds4_tokens, context: *Record, edit: *Edit) void {
    if (!edit.ready) return;
    if (!std.mem.eql(c_int, tokenSlice(transcript), tokenSlice(&edit.tokens))) {
        c.ds4_tokens_free(transcript);
        transcript.* = edit.tokens;
        edit.tokens = std.mem.zeroes(c.ds4_tokens);
    }
    if (edit.structured) {
        setTurns(context, transcript, edit.turns, edit.turn_count);
        edit.turns = null;
        edit.turn_count = 0;
    }
    edit.ready = false;
}
