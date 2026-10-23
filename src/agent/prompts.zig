//! Trusted built-in prompt examples and runtime context notes. Literal arrays
//! have one actual C definition; arbitrary user text remains plain content.
const std = @import("std");
const c = @import("agent_runtime_c");
const Trace = @import("trace.zig").Trace;
const allocator = std.heap.c_allocator;
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn fatal() noreturn {
    c.perror("ds4-agent: malloc");
    c.exit(1);
}
fn data(comptime name: []const u8) [*c]const u8 {
    return @field(c, name);
}
fn literal(comptime name: []const u8) []const u8 {
    return text(data(name));
}
fn add(output: *std.ArrayList(u8), value: []const u8) void {
    output.appendSlice(allocator, value) catch fatal();
}
fn take(output: *std.ArrayList(u8)) [*c]u8 {
    return (output.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
}

pub fn syntax(engine: ?*c.ds4_engine) c.agent_tool_syntax {
    if (c.ds4_engine_is_qwen4(engine)) return c.AGENT_TOOL_SYNTAX_QWEN;
    if (c.ds4_engine_is_glm_dsa(engine)) return c.AGENT_TOOL_SYNTAX_GLM;
    return if (c.ds4_engine_is_deepseek41(engine)) c.AGENT_TOOL_SYNTAX_DSML41 else c.AGENT_TOOL_SYNTAX_DSML;
}
fn xml(kind: c.agent_tool_syntax) bool {
    return kind == c.AGENT_TOOL_SYNTAX_GLM or kind == c.AGENT_TOOL_SYNTAX_QWEN;
}
pub fn dsml(allow_upto: bool, vision: bool) [*c]u8 {
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    add(&output, literal("agent_tools_prompt_intro"));
    add(&output, if (allow_upto) literal("agent_tools_prompt_edit_upto") else literal("agent_tools_prompt_edit_exact"));
    const after = literal("agent_tools_prompt_after_edit");
    const rules = std.mem.find(u8, after, "\n# Rules\n").?;
    add(&output, after[0..rules]);
    if (vision) {
        add(&output, "\n{\"type\":\"function\",\"function\":");
        add(&output, literal("agent_vision_tool_schema"));
        add(&output, "}\n");
    }
    add(&output, after[rules..]);
    return take(&output);
}
pub fn glm(allow_upto: bool, vision: bool) [*c]u8 {
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    add(&output, literal("agent_glm_tools_prompt_intro"));
    add(&output, literal("agent_glm_tool_schemas"));
    if (vision) {
        add(&output, literal("agent_vision_tool_schema"));
        add(&output, "\n");
    }
    add(&output, literal("agent_glm_tools_prompt_after_schemas"));
    add(&output, if (allow_upto) literal("agent_glm_tools_prompt_edit_upto") else literal("agent_glm_tools_prompt_edit_exact"));
    add(&output, literal("agent_glm_tools_prompt_rules_tail"));
    return take(&output);
}
/// Each line of the shared GLM schema list is one wrapped Qwen function.
pub fn qwen(allow_upto: bool, vision: bool) [*c]u8 {
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    const wrapper = "\n{\"type\": \"function\", \"function\": ";
    add(&output, literal("agent_qwen_tools_prompt_intro"));
    var lines = std.mem.splitScalar(u8, literal("agent_glm_tool_schemas"), '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        add(&output, wrapper);
        add(&output, line);
        add(&output, "}");
    }
    if (vision) {
        add(&output, wrapper);
        add(&output, literal("agent_vision_tool_schema"));
        add(&output, "}");
    }
    add(&output, literal("agent_qwen_tools_prompt_after_schemas"));
    add(&output, if (allow_upto) literal("agent_glm_tools_prompt_edit_upto") else literal("agent_glm_tools_prompt_edit_exact"));
    add(&output, literal("agent_glm_tools_prompt_rules_tail"));
    return take(&output);
}
/// Adapt only trusted examples, including escaped closing tags. Never rewrite
/// sampled text or arbitrary user/tool payloads between protocol formats.
pub fn dsml41(source: [*c]const u8) [*c]u8 {
    const value = text(source);
    const marker = "｜DSML｜";
    const names = [_][]const u8{ "tool_calls", "invoke", "parameter" };
    const replacements = [_][]const u8{ " calls", " invoke", " parameter" };
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    var position: usize = 0;
    while (std.mem.findPos(u8, value, position, marker)) |found| {
        const after = found + marker.len;
        add(&output, value[position..after]);
        position = after;
        for (names, replacements) |name, replacement| {
            const remaining = value[position..];
            if (remaining.len > name.len and std.mem.startsWith(u8, remaining, name) and
                (remaining[name.len] == '>' or remaining[name.len] == ' '))
            {
                add(&output, replacement);
                position += name.len;
                break;
            }
        }
    }
    add(&output, value[position..]);
    return take(&output);
}
pub fn tools(engine: ?*c.ds4_engine, allow_upto: bool) [*c]u8 {
    const vision = c.ds4_engine_has_vision(engine);
    return switch (syntax(engine)) {
        c.AGENT_TOOL_SYNTAX_QWEN => qwen(allow_upto, vision),
        c.AGENT_TOOL_SYNTAX_GLM => glm(allow_upto, vision),
        else => result: {
            const original = dsml(allow_upto, vision);
            if (!c.ds4_engine_is_deepseek41(engine)) break :result original;
            defer c.free(original);
            break :result dsml41(original);
        },
    };
}
pub fn reminder(engine: ?*c.ds4_engine, allow_upto: bool) [*c]u8 {
    const built = tools(engine, allow_upto);
    defer c.free(built);
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    add(&output, "\n\n[System prompt reminder follows.]\n");
    add(&output, text(built));
    add(&output, "[End system prompt reminder.]\n\n");
    return take(&output);
}
/// Built-in control examples are trusted rendered text; arbitrary -sys content
/// must remain ordinary message content even if it spells native role/DSML tags.
pub fn appendSystem(engine: ?*c.ds4_engine, tokens: *c.ds4_tokens, extra: [*c]const u8, allow_upto: bool) void {
    const built = tools(engine, allow_upto);
    defer c.free(built);
    if (xml(syntax(engine))) c.ds4_chat_append_message(engine, tokens, "system", built) else {
        if (c.ds4_engine_is_deepseek41(engine)) c.ds4_chat_append_message(engine, tokens, "system", "");
        c.ds4_tokenize_rendered_chat(engine, built, tokens);
    }
    if (text(extra).len == 0) return;
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    add(&output, "\n\n");
    add(&output, text(extra));
    const plain = take(&output);
    defer c.free(plain);
    c.ds4_chat_append_message(engine, tokens, "system", plain);
}
/// Full instructions are sent once per live context, not on every enable.
pub fn hintsNote(hints: *c.agent_hints) [*c]const u8 {
    const enabled: @TypeOf(hints.applied) = if (hints.enabled) c.AGENT_HINTS_ON else c.AGENT_HINTS_OFF;
    const note: [*c]const u8 = if (hints.enabled and !hints.instructed) fresh: {
        hints.instructed = true;
        break :fresh data("agent_hints_prompt");
    } else if (hints.applied != c.AGENT_HINTS_UNSET and hints.applied != enabled)
        (if (hints.enabled) "The user enabled programming hints. Produce them from now on.\n" else "Programming hints are disabled. Do not produce teaching asides; continue the task normally.\n")
    else
        return null;
    hints.applied = enabled;
    return note;
}
pub fn hintsCompacted(hints: *c.agent_hints) void {
    // The retained tail may still contain an old on/off note.
    if (hints.applied != c.AGENT_HINTS_UNSET) hints.applied = c.AGENT_HINTS_UNKNOWN;
    hints.instructed = false;
}
fn systemStatus(worker: *c.agent_worker, value: [*c]const u8) void {
    if (worker.cfg.*.non_interactive) return;
    const color = c.isatty(c.STDOUT_FILENO) != 0;
    const prefix: []const u8 = if (color) "\x1b[33m✦ \x1b[38;5;218m" else "✦ ";
    c.ds4_agent_comms_publish(worker, prefix.ptr, prefix.len);
    c.ds4_agent_comms_publish(worker, value, text(value).len);
    const tail: []const u8 = if (color) "\x1b[0m\n" else "\n";
    c.ds4_agent_comms_publish(worker, tail.ptr, tail.len);
}
pub fn hintsContext(worker: *c.agent_worker) void {
    _ = c.pthread_mutex_lock(&worker.mu);
    const note = hintsNote(&worker.hints);
    _ = c.pthread_mutex_unlock(&worker.mu);
    if (note == null) return;
    c.ds4_chat_append_message(worker.engine, &worker.transcript, "system", note);
    (Trace{ .worker = worker }).text("hints-context", text(note));
    _ = c.pthread_mutex_lock(&worker.mu);
    worker.session_dirty = true;
    _ = c.pthread_mutex_unlock(&worker.mu);
}
pub fn datetimeContext(worker: *c.agent_worker) void {
    if (worker.datetime_context_injected) return;
    var instant = c.time(null);
    // Reproducible noninteractive probes route the same datetime tokens. Trace
    // timestamps and session identities remain real time.
    const fixed = c.getenv("DS4_AGENT_TEST_TIME");
    if (worker.cfg.*.non_interactive and text(fixed).len != 0) {
        var end: [*c]u8 = null;
        c.__error().* = 0;
        const value = c.strtoll(fixed, &end, 10);
        var candidate: c.time_t = @intCast(value);
        var calendar: c.struct_tm = undefined;
        if (c.__error().* == 0 and end != fixed and end[0] == 0 and value > 0 and
            @as(c_longlong, candidate) == value and c.localtime_r(&candidate, &calendar) != null) instant = candidate;
    }
    var calendar: c.struct_tm = undefined;
    _ = c.localtime_r(&instant, &calendar);
    var when: [128]u8 = undefined;
    if (c.strftime(&when, when.len, "%Y-%m-%d %H:%M:%S %Z", &calendar) == 0)
        _ = c.snprintf(&when, when.len, "%lld", @as(c_longlong, instant));
    var note: [256]u8 = undefined;
    _ = c.snprintf(&note, note.len, "Current local date and time at session start: %s. Use this only when date or time matters.", &when);
    c.ds4_chat_append_message(worker.engine, &worker.transcript, "system", &note);
    (Trace{ .worker = worker }).text("datetime-context", text(&note));
    worker.datetime_context_injected = true;
}
pub fn systemReminder(worker: *c.agent_worker) void {
    if (worker.last_system_prompt_reminder_at <= 0) {
        worker.last_system_prompt_reminder_at = worker.transcript.len;
        return;
    }
    if (worker.transcript.len - worker.last_system_prompt_reminder_at < c.AGENT_SYSTEM_PROMPT_REMINDER_TOKENS) return;
    const note = reminder(worker.engine, worker.cfg.*.edit_upto);
    defer c.free(note);
    systemStatus(worker, "Re-injecting system prompt reminder...");
    (Trace{ .worker = worker }).log("system prompt reminder injected at transcript=%d", .{worker.transcript.len});
    if (xml(syntax(worker.engine))) c.ds4_chat_append_message(worker.engine, &worker.transcript, "system", note) else c.ds4_tokenize_rendered_chat(worker.engine, note, &worker.transcript);
    if (text(worker.cfg.*.gen.system).len != 0) c.ds4_chat_append_message(worker.engine, &worker.transcript, "system", worker.cfg.*.gen.system);
    worker.last_system_prompt_reminder_at = worker.transcript.len;
}
pub const publishSystemStatus = systemStatus;
