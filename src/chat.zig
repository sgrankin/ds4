//! Family-specific chat policy over a borrowed vocabulary and C token vector.
//! User/tool bodies stay ordinary bytes; only explicit rendered-history paths
//! interpret control spellings. No engine/model graph or global family state.
const std = @import("std");
const c = @import("chat_c");
const Vocab = c.ds4_vocab;
const Family = c.ds4_chat_family;
const Tokens = c.ds4_tokens;
const Mode = c.ds4_think_mode;
const Oom = error{OutOfMemory};

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn same(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

const Chat = struct {
    vocab: *const Vocab,
    family: Family,
    out: *Tokens,

    fn push(self: Chat, token: c_int) void {
        c.ds4_tokens_push(self.out, token);
    }
    fn emit(self: Chat, bytes: []const u8) Oom!void {
        if (bytes.len == 0) return;
        c.ds4_vocab_admit_text(bytes.len);
        const lexer: c.ds4_pretokenizer_family = switch (self.family) {
            c.DS4_CHAT_GLM4 => c.DS4_PRETOKENIZER_GLM4,
            c.DS4_CHAT_QWEN35 => c.DS4_PRETOKENIZER_QWEN35,
            else => c.DS4_PRETOKENIZER_JOYAI,
        };
        if (!c.ds4_pretokenize(lexer, self.vocab.token_to_id, self.vocab.merge_rank, bytes.ptr, bytes.len, self.out))
            return error.OutOfMemory;
    }
    fn begin(self: Chat) void {
        if (self.vocab.bos_id < 0) return;
        self.push(self.vocab.bos_id);
        if (self.family == c.DS4_CHAT_GLM4 and self.vocab.sop_id >= 0) self.push(self.vocab.sop_id);
    }
    fn thinkPrefix(self: Chat, mode: Mode) Oom!void {
        const effort = switch (self.family) {
            c.DS4_CHAT_GLM4 => c.ds4_glm_reasoning_effort_text(mode),
            c.DS4_CHAT_DEEPSEEK41 => c.ds4_deepseek41_reasoning_effort_text(mode),
            else => null,
        };
        if (effort != null) {
            self.push(self.vocab.system_id);
            try self.emit(text(effort));
        } else if (self.family != c.DS4_CHAT_GLM4 and self.family != c.DS4_CHAT_DEEPSEEK41 and mode == c.DS4_THINK_MAX) {
            try self.emit(text(c.ds4_think_max_prefix()));
        }
    }

    // Qwen ChatML thinking-on opens <think> after the effort instruction;
    // thinking-off closes an empty block with the reference's exact newlines.
    fn qwenOpen(self: Chat, role: []const u8) Oom!void {
        self.push(self.vocab.im_start_id);
        try self.emit(role);
        try self.emit("\n");
    }
    fn qwenClose(self: Chat) Oom!void {
        self.push(self.vocab.im_end_id);
        try self.emit("\n");
    }
    fn assistantPrefix(self: Chat, mode: Mode) Oom!void {
        if (self.family == c.DS4_CHAT_QWEN35) {
            try self.qwenOpen("assistant");
            self.push(self.vocab.think_start_id);
            if (c.ds4_think_mode_enabled(mode)) {
                try self.emit("\n");
            } else {
                try self.emit("\n\n");
                self.push(self.vocab.think_end_id);
                try self.emit("\n\n");
            }
            return;
        }
        self.push(self.vocab.assistant_id);
        if (self.family == c.DS4_CHAT_GLM4 and !c.ds4_think_mode_enabled(mode)) {
            self.push(self.vocab.think_start_id);
            self.push(self.vocab.think_end_id);
            return;
        }
        self.push(if (c.ds4_think_mode_enabled(mode)) self.vocab.think_start_id else self.vocab.think_end_id);
    }
    fn qwenSystem(self: Chat, system: []const u8, mode: Mode) Oom!void {
        const instruction = c.ds4_qwen4_reasoning_effort_text(mode);
        if (instruction == null and system.len == 0) return;
        try self.qwenOpen("system");
        if (instruction != null) {
            try self.emit(text(instruction));
            if (system.len != 0) try self.emit("\n\n");
        }
        try self.emit(system);
        try self.qwenClose();
    }

    fn encode(self: Chat, system: []const u8, prompt: []const u8, mode: Mode) Oom!void {
        if (self.family == c.DS4_CHAT_QWEN35) {
            try self.qwenSystem(system, mode);
            try self.qwenOpen("user");
            try self.emit(prompt);
            try self.qwenClose();
            try self.assistantPrefix(mode);
            return;
        }
        // BOS, optional effort/system text, user body, assistant mode marker.
        // Max effort is a prompt prefix; the model still enters via <think>.
        self.begin();
        try self.thinkPrefix(mode);
        if (system.len != 0) {
            if (self.family == c.DS4_CHAT_GLM4 or
                (self.family == c.DS4_CHAT_DEEPSEEK41 and !c.ds4_think_mode_enabled(mode))) self.push(self.vocab.system_id);
            try self.emit(system);
        }
        self.push(self.vocab.user_id);
        try self.emit(prompt);
        try self.assistantPrefix(mode);
    }

    const Special = struct { spelling: []const u8, token: c_int };
    fn specials(self: Chat) [25]Special {
        const v = self.vocab;
        return .{
            .{ .spelling = "<｜begin▁of▁sentence｜>", .token = v.bos_id },
            .{ .spelling = "<｜end▁of▁sentence｜>", .token = v.eos_id },
            .{ .spelling = "[gMASK]", .token = v.bos_id },
            .{ .spelling = "<sop>", .token = v.sop_id },
            .{ .spelling = "<|system|>", .token = v.system_id },
            .{ .spelling = "<｜System｜>", .token = if (self.family == c.DS4_CHAT_DEEPSEEK41) v.system_id else -1 },
            .{ .spelling = "<｜User｜>", .token = v.user_id },
            .{ .spelling = "<｜Assistant｜>", .token = v.assistant_id },
            .{ .spelling = "<|user|>", .token = v.user_id },
            .{ .spelling = "<|assistant|>", .token = v.assistant_id },
            .{ .spelling = "<|observation|>", .token = v.observation_id },
            .{ .spelling = "<think>", .token = v.think_start_id },
            .{ .spelling = "</think>", .token = v.think_end_id },
            .{ .spelling = "<tool_call>", .token = v.tool_call_start_id },
            .{ .spelling = "</tool_call>", .token = v.tool_call_end_id },
            .{ .spelling = "<tool_response>", .token = v.tool_response_start_id },
            .{ .spelling = "</tool_response>", .token = v.tool_response_end_id },
            .{ .spelling = "<arg_key>", .token = v.arg_key_start_id },
            .{ .spelling = "</arg_key>", .token = v.arg_key_end_id },
            .{ .spelling = "<arg_value>", .token = v.arg_value_start_id },
            .{ .spelling = "</arg_value>", .token = v.arg_value_end_id },
            .{ .spelling = "｜DSML｜", .token = v.dsml_id },
            .{ .spelling = "<|im_start|>", .token = v.im_start_id },
            .{ .spelling = "<|im_end|>", .token = v.im_end_id },
            .{ .spelling = "<|endoftext|>", .token = v.endoftext_id },
        };
    }
    fn rendered(self: Chat, bytes: []const u8) Oom!void {
        const markers = self.specials();
        var span: usize = 0;
        var position: usize = 0;
        while (position < bytes.len) {
            // Every spelling above starts with '<', '[', or the first byte of
            // the fullwidth bar. Skip ordinary text with the standard scanner.
            position = std.mem.findAnyPos(u8, bytes, position, "<[\xef") orelse break;
            var matched = false;
            for (markers) |marker| {
                if (marker.token < 0 or !std.mem.startsWith(u8, bytes[position..], marker.spelling)) continue;
                try self.emit(bytes[span..position]);
                self.push(marker.token);
                position += marker.spelling.len;
                span = position;
                matched = true;
                break; // Preserve the original table order, not longest match.
            }
            if (!matched) position += 1;
        }
        try self.emit(bytes[span..]);
    }
    fn systemMarker(self: Chat) void {
        // Initial system text may be built in pieces after the effort prefix.
        // Later system messages must introduce their own role token.
        var has_system = false;
        for (self.out.v[0..@intCast(self.out.len)]) |token| {
            if (token == self.vocab.user_id or token == self.vocab.assistant_id) {
                self.push(self.vocab.system_id);
                return;
            }
            if (token == self.vocab.system_id) has_system = true;
        }
        if (!has_system) self.push(self.vocab.system_id);
    }
    fn payload(self: Chat, bytes: []const u8, closing: []const u8) Oom!void {
        // Tool bodies are plain data, not HTML. Preserve literal <, > and &;
        // escape only the exact closing sentinel and colliding escape spellings.
        var span: usize = 0;
        var position: usize = 0;
        while (position < bytes.len) : (position += 1) {
            if (!needsEscape(bytes[position..], closing)) continue;
            try self.emit(bytes[span..position]);
            try self.emit(if (bytes[position] == '<') "&lt;" else "&amp;");
            span = position + 1;
        }
        try self.emit(bytes[span..]);
    }
    fn append(self: Chat, role: []const u8, content: []const u8) Oom!void {
        const system = same(role, "system") or same(role, "developer");
        const assistant = same(role, "assistant");
        const tool = same(role, "tool") or same(role, "function");
        if (self.family == c.DS4_CHAT_QWEN35) {
            try self.qwenOpen(if (tool) "user" else if (system) "system" else if (assistant) "assistant" else "user");
            if (tool) {
                try self.emit("<tool_response>\n");
                try self.payload(content, "</tool_response>");
                try self.emit("\n</tool_response>");
            } else if (assistant) {
                try self.rendered(content);
            } else {
                try self.emit(content);
            }
            try self.qwenClose();
            return;
        }
        if (self.family == c.DS4_CHAT_GLM4) {
            if (system) {
                if (self.vocab.system_id >= 0) self.push(self.vocab.system_id);
                try self.rendered(content);
            } else if (assistant) {
                self.push(self.vocab.assistant_id);
                if (!startsThinking(content)) {
                    self.push(self.vocab.think_start_id);
                    self.push(self.vocab.think_end_id);
                }
                try self.rendered(content);
            } else if (tool) {
                if (self.vocab.observation_id >= 0) self.push(self.vocab.observation_id);
                try self.rendered("<tool_response>");
                try self.payload(content, "</tool_response>");
                try self.rendered("</tool_response>");
            } else {
                self.push(self.vocab.user_id);
                try self.emit(content);
            }
            return;
        }
        if (system) {
            if (self.family == c.DS4_CHAT_DEEPSEEK41) self.systemMarker();
            try self.emit(content);
        } else if (assistant) {
            self.push(self.vocab.assistant_id);
            if (!startsThinking(content)) self.push(self.vocab.think_end_id);
            try self.emit(content); // DeepSeek assistant bodies stay ordinary BPE.
        } else if (tool) {
            self.push(self.vocab.user_id);
            try self.emit("<tool_result>");
            try self.payload(content, "</tool_result>");
            try self.emit("</tool_result>");
        } else {
            self.push(self.vocab.user_id);
            try self.emit(content);
        }
    }
};

fn startsThinking(bytes: []const u8) bool {
    return std.mem.startsWith(u8, bytes, "<think>") or std.mem.startsWith(u8, bytes, "</think>");
}
fn needsEscape(bytes: []const u8, closing: []const u8) bool {
    if (std.mem.startsWith(u8, bytes, closing)) return true;
    if (bytes.len == 0 or bytes[0] != '&') return false;
    var tail = bytes[1..];
    while (std.mem.startsWith(u8, tail, "amp;")) tail = tail[4..];
    return std.mem.startsWith(u8, tail, "lt;") and std.mem.startsWith(u8, tail[3..], closing[1..]);
}
fn chat(vocab: *const Vocab, family: Family, out: *Tokens) Chat {
    return .{ .vocab = vocab, .family = family, .out = out };
}

export fn ds4_vocab_chat_begin(vocab: *const Vocab, family: Family, out: *Tokens) void {
    chat(vocab, family, out).begin();
}
export fn ds4_vocab_chat_think_prefix(vocab: *const Vocab, family: Family, mode: Mode, out: *Tokens) bool {
    chat(vocab, family, out).thinkPrefix(mode) catch return false;
    return true;
}
export fn ds4_vocab_encode_chat(vocab: *const Vocab, family: Family, system: [*c]const u8, prompt: [*c]const u8, mode: Mode, out: *Tokens) [*c]const u8 {
    if (family == c.DS4_CHAT_QWEN35) {
        if (vocab.im_start_id < 0 or vocab.im_end_id < 0 or vocab.think_start_id < 0 or vocab.think_end_id < 0)
            return "this tokenizer does not provide the Qwen chat markers; use raw prompt tokenization";
    } else {
        const need_start = c.ds4_think_mode_enabled(mode) or family == c.DS4_CHAT_GLM4;
        if (vocab.bos_id < 0 or vocab.user_id < 0 or vocab.assistant_id < 0 or vocab.think_end_id < 0 or
            (family == c.DS4_CHAT_GLM4 and vocab.system_id < 0) or (need_start and vocab.think_start_id < 0))
            return "this tokenizer does not provide the DeepSeek chat markers; use raw prompt tokenization";
    }
    chat(vocab, family, out).encode(text(system), text(prompt), mode) catch return "out of memory";
    return null;
}
export fn ds4_vocab_tokenize_rendered(vocab: *const Vocab, family: Family, input: [*c]const u8, out: *Tokens) bool {
    chat(vocab, family, out).rendered(text(input)) catch return false;
    return true;
}
export fn ds4_vocab_chat_append(vocab: *const Vocab, family: Family, role: [*c]const u8, content: [*c]const u8, out: *Tokens) bool {
    chat(vocab, family, out).append(if (role == null) "user" else text(role), text(content)) catch return false;
    return true;
}
export fn ds4_vocab_assistant_prefix(vocab: *const Vocab, family: Family, mode: Mode, out: *Tokens) bool {
    chat(vocab, family, out).assistantPrefix(mode) catch return false;
    return true;
}
export fn ds4_vocab_tokenize_payload(vocab: *const Vocab, family: Family, content: [*c]const u8, closing: [*c]const u8, out: *Tokens) bool {
    chat(vocab, family, out).payload(text(content), text(closing)) catch return false;
    return true;
}
