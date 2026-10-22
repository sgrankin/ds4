//! Ordered JoyAI, GLM4 and Qwen35 text splitting, followed by the selected BPE.
//! Vocabulary maps remain opaque and borrowed; there is no engine/vocab layout
//! here. Unicode classification data comes from the existing generated header.
const c = @import("pretokenizer_c");
const Family = c.ds4_pretokenizer_family;
const Map = c.ds4_token_map;
const Tokens = c.ds4_tokens;

const Bpe = struct {
    tokens: *const Map,
    merges: *const Map,
    output: *Tokens,

    fn emit(self: Bpe, bytes: []const u8) bool {
        return c.ds4_bpe_emit_piece(self.tokens, self.merges, bytes.ptr, bytes.len, self.output);
    }
};

fn alpha(byte: u8) bool {
    return (byte >= 'A' and byte <= 'Z') or (byte >= 'a' and byte <= 'z');
}
fn digit(byte: u8) bool {
    return byte >= '0' and byte <= '9';
}
fn space(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r' or byte == '\x0b' or byte == '\x0c';
}
fn newline(codepoint: u32) bool {
    return codepoint == '\n' or codepoint == '\r';
}
fn punctuation(byte: u8) bool {
    return (byte >= '!' and byte <= '/') or (byte >= ':' and byte <= '@') or
        (byte >= '[' and byte <= '`') or (byte >= '{' and byte <= '~');
}

const Codepoint = struct { value: u32, next: usize };

// Deliberately NOT a validating UTF-8 decoder: the original tokenizer chooses
// width from the first byte, masks following bytes, and falls back to one byte
// for a truncated sequence. Replacing this with Unicode normalization/validation
// would change accepted input and token IDs.
fn decode(text: []const u8, position: usize) Codepoint {
    const first = text[position];
    var width: usize = if (first < 0x80) 1 else if (first & 0xe0 == 0xc0) 2 else if (first & 0xf0 == 0xe0) 3 else if (first & 0xf8 == 0xf0) 4 else 1;
    if (width > text.len - position) width = 1;
    const value: u32 = switch (width) {
        2 => (@as(u32, first & 0x1f) << 6) | (text[position + 1] & 0x3f),
        3 => (@as(u32, first & 0x0f) << 12) | (@as(u32, text[position + 1] & 0x3f) << 6) | (text[position + 2] & 0x3f),
        4 => (@as(u32, first & 0x07) << 18) | (@as(u32, text[position + 1] & 0x3f) << 12) | (@as(u32, text[position + 2] & 0x3f) << 6) | (text[position + 3] & 0x3f),
        else => first,
    };
    return .{ .value = value, .next = position + width };
}

fn lowerAscii(codepoint: u32) u32 {
    return if (codepoint >= 'A' and codepoint <= 'Z') codepoint + ('a' - 'A') else codepoint;
}

// Both GLM4 and Qwen35 give this ASCII-only case-insensitive contraction rule
// priority over their letter, number, punctuation and whitespace alternatives.
fn contractionEnd(text: []const u8, after_quote: usize) ?usize {
    if (after_quote >= text.len) return null;
    const first = decode(text, after_quote);
    const one = lowerAscii(first.value);
    if (one == 's' or one == 't' or one == 'm' or one == 'd') return first.next;
    if (first.next < text.len) {
        const second = decode(text, first.next);
        const two = lowerAscii(second.value);
        if ((one == 'r' and two == 'e') or (one == 'v' and two == 'e') or (one == 'l' and two == 'l')) return second.next;
    }
    return null;
}

fn joyLetter(text: []const u8, position: usize) bool {
    // JoyAI collapses Unicode letters into its regex alphabet. Preserve the C
    // approximation: non-ASCII bytes are letter-like; CJK/kana are split first.
    return text[position] >= 128 or alpha(text[position]);
}
fn joyCjk(text: []const u8, position: usize) bool {
    if (text[position] < 128) return false;
    const cp = decode(text, position).value;
    return (cp >= 0x4e00 and cp <= 0x9fa5) or (cp >= 0x3040 and cp <= 0x309f) or (cp >= 0x30a0 and cp <= 0x30ff);
}
fn joyLettersEnd(text: []const u8, start: usize) usize {
    var position = start;
    while (position < text.len and joyLetter(text, position)) position = decode(text, position).next;
    return position;
}

// DeepSeek V4 Flash declares tokenizer.ggml.pre="joyai-llm". Ordered rules:
//   \p{N}{1,3}; [CJK/Hiragana/Katakana]+; [P/S][A-Za-z]+;
//   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+;
//    ?[\p{P}\p{S}]+[\r\n]*; \s*[\r\n]+; \s+(?!\S); \s+.
// Punctuation keeps trailing newlines in the SAME BPE piece (e.g. ">;\n").
// Splitting those newlines changes code-prompt IDs and long-context logits.
fn joyai(bpe: Bpe, text: []const u8) bool {
    var position: usize = 0;
    while (position < text.len) {
        const start = position;
        const byte = text[position];
        if (digit(byte)) {
            var digits: usize = 0;
            while (position < text.len and digit(text[position]) and digits < 3) : (digits += 1) position += 1;
        } else if (joyCjk(text, position)) {
            position = decode(text, position).next;
            while (position < text.len and joyCjk(text, position)) position = decode(text, position).next;
        } else if (punctuation(byte) and position + 1 < text.len and alpha(text[position + 1])) {
            position += 1;
            while (position < text.len and alpha(text[position])) position += 1;
        } else if (joyLetter(text, position)) {
            position = joyLettersEnd(text, position);
        } else if (!newline(byte) and !punctuation(byte) and position + 1 < text.len and joyLetter(text, position + 1)) {
            position = joyLettersEnd(text, position + 1);
        } else if (byte == ' ' and position + 1 < text.len and punctuation(text[position + 1])) {
            position += 1;
            while (position < text.len and punctuation(text[position])) position += 1;
            while (position < text.len and newline(text[position])) position += 1;
        } else if (punctuation(byte)) {
            while (position < text.len and punctuation(text[position])) position += 1;
            while (position < text.len and newline(text[position])) position += 1;
        } else if (space(byte)) {
            var end = position;
            var last_newline: usize = 0;
            while (end < text.len and space(text[end])) {
                const current = text[end];
                end += 1;
                if (newline(current)) last_newline = end;
            }
            if (last_newline != 0) {
                position = last_newline;
            } else if (end < text.len and end > position + 1 and (joyLetter(text, end) or punctuation(text[end]))) {
                // One leading space joins the next word/punctuation run:
                // "    int" becomes "   " then " int", not "    " + "int".
                position = end - 1;
            } else position = end;
        } else position = decode(text, position).next;
        if (position == start) position = decode(text, position).next;
        if (!bpe.emit(text[start..position])) return false;
    }
    return true;
}

const Character = struct {
    codepoint: u32,
    next: usize,
    letter: bool,
    number: bool,
    whitespace: bool,
    mark: bool = false,
};

fn glmWhitespace(cp: u32) bool {
    if (cp < 128) return space(@intCast(cp));
    return cp == 0x85 or cp == 0xa0 or cp == 0x1680 or (cp >= 0x2000 and cp <= 0x200a) or
        cp == 0x2028 or cp == 0x2029 or cp == 0x202f or cp == 0x205f or cp == 0x3000;
}

fn inRanges(cp: u32, comptime ranges: []const [2]u32) bool {
    inline for (ranges) |range| if (cp >= range[0] and cp <= range[1]) return true;
    return false;
}

fn glmNumber(cp: u32) bool {
    if (cp < 128) return digit(@intCast(cp));
    // Preserve the original GLM Unicode classification, not a new Unicode DB.
    return inRanges(cp, &.{
        .{ 0x0660, 0x0669 }, .{ 0x06f0, 0x06f9 }, .{ 0x07c0, 0x07c9 },
        .{ 0x0966, 0x096f }, .{ 0x09e6, 0x09ef }, .{ 0x0a66, 0x0a6f },
        .{ 0x0ae6, 0x0aef }, .{ 0x0b66, 0x0b6f }, .{ 0x0be6, 0x0bef },
        .{ 0x0c66, 0x0c6f }, .{ 0x0ce6, 0x0cef }, .{ 0x0d66, 0x0d6f },
        .{ 0x0de6, 0x0def }, .{ 0x0e50, 0x0e59 }, .{ 0x0ed0, 0x0ed9 },
        .{ 0x0f20, 0x0f29 }, .{ 0x1040, 0x1049 }, .{ 0x1090, 0x1099 },
        .{ 0x17e0, 0x17e9 }, .{ 0x1810, 0x1819 }, .{ 0xff10, 0xff19 },
    });
}

fn glmPunctuation(cp: u32) bool {
    if (cp < 128) return punctuation(@intCast(cp));
    return inRanges(cp, &.{
        .{ 0x00a1, 0x00a9 },   .{ 0x00ab, 0x00ac }, .{ 0x00ae, 0x00b1 },
        .{ 0x00b4, 0x00b4 },   .{ 0x00b6, 0x00b8 }, .{ 0x00bb, 0x00bb },
        .{ 0x00bf, 0x00bf },   .{ 0x00d7, 0x00d7 }, .{ 0x00f7, 0x00f7 },
        .{ 0x02c2, 0x02df },   .{ 0x02e5, 0x02eb }, .{ 0x02ed, 0x02ff },
        .{ 0x0375, 0x037e },   .{ 0x0384, 0x0385 }, .{ 0x0387, 0x0387 },
        .{ 0x055a, 0x055f },   .{ 0x0589, 0x058a }, .{ 0x05be, 0x05c0 },
        .{ 0x05c3, 0x05c3 },   .{ 0x05c6, 0x05c7 }, .{ 0x0609, 0x060a },
        .{ 0x060c, 0x060d },   .{ 0x061b, 0x061b }, .{ 0x061e, 0x061f },
        .{ 0x066a, 0x066a },   .{ 0x066d, 0x066d }, .{ 0x06d4, 0x06d4 },
        .{ 0x2000, 0x206f },   .{ 0x20a0, 0x20cf }, .{ 0x2100, 0x214f },
        .{ 0x2190, 0x23ff },   .{ 0x2460, 0x24ff }, .{ 0x2500, 0x2775 },
        .{ 0x2794, 0x2bff },   .{ 0x2e00, 0x2e7f }, .{ 0x3000, 0x303f },
        .{ 0xfd3e, 0xfd3f },   .{ 0xfe10, 0xfe6f }, .{ 0xff01, 0xff0f },
        .{ 0xff1a, 0xff20 },   .{ 0xff3b, 0xff40 }, .{ 0xff5b, 0xff65 },
        .{ 0x1f000, 0x1faff },
    });
}

fn glmAt(text: []const u8, position: usize) ?Character {
    if (position >= text.len) return null;
    const decoded = decode(text, position);
    const cp = decoded.value;
    const whitespace = glmWhitespace(cp);
    const number = glmNumber(cp);
    return .{
        .codepoint = cp,
        .next = decoded.next,
        .letter = if (cp < 128) alpha(@intCast(cp)) else !whitespace and !number and !glmPunctuation(cp),
        .number = number,
        .whitespace = whitespace,
    };
}

// GLM GGUFs use tokenizer.ggml.pre="glm4", the llama3-style ordered split
// shape from the original CHATGLM4 path. Numeric pieces contain up to 3 digits.
fn glm4(bpe: Bpe, text: []const u8) bool {
    var position: usize = 0;
    while (glmAt(text, position)) |current| {
        const start = position;
        if (current.codepoint == '\'') {
            if (contractionEnd(text, current.next)) |end| {
                position = end;
                if (!bpe.emit(text[start..position])) return false;
                continue;
            }
        }
        if (!newline(current.codepoint) and !current.number) {
            const following_letter = if (glmAt(text, current.next)) |next| next.letter else false;
            if (current.letter or following_letter) {
                position = current.next;
                while (glmAt(text, position)) |scan| {
                    if (!scan.letter) break;
                    position = scan.next;
                }
                if (!bpe.emit(text[start..position])) return false;
                continue;
            }
        }
        if (current.number) {
            var digits: usize = 0;
            while (glmAt(text, position)) |scan| {
                if (!scan.number or digits == 3) break;
                position = scan.next;
                digits += 1;
            }
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        var punct: ?Character = current;
        var punct_position = position;
        if (current.codepoint == ' ') {
            punct_position = current.next;
            punct = glmAt(text, punct_position);
        }
        if (punct != null and !punct.?.whitespace and !punct.?.letter and !punct.?.number) {
            position = punct_position;
            while (glmAt(text, position)) |scan| {
                if (scan.whitespace or scan.letter or scan.number) break;
                position = scan.next;
            }
            while (glmAt(text, position)) |scan| {
                if (!newline(scan.codepoint)) break;
                position = scan.next;
            }
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        if (current.whitespace) {
            var end = position;
            var last_newline: usize = 0;
            var last_space = position;
            var spaces: usize = 0;
            while (glmAt(text, end)) |scan| {
                if (!scan.whitespace) break;
                last_space = end;
                if (newline(scan.codepoint)) last_newline = scan.next;
                end = scan.next;
                spaces += 1;
            }
            position = if (last_newline != 0) last_newline else if (spaces > 1 and end < text.len) last_space else end;
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        position = current.next;
        if (position == start) position = decode(text, position).next;
        if (!bpe.emit(text[start..position])) return false;
    }
    return true;
}

fn unicodeRange(ranges: []const c.ds4_qwen4_urange, cp: u32) bool {
    var low: usize = 0;
    var high = ranges.len;
    while (low < high) {
        const middle = low + (high - low) / 2;
        if (cp < ranges[middle].lo) high = middle else if (cp > ranges[middle].hi) low = middle + 1 else return true;
    }
    return false;
}

fn qwenAt(text: []const u8, position: usize) ?Character {
    if (position >= text.len) return null;
    const decoded = decode(text, position);
    const cp = decoded.value;
    return .{
        .codepoint = cp,
        .next = decoded.next,
        .letter = unicodeRange(&c.ds4_qwen4_letter_ranges, cp),
        .mark = unicodeRange(&c.ds4_qwen4_mark_ranges, cp),
        .number = unicodeRange(&c.ds4_qwen4_number_ranges, cp),
        .whitespace = unicodeRange(&c.ds4_qwen4_space_ranges, cp),
    };
}

// Qwen3.8 tokenizer.ggml.pre="qwen35", ordered alternation:
// (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ |
// \p{N} |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+.
// These classes use the existing regex-generated Unicode tables, not GLM's
// approximations. In particular, each number is ONE code point, not 3 digits.
fn qwen35(bpe: Bpe, text: []const u8) bool {
    var position: usize = 0;
    while (qwenAt(text, position)) |current| {
        const start = position;
        if (current.codepoint == '\'') {
            if (contractionEnd(text, current.next)) |end| {
                position = end;
                if (!bpe.emit(text[start..position])) return false;
                continue;
            }
        }
        // Optional non-letter prefix, then a run of letters and combining marks.
        var run: ?usize = null;
        if (current.letter or current.mark) {
            run = current.next;
        } else if (!newline(current.codepoint) and !current.number) {
            if (qwenAt(text, current.next)) |following| if (following.letter or following.mark) {
                run = following.next;
            };
        }
        if (run) |end| {
            position = end;
            while (qwenAt(text, position)) |scan| {
                if (!scan.letter and !scan.mark) break;
                position = scan.next;
            }
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        if (current.number) {
            position = current.next;
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        var punct: ?Character = current;
        var punct_position = position;
        if (current.codepoint == ' ') {
            punct_position = current.next;
            punct = qwenAt(text, punct_position);
        }
        if (punct != null and !punct.?.whitespace and !punct.?.letter and !punct.?.mark and !punct.?.number) {
            position = punct_position;
            while (qwenAt(text, position)) |scan| {
                if (scan.whitespace or scan.letter or scan.mark or scan.number) break;
                position = scan.next;
            }
            while (qwenAt(text, position)) |scan| {
                if (!newline(scan.codepoint)) break;
                position = scan.next;
            }
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        if (current.whitespace) {
            var end = position;
            var last_newline: usize = 0;
            var last_space = position;
            var spaces: usize = 0;
            while (qwenAt(text, end)) |scan| {
                if (!scan.whitespace) break;
                last_space = end;
                if (newline(scan.codepoint)) last_newline = scan.next;
                end = scan.next;
                spaces += 1;
            }
            position = if (last_newline != 0) last_newline else if (spaces > 1 and end < text.len) last_space else end;
            if (!bpe.emit(text[start..position])) return false;
            continue;
        }
        position = current.next;
        if (position == start) position = decode(text, position).next;
        if (!bpe.emit(text[start..position])) return false;
    }
    return true;
}

export fn ds4_pretokenize(family: Family, tokens: *const Map, merges: *const Map, raw: [*c]const u8, length: u64, output: *Tokens) bool {
    const text: []const u8 = if (length == 0) "" else raw[0..@as(usize, @intCast(length))];
    const bpe = Bpe{ .tokens = tokens, .merges = merges, .output = output };
    return switch (family) {
        c.DS4_PRETOKENIZER_JOYAI => joyai(bpe, text),
        c.DS4_PRETOKENIZER_GLM4 => glm4(bpe, text),
        c.DS4_PRETOKENIZER_QWEN35 => qwen35(bpe, text),
        else => false,
    };
}
