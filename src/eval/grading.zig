//! Benchmark answer policy. Old traces intentionally retain loose extraction;
//! live runs require a final Answer: line. These lexical rules are small and
//! high-precision, not a general prose/math parser. Keep their order intact.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
const Case = c.eval_case;
const allocator = std.heap.c_allocator;
fn whitespace(byte: u8) bool {
    return c.isspace(byte) != 0;
}
fn digit(byte: u8) bool {
    return c.isdigit(byte) != 0;
}
fn alphabetic(byte: u8) bool {
    return c.isalpha(byte) != 0;
}
fn lower(byte: u8) u8 {
    return @intCast(c.tolower(byte));
}
fn upper(byte: u8) u8 {
    return @intCast(c.toupper(byte));
}
fn boundary(before: u8, after: u8) bool {
    return !alphabetic(before) and !alphabetic(after);
}
fn byteAt(value: []const u8, at: usize) u8 {
    return if (at < value.len) value[at] else 0;
}
fn equal(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn insensitive(value: []const u8, pattern: []const u8) ?usize {
    if (pattern.len == 0) return 0;
    if (value.len < pattern.len) return null;
    for (0..value.len - pattern.len + 1) |index| {
        if (c.strncasecmp(value.ptr + index, pattern.ptr, pattern.len) == 0) return index;
    }
    return null;
}
pub fn choices(case: *const Case) c_int {
    var count: usize = 0;
    while (count < c.EVAL_MAX_CHOICES and case.choice[count] != null) count += 1;
    return @intCast(count);
}
pub fn answerKind(case: *const Case) c.eval_answer_kind {
    if (case.answer_kind != c.EVAL_ANSWER_AUTO) return case.answer_kind;
    if (choices(case) != 0) return c.EVAL_ANSWER_CHOICE;
    if (equal(Common.text(case.source), "COMPSEC")) return c.EVAL_ANSWER_LINE_SET;
    return c.EVAL_ANSWER_INTEGER;
}
pub fn kindName(kind: c.eval_answer_kind) [*:0]const u8 {
    return switch (kind) {
        c.EVAL_ANSWER_CHOICE => "choice",
        c.EVAL_ANSWER_INTEGER => "integer",
        c.EVAL_ANSWER_RATIONAL => "reduced_rational",
        c.EVAL_ANSWER_EXACT_TEXT => "exact_text",
        c.EVAL_ANSWER_ORDERED_SEQUENCE => "ordered_sequence",
        c.EVAL_ANSWER_LINE_SET => "security_line_set",
        else => "auto",
    };
}
fn visible(generated: []const u8) []const u8 {
    return if (std.mem.find(u8, generated, "</think>")) |index| generated[index + 8 ..] else generated;
}
fn lastMarker(value: []const u8) ?usize {
    var last: ?usize = null;
    var position: usize = 0;
    while (position + 6 <= value.len) : (position += 1) {
        if (lower(value[position]) != 'a' or c.strncasecmp(value.ptr + position, "answer", 6) != 0) continue;
        if (!boundary(if (position == 0) ' ' else value[position - 1], byteAt(value, position + 6))) continue;
        var after = position + 6;
        while (after < value.len and whitespace(value[after])) after += 1;
        if (after < value.len and value[after] == ':') last = position;
    }
    if (last) |found| return found;
    return insensitive(value, "answer");
}
/// Reject only explicit same-line distractors ("not B", "rules out C"). Never
/// cross the selected marker or a newline. "D, not B" must still select D.
fn negated(value: []const u8, marker: usize, letter_index: usize) bool {
    var position = letter_index;
    while (position > marker) {
        const byte = value[position - 1];
        if (byte == '\n') return false;
        if (byte == ' ' or byte == '\t' or byte == ',' or byte == ';') position -= 1 else break;
    }
    const end = position;
    while (position > marker and (alphabetic(value[position - 1]) or value[position - 1] == '\'')) position -= 1;
    const length = end - position;
    if (length == 0 or length >= 16) return false;
    var word_buffer: [16]u8 = undefined;
    for (value[position..end], 0..) |byte, index| word_buffer[index] = lower(byte);
    const word = word_buffer[0..length];
    if (std.mem.endsWith(u8, word, "n't")) return true;
    for ([_][]const u8{ "not", "except", "excluding", "exclude", "excludes", "eliminate", "eliminates", "eliminated", "reject", "rejects", "rejected", "rejecting" }) |cue| {
        if (equal(word, cue)) return true;
    }
    if (equal(word, "out")) {
        while (position > marker and (value[position - 1] == ' ' or value[position - 1] == '\t')) position -= 1;
        const previous_end = position;
        while (position > marker and alphabetic(value[position - 1])) position -= 1;
        const previous_length = previous_end - position;
        if (previous_length != 0 and previous_length < 8) {
            var previous: [8]u8 = undefined;
            for (value[position..previous_end], 0..) |byte, index| previous[index] = lower(byte);
            const cue = previous[0..previous_length];
            return equal(cue, "rule") or equal(cue, "rules") or equal(cue, "ruled");
        }
    }
    return false;
}
pub fn letter(generated: []const u8, count: c_int) u8 {
    if (count <= 0) return '?';
    const value = visible(generated);
    const maximum: u8 = @intCast('A' + count - 1);
    if (lastMarker(value)) |marker| {
        const end = marker + @min(value.len - marker, 96);
        for (marker..end) |position| {
            const candidate = upper(value[position]);
            if (candidate < 'A' or candidate > maximum) continue;
            const after = byteAt(value, position + 1);
            if (!boundary(if (position == 0) ' ' else value[position - 1], after)) continue;
            // A capital introducing prose ("A careful ...") or a contraction
            // ("I'll ...") is not the pick. The reverse fallback still recovers
            // a sole candidate such as "Answer: A is correct".
            if (after == '\'') continue;
            if (after == ' ' or after == '\t') {
                var next = position + 1;
                while (next < value.len and (value[next] == ' ' or value[next] == '\t')) next += 1;
                if (c.islower(byteAt(value, next)) != 0) continue;
            }
            if (!negated(value, marker, position)) return candidate;
        }
    }
    var position = value.len;
    while (position > 0) {
        position -= 1;
        const candidate = upper(value[position]);
        if (candidate >= 'A' and candidate <= maximum and boundary(if (position == 0) ' ' else value[position - 1], byteAt(value, position + 1))) return candidate;
    }
    return '?';
}
pub fn normalizeInteger(value: []const u8, output: []u8) void {
    var start: usize = 0;
    while (value.len - start > 1 and value[start] == '0') start += 1;
    Common.copy(output, value[start..]);
}
fn firstInteger(value: []const u8, output: []u8) bool {
    for (value, 0..) |byte, start| {
        if (!digit(byte)) continue;
        var end = start + 1;
        while (end < value.len and digit(value[end])) end += 1;
        normalizeInteger(value[start..end], output);
        return true;
    }
    return false;
}
pub fn integer(generated: []const u8, output: []u8) void {
    if (output.len == 0) return;
    Common.copy(output, "?");
    const value = visible(generated);
    if (lastMarker(value)) |marker| {
        var line = value[marker..][0..@min(value.len - marker, 160)];
        if (std.mem.findScalar(u8, line, '\n')) |newline| line = line[0..newline];
        // Only the final answer line counts here. On an arithmetic line use
        // the LAST '=' result; otherwise keep the first integer ("082").
        const equals = std.mem.findScalarLast(u8, line, '=');
        if (firstInteger(if (equals) |index| line[index + 1 ..] else line, output)) return;
        if (equals != null and firstInteger(line, output)) return;
    }
    var position: usize = 0;
    var last: ?[]const u8 = null;
    while (position < value.len) {
        if (!digit(value[position])) {
            position += 1;
            continue;
        }
        const start = position;
        while (position < value.len and digit(value[position])) position += 1;
        last = value[start..position];
    }
    if (last) |digits| normalizeInteger(digits, output);
}
fn put(output: []u8, length: *usize, byte: u8) void {
    if (length.* + 1 < output.len) {
        output[length.*] = byte;
        length.* += 1;
    }
}
fn lineSpec(value: []const u8, output: []u8) void {
    if (output.len == 0) return;
    var length: usize = 0;
    var position: usize = 0;
    // Accept harmless final-line spellings ("line 9", "9 and 15", "20-22").
    // The audited key, not this normalizer, decides which lines are acceptable.
    while (position < value.len) : (position += 1) {
        if (!digit(value[position])) continue;
        if (length != 0) put(output, &length, ',');
        while (position < value.len and digit(value[position])) : (position += 1) put(output, &length, value[position]);
        while (position < value.len and whitespace(value[position])) position += 1;
        if (position < value.len and value[position] == '-') {
            put(output, &length, '-');
            position += 1;
            while (position < value.len and whitespace(value[position])) position += 1;
            while (position < value.len and digit(value[position])) : (position += 1) put(output, &length, value[position]);
        }
        if (position == value.len) break;
    }
    while (length > 0 and (output[length - 1] == ',' or output[length - 1] == '-')) length -= 1;
    output[length] = 0;
    if (length == 0) Common.copy(output, "?");
}
pub fn compsec(generated: []const u8, output: []u8) void {
    if (output.len == 0) return;
    Common.copy(output, "?");
    const value = visible(generated);
    if (lastMarker(value)) |marker| {
        var line = value[marker..][0..@min(value.len - marker, 160)];
        if (std.mem.findScalar(u8, line, '\n')) |newline| line = line[0..newline];
        lineSpec(line, output);
        if (!equal(Common.written(output), "?")) return;
    }
    integer(generated, output);
}
/// The libc parser keeps the original permissive range grammar, including
/// reversed ranges and clamping to the fixed line-number domain.
pub fn parseLineSet(value: [*c]const u8, set: []bool) bool {
    var any = false;
    var cursor = value;
    while (cursor != null and cursor[0] != 0) {
        while (cursor[0] != 0 and !digit(cursor[0])) cursor += 1;
        if (cursor[0] == 0) break;
        var end: [*c]u8 = null;
        var first = c.strtol(cursor, &end, 10);
        var last = first;
        cursor = end;
        if (cursor[0] == '-') {
            cursor += 1;
            last = c.strtol(cursor, &end, 10);
            cursor = end;
        }
        if (first > last) std.mem.swap(c_long, &first, &last);
        first = @max(first, 0);
        last = @min(last, @as(c_long, @intCast(set.len)) - 1);
        var number = first;
        while (number <= last) : (number += 1) {
            set[@intCast(number)] = true;
            any = true;
        }
    }
    return any;
}
fn lineSetMatches(expected: [*c]const u8, actual: [*c]const u8) bool {
    var expected_lines: [256]bool = @splat(false);
    var actual_lines: [256]bool = @splat(false);
    if (!parseLineSet(expected, &expected_lines) or !parseLineSet(actual, &actual_lines)) return false;
    var hit = false;
    // The model may give the smallest exact subset of an audited equivalent
    // range. At least one line must be present; every supplied line must match.
    for (actual_lines, expected_lines) |supplied, accepted| {
        if (supplied) {
            if (!accepted) return false;
            hit = true;
        }
    }
    return hit;
}
fn payload(value: []const u8) ?[]const u8 {
    const marker = lastMarker(value) orelse return null;
    var position = marker + 6;
    while (position < value.len and whitespace(value[position])) position += 1;
    if (position == value.len or value[position] != ':') return null;
    position += 1;
    while (position < value.len and (whitespace(value[position]) or value[position] == '*')) position += 1;
    return value[position..];
}
pub fn normalizeExact(value: []const u8, output: []u8) void {
    if (output.len == 0) return;
    var length: usize = 0;
    var position: usize = 0;
    while (position < value.len) : (position += 1) {
        const byte = value[position];
        if (whitespace(byte) or byte == '$' or byte == '*' or byte == '{' or byte == '}') continue;
        if (byte == '\\') {
            const tail = value[position..];
            if (std.mem.startsWith(u8, tail, "\\boxed")) position += 5 else if (std.mem.startsWith(u8, tail, "\\left")) position += 4 else if (std.mem.startsWith(u8, tail, "\\right")) position += 5;
            continue;
        }
        put(output, &length, lower(byte));
    }
    while (length > 0 and (output[length - 1] == '.' or output[length - 1] == ';')) length -= 1;
    output[length] = 0;
}
pub fn normalizeOrdered(value: []const u8, output: []u8) void {
    if (output.len == 0) return;
    var length: usize = 0;
    var in_tag = false;
    for (value) |byte| {
        if (byte == '<') {
            in_tag = true;
            continue;
        }
        if (in_tag) {
            if (byte == '>') in_tag = false;
            continue;
        }
        if (c.isalnum(byte) != 0 or byte == ',' or byte == '-' or byte == '+' or byte == '/' or byte == '.') put(output, &length, lower(byte));
    }
    while (length > 0 and (output[length - 1] == '.' or output[length - 1] == ',')) length -= 1;
    output[length] = 0;
}
pub fn normalizeRational(value: []const u8, output: []u8) void {
    if (output.len == 0) return;
    if (std.mem.find(u8, value, "\\frac")) |fraction| {
        if (std.mem.findScalarPos(u8, value, fraction, '{')) |first| {
            if (std.mem.findScalarPos(u8, value, first + 1, '}')) |first_end| {
                if (std.mem.findScalarPos(u8, value, first_end + 1, '{')) |second| {
                    if (std.mem.findScalarPos(u8, value, second + 1, '}')) |second_end| {
                        var length: usize = 0;
                        for (value[first + 1 .. first_end]) |byte| {
                            if (!whitespace(byte)) put(output, &length, byte);
                        }
                        put(output, &length, '/');
                        for (value[second + 1 .. second_end]) |byte| {
                            if (!whitespace(byte)) put(output, &length, byte);
                        }
                        output[length] = 0;
                        return;
                    }
                }
            }
        }
    }
    var length: usize = 0;
    for (value) |byte| {
        if (digit(byte) or byte == '-' or byte == '+' or byte == '/' or byte == '.') put(output, &length, byte);
    }
    while (length > 0 and output[length - 1] == '.') length -= 1;
    output[length] = 0;
}
fn payloadAnswer(case: *const Case, generated: []const u8, output: []u8) void {
    Common.copy(output, "?");
    const answer = payload(visible(generated)) orelse return;
    switch (answerKind(case)) {
        c.EVAL_ANSWER_RATIONAL => normalizeRational(answer, output),
        c.EVAL_ANSWER_ORDERED_SEQUENCE => normalizeOrdered(answer, output),
        c.EVAL_ANSWER_EXACT_TEXT => normalizeExact(answer, output),
        else => {},
    }
    if (output[0] == 0) Common.copy(output, "?");
}
pub fn extract(case: *const Case, generated: []const u8, output: []u8) void {
    if (output.len == 0) return;
    switch (answerKind(case)) {
        c.EVAL_ANSWER_CHOICE => {
            output[0] = letter(generated, choices(case));
            if (output.len > 1) output[1] = 0;
        },
        c.EVAL_ANSWER_LINE_SET => compsec(generated, output),
        c.EVAL_ANSWER_INTEGER => integer(generated, output),
        c.EVAL_ANSWER_RATIONAL, c.EVAL_ANSWER_EXACT_TEXT, c.EVAL_ANSWER_ORDERED_SEQUENCE => payloadAnswer(case, generated, output),
        else => Common.copy(output, "?"),
    }
}
pub fn strict(case: *const Case, generated: []const u8, output: []u8) bool {
    if (output.len == 0) return false;
    Common.copy(output, "?");
    var end = generated.len;
    while (end > 0 and whitespace(generated[end - 1])) end -= 1;
    if (end == 0) return false;
    var start = end;
    while (start > 0 and generated[start - 1] != '\n' and generated[start - 1] != '\r') start -= 1;
    const line = allocator.dupeSentinel(u8, generated[start..end], 0) catch {
        _ = c.fprintf(c.__stderrp, "ds4-eval: out of memory while grading answer\n");
        return false;
    };
    defer allocator.free(line);
    if (payload(line) == null) return false;
    extract(case, line, output);
    return true;
}
pub fn normalizeExpected(kind: c.eval_answer_kind, value: []const u8, output: []u8) void {
    switch (kind) {
        c.EVAL_ANSWER_RATIONAL => normalizeRational(value, output),
        c.EVAL_ANSWER_EXACT_TEXT => normalizeExact(value, output),
        c.EVAL_ANSWER_ORDERED_SEQUENCE => normalizeOrdered(value, output),
        c.EVAL_ANSWER_INTEGER => normalizeInteger(value, output),
        else => Common.copy(output, value),
    }
}
pub fn matches(case: *const Case, actual: [*c]const u8) bool {
    const kind = answerKind(case);
    if (kind == c.EVAL_ANSWER_CHOICE) return actual != null and actual[0] != 0 and case.answer != null and actual[0] == case.answer[0];
    if (kind == c.EVAL_ANSWER_LINE_SET) return actual != null and case.answer != null and lineSetMatches(case.answer, actual);
    if (actual == null or case.answer == null) return false;
    var expected: [c.EVAL_ANSWER_MAX]u8 = undefined;
    normalizeExpected(kind, Common.text(case.answer), &expected);
    if (equal(Common.text(actual), Common.written(&expected))) return true;
    for (case.alias) |alias| {
        if (alias == null) break;
        normalizeExpected(kind, Common.text(alias), &expected);
        if (equal(Common.text(actual), Common.written(&expected))) return true;
    }
    return false;
}
