const std = @import("std");
const grammar = @import("grammar");
const format = @import("server/grammar_format.zig");
const expect = std.testing.expect;

const vocabulary = [_][]const u8{
    "{",     "}",      "\"",   "x",    ":", "1", "2", "a", "b", "c", "<eos>", "\"x\"", "null",
    "\x00",  "\xc3",   "\xa9",
    "é",
    "<tag>", "</tag>", "<",    "tag>",
};
const eos = 10;

fn compiler(diagnostic: *grammar.Diagnostic) !grammar.Compiler {
    return grammar.Compiler.init(std.testing.allocator, &vocabulary, &.{eos}, diagnostic);
}

fn accept(matcher: *grammar.Matcher, token: i32, diagnostic: *grammar.Diagnostic) !void {
    if (!try matcher.accept(token, diagnostic)) return error.UnexpectedGrammarRejection;
}

test "regex mask advances, rejects and rolls back through stop" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    defer owner.deinit();
    var compiled = try owner.compile(.regex, "a(b|c)", true, &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    var words: [1]u32 = undefined;
    try expect(matcher.wordCount() == words.len);
    try expect(try matcher.fill(&words, &diagnostic));
    try expect(grammar.allowed(&words, 7));
    try expect(!grammar.allowed(&words, 8));
    try accept(&matcher, 7, &diagnostic);
    _ = try matcher.fill(&words, &diagnostic);
    try expect(grammar.allowed(&words, 8) and grammar.allowed(&words, 9));
    try expect(!try matcher.accept(3, &diagnostic));
    try accept(&matcher, 8, &diagnostic);
    try expect(matcher.completed() and !matcher.terminated());
    _ = try matcher.fill(&words, &diagnostic);
    try expect(grammar.allowed(&words, eos));
    try expect(!grammar.allowed(&words, 7));
    try accept(&matcher, eos, &diagnostic);
    try expect(matcher.terminated());
    try matcher.rollback(2, &diagnostic);
    try expect(!matcher.completed() and !matcher.terminated());
    _ = try matcher.fill(&words, &diagnostic);
    try expect(grammar.allowed(&words, 8) and grammar.allowed(&words, 9));
    try accept(&matcher, 9, &diagnostic);
    try accept(&matcher, eos, &diagnostic);
    try expect(matcher.terminated());
}

test "packet rollback retains more than seventeen steps including EOS" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    defer owner.deinit();
    var compiled = try owner.compile(.regex, "a{40}b", true, &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    for (0..24) |_| try accept(&matcher, 7, &diagnostic);
    var before: [1]u32 = undefined;
    _ = try matcher.fill(&before, &diagnostic);
    for (0..16) |_| try accept(&matcher, 7, &diagnostic);
    try accept(&matcher, 8, &diagnostic);
    try accept(&matcher, eos, &diagnostic);
    try expect(matcher.terminated());
    // Eighteen accepted steps cross the ordinary 17-token packet capacity and
    // remove the terminal stop. The next mask must equal the saved frontier.
    try matcher.rollback(18, &diagnostic);
    try expect(!matcher.completed() and !matcher.terminated());
    var after: [1]u32 = undefined;
    _ = try matcher.fill(&after, &diagnostic);
    try std.testing.expectEqualSlices(u32, &before, &after);
    for (0..16) |_| try accept(&matcher, 7, &diagnostic);
    try accept(&matcher, 8, &diagnostic);
    try accept(&matcher, eos, &diagnostic);
    try expect(matcher.terminated());
}

test "JSON schema masks numeric alternatives and enforces object shape" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    defer owner.deinit();
    var compiled = try owner.compile(.json_schema,
        \\{"type":"object","properties":{"x":{"type":"integer","enum":[1]}},"required":["x"],"additionalProperties":false}
    , true, &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    for ([_]i32{ 0, 11, 4 }) |token| try accept(&matcher, token, &diagnostic);
    var words: [1]u32 = undefined;
    _ = try matcher.fill(&words, &diagnostic);
    try expect(grammar.allowed(&words, 5));
    try expect(!grammar.allowed(&words, 6));
    try accept(&matcher, 5, &diagnostic);
    try accept(&matcher, 1, &diagnostic);
    try expect(matcher.completed());
    try accept(&matcher, eos, &diagnostic);
}

test "structural token markers leave ordinary spelling inside raw content" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    defer owner.deinit();
    var compiled = try owner.compile(.structural,
        \\{"type":"structural_tag","format":{"type":"tag","begin":{"type":"token","token":17},"content":{"type":"any_tokens","exclude_tokens":[18]},"end":{"type":"token","token":18}}}
    , true, &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    try accept(&matcher, 17, &diagnostic);
    // These ordinary tokens spell the same opener but remain payload.
    try accept(&matcher, 19, &diagnostic);
    try accept(&matcher, 20, &diagnostic);
    try accept(&matcher, 18, &diagnostic);
    try expect(matcher.completed());
    try accept(&matcher, eos, &diagnostic);
}

test "raw vocabulary preserves NUL and split UTF-8 pieces" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    defer owner.deinit();
    var nul = try owner.compile(.regex, "\\x00", true, &diagnostic);
    defer nul.deinit();
    var nul_matcher = try nul.matcher(&diagnostic);
    defer nul_matcher.deinit();
    try accept(&nul_matcher, 13, &diagnostic);
    try expect(nul_matcher.completed());

    var unicode = try owner.compile(.regex, "é", true, &diagnostic);
    defer unicode.deinit();
    var unicode_matcher = try unicode.matcher(&diagnostic);
    defer unicode_matcher.deinit();
    var words: [1]u32 = undefined;
    _ = try unicode_matcher.fill(&words, &diagnostic);
    try expect(grammar.allowed(&words, 14) and grammar.allowed(&words, 16));
    try accept(&unicode_matcher, 14, &diagnostic);
    _ = try unicode_matcher.fill(&words, &diagnostic);
    try expect(grammar.allowed(&words, 15));
    try accept(&unicode_matcher, 15, &diagnostic);
    try expect(unicode_matcher.completed());
}

test "compiled and matcher ownership survives compiler retirement" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    var compiled = try owner.compile(.regex, "a", true, &diagnostic);
    owner.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    compiled.deinit();
    defer matcher.deinit();
    try accept(&matcher, 7, &diagnostic);
    try accept(&matcher, eos, &diagnostic);
    try expect(matcher.terminated());
}

test "schema, regex and structural errors reach the caller" {
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try compiler(&diagnostic);
    defer owner.deinit();
    try std.testing.expectError(error.GrammarCompile, owner.compile(.json_schema, "{", true, &diagnostic));
    try expect(diagnostic.text().len != 0);
    try std.testing.expectError(error.GrammarCompile, owner.compile(.regex, "[", true, &diagnostic));
    try expect(diagnostic.text().len != 0);
    try std.testing.expectError(error.GrammarCompile, owner.compile(.structural, "{}", true, &diagnostic));
    try expect(diagnostic.text().len != 0);
}

const ByteVocabulary = struct {
    bytes: [256][1]u8,
    pieces: [262][]const u8,

    fn init(self: *ByteVocabulary) void {
        for (&self.bytes, self.pieces[0..256], 0..) |*byte, *piece, id| {
            byte[0] = @intCast(id);
            piece.* = byte;
        }
        self.pieces[256..].* = .{ "｜DSML｜", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<eos>" };
    }
};

fn acceptBytes(matcher: *grammar.Matcher, bytes: []const u8, diagnostic: *grammar.Diagnostic) !void {
    for (bytes) |byte| try accept(matcher, byte, diagnostic);
}

fn xml(owner: grammar.Compiler, schema: []const u8, style: []const u8, diagnostic: *grammar.Diagnostic) !grammar.Compiled {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, schema, .{});
    defer parsed.deinit();
    const source = try std.json.Stringify.valueAlloc(a, .{
        .type = "structural_tag",
        .format = .{ .type = "json_schema", .json_schema = parsed.value, .style = style },
    }, .{});
    defer a.free(source);
    return owner.compile(.structural, source, true, diagnostic) catch |err| {
        std.debug.print("Native XML compile: {s}\n", .{diagnostic.text()});
        return err;
    };
}

test "DS41 XML requires control identity and preserves literal delimiter strings" {
    var bytes: ByteVocabulary = undefined;
    bytes.init();
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try grammar.Compiler.initTokenAware(std.testing.allocator, &bytes.pieces, &.{261}, &.{256}, &diagnostic);
    defer owner.deinit();
    // Exercise an unrestricted raw body, a strict enum, and the referenced enum
    // branch where the converter creates a separate parameter-wrapper rule.
    for ([_][]const u8{
        \\{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}
        ,
        \\{"type":"object","properties":{"x":{"type":"string","minLength":1,"maxLength":128}},"required":["x"],"additionalProperties":false}
        ,
        \\{"type":"object","properties":{"x":{"type":"string","pattern":".*"}},"required":["x"],"additionalProperties":false}
        ,
        \\{"type":"object","properties":{"x":{"type":"string","enum":["</｜DSML｜ parameter>"]}},"required":["x"],"additionalProperties":false}
        ,
        \\{"$defs":{"raw":{"const":"</｜DSML｜ parameter>"}},"type":"object","properties":{"x":{"$ref":"#/$defs/raw"}},"required":["x"],"additionalProperties":false}
        ,
    }) |schema| {
        var compiled = try xml(owner, schema, "ds4_deepseek41_xml", &diagnostic);
        defer compiled.deinit();
        var matcher = try compiled.matcher(&diagnostic);
        defer matcher.deinit();
        try acceptBytes(&matcher, "<", &diagnostic);
        var mask: [9]u32 = undefined;
        _ = try matcher.fill(&mask, &diagnostic);
        try expect(grammar.allowed(&mask, 256));
        try expect(!grammar.allowed(&mask, 0xef)); // Ordinary UTF-8 marker bytes cannot frame the parameter.
        try accept(&matcher, 256, &diagnostic);
        try acceptBytes(&matcher, " parameter name=\"x\" string=\"true\">", &diagnostic);
        _ = try matcher.fill(&mask, &diagnostic);
        try expect(!grammar.allowed(&mask, 256));
        try acceptBytes(&matcher, "</｜DSML｜ parameter>", &diagnostic);
        try acceptBytes(&matcher, "</", &diagnostic);
        try accept(&matcher, 256, &diagnostic);
        try acceptBytes(&matcher, " parameter>", &diagnostic);
        try expect(matcher.completed());
        try accept(&matcher, 261, &diagnostic);
        try expect(matcher.terminated());
    }
}

test "Flint XML token wrappers preserve strict enum delimiter text" {
    var bytes: ByteVocabulary = undefined;
    bytes.init();
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try grammar.Compiler.initTokenAware(std.testing.allocator, &bytes.pieces, &.{261}, &.{ 257, 258, 259, 260 }, &diagnostic);
    defer owner.deinit();
    var compiled = try xml(owner,
        \\{"type":"object","properties":{"x":{"type":"string","enum":["</arg_value>"]}},"required":["x"],"additionalProperties":false}
    , "ds4_glm_xml", &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    var mask: [9]u32 = undefined;
    _ = try matcher.fill(&mask, &diagnostic);
    try expect(grammar.allowed(&mask, 257) and !grammar.allowed(&mask, '<'));
    try accept(&matcher, 257, &diagnostic);
    try acceptBytes(&matcher, "x", &diagnostic);
    try accept(&matcher, 258, &diagnostic);
    try accept(&matcher, 259, &diagnostic);
    _ = try matcher.fill(&mask, &diagnostic);
    try expect(grammar.allowed(&mask, '<') and !grammar.allowed(&mask, 260));
    try acceptBytes(&matcher, "</arg_value>", &diagnostic);
    try accept(&matcher, 260, &diagnostic);
    try expect(matcher.completed());
    try accept(&matcher, 261, &diagnostic);
}

test "non-strict DS41 parameter lists keep nested values in JSON" {
    var bytes: ByteVocabulary = undefined;
    bytes.init();
    var diagnostic: grammar.Diagnostic = .{};
    var owner = try grammar.Compiler.initTokenAware(std.testing.allocator, &bytes.pieces, &.{261}, &.{256}, &diagnostic);
    defer owner.deinit();
    var compiled = try xml(owner, "true", "ds4_deepseek41_xml", &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    try acceptBytes(&matcher, "<", &diagnostic);
    try accept(&matcher, 256, &diagnostic);
    try acceptBytes(&matcher, " parameter name=\"x\" string=\"false\">[\"</｜DSML｜ parameter>\",{\"x\":1}]</", &diagnostic);
    try accept(&matcher, 256, &diagnostic);
    try acceptBytes(&matcher, " parameter>", &diagnostic);
    try expect(matcher.completed());
    try accept(&matcher, 261, &diagnostic);
}

const ModelVocabulary = struct {
    bytes: [256][1]u8,
    pieces: [256 + format.markers.len][]const u8,
    ids: format.Ids,
    controls: [format.markers.len]i32,
    control_count: usize,

    fn init(self: *ModelVocabulary, flint: bool) void {
        self.ids = @splat(-1);
        self.control_count = 0;
        for (&self.bytes, self.pieces[0..256], 0..) |*byte, *piece, id| {
            byte[0] = @intCast(id);
            piece.* = byte;
        }
        for (format.markers, 0..) |marker, index| {
            const applies = if (flint) (index <= 1 or (index >= 3 and index <= 8) or index >= 14) else (index <= 2 or (index >= 9 and index <= 13));
            if (!applies) continue;
            const id: i32 = @intCast(256 + self.control_count);
            self.ids[index] = id;
            self.controls[self.control_count] = id;
            self.pieces[@intCast(id)] = marker;
            self.control_count += 1;
        }
    }

    fn compiler(self: *ModelVocabulary, eos_id: i32, diagnostic: *grammar.Diagnostic) !grammar.Compiler {
        return grammar.Compiler.initTokenAware(std.testing.allocator, self.pieces[0 .. 256 + self.control_count], &.{eos_id}, self.controls[0..self.control_count], diagnostic);
    }
};

fn fullFormat(owner: grammar.Compiler, ids: format.Ids, flint: bool, diagnostic: *grammar.Diagnostic) !grammar.Compiled {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const descriptor = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), if (flint)
        \\{"tools":[{"type":"function","function":{"name":"echo","strict":true,"parameters":{"type":"object","properties":{"x":{"type":"string","enum":["</arg_value>"]}},"required":["x"],"additionalProperties":false}}}],"required":false,"thinking":true,"format":null}
    else
        \\{"tools":[{"type":"function","function":{"name":"echo","strict":true,"parameters":{"type":"object","properties":{"x":{"type":"string","enum":["</｜DSML｜ parameter>"]}},"required":["x"],"additionalProperties":false}}}],"required":false,"thinking":true,"format":null}
    , .{});
    const source = try format.build(arena.allocator(), descriptor, flint, ids);
    return owner.compile(.structural, source, true, diagnostic) catch |err| {
        std.debug.print("Composed format compile: {s}\n", .{diagnostic.text()});
        return err;
    };
}

test "serving DS41 thinking/auto format accepts a full strict call and literal controls" {
    var bytes: ModelVocabulary = undefined;
    bytes.init(false);
    var diagnostic: grammar.Diagnostic = .{};
    const ids = bytes.ids;
    const stop = ids[10];
    var owner = try bytes.compiler(stop, &diagnostic);
    defer owner.deinit();
    var compiled = try fullFormat(owner, ids, false, &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    try acceptBytes(&matcher, "A literal </think> remains reasoning.", &diagnostic);
    try accept(&matcher, ids[1], &diagnostic);
    try acceptBytes(&matcher, "\n\n<", &diagnostic);
    try accept(&matcher, ids[2], &diagnostic);
    try acceptBytes(&matcher, " calls>\n<", &diagnostic);
    try accept(&matcher, ids[2], &diagnostic);
    try acceptBytes(&matcher, " invoke name=\"echo\">\n<", &diagnostic);
    try accept(&matcher, ids[2], &diagnostic);
    try acceptBytes(&matcher, " parameter name=\"x\" string=\"true\"></｜DSML｜ parameter></", &diagnostic);
    try accept(&matcher, ids[2], &diagnostic);
    try acceptBytes(&matcher, " parameter>\n</", &diagnostic);
    try accept(&matcher, ids[2], &diagnostic);
    try acceptBytes(&matcher, " invoke>\n</", &diagnostic);
    try accept(&matcher, ids[2], &diagnostic);
    try acceptBytes(&matcher, " calls>", &diagnostic);
    try expect(matcher.completed());
    try accept(&matcher, stop, &diagnostic);
    try expect(matcher.terminated());

    // Auto permits an ordinary answer too; reasoning is still present.
    var answer = try compiled.matcher(&diagnostic);
    defer answer.deinit();
    try acceptBytes(&answer, "Think first", &diagnostic);
    try accept(&answer, ids[1], &diagnostic);
    try acceptBytes(&answer, "Ordinary answer with literal <｜DSML｜ calls>.", &diagnostic);
    try accept(&answer, stop, &diagnostic);
}

test "serving Flint auto format accepts reasoning/strict tools and an ordinary answer" {
    var bytes: ModelVocabulary = undefined;
    bytes.init(true);
    var diagnostic: grammar.Diagnostic = .{};
    const ids = bytes.ids;
    const stop = ids[15];
    var owner = try bytes.compiler(stop, &diagnostic);
    defer owner.deinit();
    var compiled = try fullFormat(owner, ids, true, &diagnostic);
    defer compiled.deinit();
    var matcher = try compiled.matcher(&diagnostic);
    defer matcher.deinit();
    try accept(&matcher, ids[0], &diagnostic);
    try acceptBytes(&matcher, "Literal </think> remains reasoning.", &diagnostic);
    try accept(&matcher, ids[1], &diagnostic);
    try acceptBytes(&matcher, "Checking that value.", &diagnostic);
    try accept(&matcher, ids[3], &diagnostic);
    try acceptBytes(&matcher, "echo", &diagnostic);
    try accept(&matcher, ids[5], &diagnostic);
    try acceptBytes(&matcher, "x", &diagnostic);
    try accept(&matcher, ids[6], &diagnostic);
    try accept(&matcher, ids[7], &diagnostic);
    try acceptBytes(&matcher, "</arg_value>", &diagnostic);
    try accept(&matcher, ids[8], &diagnostic);
    try accept(&matcher, ids[4], &diagnostic);
    try accept(&matcher, ids[20], &diagnostic);
    try expect(matcher.completed());
    try accept(&matcher, stop, &diagnostic);
    try expect(matcher.terminated());

    var answer = try compiled.matcher(&diagnostic);
    defer answer.deinit();
    try acceptBytes(&answer, "Ordinary answer with literal <tool_call>.", &diagnostic);
    try accept(&answer, stop, &diagnostic);
}
