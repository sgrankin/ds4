//! Native grammar ownership. One compiler/cache per vocabulary, immutable
//! compiled grammars shared by generations, and one mutable matcher per decode.
const std = @import("std");
const c = @import("grammar_c");

pub const Diagnostic = struct {
    bytes: [1024]u8 = @splat(0),

    pub fn text(self: *const Diagnostic) []const u8 {
        return std.mem.sliceTo(&self.bytes, 0);
    }
};

pub const Kind = enum(c_uint) {
    structural = c.DS4_GRAMMAR_STRUCTURAL,
    json_schema = c.DS4_GRAMMAR_JSON_SCHEMA,
    regex = c.DS4_GRAMMAR_REGEX,
};

pub const Compiler = struct {
    handle: *c.ds4_grammar_compiler,

    pub fn init(allocator: std.mem.Allocator, vocabulary: []const []const u8, stops: []const i32, diagnostic: *Diagnostic) !Compiler {
        return initTokenAware(allocator, vocabulary, stops, &.{}, diagnostic);
    }

    /// Controls are dedicated IDs. Structural framing uses integer TokenFormat
    /// nodes; ordinary IDs spelling the same bytes can still carry literal data.
    /// Native XML styles are ds4_deepseek41_xml and ds4_glm_xml.
    pub fn initTokenAware(allocator: std.mem.Allocator, vocabulary: []const []const u8, stops: []const i32, controls: []const i32, diagnostic: *Diagnostic) !Compiler {
        const pieces = try allocator.alloc(c.ds4_grammar_piece, vocabulary.len);
        defer allocator.free(pieces);
        for (vocabulary, pieces) |bytes, *piece| piece.* = .{ .bytes = bytes.ptr, .len = bytes.len };
        return .{ .handle = c.ds4_grammar_compiler_new_token_aware(pieces.ptr, pieces.len, stops.ptr, stops.len, controls.ptr, controls.len, &diagnostic.bytes, diagnostic.bytes.len) orelse return error.GrammarTokenizer };
    }

    pub fn deinit(self: *Compiler) void {
        c.ds4_grammar_compiler_free(self.handle);
        self.* = undefined;
    }

    pub fn compile(self: Compiler, kind: Kind, source: []const u8, strict_json: bool, diagnostic: *Diagnostic) !Compiled {
        const handle = c.ds4_grammar_compile(self.handle, @backingInt(kind), source.ptr, source.len, strict_json, &diagnostic.bytes, diagnostic.bytes.len) orelse return error.GrammarCompile;
        return .{ .handle = handle, .vocab_size = c.ds4_grammar_compiled_vocab_size(handle) };
    }
};

pub const Compiled = struct {
    handle: *c.ds4_grammar_compiled,
    vocab_size: usize,

    pub fn deinit(self: *Compiled) void {
        c.ds4_grammar_compiled_free(self.handle);
        self.* = undefined;
    }

    pub fn matcher(self: Compiled, diagnostic: *Diagnostic) !Matcher {
        return .{
            .handle = c.ds4_grammar_matcher_new(self.handle, &diagnostic.bytes, diagnostic.bytes.len) orelse return error.GrammarMatcher,
            .vocab_size = self.vocab_size,
        };
    }
};

pub const Matcher = struct {
    handle: *c.ds4_grammar_matcher,
    vocab_size: usize,

    pub fn deinit(self: *Matcher) void {
        c.ds4_grammar_matcher_free(self.handle);
        self.* = undefined;
    }

    pub fn wordCount(self: Matcher) usize {
        return (self.vocab_size + 31) / 32;
    }

    pub fn accept(self: *Matcher, token: i32, diagnostic: *Diagnostic) !bool {
        return switch (c.ds4_grammar_accept(self.handle, token, &diagnostic.bytes, diagnostic.bytes.len)) {
            0 => false,
            1 => true,
            else => error.GrammarAccept,
        };
    }

    /// Returns whether the mask contains any disabled tokens. A true bit allows
    /// a token; apply it before top-k/top-p/temperature filtering.
    pub fn fill(self: *Matcher, words: []u32, diagnostic: *Diagnostic) !bool {
        return switch (c.ds4_grammar_fill(self.handle, words.ptr, words.len, &diagnostic.bytes, diagnostic.bytes.len)) {
            0 => false,
            1 => true,
            else => error.GrammarMask,
        };
    }

    pub fn rollback(self: *Matcher, tokens: usize, diagnostic: *Diagnostic) !void {
        if (!c.ds4_grammar_rollback(self.handle, tokens, &diagnostic.bytes, diagnostic.bytes.len)) return error.GrammarRollback;
    }

    pub fn completed(self: Matcher) bool {
        return c.ds4_grammar_completed(self.handle);
    }

    pub fn terminated(self: Matcher) bool {
        return c.ds4_grammar_terminated(self.handle);
    }
};

pub fn allowed(words: []const u32, token: usize) bool {
    return (words[token / 32] & (@as(u32, 1) << @intCast(token % 32))) != 0;
}
