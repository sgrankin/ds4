//! Streaming terminal presentation, not a Markdown document parser. Plain text
//! is immediate; only ambiguous inline spans and split controls are buffered.
//! The record is shared once with the C fallback. Worker remains an opaque
//! output context; no session, tool execution or UI policy is inspected here.
const std = @import("std");
const c = @import("agent_presentation_c");
const allocator = std.heap.c_allocator;
pub const Record = c.agent_token_renderer;
pub const Capture = c.agent_tail_capture;

fn fail(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn allocate(count: usize) [*]u8 {
    return @ptrCast(c.malloc(@max(count, 1)) orelse fail("ds4-agent: malloc"));
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn equalFold(a: []const u8, b: []const u8) bool {
    return a.len == b.len and c.strncasecmp(a.ptr, b.ptr, a.len) == 0;
}
fn space(byte: u8) bool {
    return c.isspace(byte) != 0;
}
fn punctuation(byte: u8) bool {
    return c.ispunct(byte) != 0;
}
fn alphaNumeric(byte: u8) bool {
    return c.isalnum(byte) != 0;
}

pub fn captureAppend(capture: *Capture, bytes: []const u8) void {
    if (bytes.len == 0 or capture.cap == 0) return;
    if (capture.buf == null) capture.buf = allocate(capture.cap);
    capture.total +%= bytes.len;
    const ring = capture.buf[0..capture.cap];
    if (bytes.len >= ring.len) {
        @memcpy(ring, bytes[bytes.len - ring.len ..]);
        capture.start = 0;
        capture.len = ring.len;
        return;
    }
    var remaining = bytes;
    if (capture.len < ring.len) {
        const first = @min(remaining.len, ring.len - capture.len);
        const position = (capture.start + capture.len) % ring.len;
        const chunk = @min(first, ring.len - position);
        @memcpy(ring[position..][0..chunk], remaining[0..chunk]);
        @memcpy(ring[0 .. first - chunk], remaining[chunk..first]);
        capture.len += first;
        remaining = remaining[first..];
    }
    while (remaining.len != 0) {
        const position = (capture.start + capture.len) % ring.len;
        const chunk = @min(remaining.len, ring.len - position);
        @memcpy(ring[position..][0..chunk], remaining[0..chunk]);
        capture.start = (capture.start + chunk) % ring.len;
        remaining = remaining[chunk..];
    }
}
pub fn captureTake(capture: *Capture, length: ?*usize) [*:0]u8 {
    const output = allocate(capture.len + 1);
    const first = @min(capture.len, capture.cap - capture.start);
    if (capture.len != 0) {
        @memcpy(output[0..first], capture.buf[capture.start..][0..first]);
        @memcpy(output[first..capture.len], capture.buf[0 .. capture.len - first]);
    }
    output[capture.len] = 0;
    if (length) |out| out.* = capture.len;
    c.free(capture.buf);
    capture.* = std.mem.zeroes(Capture);
    return @ptrCast(output);
}

// A tiny language table plus one line-oriented tokenizer for comments, strings,
// numbers and separator-bounded keywords (inspired by antirez/kilo). This is
// deliberately not a full parser: it only makes fenced code easier to read.
pub const Syntax = struct {
    name: []const u8,
    aliases: []const u8,
    keywords: []const []const u8,
    comments: []const []const u8,
    multiline_start: ?[]const u8,
    multiline_end: ?[]const u8,
    flags: u8,
};
const numbers = 1;
const strings = 2;
const backtick_strings = 4;
const case_insensitive = 8;
const Highlight = enum { normal, comment, keyword1, keyword2, string, number };
// Fixed language data from the original deliberately small highlighter.
const kw_generic = [_][]const u8{
    "if",        "else",      "for",    "while",     "do",    "switch",  "case",    "default",
    "break",     "continue",  "return", "try",       "catch", "finally", "throw",   "throws",
    "class",     "struct",    "enum",   "interface", "trait", "impl",    "fn",      "func",
    "function",  "def",       "lambda", "let",       "var",   "const",   "static",  "public",
    "private",   "protected", "import", "include",   "from",  "export",  "package", "module",
    "namespace", "new",       "delete", "async",     "await", "yield",   "match",   "type",
    "true|",     "false|",    "null|",  "nil|",      "none|", "None|",   "NULL|",   "void|",
    "int|",      "long|",     "float|", "double|",   "char|", "bool|",   "string|", "String|",
    "usize|",    "isize|",    "u8|",    "u16|",      "u32|",  "u64|",    "i8|",     "i16|",
    "i32|",      "i64|",
};

const kw_c = [_][]const u8{
    "auto",        "break",     "case",      "continue",     "default",   "do",      "else",             "enum",
    "extern",      "for",       "goto",      "if",           "register",  "return",  "sizeof",           "static",
    "struct",      "switch",    "typedef",   "union",        "volatile",  "while",   "alignas",          "alignof",
    "and",         "and_eq",    "asm",       "bitand",       "bitor",     "class",   "compl",            "constexpr",
    "const_cast",  "decltype",  "delete",    "dynamic_cast", "explicit",  "export",  "false",            "friend",
    "inline",      "mutable",   "namespace", "new",          "noexcept",  "not",     "not_eq",           "nullptr",
    "operator",    "or",        "or_eq",     "private",      "protected", "public",  "reinterpret_cast", "static_assert",
    "static_cast", "template",  "this",      "thread_local", "throw",     "true",    "try",              "typeid",
    "typename",    "virtual",   "xor",       "xor_eq",       "NULL|",     "bool|",   "char|",            "const|",
    "double|",     "float|",    "int|",      "long|",        "short|",    "signed|", "size_t|",          "ssize_t|",
    "uint8_t|",    "uint16_t|", "uint32_t|", "uint64_t|",    "unsigned|", "void|",
};

const kw_python = [_][]const u8{
    "and",      "as",     "assert", "async",   "await", "break",  "case",    "class",
    "continue", "def",    "del",    "elif",    "else",  "except", "finally", "for",
    "from",     "global", "if",     "import",  "in",    "is",     "lambda",  "match",
    "nonlocal", "not",    "or",     "pass",    "raise", "return", "try",     "while",
    "with",     "yield",  "False|", "None|",   "True|", "bool|",  "bytes|",  "dict|",
    "float|",   "int|",   "list|",  "object|", "set|",  "str|",   "tuple|",
};

const kw_js = [_][]const u8{
    "async",     "await",    "break",     "case",    "catch",    "class",      "const",     "continue",
    "debugger",  "default",  "delete",    "do",      "else",     "export",     "extends",   "finally",
    "for",       "from",     "function",  "get",     "if",       "import",     "in",        "instanceof",
    "let",       "new",      "of",        "return",  "set",      "static",     "super",     "switch",
    "this",      "throw",    "try",       "typeof",  "var",      "void",       "while",     "with",
    "yield",     "abstract", "as",        "declare", "enum",     "implements", "interface", "keyof",
    "namespace", "private",  "protected", "public",  "readonly", "type",       "any|",      "boolean|",
    "false|",    "never|",   "null|",     "number|", "string|",  "symbol|",    "true|",     "undefined|",
    "unknown|",  "void|",
};

const kw_java = [_][]const u8{
    "abstract", "assert",       "break",      "case",   "catch",      "class",     "const",    "continue",
    "default",  "do",           "else",       "enum",   "extends",    "final",     "finally",  "for",
    "goto",     "if",           "implements", "import", "instanceof", "interface", "native",   "new",
    "package",  "private",      "protected",  "public", "return",     "static",    "strictfp", "super",
    "switch",   "synchronized", "this",       "throw",  "throws",     "transient", "try",      "volatile",
    "while",    "boolean|",     "byte|",      "char|",  "double|",    "false|",    "float|",   "int|",
    "long|",    "null|",        "short|",     "true|",  "void|",
};

const kw_csharp = [_][]const u8{
    "abstract", "as",       "base",      "break",    "case",       "catch",     "checked",   "class",
    "const",    "continue", "default",   "delegate", "do",         "else",      "enum",      "event",
    "explicit", "extern",   "finally",   "fixed",    "for",        "foreach",   "goto",      "if",
    "implicit", "in",       "interface", "internal", "is",         "lock",      "namespace", "new",
    "operator", "out",      "override",  "params",   "private",    "protected", "public",    "readonly",
    "ref",      "return",   "sealed",    "sizeof",   "stackalloc", "static",    "struct",    "switch",
    "this",     "throw",    "try",       "typeof",   "unchecked",  "unsafe",    "using",     "virtual",
    "volatile", "while",    "async",     "await",    "get",        "init",      "record",    "set",
    "var",      "bool|",    "byte|",     "char|",    "decimal|",   "double|",   "false|",    "float|",
    "int|",     "long|",    "null|",     "object|",  "sbyte|",     "short|",    "string|",   "true|",
    "uint|",    "ulong|",   "ushort|",   "void|",
};

const kw_go = [_][]const u8{
    "break",       "case",    "chan",  "const",      "continue",    "default", "defer",   "else",
    "fallthrough", "for",     "func",  "go",         "goto",        "if",      "import",  "interface",
    "map",         "package", "range", "return",     "select",      "struct",  "switch",  "type",
    "var",         "bool|",   "byte|", "complex64|", "complex128|", "error|",  "false|",  "float32|",
    "float64|",    "int|",    "int8|", "int16|",     "int32|",      "int64|",  "nil|",    "rune|",
    "string|",     "true|",   "uint|", "uint8|",     "uint16|",     "uint32|", "uint64|", "uintptr|",
};

const kw_rust = [_][]const u8{
    "as",      "async", "await",  "break",  "const",  "continue", "crate",  "dyn",
    "else",    "enum",  "extern", "fn",     "for",    "if",       "impl",   "in",
    "let",     "loop",  "match",  "mod",    "move",   "mut",      "pub",    "ref",
    "return",  "self",  "Self",   "static", "struct", "super",    "trait",  "type",
    "unsafe",  "use",   "where",  "while",  "bool|",  "char|",    "false|", "f32|",
    "f64|",    "i8|",   "i16|",   "i32|",   "i64|",   "i128|",    "isize|", "str|",
    "String|", "true|", "u8|",    "u16|",   "u32|",   "u64|",     "u128|",  "usize|",
};

const kw_shell = [_][]const u8{
    "case",     "do",       "done",   "elif",    "else",  "esac",     "fi",     "for",
    "function", "if",       "in",     "select",  "then",  "time",     "until",  "while",
    "break",    "continue", "return", "export",  "local", "readonly", "source", "test",
    "true|",    "false|",   "echo|",  "printf|", "cd|",   "pwd|",     "read|",  "set|",
    "unset|",   "shift|",
};

const kw_sql = [_][]const u8{
    "add",        "alter",  "and",        "as",       "asc",     "between",  "by",         "case",
    "check",      "column", "constraint", "create",   "delete",  "desc",     "distinct",   "drop",
    "else",       "end",    "exists",     "foreign",  "from",    "group",    "having",     "in",
    "index",      "insert", "into",       "is",       "join",    "key",      "left",       "like",
    "limit",      "not",    "null",       "on",       "or",      "order",    "outer",      "primary",
    "references", "right",  "select",     "set",      "table",   "then",     "union",      "unique",
    "update",     "values", "view",       "where",    "bigint|", "boolean|", "date|",      "decimal|",
    "false|",     "int|",   "integer|",   "numeric|", "real|",   "text|",    "timestamp|", "true|",
    "varchar|",
};

const kw_ruby = [_][]const u8{
    "BEGIN", "END",      "alias",  "and",    "begin", "break", "case",   "class",
    "def",   "defined?", "do",     "else",   "elsif", "end",   "ensure", "for",
    "if",    "in",       "module", "next",   "not",   "or",    "redo",   "rescue",
    "retry", "return",   "self",   "super",  "then",  "undef", "unless", "until",
    "when",  "while",    "yield",  "false|", "nil|",  "true|",
};

const kw_php = [_][]const u8{
    "abstract",     "and",        "array",     "as",        "break",      "callable", "case",       "catch",
    "class",        "clone",      "const",     "continue",  "declare",    "default",  "die",        "do",
    "echo",         "else",       "elseif",    "empty",     "enddeclare", "endfor",   "endforeach", "endif",
    "endswitch",    "endwhile",   "eval",      "exit",      "extends",    "final",    "finally",    "fn",
    "for",          "foreach",    "function",  "global",    "goto",       "if",       "implements", "include",
    "include_once", "instanceof", "insteadof", "interface", "isset",      "list",     "match",      "namespace",
    "new",          "or",         "print",     "private",   "protected",  "public",   "readonly",   "require",
    "require_once", "return",     "static",    "switch",    "throw",      "trait",    "try",        "unset",
    "use",          "var",        "while",     "xor",       "bool|",      "false|",   "float|",     "int|",
    "null|",        "string|",    "true|",     "void|",
};

const kw_swift = [_][]const u8{
    "actor",       "as",       "associatedtype", "async",       "await",     "break",   "case",      "catch",
    "class",       "continue", "default",        "defer",       "do",        "else",    "enum",      "extension",
    "fallthrough", "for",      "func",           "guard",       "if",        "import",  "in",        "init",
    "inout",       "is",       "let",            "nonisolated", "operator",  "private", "protocol",  "public",
    "repeat",      "return",   "self",           "Self",        "static",    "struct",  "subscript", "super",
    "switch",      "throw",    "throws",         "try",         "typealias", "var",     "where",     "while",
    "Any|",        "Bool|",    "Double|",        "false|",      "Float|",    "Int|",    "nil|",      "String|",
    "true|",       "Void|",
};

const kw_kotlin = [_][]const u8{
    "as",        "break",    "class",       "continue",    "do",      "else",       "false",     "for",
    "fun",       "if",       "in",          "interface",   "is",      "null",       "object",    "package",
    "return",    "super",    "this",        "throw",       "true",    "try",        "typealias", "typeof",
    "val",       "var",      "when",        "while",       "actual",  "annotation", "by",        "catch",
    "companion", "const",    "constructor", "crossinline", "data",    "enum",       "expect",    "external",
    "final",     "finally",  "import",      "infix",       "init",    "inline",     "inner",     "internal",
    "lateinit",  "noinline", "open",        "operator",    "out",     "override",   "private",   "protected",
    "public",    "reified",  "sealed",      "suspend",     "tailrec", "vararg",     "Any|",      "Boolean|",
    "Byte|",     "Char|",    "Double|",     "Float|",      "Int|",    "Long|",      "Short|",    "String|",
    "Unit|",
};

const kw_zig = [_][]const u8{
    "addrspace", "align",       "allowzero",   "and",     "anyframe",    "anytype",        "asm",      "async",
    "await",     "break",       "callconv",    "catch",   "comptime",    "const",          "continue", "defer",
    "else",      "enum",        "errdefer",    "error",   "export",      "extern",         "fn",       "for",
    "if",        "inline",      "linksection", "noalias", "noinline",    "nosuspend",      "opaque",   "or",
    "orelse",    "packed",      "pub",         "resume",  "return",      "struct",         "suspend",  "switch",
    "test",      "threadlocal", "try",         "union",   "unreachable", "usingnamespace", "var",      "volatile",
    "while",     "bool|",       "false|",      "f32|",    "f64|",        "i32|",           "i64|",     "null|",
    "true|",     "u8|",         "u16|",        "u32|",    "u64|",        "usize|",         "void|",
};

const kw_lua = [_][]const u8{
    "and",      "break",  "do",   "else", "elseif", "end",   "false", "for",
    "function", "goto",   "if",   "in",   "local",  "nil",   "not",   "or",
    "repeat",   "return", "then", "true", "until",  "while",
};

const kw_html = [_][]const u8{
    "a",      "body",   "button", "div",   "doctype", "form",    "h1",     "h2",
    "h3",     "head",   "html",   "input", "label",   "li",      "link",   "main",
    "meta",   "ol",     "option", "p",     "script",  "section", "select", "span",
    "style",  "table",  "tbody",  "td",    "th",      "thead",   "title",  "tr",
    "ul",     "class|", "href|",  "id|",   "name|",   "rel|",    "src|",   "type|",
    "value|",
};

const kw_css = [_][]const u8{
    "align-items", "background", "border",   "bottom", "color",           "display",   "flex",    "font",
    "font-size",   "gap",        "grid",     "height", "justify-content", "left",      "margin",  "max-width",
    "min-width",   "padding",    "position", "right",  "top",             "transform", "width",   "z-index",
    "absolute|",   "auto|",      "block|",   "flex|",  "grid|",           "hidden|",   "inline|", "none|",
    "relative|",   "solid|",
};

const languages = [_]Syntax{
    .{ .name = "generic", .aliases = " text txt", .keywords = &kw_generic, .comments = &.{ "//", "#" }, .multiline_start = "/*", .multiline_end = "*/", .flags = 7 },
    .{ .name = "c", .aliases = " c h cpp c++ cc cxx hpp hxx objc objective-c", .keywords = &kw_c, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "python", .aliases = " py python py3", .keywords = &kw_python, .comments = &.{"#"}, .multiline_start = null, .multiline_end = null, .flags = 3 },
    .{ .name = "javascript", .aliases = " js jsx javascript typescript ts tsx node mjs cjs", .keywords = &kw_js, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 7 },
    .{ .name = "java", .aliases = " java", .keywords = &kw_java, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "csharp", .aliases = " cs c# csharp dotnet", .keywords = &kw_csharp, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "go", .aliases = " go golang", .keywords = &kw_go, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 7 },
    .{ .name = "rust", .aliases = " rs rust", .keywords = &kw_rust, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "shell", .aliases = " sh bash zsh shell fish ksh", .keywords = &kw_shell, .comments = &.{"#"}, .multiline_start = null, .multiline_end = null, .flags = 7 },
    .{ .name = "sql", .aliases = " sql postgres mysql sqlite", .keywords = &kw_sql, .comments = &.{"--"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 11 },
    .{ .name = "ruby", .aliases = " rb ruby", .keywords = &kw_ruby, .comments = &.{"#"}, .multiline_start = null, .multiline_end = null, .flags = 3 },
    .{ .name = "php", .aliases = " php", .keywords = &kw_php, .comments = &.{ "//", "#" }, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "swift", .aliases = " swift", .keywords = &kw_swift, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "kotlin", .aliases = " kt kts kotlin", .keywords = &kw_kotlin, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "zig", .aliases = " zig", .keywords = &kw_zig, .comments = &.{"//"}, .multiline_start = null, .multiline_end = null, .flags = 3 },
    .{ .name = "lua", .aliases = " lua", .keywords = &kw_lua, .comments = &.{"--"}, .multiline_start = null, .multiline_end = null, .flags = 3 },
    .{ .name = "html", .aliases = " html htm xml svg", .keywords = &kw_html, .comments = &.{}, .multiline_start = "<!--", .multiline_end = "-->", .flags = 3 },
    .{ .name = "css", .aliases = " css scss sass", .keywords = &kw_css, .comments = &.{}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "json", .aliases = " json jsonc", .keywords = &.{}, .comments = &.{"//"}, .multiline_start = "/*", .multiline_end = "*/", .flags = 3 },
    .{ .name = "yaml", .aliases = " yaml yml toml ini", .keywords = &.{}, .comments = &.{"#"}, .multiline_start = null, .multiline_end = null, .flags = 3 },
    .{ .name = "markdown", .aliases = " md markdown", .keywords = &kw_generic, .comments = &.{}, .multiline_start = "<!--", .multiline_end = "-->", .flags = 3 },
};
fn syntaxForLanguage(language: []const u8) *const Syntax {
    if (language.len != 0) for (&languages) |*syntax| {
        if (equalFold(syntax.name, language)) return syntax;
        var aliases = std.mem.tokenizeScalar(u8, syntax.aliases, ' ');
        while (aliases.next()) |alias| if (equalFold(alias, language)) return syntax;
    };
    return &languages[0];
}
pub fn syntaxForPath(path: []const u8) *const Syntax {
    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |slash| path[slash + 1 ..] else path;
    if (equalFold(base, "Dockerfile") or equalFold(base, "Makefile")) return syntaxForLanguage("sh");
    const dot = std.mem.lastIndexOfScalar(u8, base, '.') orelse return syntaxForLanguage("");
    return syntaxForLanguage(base[dot + 1 ..]);
}
pub fn columns() c_int {
    var size: c.struct_winsize = undefined;
    if (c.ioctl(c.STDOUT_FILENO, c.TIOCGWINSZ, &size) == 0 and size.ws_col > 0) return size.ws_col;
    return 80;
}
fn utf8Width(byte: u8) usize {
    return if (byte >= 0xc2 and byte <= 0xdf) 2 else if (byte >= 0xe0 and byte <= 0xef) 3 else if (byte >= 0xf0 and byte <= 0xf4) 4 else 1;
}
fn separator(byte: u8) bool {
    return byte == 0 or space(byte) or std.mem.findScalar(u8, ",.()+-/*=~%[]{}<>:;!&|^?", byte) != null;
}
fn syntaxHandle(syntax: *const Syntax) *const c.agent_syntax {
    return @ptrCast(syntax);
}
fn nativeSyntax(syntax: ?*const c.agent_syntax) *const Syntax {
    return if (syntax) |handle| @ptrCast(@alignCast(handle)) else &languages[0];
}

pub const Renderer = struct {
    record: *Record,

    pub fn write(self: Renderer, bytes: []const u8) void {
        if (self.record.capture != null) captureAppend(@ptrCast(self.record.capture), bytes) else if (self.record.emit) |emit| emit(self.record.worker, bytes.ptr, bytes.len);
    }
    pub fn reset(self: Renderer) void {
        if (self.record.use_color) self.write("\x1b[0m");
        self.record.color_open = false;
    }
    fn hasAttributes(self: Renderer) bool {
        const r = self.record;
        return r.in_think or r.md_bold or r.md_italic or r.md_inline_code or r.md_code_block;
    }
    fn setAttributes(self: Renderer) void {
        const r = self.record;
        if (!r.use_color) return;
        if (r.in_think) {
            self.write("\x1b[38;5;245m");
            return;
        }
        if (r.md_code_block) {
            self.write("\x1b[38;5;75m");
            return;
        }
        if (r.md_inline_code) self.write("\x1b[36m");
        if (r.md_bold) self.write("\x1b[1m");
        if (r.md_italic) self.write("\x1b[3m");
    }
    pub fn restore(self: Renderer) void {
        if (self.record.use_color and self.record.color_open and self.hasAttributes()) self.setAttributes();
    }
    fn completeCharacter(self: Renderer, bytes: []const u8) void {
        const r = self.record;
        const styled = r.use_color and self.hasAttributes();
        if (styled and !r.color_open) {
            self.setAttributes();
            r.color_open = true;
        } else if (!styled and r.color_open) self.reset();
        self.write(bytes);
        if (bytes.len != 0) r.wrote_visible_output = true;
        r.last_output_newline = bytes.len == 1 and bytes[0] == '\n';
    }
    pub fn flushUtf8(self: Renderer) void {
        const r = self.record;
        if (r.utf8_pending_len == 0) return;
        self.completeCharacter(r.utf8_pending[0..r.utf8_pending_len]);
        r.utf8_pending_len = 0;
        r.utf8_pending_need = 0;
    }
    fn rawByte(self: Renderer, byte: u8) void {
        const r = self.record;
        if (r.utf8_pending_len != 0) {
            if (byte & 0xc0 == 0x80 and r.utf8_pending_len < r.utf8_pending.len) {
                r.utf8_pending[r.utf8_pending_len] = byte;
                r.utf8_pending_len += 1;
                if (r.utf8_pending_len == r.utf8_pending_need) self.flushUtf8();
                return;
            }
            self.flushUtf8();
        }
        const width = utf8Width(byte);
        if (width == 1) {
            self.completeCharacter(&.{byte});
            return;
        }
        r.utf8_pending[0] = byte;
        r.utf8_pending_len = 1;
        r.utf8_pending_need = width;
    }
    fn plainByte(self: Renderer, byte: u8) void {
        const r = self.record;
        const bold = r.md_bold;
        const italic = r.md_italic;
        const inline_code = r.md_inline_code;
        const code_block = r.md_code_block;
        // Code streams immediately, then may repaint a complete safe line.
        // Suppress Markdown attributes only for this byte, preserving thinking.
        r.md_bold = false;
        r.md_italic = false;
        r.md_inline_code = false;
        r.md_code_block = false;
        self.rawByte(byte);
        r.md_bold = bold;
        r.md_italic = italic;
        r.md_inline_code = inline_code;
        r.md_code_block = code_block;
    }
    fn syntaxWrite(self: Renderer, highlight: Highlight, bytes: []const u8) void {
        if (bytes.len == 0) return;
        const r = self.record;
        if (highlight != .normal) r.md_syntax_has_highlight = true;
        if (r.md_syntax_silent) return;
        if (r.use_color and highlight != .normal) {
            const palette_index: c_int = switch (highlight) {
                .comment => 244,
                .keyword1 => 214,
                .keyword2 => 81,
                .string => 150,
                .number => 203,
                else => 252,
            };
            var sequence: [32]u8 = undefined;
            const length = c.snprintf(&sequence, sequence.len, "\x1b[38;5;%dm", palette_index);
            self.write(sequence[0..@intCast(length)]);
        }
        self.write(bytes);
        if (r.use_color and highlight != .normal) self.write("\x1b[0m");
        r.wrote_visible_output = true;
        r.last_output_newline = false;
    }
    fn upto(self: Renderer) void {
        const r = self.record;
        r.md_syntax_has_highlight = true;
        if (r.md_syntax_silent) return;
        if (r.use_color) self.write("\x1b[38;5;244m[\x1b[1;38;5;177mupto\x1b[38;5;244m]\x1b[0m") else self.write("[upto]");
        r.wrote_visible_output = true;
        r.last_output_newline = false;
    }
    fn syntaxLine(self: Renderer, line: []const u8) void {
        const r = self.record;
        const syntax = nativeSyntax(r.md_syntax);
        var position: usize = 0;
        var previous_separator = true;
        var previous_highlight: Highlight = .normal;
        while (position < line.len) {
            const remaining = line[position..];
            if (r.md_code_highlight_upto and std.mem.startsWith(u8, remaining, "[upto]")) {
                self.upto();
                position += 6;
                previous_separator = true;
                previous_highlight = .normal;
                continue;
            }
            if (r.md_code_in_ml_comment) {
                if (syntax.multiline_end) |ending| {
                    if (ending.len != 0) if (std.mem.find(u8, remaining, ending)) |offset| {
                        const length = offset + ending.len;
                        self.syntaxWrite(.comment, remaining[0..length]);
                        position += length;
                        r.md_code_in_ml_comment = false;
                        previous_separator = true;
                        previous_highlight = .comment;
                        continue;
                    };
                }
                self.syntaxWrite(.comment, remaining);
                return;
            }
            for (syntax.comments) |marker| {
                if (marker.len != 0 and std.mem.startsWith(u8, remaining, marker)) {
                    self.syntaxWrite(.comment, remaining);
                    return;
                }
            }
            if (syntax.multiline_start != null and syntax.multiline_end != null and
                std.mem.startsWith(u8, remaining, syntax.multiline_start.?))
            {
                const start = syntax.multiline_start.?.len;
                const end = if (std.mem.findPos(u8, remaining, start, syntax.multiline_end.?)) |offset| offset + syntax.multiline_end.?.len else remaining.len;
                if (std.mem.findPos(u8, remaining, start, syntax.multiline_end.?) == null) r.md_code_in_ml_comment = true;
                self.syntaxWrite(.comment, remaining[0..end]);
                position += end;
                previous_separator = false;
                previous_highlight = .comment;
                continue;
            }
            const byte = remaining[0];
            if (syntax.flags & strings != 0 and (byte == '"' or byte == '\'' or (syntax.flags & backtick_strings != 0 and byte == '`'))) {
                var end: usize = 1;
                while (end < remaining.len) {
                    if (remaining[end] == '\\' and end + 1 < remaining.len) {
                        end += 2;
                        continue;
                    }
                    end += 1;
                    if (remaining[end - 1] == byte) break;
                }
                self.syntaxWrite(.string, remaining[0..end]);
                position += end;
                previous_separator = false;
                previous_highlight = .string;
                continue;
            }
            if (syntax.flags & numbers != 0 and ((c.isdigit(byte) != 0 and (previous_separator or previous_highlight == .number)) or (byte == '.' and position > 0 and previous_highlight == .number))) {
                var end: usize = 0;
                while (end < remaining.len and (alphaNumeric(remaining[end]) or std.mem.findScalar(u8, "_.+-", remaining[end]) != null)) end += 1;
                self.syntaxWrite(.number, remaining[0..end]);
                position += end;
                previous_separator = false;
                previous_highlight = .number;
                continue;
            }
            if (previous_separator) {
                var matched: usize = 0;
                for (syntax.keywords) |word| {
                    const secondary = std.mem.endsWith(u8, word, "|");
                    const keyword = if (secondary) word[0 .. word.len - 1] else word;
                    if (keyword.len > remaining.len) continue;
                    const matches = if (syntax.flags & case_insensitive != 0) equalFold(remaining[0..keyword.len], keyword) else std.mem.startsWith(u8, remaining, keyword);
                    const next = if (keyword.len < remaining.len) remaining[keyword.len] else 0;
                    if (!matches or !separator(next)) continue;
                    self.syntaxWrite(if (secondary) .keyword2 else .keyword1, remaining[0..keyword.len]);
                    matched = keyword.len;
                    break;
                }
                if (matched != 0) {
                    position += matched;
                    previous_separator = false;
                    previous_highlight = .normal; // only separator state is used after a keyword
                    continue;
                }
            }
            self.syntaxWrite(.normal, remaining[0..1]);
            previous_separator = separator(byte);
            previous_highlight = .normal;
            position += 1;
        }
    }
    fn codeLine(self: Renderer) []const u8 {
        const r = self.record;
        return if (r.md_code_line_len == 0) &.{} else r.md_code_line[0..r.md_code_line_len];
    }
    fn appendCodeByte(self: Renderer, byte: u8) void {
        const r = self.record;
        var line: std.ArrayList(u8) = .empty;
        if (r.md_code_line != null) line.items = r.md_code_line[0..r.md_code_line_len];
        line.capacity = r.md_code_line_cap;
        line.append(allocator, byte) catch fail("ds4-agent: realloc");
        line.ensureUnusedCapacity(allocator, 1) catch fail("ds4-agent: realloc");
        line.items.ptr[line.items.len] = 0;
        r.md_code_line = line.items.ptr;
        r.md_code_line_len = line.items.len;
        r.md_code_line_cap = line.capacity;
    }
    fn canRepaint(self: Renderer) bool {
        const r = self.record;
        if (!r.use_color or r.capture != null or r.md_code_line_len == 0) return false;
        const width = columns();
        if (width <= 1 or text(r.md_code_line_prefix).len + r.md_code_line_len >= @as(usize, @intCast(width))) return false;
        for (self.codeLine()) |byte| if (byte == '\t' or byte == 0x1b or byte >= 0x80 or (byte < 0x20 and byte != '\r')) return false;
        return true;
    }
    fn codePrefix(self: Renderer) void {
        const r = self.record;
        if (r.md_code_line_prefix == null) return;
        if (r.use_color and r.md_code_line_prefix_color != null) self.write(text(r.md_code_line_prefix_color));
        self.write(text(r.md_code_line_prefix));
        if (r.use_color and r.md_code_line_prefix_color != null) self.write("\x1b[0m");
        r.color_open = false;
    }
    fn scanCodeLine(self: Renderer, final_comment: *bool) bool {
        // Silent scanning learns whether repaint changes the already shown row,
        // without publishing its multiline-comment state prematurely.
        const r = self.record;
        const silent = r.md_syntax_silent;
        const highlighted = r.md_syntax_has_highlight;
        const comment = r.md_code_in_ml_comment;
        r.md_syntax_silent = true;
        r.md_syntax_has_highlight = false;
        self.syntaxLine(self.codeLine());
        const changed = r.md_syntax_has_highlight;
        final_comment.* = r.md_code_in_ml_comment;
        r.md_code_in_ml_comment = comment;
        r.md_syntax_silent = silent;
        r.md_syntax_has_highlight = highlighted;
        return changed;
    }
    fn emitCodeLine(self: Renderer, newline: bool) void {
        // Repaint only a simple, safe single-row ASCII line. Long/tabbed/escaped
        // or UTF-8 lines remain as streamed and only advance syntax state.
        const r = self.record;
        var final_comment = r.md_code_in_ml_comment;
        const changed = self.scanCodeLine(&final_comment);
        if (changed and self.canRepaint()) {
            var frame = std.mem.zeroes(Capture);
            frame.cap = 64 * (r.md_code_line_len + text(r.md_code_line_prefix).len + 1) + 256;
            const capture = r.capture;
            r.capture = &frame;
            self.reset();
            self.write("\r\x1b[0K");
            self.codePrefix();
            self.syntaxLine(self.codeLine());
            r.capture = capture;
            var length: usize = 0;
            const bytes = captureTake(&frame, &length);
            defer c.free(bytes);
            self.write(bytes[0..length]);
        } else r.md_code_in_ml_comment = final_comment;
        r.md_code_line_len = 0;
        if (newline) {
            self.plainByte('\n');
            r.wrote_visible_output = true;
            r.last_output_newline = true;
            r.md_code_line_start = true;
        }
    }
    pub fn codeByte(self: Renderer, byte: u8) void {
        if (byte == '\n') {
            self.emitCodeLine(true);
            return;
        }
        self.appendCodeByte(byte);
        self.plainByte(byte);
        if (byte != ' ' and byte != '\t' and byte != '\r') self.record.md_code_line_start = false;
    }
    fn fenceBegin(self: Renderer) void {
        self.codeBegin(null);
        self.record.md_fence_info = true;
    }
    pub fn codeBegin(self: Renderer, syntax: ?*const Syntax) void {
        const r = self.record;
        self.reset();
        r.md_code_block = true;
        r.md_inline_code = false;
        r.md_fence_info = false;
        r.md_code_line_start = true;
        r.md_code_in_ml_comment = false;
        r.md_syntax = syntaxHandle(syntax orelse &languages[0]);
        r.md_fence_lang_len = 0;
        r.md_fence_lang[0] = 0;
        r.md_code_line_prefix = null;
        r.md_code_line_prefix_color = null;
        r.md_code_highlight_upto = false;
        r.md_code_line_len = 0;
    }
    pub fn setCodePrefix(self: Renderer, prefix: [*c]const u8, prefix_color: [*c]const u8) void {
        self.record.md_code_line_prefix = prefix;
        self.record.md_code_line_prefix_color = prefix_color;
    }
    pub fn setCodeUpto(self: Renderer, enabled: bool) void {
        self.record.md_code_highlight_upto = enabled;
    }
    pub fn codeEnd(self: Renderer) void {
        const r = self.record;
        var only_space = true;
        for (self.codeLine()) |byte| if (byte != ' ' and byte != '\t' and byte != '\r') {
            only_space = false;
            break;
        };
        if (r.md_code_line_len != 0 and !only_space) self.emitCodeLine(false) else r.md_code_line_len = 0;
        r.md_code_block = false;
        r.md_inline_code = false;
        r.md_fence_info = false;
        r.md_code_line_start = true;
        r.md_code_in_ml_comment = false;
        r.md_syntax = null;
        r.md_fence_lang_len = 0;
        r.md_fence_lang[0] = 0;
        r.md_code_line_prefix = null;
        r.md_code_line_prefix_color = null;
    }
    fn clearPending(self: Renderer) void {
        self.record.md_pending = c.AGENT_MD_PENDING_NONE;
        self.record.md_pending_len = 0;
    }
    fn inlinePair(self: Renderer) ?usize {
        const inline_text = self.record.md_inline[0..self.record.md_inline_len];
        if (inline_text.len == 0) return null;
        const byte = inline_text[0];
        var opening: usize = 1;
        var closing: usize = 0;
        while (opening < inline_text.len and inline_text[opening] == byte) opening += 1;
        while (closing < inline_text.len and inline_text[inline_text.len - closing - 1] == byte) closing += 1;
        if (opening > 2 or closing != opening or inline_text.len <= opening + closing) return null;
        if (byte == '*') {
            if (space(inline_text[opening]) or space(inline_text[inline_text.len - closing - 1])) return null;
            var escapes: usize = 0;
            var position = inline_text.len - closing;
            while (position > 0) {
                position -= 1;
                if (inline_text[position] != '\\') break;
                escapes += 1;
            }
            if (escapes & 1 != 0) return null;
        }
        return opening;
    }
    fn flushInline(self: Renderer, format: bool) void {
        const r = self.record;
        if (r.md_inline_len == 0) return;
        const delimiter = if (format) self.inlinePair() else null;
        const paired = delimiter != null;
        const code = paired and r.md_inline[0] == '`';
        r.md_inline_code = code;
        r.md_bold = paired and !code and delimiter.? == 2;
        r.md_italic = paired and !code and delimiter.? == 1;
        var position = delimiter orelse 0;
        const end = r.md_inline_len - (delimiter orelse 0);
        while (position < end) : (position += 1) {
            if (!code and r.md_inline[position] == '\\' and position + 1 < end and punctuation(r.md_inline[position + 1])) position += 1;
            self.rawByte(r.md_inline[position]);
        }
        r.md_inline_len = 0;
        r.md_inline_code = false;
        r.md_bold = false;
        r.md_italic = false;
        self.reset();
    }
    fn flushHintPrefix(self: Renderer) void {
        const r = self.record;
        for (r.md_hint_prefix[0..r.md_hint_prefix_len]) |byte| self.rawByte(byte);
        r.md_hint_prefix_len = 0;
    }
    pub fn hintEnd(self: Renderer) void {
        self.flushHintPrefix();
        if (self.record.md_hint) self.reset();
        self.record.md_hint = false;
        self.record.md_line_started = false;
    }
    // Tool visualization and redirected output bypass Markdown formatting.
    fn pendingLiterals(self: Renderer) void {
        const r = self.record;
        self.flushHintPrefix();
        self.flushInline(false);
        if (r.md_escape) {
            self.rawByte('\\');
            r.md_escape = false;
        }
        const count = r.md_pending_len;
        self.clearPending();
        for (0..count) |_| if (r.md_code_block) self.codeByte('`') else self.rawByte('`');
    }
    fn commitBackticks(self: Renderer) void {
        const r = self.record;
        const count = r.md_pending_len;
        self.clearPending();
        if (count >= 3) {
            for (0..count) |_| self.plainByte('`');
            if (r.md_code_block) self.codeEnd() else self.fenceBegin();
        } else if (r.md_code_block) for (0..count) |_| self.codeByte('`');
    }
    // An ambiguous inline span is held until its closing delimiter, newline or
    // 4 KiB bound. Without a pair it is literal. Fences keep line-based syntax.
    pub fn markdownByte(self: Renderer, byte: u8) void {
        const r = self.record;
        if (r.md_fence_info) {
            if (byte == '\n') {
                if (r.md_code_block) {
                    r.md_fence_lang[r.md_fence_lang_len] = 0;
                    r.md_syntax = syntaxHandle(syntaxForLanguage(r.md_fence_lang[0..r.md_fence_lang_len]));
                }
                self.plainByte('\n');
                r.md_fence_info = false;
            } else if (r.md_code_block) {
                if (r.md_fence_lang_len + 1 < r.md_fence_lang.len and
                    (alphaNumeric(byte) or std.mem.findScalar(u8, "_-+#", byte) != null))
                {
                    r.md_fence_lang[r.md_fence_lang_len] = byte;
                    r.md_fence_lang_len += 1;
                }
                self.plainByte(byte);
            }
            return;
        }
        if (r.md_pending == c.AGENT_MD_PENDING_BACKTICK) {
            if (byte == '`') {
                r.md_pending_len += 1;
                return;
            }
            self.commitBackticks();
            self.markdownByte(byte);
            return;
        }
        if (r.md_code_block) {
            if (byte == '`' and r.md_code_line_start) {
                r.md_pending = c.AGENT_MD_PENDING_BACKTICK;
                r.md_pending_len = 1;
            } else self.codeByte(byte);
            return;
        }
        if (r.md_inline_len != 0) {
            if (byte != r.md_inline[0] and self.inlinePair() != null) {
                self.flushInline(true);
                self.markdownByte(byte);
                return;
            }
            var run: usize = 0;
            while (run < r.md_inline_len and r.md_inline[run] == r.md_inline[0]) run += 1;
            if (run == r.md_inline_len and byte != r.md_inline[0]) {
                if (r.md_inline[0] == '`' and run >= 3) {
                    r.md_inline_len = 0;
                    for (0..run) |_| self.plainByte('`');
                    self.fenceBegin();
                    self.markdownByte(byte);
                    return;
                }
                if (space(byte) or run > 2) {
                    self.flushInline(false);
                    self.markdownByte(byte);
                    return;
                }
            }
            if (byte == '\n' or r.md_inline_len == r.md_inline.len) {
                self.flushInline(false);
                self.markdownByte(byte);
                return;
            }
            r.md_inline[r.md_inline_len] = byte;
            r.md_inline_len += 1;
            return;
        }
        if (r.md_escape) {
            r.md_escape = false;
            if (punctuation(byte)) {
                self.rawByte(byte);
                return;
            }
            self.rawByte('\\');
        }
        if (byte == '\\') {
            r.md_escape = true;
            return;
        }
        if (byte == '*' or byte == '`') {
            r.md_inline[0] = byte;
            r.md_inline_len = 1;
            return;
        }
        self.rawByte(byte);
    }
    pub fn finishMarkdown(self: Renderer) void {
        const r = self.record;
        self.flushHintPrefix();
        self.flushInline(true);
        // A closing fence may be the final bytes of a reply. Commit it without
        // waiting for a following byte, rather than leaking its pending marker.
        if (r.md_pending == c.AGENT_MD_PENDING_BACKTICK and r.md_pending_len >= 3)
            self.commitBackticks()
        else
            self.pendingLiterals();
        if (r.md_code_block and r.md_code_line_len != 0) self.emitCodeLine(false);
        r.md_bold = false;
        r.md_italic = false;
        r.md_inline_code = false;
        r.md_code_block = false;
        r.md_fence_info = false;
        r.md_code_line_start = false;
        r.md_code_in_ml_comment = false;
        r.md_syntax = null;
        r.md_fence_lang_len = 0;
        r.md_fence_lang[0] = 0;
        r.md_code_line_prefix = null;
        r.md_code_line_prefix_color = null;
        r.md_code_highlight_upto = false;
        c.free(r.md_code_line);
        r.md_code_line = null;
        r.md_code_line_len = 0;
        r.md_code_line_cap = 0;
        self.hintEnd();
    }
    pub fn character(self: Renderer, byte: u8) void {
        const r = self.record;
        if (!r.format_markdown or r.in_think) {
            self.pendingLiterals();
            self.hintEnd();
            self.rawByte(byte);
            return;
        }
        // Only the advertised marker at a prose boundary opens a hint aside.
        // Hold that short prefix; explicit quote lines continue the same aside.
        if (r.use_color and !r.md_line_started and !r.md_code_block and
            !r.md_fence_info and r.md_inline_len == 0 and !r.md_escape and
            r.md_pending == c.AGENT_MD_PENDING_NONE)
        {
            const prefix: []const u8 = if (r.md_hint) "> " else "> **Hint:**";
            if (byte == prefix[r.md_hint_prefix_len]) {
                r.md_hint_prefix[r.md_hint_prefix_len] = byte;
                r.md_hint_prefix_len += 1;
                if (r.md_hint_prefix_len < prefix.len) return;
                const continuation = r.md_hint;
                r.md_hint_prefix_len = 0;
                r.md_hint = true;
                r.md_line_started = true;
                self.reset();
                self.write("\x1b[38;5;30m\xe2\x94\x82 \x1b[0m");
                r.wrote_visible_output = true;
                r.last_output_newline = false;
                if (!continuation) {
                    self.write("\x1b[1;97;48;5;23m Hint ");
                    self.reset();
                }
                return;
            }
            if (r.md_hint) self.reset();
            r.md_hint = false;
            const count = r.md_hint_prefix_len;
            r.md_hint_prefix_len = 0;
            for (r.md_hint_prefix[0..count]) |held| self.markdownByte(held);
        }
        r.md_line_started = byte != '\n';
        self.markdownByte(byte);
    }
    // Think tags can split across model tokens. They are hidden, and their
    // contents use the existing dimmed style rather than entering Markdown.
    pub fn process(self: Renderer, bytes: []const u8, final: bool) void {
        const r = self.record;
        const count = r.pending_len + bytes.len;
        const storage = allocate(count);
        defer c.free(storage);
        @memcpy(storage[0..r.pending_len], r.pending[0..r.pending_len]);
        @memcpy(storage[r.pending_len..count], bytes);
        r.pending_len = 0;
        var position: usize = 0;
        while (position < count) {
            const remaining = storage[position..count];
            if (std.mem.startsWith(u8, remaining, "<think>")) {
                self.hintEnd();
                r.in_think = true;
                position += 7;
                continue;
            }
            if (std.mem.startsWith(u8, remaining, "</think>")) {
                self.hintEnd();
                r.in_think = false;
                self.reset();
                if (!r.last_output_newline) self.write("\n");
                self.write("\n");
                r.last_output_newline = true;
                position += 8;
                continue;
            }
            if (!final and remaining[0] == '<' and
                (std.mem.startsWith(u8, "<think>", remaining) or std.mem.startsWith(u8, "</think>", remaining)))
            {
                if (remaining.len < r.pending.len) {
                    @memcpy(r.pending[0..remaining.len], remaining);
                    r.pending_len = remaining.len;
                }
                break;
            }
            self.character(remaining[0]);
            position += 1;
        }
    }
    pub fn finish(self: Renderer) void {
        const r = self.record;
        if (r.format_thinking) self.process(&.{}, true);
        self.finishMarkdown();
        self.flushUtf8();
        self.reset();
        if (r.wrote_visible_output) {
            if (!r.last_output_newline) self.write("\n");
            self.write("\n");
            r.last_output_newline = true;
        }
    }
    pub fn color(self: Renderer, sequence: []const u8) void {
        self.pendingLiterals();
        self.hintEnd();
        self.flushUtf8();
        const reset_color = sequence.len == 0 or std.mem.eql(u8, sequence, "\x1b[0m");
        if (self.record.use_color and sequence.len != 0) self.write(sequence);
        self.record.color_open = self.record.use_color and !reset_color;
    }
    pub fn plain(self: Renderer, bytes: []const u8) void {
        self.pendingLiterals();
        self.hintEnd();
        self.flushUtf8();
        self.write(bytes);
        if (bytes.len != 0) {
            self.record.wrote_visible_output = true;
            self.record.last_output_newline = bytes[bytes.len - 1] == '\n';
        }
    }
};
