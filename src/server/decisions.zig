//! Two decision wire formats over one native request/result. Model execution
//! owns logits; this module owns typed values, answer order and wire statistics.
const std = @import("std");
const types = @import("../models/clef/types.zig");
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

pub const Protocol = enum { systemone, openai };
pub const Parsed = struct {
    model: []const u8,
    request: types.Request,
    /// OpenAI names need not be unique, and unnamed answers remain in order.
    names: []const Value,
};

fn object(value: Value) !std.json.ObjectMap {
    return if (value == .object) value.object else error.ExpectedObject;
}
fn field(value: Value, name: []const u8) !Value {
    return (try object(value)).get(name) orelse error.MissingField;
}
fn string(value: Value) ![]const u8 {
    return if (value == .string) value.string else error.ExpectedString;
}
fn array(value: Value) ![]const Value {
    return if (value == .array) value.array.items else error.ExpectedArray;
}
fn optional(value: Value, name: []const u8) Value {
    return if (value == .object) value.object.get(name) orelse .null else .null;
}
fn eql(value: Value, expected: []const u8) bool {
    return value == .string and std.mem.eql(u8, value.string, expected);
}
fn numbered(a: Allocator, index: usize) ![]const u8 {
    return std.fmt.allocPrint(a, "{d}", .{index});
}

/// Request and parsed JSON borrow the caller's arena for their whole lifetime.
pub fn parse(a: Allocator, body: []const u8, protocol: Protocol) !Parsed {
    const root = try std.json.parseFromSliceLeaky(Value, a, body, .{ .allocate = .alloc_always });
    const model = try string(try field(root, "model"));
    if (model.len == 0) return error.EmptyModel;
    const streaming = optional(root, "stream");
    if (streaming != .null and streaming != .bool) return error.InvalidStream;
    if (streaming == .bool and streaming.bool) return error.DecisionsDoNotStream;
    return switch (protocol) {
        .systemone => try parseSystemOne(a, root, model),
        .openai => try parseOpenAI(a, root, model),
    };
}

fn binaryOptions(a: Allocator, criteria: Value) ![]const types.Option {
    if (criteria != .null and criteria != .object) return error.InvalidNoulCriteria;
    const options = try a.alloc(types.Option, 2);
    for ([_][]const u8{ "true", "false" }, options) |id, *out| {
        const fallback: Value = .{ .string = if (std.mem.eql(u8, id, "true")) "The proposition is true or the answer is yes." else "The proposition is false or the answer is no." };
        out.* = .{ .id = id, .value = .{ .bool = std.mem.eql(u8, id, "true") }, .description = if (criteria == .object) criteria.object.get(id) orelse fallback else fallback };
    }
    return options;
}

fn parseSystemOne(a: Allocator, root: Value, model: []const u8) !Parsed {
    var media = try nativeMedia(a, root);
    const assets = optional(root, "assets");
    if (assets != .null) {
        var iter = (try object(assets)).iterator();
        while (iter.next()) |entry| {
            if (!eql(try field(entry.value_ptr.*, "type"), "image")) return error.UnsupportedMediaType;
            if (optional(entry.value_ptr.*, "part") != .null) return error.MultipartNotSupported;
            try media.append(a, .{ .kind = .image, .source = entry.value_ptr.*, .label = entry.key_ptr.* });
        }
    }
    const input = try field(root, "state");
    const supplied = try object(try field(root, "questions"));
    if (supplied.count() == 0) return error.QuestionCount;
    const questions = try a.alloc(types.Question, supplied.count());
    const names = try a.alloc(Value, questions.len);
    var entries = supplied.iterator();
    var index: usize = 0;
    while (entries.next()) |entry| : (index += 1) {
        const question = entry.value_ptr.*;
        const kind = try string(try field(question, "type"));
        const criteria = optional(question, "criteria");
        const selected: types.Kind = if (std.mem.eql(u8, kind, "noul")) .noul else if (std.mem.eql(u8, kind, "choice")) .choice else if (std.mem.eql(u8, kind, "score")) .score else return error.UnsupportedQuestionType;
        const options = switch (selected) {
            .noul => try binaryOptions(a, criteria),
            .choice => choice: {
                const choices = try object(criteria);
                if (choices.count() == 0 or choices.count() > 255) return error.ChoiceCount;
                const options = try a.alloc(types.Option, choices.count());
                var iter = choices.iterator();
                var i: usize = 0;
                while (iter.next()) |choice| : (i += 1) options[i] = .{ .id = choice.key_ptr.*, .description = choice.value_ptr.*, .value = .{ .string = choice.key_ptr.* } };
                break :choice options;
            },
            .score => score: {
                const levels = try array(criteria);
                if (levels.len == 0 or levels.len > 10) return error.ScoreLevelCount;
                const options = try a.alloc(types.Option, levels.len);
                for (levels, options, 0..) |level, *out, i| out.* = .{ .id = try numbered(a, i), .description = level, .value = .{ .integer = @intCast(i) } };
                break :score options;
            },
        };
        questions[index] = .{ .id = entry.key_ptr.*, .kind = selected, .instructions = optional(question, "instructions"), .options = options };
        names[index] = .{ .string = entry.key_ptr.* };
    }
    return .{ .model = model, .names = names, .request = .{ .state = input, .questions = questions, .media = try media.toOwnedSlice(a) } };
}

fn nativeMedia(a: Allocator, root: Value) !std.ArrayList(types.MediaSource) {
    var media: std.ArrayList(types.MediaSource) = .empty;
    for ([_][]const u8{ "images", "videos", "audio" }, [_]types.MediaKind{ .image, .video, .audio }) |key, kind| {
        const values = optional(root, key);
        if (values == .null) continue;
        for (try array(values)) |source| try media.append(a, .{ .kind = kind, .source = source });
    }
    return media;
}

/// Preserve ordered message/tool evidence, extracting image bytes/URLs before
/// model rendering. JSON parts are evidence and are never reinterpreted as API
/// content envelopes. Labels bind question-specific images to their prefix.
fn evidence(a: Allocator, value: Value, media: *std.ArrayList(types.MediaSource)) !Value {
    switch (value) {
        .array => {
            var out = jsonArray(a);
            for (value.array.items) |item| try out.array.append(try evidence(a, item, media));
            return out;
        },
        .object => {
            const kind = optional(value, "type");
            if (eql(kind, "input_json")) return value;
            if (eql(kind, "input_image")) {
                const source = try field(value, "image_url");
                if (source != .string) return error.InvalidImageUrl;
                const label = try std.fmt.allocPrint(a, "image_{d}", .{media.items.len});
                try media.append(a, .{ .kind = .image, .source = source, .label = label });
                var out = jsonObject();
                try put(a, &out, "type", .{ .string = "image" });
                try put(a, &out, "id", .{ .string = label });
                return out;
            }
            if (kind == .string) {
                for ([_][]const u8{ "input_audio", "input_file", "item_reference", "image", "video", "audio" }) |unsupported| {
                    if (eql(kind, unsupported)) return error.MediaNotSupported;
                }
            }
            var out = jsonObject();
            var iterator = value.object.iterator();
            while (iterator.next()) |entry| {
                try put(a, &out, entry.key_ptr.*, try evidence(a, entry.value_ptr.*, media));
            }
            return out;
        },
        else => return value,
    }
}

fn parseOpenAI(a: Allocator, root: Value, model: []const u8) !Parsed {
    var media = try nativeMedia(a, root);
    const raw_input = try field(root, "input");
    if (raw_input != .string and raw_input != .array) return error.InvalidInput;
    const input = try evidence(a, raw_input, &media);
    const supplied = try array(try field(root, "questions"));
    if (supplied.len == 0 or supplied.len > 400) return error.QuestionCount;
    const questions = try a.alloc(types.Question, supplied.len);
    const names = try a.alloc(Value, supplied.len);
    for (supplied, questions, names, 0..) |question, *out, *name, index| {
        name.* = optional(question, "name");
        if (name.* != .null and name.* != .string) return error.InvalidQuestionName;
        const id = try std.fmt.allocPrint(a, "q{d}", .{index});
        const instructions = try evidence(a, try field(question, "instructions"), &media);
        if (!(instructions == .string and instructions.string.len != 0) and !(instructions == .array and instructions.array.items.len != 0)) return error.EmptyInstructions;
        const kind = try string(try field(question, "type"));
        const selected: types.Kind = if (std.mem.eql(u8, kind, "predicate")) .noul else if (std.mem.eql(u8, kind, "choice")) .choice else if (std.mem.eql(u8, kind, "score")) .score else return error.UnsupportedQuestionType;
        const options = switch (selected) {
            .noul => try binaryOptions(a, .null),
            .choice => choice: {
                const choices = try array(try field(question, "choices"));
                if (choices.len < 2 or choices.len > 255) return error.ChoiceCount;
                const options = try a.alloc(types.Option, choices.len);
                for (choices, options, 0..) |item, *option, i| {
                    const value = try field(item, "value");
                    if (value != .string and value != .bool) return error.InvalidChoiceValue;
                    for (options[0..i]) |previous| {
                        if ((value == .bool and previous.value == .bool and value.bool == previous.value.bool) or (value == .string and previous.value == .string and std.mem.eql(u8, value.string, previous.value.string))) return error.DuplicateChoiceValue;
                    }
                    // Lexical IDs preserve caller labels where possible; typed
                    // values remain separate so true and "true" never collide.
                    const lexical = if (value == .string) value.string else if (value.bool) "true" else "false";
                    var option_id = lexical;
                    // Only a typed-value collision needs an internal spelling.
                    // Ordinary string labels keep their trained lexical anchor.
                    for (choices) |other| {
                        const other_value = try field(other, "value");
                        if (value == .string and other_value == .bool and std.mem.eql(u8, lexical, if (other_value.bool) "true" else "false")) {
                            option_id = try std.fmt.allocPrint(a, "string:{s}", .{lexical});
                        }
                    }
                    option.* = .{ .id = option_id, .description = try field(item, "description"), .value = value };
                }
                break :choice options;
            },
            .score => score: {
                const levels = try array(try field(question, "levels"));
                if (levels.len < 2 or levels.len > 10) return error.ScoreLevelCount;
                const options = try a.alloc(types.Option, levels.len);
                for (levels, options, 0..) |level, *option, i| {
                    const label = try string(try field(level, "label"));
                    const description = try string(try field(level, "description"));
                    option.* = .{ .id = try numbered(a, i), .description = .{ .string = try std.fmt.allocPrint(a, "{s}: {s}", .{ label, description }) }, .value = .{ .string = label } };
                }
                break :score options;
            },
        };
        out.* = .{ .id = id, .kind = selected, .instructions = instructions, .options = options };
    }
    return .{ .model = model, .names = names, .request = .{ .state = input, .questions = questions, .media = try media.toOwnedSlice(a) } };
}

fn jsonObject() Value {
    return .{ .object = .empty };
}
fn jsonArray(a: Allocator) Value {
    return .{ .array = std.json.Array.init(a) };
}
fn put(a: Allocator, object_value: *Value, name: []const u8, value: Value) !void {
    try object_value.object.put(a, name, value);
}
fn number(value: f64) Value {
    return .{ .float = value };
}

fn maximum(probabilities: []const f32) usize {
    var best: usize = 0;
    for (probabilities, 0..) |probability, i| if (probability > probabilities[best]) {
        best = i;
    };
    return best;
}

/// TypeSafe's confidence removes uniform-chance certainty. The OpenAI wire
/// uses Clef's native max probability, not a claim of Luna calibration.
fn confidence(probabilities: []const f32, kind: types.Kind, protocol: Protocol) f64 {
    const best = maximum(probabilities);
    const peak: f64 = probabilities[best];
    if (protocol == .openai or probabilities.len == 1) return peak;
    if (kind == .choice) {
        const uniform = 1.0 / @as(f64, @floatFromInt(probabilities.len));
        return std.math.clamp((peak - uniform) / (1 - uniform), 0, 1);
    }
    var distance: f64 = 0;
    var uniform_distance: f64 = 0;
    for (probabilities, 0..) |probability, i| {
        const delta: f64 = @floatFromInt(if (i > best) i - best else best - i);
        distance += probability * delta;
        uniform_distance += delta;
    }
    uniform_distance /= @floatFromInt(probabilities.len);
    return if (uniform_distance == 0) 1 else std.math.clamp(1 - distance / uniform_distance, 0, 1);
}

pub fn serialize(a: Allocator, parsed: Parsed, result: types.Result, protocol: Protocol) ![]u8 {
    if (result.answers.len != parsed.request.questions.len) return error.AnswerCountMismatch;
    var root = jsonObject();
    try put(a, &root, "model", .{ .string = parsed.model });
    var answers = if (protocol == .systemone) jsonObject() else jsonArray(a);
    for (parsed.request.questions, parsed.names, result.answers) |question, name, answer| {
        const p = answer.probabilities;
        if (p.len == 0 or p.len != question.options.len) return error.OptionCountMismatch;
        for (p) |probability| if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.InvalidProbability;
        const best = maximum(p);
        var item = jsonObject();
        try put(a, &item, "type", .{ .string = switch (question.kind) {
            .noul => if (protocol == .systemone) "noul" else "predicate",
            .choice => "choice",
            .score => "score",
        } });
        if (protocol == .openai) try put(a, &item, "name", name);
        if (question.kind == .noul) {
            try put(a, &item, if (protocol == .systemone) "noul" else "probability", number(p[0]));
        } else {
            try put(a, &item, "confidence", number(confidence(p, question.kind, protocol)));
            if (question.kind == .choice) try put(a, &item, "choice", question.options[best].value) else {
                var score: f64 = 0;
                for (p, 0..) |probability, i| score += @as(f64, probability) * @as(f64, @floatFromInt(i));
                try put(a, &item, "score", number(score));
            }
            var probabilities = if (protocol == .systemone) jsonObject() else jsonArray(a);
            var legend = jsonObject();
            for (question.options, p, 0..) |option, probability, i| {
                if (protocol == .systemone) {
                    try put(a, &probabilities, option.id, number(probability));
                    if (question.kind == .score) try put(a, &legend, option.id, option.description);
                } else {
                    var pair = jsonObject();
                    try put(a, &pair, "value", if (question.kind == .score) .{ .integer = @intCast(i) } else option.value);
                    try put(a, &pair, "probability", number(probability));
                    if (question.kind == .score) try put(a, &pair, "label", option.value);
                    try probabilities.array.append(pair);
                }
            }
            try put(a, &item, "probabilities", probabilities);
            if (protocol == .systemone and question.kind == .score) try put(a, &item, "legend", legend);
        }
        if (protocol == .systemone) try put(a, &answers, question.id, item) else try answers.array.append(item);
    }
    try put(a, &root, "answers", answers);
    var usage = jsonObject();
    try put(a, &usage, "input_tokens", .{ .integer = @intCast(result.input_tokens) });
    try put(a, &usage, "output_tokens", .{ .integer = 0 });
    if (protocol == .openai) {
        try put(a, &usage, "total_tokens", .{ .integer = @intCast(result.input_tokens) });
        var input_details = jsonObject();
        try put(a, &input_details, "cached_tokens", .{ .integer = 0 });
        try put(a, &usage, "input_tokens_details", input_details);
        var output_details = jsonObject();
        try put(a, &output_details, "reasoning_tokens", .{ .integer = 0 });
        try put(a, &usage, "output_tokens_details", output_details);
    }
    try put(a, &root, "usage", usage);
    return std.json.Stringify.valueAlloc(a, root, .{});
}

test "ordered unnamed OpenAI answers preserve boolean versus string choices" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try parse(a,
        \\{"model":"clef-flash","input":"evidence","questions":[{"type":"choice","instructions":"Pick","choices":[{"value":true,"description":"bool"},{"value":"true","description":"text"}]},{"type":"predicate","instructions":"Is it true?"}]}
    , .openai);
    const encoded = try serialize(a, parsed, .{ .input_tokens = 128, .answers = &.{ .{ .probabilities = &.{ 0.25, 0.75 } }, .{ .probabilities = &.{ 0.8, 0.2 } } } }, .openai);
    const root = try std.json.parseFromSliceLeaky(Value, a, encoded, .{});
    const answers = (try field(root, "answers")).array.items;
    try std.testing.expectEqual(@as(usize, 2), answers.len);
    try std.testing.expect(eql(try field(answers[0], "choice"), "true"));
    try std.testing.expect((try field(answers[0], "name")) == .null);
    const probabilities = (try field(answers[0], "probabilities")).array.items;
    try std.testing.expect((try field(probabilities[0], "value")) == .bool);
    try std.testing.expect((try field(probabilities[1], "value")) == .string);
}

test "SystemOne confidence uses chance correction and score dispersion" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), confidence(&.{ 0.75, 0.25 }, .choice, .systemone), 0.000001);
    try std.testing.expectApproxEqAbs(@as(f64, 0), confidence(&.{ 0.5, 0.5 }, .choice, .systemone), 0.000001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), confidence(&.{ 0.1, 0.8, 0.1 }, .score, .systemone), 0.000001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), confidence(&.{ 0.1, 0.8, 0.1 }, .score, .openai), 0.000001);
}

test "SystemOne permits a single choice or score level" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try parse(a,
        \\{"model":"clef","state":"x","questions":{"one":{"type":"choice","criteria":{"only":"Available"}},"level":{"type":"score","criteria":["Only level"]}}}
    , .systemone);
    const encoded = try serialize(a, parsed, .{ .input_tokens = 1, .answers = &.{ .{ .probabilities = &.{1} }, .{ .probabilities = &.{1} } } }, .systemone);
    const root = try std.json.parseFromSliceLeaky(Value, a, encoded, .{});
    const answers = try field(root, "answers");
    try std.testing.expect(eql(try field(try field(answers, "one"), "choice"), "only"));
    const certainty = try field(try field(answers, "level"), "confidence");
    const actual: f64 = switch (certainty) {
        .integer => @floatFromInt(certainty.integer),
        .float => certainty.float,
        else => return error.ExpectedNumber,
    };
    try std.testing.expectEqual(@as(f64, 1), actual);
}
