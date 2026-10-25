//! Native typed decisions, independent of the two HTTP wire formats.
const std = @import("std");
pub const Value = std.json.Value;
pub const Kind = enum(u8) { noul, choice, score };

pub const Option = struct {
    /// Stable model-facing ID. Wire values remain separately typed.
    id: []const u8,
    description: Value = .null,
    value: Value = .null,
};

pub const Question = struct {
    id: []const u8,
    kind: Kind,
    instructions: Value = .null,
    /// Caller order. Choice IDs are sorted only for model preparation.
    options: []const Option = &.{},
};

pub const MediaKind = enum { image, audio, video };
pub const MediaSource = struct {
    kind: MediaKind,
    source: Value,
    label: ?[]const u8 = null,
};

pub const Request = struct {
    state: Value,
    questions: []const Question,
    media: []const MediaSource = &.{},
};

pub const Answer = struct {
    /// Probabilities in the original Request.Question.options order.
    /// Noul with default options is true, false; score levels retain order.
    probabilities: []const f32,
};

pub const Result = struct {
    answers: []const Answer,
    input_tokens: usize,
};

pub const Span = struct {
    start: usize,
    end: usize,
    pub fn len(self: Span) usize {
        return self.end - self.start;
    }
};

/// Borrowed memory-side features for the small joint head. The backend retains
/// their storage until its next model call; caller keeps the engine lock.
pub const HeadFeatures = struct {
    normalized_hidden: []const f32,
    memory: []const f32,
    /// Evidence K/V first, decoder cross-attention K/V next; rows are [T,2W].
    /// Key/value biases are deliberately omitted and handled by the CPU head.
    kv: []const []const f32,
};
