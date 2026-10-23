//! Serving policy for the model families supported by the retained DS4 runtime.
//! No wire protocol, server records or engine headers enter this module. The
//! native DS4.1 prompt implementation lives beside it; legacy renderers remain
//! behind the runtime adapter until their implementations are needed.
const std = @import("std");
const serving = @import("serving_types");
const ds41 = @import("ds41_prompt");
const flint = @import("flint_prompt");

pub const Family = enum { deepseek, deepseek41, glm, glm53, qwen, flint };
pub const Capabilities = struct {
    reasoning_budget: bool,
    context_relocation: bool,
};

pub const Model = struct {
    family: Family = .deepseek41,
    legacy_pro: bool = false,
    max_effort_min_context: usize = 393216,

    pub fn id(self: Model) [:0]const u8 {
        return switch (self.family) {
            .deepseek41 => "deepseek-v4.1-flash",
            .flint => "flint",
            .qwen => "qwen3.8-flash-next",
            .glm53 => "glm-5.3-flash",
            .glm => "glm-5.2",
            .deepseek => if (self.legacy_pro) "deepseek-v4-pro" else "deepseek-v4-flash",
        };
    }

    pub fn capabilities(self: Model) Capabilities {
        return .{
            .reasoning_budget = self.family == .deepseek41,
            .context_relocation = self.family == .deepseek41,
        };
    }

    pub fn incrementalTools(self: Model) bool {
        return self.family == .deepseek41;
    }

    /// Native model preparation takes logical values, independent of HTTP,
    /// persisted record layouts and the engine's C ABI.
    pub fn prepare(self: Model, allocator: std.mem.Allocator, request: serving.Request, options: ds41.Options, context: usize) ![:0]u8 {
        if (self.family == .flint) return flint.render(allocator, request.messages, .{
            .start = options.start,
            .schemas = request.tool_schemas,
        });
        if (self.family != .deepseek41) return error.LegacyModelAdapterRequired;
        var selected = options;
        selected.schemas = request.tool_schemas;
        selected.reasoning = self.reasoning(request.reasoning, request.model, context);
        return ds41.render(allocator, request.messages, selected);
    }

    /// An explicit client switch wins over a model alias. Context restrictions
    /// belong to the model, not to Responses or another transport.
    pub fn reasoning(self: Model, controls: serving.ReasoningControls, alias: ?[]const u8, context: usize) serving.Reasoning {
        const enabled = controls.enabled orelse aliasThinking(alias) orelse true;
        if (!enabled) return .none;
        return switch (controls.effort) {
            .max => if (self.family != .deepseek41 and context < self.max_effort_min_context) .high else .max,
            else => controls.effort,
        };
    }
};

fn aliasThinking(alias: ?[]const u8) ?bool {
    const name = alias orelse return null;
    const disabled = [_][]const u8{
        "deepseek-chat",              "qwen3.8-flash-next-chat",      "qwen3.8-flash-next-no-think",
        "qwen3.8-flash-next-nothink", "qwen/qwen3.8-flash-next-chat", "glm-5.2-chat",
        "glm-5.2-no-think",           "glm-5.2-nothink",              "zai/glm-5.2-chat",
        "glm-5.3-flash-chat",         "glm-5.3-flash-no-think",       "glm-5.3-flash-nothink",
        "zai/glm-5.3-flash-chat",
    };
    for (disabled) |candidate| if (std.mem.eql(u8, name, candidate)) return false;
    const enabled = [_][]const u8{
        "deepseek-reasoner",          "qwen3.8-flash-next-reasoner", "qwen/qwen3.8-flash-next-reasoner",
        "glm-5.2-reasoner",           "zai/glm-5.2-reasoner",        "glm-5.3-flash-reasoner",
        "zai/glm-5.3-flash-reasoner",
    };
    for (enabled) |candidate| if (std.mem.eql(u8, name, candidate)) return true;
    return null;
}

pub fn knownModel(name: []const u8) bool {
    const names = [_][]const u8{
        "flint",                            "flint-int4",
        "deepseek-v4-flash",                "deepseek-v4.1-flash",
        "deepseek-v4-pro",                  "qwen3.8-flash-next",
        "qwen3.8-flash-next-chat",          "qwen3.8-flash-next-no-think",
        "qwen3.8-flash-next-nothink",       "qwen3.8-flash-next-reasoner",
        "qwen/qwen3.8-flash-next",          "qwen/qwen3.8-flash-next-chat",
        "qwen/qwen3.8-flash-next-reasoner", "glm-5.2",
        "glm-5.2-chat",                     "glm-5.2-no-think",
        "glm-5.2-nothink",                  "glm-5.2-reasoner",
        "zai/glm-5.2",                      "zai/glm-5.2-chat",
        "zai/glm-5.2-reasoner",             "glm-5.3-flash",
        "glm-5.3-flash-chat",               "glm-5.3-flash-no-think",
        "glm-5.3-flash-nothink",            "glm-5.3-flash-reasoner",
        "zai/glm-5.3-flash",                "zai/glm-5.3-flash-chat",
        "zai/glm-5.3-flash-reasoner",
    };
    for (names) |candidate| if (std.mem.eql(u8, name, candidate)) return true;
    return false;
}
