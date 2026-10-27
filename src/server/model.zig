//! Adapter from canonical serving values to the retained DS4 runtime.
//! Model policy and DS4.1 formatting are native modules; C record/enum and
//! allocation conversions stay here at the compatibility boundary.
const std = @import("std");
const types = @import("serving_types");
const policy = @import("ds4_model");
const Canonical = @import("canonical.zig");

pub fn Backend(comptime c: type) type {
    return struct {
        engine: ?*c.ds4_engine,
        const Self = @This();

        pub fn model(self: Self) policy.Model {
            return .{
                .family = if (c.ds4_engine_is_flint(self.engine)) .flint else if (c.ds4_engine_is_deepseek41(self.engine)) .deepseek41 else if (c.ds4_engine_is_qwen4(self.engine)) .qwen else if (c.ds4_engine_is_glm53(self.engine)) .glm53 else if (c.ds4_engine_is_glm_dsa(self.engine)) .glm else .deepseek,
                .max_effort_min_context = c.ds4_think_max_min_context(),
            };
        }

        pub fn syntax(self: Self) c.server_model_syntax {
            return switch (self.model().family) {
                .deepseek41 => c.SERVER_MODEL_SYNTAX_DEEPSEEK41,
                .flint => c.SERVER_MODEL_SYNTAX_FLINT,
                .qwen => c.SERVER_MODEL_SYNTAX_QWEN,
                .glm, .glm53 => c.SERVER_MODEL_SYNTAX_GLM,
                .deepseek => c.SERVER_MODEL_SYNTAX_DEEPSEEK,
            };
        }

        pub fn incrementalTools(self: Self) bool {
            const enabled = c.getenv("DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS");
            return self.model().incrementalTools() and (enabled == null or !std.mem.eql(u8, std.mem.span(enabled), "0"));
        }

        pub fn thinking(self: Self, input: *const c.ds4_server_input, req: *c.request, context: c_int) void {
            const controls: types.ReasoningControls = .{
                .enabled = if (input.got_thinking) input.thinking_enabled else null,
                .effort = Canonical.reasoning(c, input.reasoning_effort),
            };
            const mode = self.model().reasoning(controls, optional(req.model), @intCast(@max(context, 0)));
            req.think_mode = Canonical.reasoningMode(c, mode);
        }

        pub fn render(self: Self, req: *const c.request, messages: *const c.chat_msgs, start: usize, schemas: [*c]const u8, live: bool, order: ?*const c.stop_list, context: usize) [*c]u8 {
            return self.renderMapped(req, messages, start, schemas, live, order, context, null);
        }

        pub fn renderTracked(self: Self, req: *c.request, messages: *const c.chat_msgs, start: usize, schemas: [*c]const u8, live: bool, order: ?*const c.stop_list, context: usize) [*c]u8 {
            var map: types.PromptMap = .{};
            defer map.deinit(std.heap.c_allocator);
            const rendered = self.renderMapped(req, messages, start, schemas, live, order, context, &map);
            if (req.model_syntax != c.SERVER_MODEL_SYNTAX_DEEPSEEK41 and req.model_syntax != c.SERVER_MODEL_SYNTAX_FLINT) return rendered;
            const ranges = std.heap.c_allocator.alloc(c.ds4_prompt_range, map.special_ranges.items.len) catch fatal();
            for (map.special_ranges.items, ranges) |range, *target| target.* = .{ .begin = range.begin, .end = range.end };
            const pointer: [*c]c.ds4_prompt_range = if (ranges.len == 0) null else ranges.ptr;
            if (!live) {
                c.free(req.prompt_special_ranges);
                req.prompt_special_ranges = pointer;
                req.prompt_special_count = ranges.len;
                req.cache_breakpoint_bytes.len = 0;
                for (map.cache_breakpoints.items) |position| c.ds4_tokens_push(&req.cache_breakpoint_bytes, @intCast(position));
            } else if (req.api == c.API_RESPONSES) {
                c.free(req.responses_live_special_ranges);
                req.responses_live_special_ranges = pointer;
                req.responses_live_special_count = ranges.len;
            } else {
                c.free(req.anthropic_live_special_ranges);
                req.anthropic_live_special_ranges = pointer;
                req.anthropic_live_special_count = ranges.len;
            }
            return rendered;
        }

        fn renderMapped(self: Self, req: *const c.request, messages: *const c.chat_msgs, start: usize, schemas: [*c]const u8, live: bool, order: ?*const c.stop_list, context: usize, map: ?*types.PromptMap) [*c]u8 {
            _ = self;
            if (req.model_syntax != c.SERVER_MODEL_SYNTAX_DEEPSEEK41 and req.model_syntax != c.SERVER_MODEL_SYNTAX_FLINT and req.model_syntax != c.SERVER_MODEL_SYNTAX_QWEN and req.model_syntax != c.SERVER_MODEL_SYNTAX_GLM)
                return c.ds4_server_materialize_render_other(req.model_syntax, messages, @intCast(start), schemas, &req.tool_orders, req.think_mode, live);

            var view = Canonical.View.init(std.heap.c_allocator, c, messages, &req.tool_orders) catch fatal();
            defer view.deinit();
            const initial = view.strings(order) catch fatal();
            const request: types.Request = .{
                .model = optional(req.model),
                .messages = view.messages,
                .tools = view.tools,
                .tool_schemas = optional(schemas),
                // Reasoning has already been resolved before continuation
                // validation. Preserve that exact decision at rendering.
                .reasoning = .{ .enabled = true, .effort = Canonical.reasoning(c, req.think_mode) },
                .response_format = responseFormat(view.arena.allocator(), optional(req.response_format_json)),
            };
            const target: policy.Model = .{ .family = switch (req.model_syntax) {
                c.SERVER_MODEL_SYNTAX_FLINT => .flint,
                c.SERVER_MODEL_SYNTAX_QWEN => .qwen,
                c.SERVER_MODEL_SYNTAX_GLM => .glm53,
                else => .deepseek41,
            } };
            const rendered = target.prepare(std.heap.c_allocator, request, .{ .start = start, .live_tail = live, .initial_tool_order = initial, .prompt_map = map }, context) catch fatal();
            return rendered.ptr;
        }

        pub fn tokenize(self: Self, runtime: ?*c.server, req: *c.request, messages: *const c.chat_msgs, errors: []u8) bool {
            return c.ds4_server_materialize_tokenize(self.engine, runtime, req, messages, if (errors.len == 0) null else errors.ptr, errors.len);
        }

        pub fn tokenizeText(self: Self, text: [*c]const u8, out: *c.ds4_tokens) void {
            c.ds4_tokenize_rendered_chat(self.engine, text, out);
        }

        fn optional(value: [*c]const u8) ?[]const u8 {
            return if (value == null) null else std.mem.span(value);
        }

        fn responseFormat(arena: std.mem.Allocator, raw: ?[]const u8) types.ResponseFormat {
            const source = raw orelse return .text;
            // The common contract already validated the request. Keep the
            // original answer schema in the prompt, as the recipe does.
            const value = std.json.parseFromSliceLeaky(std.json.Value, arena, source, .{}) catch fatal();
            const kind = value.object.get("type").?.string;
            if (std.mem.eql(u8, kind, "json_object")) return .json_object;
            if (std.mem.eql(u8, kind, "regex")) return .regex;
            if (!std.mem.eql(u8, kind, "json_schema")) return .text;
            const object = value.object.get("json_schema") orelse value;
            const schema = object.object.get("schema").?;
            return .{ .json_schema = std.json.Stringify.valueAlloc(arena, schema, .{}) catch fatal() };
        }

        fn fatal() noreturn {
            _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
            c.exit(1);
        }
    };
}
