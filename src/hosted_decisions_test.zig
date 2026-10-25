//! Model-free checks use the actual input/output ownership API linked into the
//! server. A registry stub keeps these protocol tests independent of Clef.
const std = @import("std");
const c = @import("hosted_test_c");

comptime {
    _ = @import("server/hosted_decisions.zig");
    _ = @import("server/hosted_tools_output_test.zig");
}

const descriptor =
    \\{"type":"ds4:decision","model":"clef-flash","input":"Client-bound invoice evidence"}
;
const ordinary =
    \\{"type":"function","name":"ds4_decide","parameters":{"type":"object","properties":{"ordinary":{"type":"string"}}}}
;
const program =
    \\{"type":"programmatic_tool_calling","ds4":{"inputs":{"queue":[{"id":"A","note":"SECRET_EVIDENCE"}]},"schemas":{"triage":{"type":"choice","instructions":"SECRET_POLICY","choices":[{"value":"active","description":"SECRET_CRITERIA"},{"value":"other","description":"other"}]}}}}
;
const program_helper =
    \\{"type":"ds4:decision","model":"clef-flash","allowed_callers":["programmatic"]}
;

test "program exposes manifests and keeps bound evidence and criteria out of the model schema" {
    var input = try decode("{\"input\":\"Review the queue\",\"tools\":[" ++ program ++ "," ++ program_helper ++ "]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expectEqual(@as(c_int, 2), input.req.tool_orders.len);
    const schema = std.mem.span(input.tool_schemas);
    try std.testing.expect(std.mem.indexOf(u8, schema, "SECRET_") == null);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, schema, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("ds4_exec", parsed.value.object.get("name").?.string);
    const execution = input.req.tool_orders.v[0];
    const helper = input.req.tool_orders.v[1];
    try std.testing.expect(execution.responses_programmatic);
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(execution.responses_program_bindings), "SECRET_EVIDENCE") != null);
    try std.testing.expect(helper.responses_decision_programmatic);
    try std.testing.expect(!helper.responses_decision_direct);
}

test "programmatic routing requires the runtime and excludes client tool suspension" {
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"tools\":[" ++ program_helper ++ "]}"));
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"tools\":[" ++ program ++ ",{\"type\":\"function\",\"name\":\"client\",\"allowed_callers\":[\"programmatic\"]}]}"));
    var input = try decode("{\"input\":\"x\",\"tools\":[" ++ program ++ ",{\"type\":\"ds4:decision\",\"model\":\"clef-flash\",\"allowed_callers\":[\"direct\",\"programmatic\"]}]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expect(input.req.tool_orders.v[1].responses_decision_direct);
    try std.testing.expect(input.req.tool_orders.v[1].responses_decision_programmatic);
}

test "predicate templates describe probability outputs without inventing a choice field" {
    var input = try decode("{\"input\":\"x\",\"tools\":[{\"type\":\"programmatic_tool_calling\",\"ds4\":{\"inputs\":{\"data\":{\"records\":[{\"id\":\"A\"}]}},\"schemas\":{\"check\":{\"type\":\"predicate\",\"instructions\":\"Is it active?\"}}}}]}");
    defer c.ds4_server_input_deinit(&input);
    const schema = std.mem.span(input.tool_schemas);
    try std.testing.expect(std.mem.indexOf(u8, schema, "predicates return probability") != null);
    try std.testing.expect(std.mem.indexOf(u8, schema, "choice:a.choice") == null);
}

test "program declaration cannot shadow a client function or be enabled by history" {
    const client = "{\"type\":\"function\",\"name\":\"ds4_exec\",\"parameters\":{\"type\":\"object\"}}";
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"tools\":[" ++ program ++ "," ++ client ++ "]}"));
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"tools\":[" ++ client ++ "," ++ program ++ "]}"));
    var req = std.mem.zeroes(c.request);
    defer c.ds4_server_input_request_free(&req);
    var schema: [*c]u8 = null;
    defer c.free(schema);
    var cursor: [*c]const u8 = "[" ++ program ++ "]";
    try std.testing.expect(!c.ds4_server_json_tools(&cursor, &schema, &req.tool_orders));
}

fn decode(body: [:0]const u8) !c.ds4_server_input {
    const defaults = c.ds4_server_input_defaults{
        .max_tokens = 128,
        .ctx_size = 8192,
        .syntax = c.SERVER_MODEL_SYNTAX_DEEPSEEK41,
        .responses_store_dir = null,
        .incremental_tool_schemas = false,
    };
    var input = std.mem.zeroes(c.ds4_server_input);
    var message: [512]u8 = @splat(0);
    if (!c.ds4_server_input_decode(c.DS4_SERVER_INPUT_RESPONSES, body, &defaults, &input, &message, message.len)) {
        // The public decoder clears all partially owned data on rejection.
        try std.testing.expect(input.req.tool_orders.v == null);
        try std.testing.expectEqual(@as(c_int, 0), input.req.tool_orders.len);
        return error.InvalidRequest;
    }
    return input;
}

test "current Responses declaration owns the model binding and generated schema" {
    var input = try decode("{\"input\":\"Route the invoice\",\"tools\":[" ++ descriptor ++ "]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expectEqual(@as(c_int, 1), input.req.tool_orders.len);
    const order = input.req.tool_orders.v[0];
    try std.testing.expectEqualStrings("ds4_decide", std.mem.span(order.name));
    try std.testing.expectEqualStrings("clef-flash", std.mem.span(order.responses_decision_model));
    try std.testing.expectEqualStrings("\"Client-bound invoice evidence\"", std.mem.span(order.responses_decision_input));
    try std.testing.expect(order.responses_decision_model != null);
    const schema = std.mem.span(input.tool_schemas);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, schema, .{});
    defer parsed.deinit();
    const properties = parsed.value.object.get("parameters").?.object.get("properties").?.object;
    try std.testing.expect(properties.contains("questions"));
    try std.testing.expect(!properties.contains("model"));
}

test "hosted binding rejects a same-name client function in either declaration order" {
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"tools\":[" ++ ordinary ++ "," ++ descriptor ++ "]}"));
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"tools\":[" ++ descriptor ++ "," ++ ordinary ++ "]}"));
}

test "ordinary function with the provider spelling remains client-owned" {
    var input = try decode("{\"input\":\"x\",\"tools\":[" ++ ordinary ++ "]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expectEqual(@as(c_int, 1), input.req.tool_orders.len);
    const order = input.req.tool_orders.v[0];
    try std.testing.expect(order.responses_decision_model == null);
    try std.testing.expect(order.responses_decision_input == null);
    try std.testing.expectEqualStrings("ordinary", std.mem.span(order.prop[0]));
}

test "restoring explicit current tools preserves the hosted binding" {
    for ([_]bool{ false, true }) |historical_client_tool| {
        var input = try decode("{\"input\":\"x\",\"tools\":[" ++ descriptor ++ "]}");
        defer c.ds4_server_input_deinit(&input);
        if (historical_client_tool) {
            input.messages.v[0].discovered_tools = c.strdup("[" ++ ordinary ++ "]");
            try std.testing.expect(input.messages.v[0].discovered_tools != null);
        }
        try std.testing.expectEqual(@as(c_int, if (historical_client_tool) 1 else 0), c.ds4_server_input_restore_discovered_tools(&input.messages, &input.req.tool_orders, false, input.explicit_tools));
        try std.testing.expectEqual(@as(c_int, 1), input.req.tool_orders.len);
        const order = input.req.tool_orders.v[0];
        try std.testing.expectEqualStrings("clef-flash", std.mem.span(order.responses_decision_model));
        try std.testing.expectEqualStrings("\"Client-bound invoice evidence\"", std.mem.span(order.responses_decision_input));
    }
}

test "historical tools cannot enable hosting but can replay a client function" {
    var request = std.mem.zeroes(c.request);
    defer c.ds4_server_input_request_free(&request);
    var schemas: [*c]u8 = null;
    defer c.free(schemas);
    var hosted_cursor: [*c]const u8 = "[" ++ descriptor ++ "]";
    try std.testing.expect(!c.ds4_server_json_tools(&hosted_cursor, &schemas, &request.tool_orders));
    try std.testing.expectEqual(@as(c_int, 0), request.tool_orders.len);
    var client_cursor: [*c]const u8 = "[" ++ ordinary ++ "]";
    try std.testing.expect(c.ds4_server_json_tools(&client_cursor, &schemas, &request.tool_orders));
    try std.testing.expectEqual(@as(c_int, 1), request.tool_orders.len);
    try std.testing.expect(request.tool_orders.v[0].responses_decision_model == null);
}

test "hosted tool call budget defaults to four and accepts a positive override" {
    var default_input = try decode("{\"input\":\"x\",\"tools\":[" ++ descriptor ++ "]}");
    defer c.ds4_server_input_deinit(&default_input);
    try std.testing.expectEqual(@as(c_int, 4), default_input.req.max_tool_calls);
    var configured = try decode("{\"input\":\"x\",\"max_tool_calls\":2,\"tools\":[" ++ descriptor ++ "]}");
    defer c.ds4_server_input_deinit(&configured);
    try std.testing.expectEqual(@as(c_int, 2), configured.req.max_tool_calls);
    try std.testing.expectError(error.InvalidRequest, decode("{\"input\":\"x\",\"max_tool_calls\":0,\"tools\":[" ++ descriptor ++ "]}"));
}

test "tool choice none preserves declaration metadata without enabling execution" {
    var input = try decode("{\"input\":\"x\",\"tool_choice\":\"none\",\"tools\":[" ++ descriptor ++ "]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expect(input.tool_choice_none);
    try std.testing.expectEqual(@as(c_int, 1), input.req.tool_orders.len);
    try std.testing.expect(!input.req.responses_hosted_tools);
}
