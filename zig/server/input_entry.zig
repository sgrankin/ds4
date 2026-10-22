//! Single C export facade for the selected server input stage.
//! Pure owners can import each other without emitting duplicate C symbols.
const c = @import("server_input_c");
const Record = c.ds4_server_input;
const Request = c.request;
const Messages = c.chat_msgs;
const Endpoint = c.ds4_server_input_endpoint;

const Input = @import("input.zig");

export fn ds4_server_input_decode(endpoint: Endpoint, body: [*c]const u8, defaults: *const c.ds4_server_input_defaults, out: *Record, error_text: [*c]u8, error_length: usize) bool {
    return Input.ds4_server_input_decode(endpoint, body, defaults, out, error_text, error_length);
}
export fn ds4_server_input_deinit(record: *Record) void {
    return Input.ds4_server_input_deinit(record);
}
export fn ds4_server_input_messages_free(messages: *Messages) void {
    return Input.ds4_server_input_messages_free(messages);
}
export fn ds4_server_input_request_free(request: *Request) void {
    return Input.ds4_server_input_request_free(request);
}

const Values = @import("input_values.zig");

export fn ds4_server_json_tools(cursor: *[*c]const u8, output: *[*c]u8, orders: [*c]c.tool_schema_orders) bool {
    return Values.ds4_server_json_tools(cursor, output, orders);
}
export fn ds4_server_input_schema_order_add(orders: [*c]c.tool_schema_orders, raw: [*c]const u8, namespace: [*c]const u8, wire_name: [*c]const u8, hosted_search: bool) void {
    return Values.ds4_server_input_schema_order_add(orders, raw, namespace, wire_name, hosted_search);
}

const MessagesOwner = @import("message_values.zig");

export fn ds4_server_json_messages(cursor: *[*c]const u8, messages: *c.chat_msgs, trusted_store: bool) bool {
    return MessagesOwner.ds4_server_json_messages(cursor, messages, trusted_store);
}
export fn ds4_server_json_anthropic_messages(cursor: *[*c]const u8, messages: *c.chat_msgs) bool {
    return MessagesOwner.ds4_server_json_anthropic_messages(cursor, messages);
}
export fn ds4_server_json_anthropic_system(cursor: *[*c]const u8, output: *[*c]u8) bool {
    return MessagesOwner.ds4_server_json_anthropic_system(cursor, output);
}
export fn ds4_server_json_content(cursor: *[*c]const u8, output: *[*c]u8) bool {
    return MessagesOwner.ds4_server_json_content(cursor, output);
}
export fn ds4_server_json_function_call(cursor: *[*c]const u8, call: *c.tool_call) bool {
    return MessagesOwner.ds4_server_json_function_call(cursor, call);
}
export fn ds4_server_json_tool_calls(cursor: *[*c]const u8, calls: *c.tool_calls) bool {
    return MessagesOwner.ds4_server_json_tool_calls(cursor, calls);
}
export fn ds4_server_json_image_url(cursor: *[*c]const u8, output: *[*c]u8) bool {
    return MessagesOwner.ds4_server_json_image_url(cursor, output);
}

const Responses = @import("responses_values.zig");

export fn ds4_server_json_responses_input(cursor: *[*c]const u8, list: *c.chat_msgs, loaded_schemas: [*c][*c]u8, orders: [*c]c.tool_schema_orders, incremental: bool) bool {
    return Responses.ds4_server_json_responses_input(cursor, list, loaded_schemas, orders, incremental);
}
export fn ds4_server_json_responses_content(cursor: *[*c]const u8, output: *[*c]u8, images: [*c]c.server_image_inputs) bool {
    return Responses.ds4_server_json_responses_content(cursor, output, images);
}
export fn ds4_server_input_restore_discovered_tools(list: *c.chat_msgs, orders: [*c]c.tool_schema_orders, tool_choice_none: bool, explicit_tools: [*c]const u8) c_int {
    return Responses.ds4_server_input_restore_discovered_tools(list, orders, tool_choice_none, explicit_tools);
}
