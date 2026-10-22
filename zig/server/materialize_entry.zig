//! The sole C facade; all materialization policy lives in the pure owner.
const c = @import("server_materialize_c");
const owner = @import("materialize.zig");

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    return if (capacity == 0) &.{} else pointer[0..capacity];
}
export fn ds4_server_materialize(engine: ?*c.ds4_engine, runtime: ?*c.server, endpoint: c.ds4_server_input_endpoint, input: *c.ds4_server_input, context: c_int, incremental: bool, out: *c.request, error_text: [*c]u8, capacity: usize) bool {
    return owner.materialize(engine, runtime, endpoint, input, context, incremental, out, errors(error_text, capacity));
}
export fn ds4_server_materialize_validate(runtime: ?*c.server, messages: ?*const c.chat_msgs, api: c.api_style, mode: c.ds4_think_mode, live: ?*bool, reasoning: ?*bool, error_text: [*c]u8, capacity: usize) bool {
    return owner.validate(runtime, messages, api, mode, live, reasoning, errors(error_text, capacity));
}
export fn ds4_server_materialize_live(runtime: ?*c.server, req: ?*c.request, messages: ?*const c.chat_msgs) void {
    owner.prepareLive(runtime, req, messages);
}
export fn ds4_server_response_string_field(object: [*c]const u8, wanted: [*c]const u8) [*c]u8 {
    return owner.stringField(object, wanted);
}
export fn ds4_server_response_record_load(directory: [*c]const u8, id: [*c]const u8, response: *[*c]u8, replay: *[*c]u8, error_text: [*c]u8, capacity: usize) c_int {
    return owner.loadResponse(directory, id, response, replay, errors(error_text, capacity));
}
