//! The sole C facade for complete Conversation persistence/route actions.
const c = @import("server_conversations_c");
const owner = @import("conversations.zig");
fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    return if (pointer == null or capacity == 0) &.{} else pointer[0..capacity];
}
export fn ds4_conversation_json_field(object: [*c]const u8, wanted: [*c]const u8) [*c]u8 {
    return owner.jsonField(object, wanted);
}
export fn ds4_conversation_free(record: ?*c.ds4_conversation) void {
    owner.deinit(record);
}
export fn ds4_conversation_load(root: [*c]const u8, id: [*c]const u8, record: *c.ds4_conversation, error_text: [*c]u8, capacity: usize) c_int {
    return owner.load(root, id, record, errors(error_text, capacity));
}
export fn ds4_conversation_input_items(raw: [*c]const u8, normalized: *[*c]u8, error_text: [*c]u8, capacity: usize) bool {
    return owner.inputItems(raw, normalized, errors(error_text, capacity));
}
export fn ds4_conversation_publish(root: [*c]const u8, id: [*c]const u8, expected: c_ulonglong, input: [*c]const u8, output: [*c]const u8, replay: [*c]const u8, error_text: [*c]u8, capacity: usize) c_int {
    return owner.publish(root, id, expected, input, output, replay, errors(error_text, capacity));
}
export fn ds4_conversation_route(root: [*c]const u8, method: [*c]const u8, path: [*c]const u8, body: [*c]const u8, public_json: *[*c]u8, error_text: [*c]u8, capacity: usize) c_int {
    return owner.route(root, method, path, body, public_json, errors(error_text, capacity));
}
