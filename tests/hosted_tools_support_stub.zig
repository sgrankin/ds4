//! Cold C integration seams are not exercised by the model-free hosted tool
//! fixture. Keep accidental model materialization or persistence work obvious.
const c = @import("hosted_test_c");

export fn ds4_server_materialize_render_other(_: c.server_model_syntax, _: [*c]const c.chat_msgs, _: c_int, _: [*c]const u8, _: [*c]const c.tool_schema_orders, _: c.ds4_think_mode, _: bool) [*c]u8 {
    @panic("hosted tool fixture unexpectedly rendered a model prompt");
}

export fn ds4_server_materialize_restore_tools(_: [*c]c.server, _: [*c]const c.chat_msgs) void {
    @panic("hosted tool fixture unexpectedly restored live model tools");
}

export fn ds4_server_persist_log(_: c.ds4_log_type, _: [*c]const u8) void {
    @panic("hosted tool fixture unexpectedly entered persistence logging");
}
