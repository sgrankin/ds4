//! Sole frontend facade. The vendor completion API has no userdata and SIGINT
//! must only set sig_atomic_t: their two original slots live exactly here.
const c = @import("agent_runtime_c");
const Ui = @import("ui_text.zig");
const Frontend = @import("frontend.zig");
const Options = @import("options.zig");
const Commands = @import("commands.zig");
const Bootstrap = @import("bootstrap.zig");
export var ds4_agent_completion_worker: ?*c.agent_worker = null;
var interrupted: c.sig_atomic_t = 0;
fn signalHandler(_: c_int) callconv(.c) void {
    const flag: *volatile c.sig_atomic_t = &interrupted;
    flag.* = 1;
}
export fn ds4_agent_frontend_complete(input: [*c]const u8, output: [*c]c.linenoiseCompletions) callconv(.c) void {
    const worker = ds4_agent_completion_worker orelse return;
    c.ds4_agent_sessions_complete(worker.cache_dir, @intCast(c.ds4_engine_model_id(worker.engine)), input, output);
}
fn frontend() Frontend.Frontend {
    return .{ .completion_slot = &ds4_agent_completion_worker, .completion_callback = ds4_agent_frontend_complete, .interrupt = &interrupted };
}
export fn ds4_agent_frontend_main(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend) c_int {
    return Bootstrap.run(argc, argv, default_backend, frontend(), signalHandler);
}
export fn ds4_agent_frontend_options(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend) c.agent_config {
    return Options.parse(argc, argv, default_backend);
}
export fn ds4_agent_frontend_defaults(engine: ?*c.ds4_engine, options: ?*c.agent_generation_options) void {
    Options.modelDefaults(engine, options);
}
export fn ds4_agent_frontend_headless(engine: ?*c.ds4_engine, config: *c.agent_config) c_int {
    return Frontend.headless(engine, config);
}
export fn ds4_agent_frontend_interactive(engine: ?*c.ds4_engine, config: *c.agent_config) c_int {
    return frontend().interactive(engine, config);
}
export fn ds4_agent_frontend_power(value: [*c]const u8, output: *c_int) bool {
    return Commands.power(value, output);
}
export fn ds4_agent_frontend_steering(value: [*c]const u8, output: *f32) bool {
    return Commands.steering(value, output);
}
export fn ds4_agent_frontend_hints(value: [*c]const u8, output: *bool) bool {
    return Commands.hints(value, output);
}
export fn ds4_agent_frontend_with_args(command: [*c]const u8, name: [*c]const u8) bool {
    return Commands.withArguments(Ui.text(command), Ui.text(name));
}
export fn ds4_agent_frontend_known(command: [*c]const u8) bool {
    return Commands.known(Ui.text(command));
}
