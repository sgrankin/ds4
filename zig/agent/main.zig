//! Native program entry. CLI policy, signals and worker lifetime are owned by
//! the Zig frontend; model selection can still dispatch to C Qwen/GLM engines.
const c = @import("agent_runtime_c");

export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    return c.ds4_agent_frontend_main(argc, argv, c.DS4_BACKEND_METAL);
}
