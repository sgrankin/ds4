//! GPU startup and retirement order. Actual objects are owned by fixed ARC
//! factories; policy uses one shared scalar record, not shadow driver state.
const std = @import("std");
const c = @import("metal_c");
const Schedule = @import("command_schedule.zig");
const Cache = @import("expert_cache.zig");
const IO = @import("expert_io.zig");
const Context = c.ds4_metal_bootstrap_context;
fn state(context: *const Context) *c.ds4_metal_bootstrap_state {
    return @ptrCast(context.state);
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn enabled(name: [*:0]const u8) c_int {
    return c.ds4_zig_metal_env_bool(name);
}
fn detect(s: *c.ds4_metal_bootstrap_state) void {
    s.runtime_available = 0;
    s.family_supported = 0;
    s.queue_supported = 0;
    s.neural_accelerators_hint = 0;
    s.tensor_enabled = 0;
    s.tensor_compile_supported = 0;
    s.device_name[0] = 0;
    const name = c.ds4_metal_bootstrap_device_name();
    if (name != null) _ = c.snprintf(&s.device_name, s.device_name.len, "%s", name);
    const capabilities = c.ds4_metal_bootstrap_features_query(enabled("DS4_METAL_DISABLE_METAL4") > 0);
    s.runtime_available = @intFromBool(capabilities.runtime);
    s.family_supported = @intFromBool(capabilities.family);
    s.queue_supported = @intFromBool(capabilities.queue);
    const text = std.mem.sliceTo(&s.device_name, 0);
    if (capabilities.family and std.mem.find(u8, text, "M5") != null) s.neural_accelerators_hint = 1;
    if (!capabilities.family) return;
    // Metal exposes no independent Neural Accelerator bit. Only generations
    // expected to benefit select TensorOps automatically; earlier chips retain
    // established kernels even though they can compile the portable source.
    const automatic = std.mem.find(u8, text, "M5") != null or std.mem.find(u8, text, "M6") != null or
        std.mem.find(u8, text, "A19") != null or std.mem.find(u8, text, "A20") != null;
    if (automatic) {
        s.tensor_compile_supported = c.ds4_metal_bootstrap_tensor_probe();
        s.tensor_enabled = s.tensor_compile_supported;
        if (s.tensor_enabled == 0) _ = c.fprintf(c.__stderrp, "ds4: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
    } else {
        _ = c.fprintf(c.__stderrp, "ds4: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
    }
}
fn preM5(name: []const u8) bool {
    return name.len >= 8 and std.mem.startsWith(u8, name, "Apple M") and
        name[7] >= '1' and name[7] <= '4' and (name.len == 8 or name[8] == ' ');
}
pub fn init(context: *const Context) bool {
    const s = state(context);
    if (s.initialized != 0) return true;
    // A failed partial startup owns no surviving model resources or runtime borrow.
    var complete = false;
    defer if (!complete) cleanup(context);
    c.ds4_metal_bootstrap_reset_fast_cache();
    if (!c.ds4_metal_bootstrap_device()) return false;
    c.ds4_metal_bootstrap_diagnostics();
    detect(s);
    if (!c.ds4_metal_bootstrap_queue()) return false;
    if (!c.ds4_metal_bootstrap_bookkeeping()) return false;
    var source = std.mem.zeroes(c.ds4_metal_shader_resources);
    if (!c.ds4_metal_bootstrap_source(&source)) return false;
    defer c.ds4_metal_bootstrap_source_release(&source);
    // Missing env is -1: the first two flags default ON; the rest default OFF.
    const flags = c.ds4_metal_shader_flags{
        .hc_stable = enabled("DS4_METAL_HC_STABLE") != 0,
        .norm_unify = enabled("DS4_METAL_NORM_RSQRT_DISABLE") != 0,
        .raw_f32 = enabled("DS4_METAL_KV_RAW_F32") > 0,
        .rope_exp2 = enabled("DS4_METAL_ROPE_EXP2_LOG2") > 0,
        .math_safe = enabled("DS4_METAL_MATH_SAFE") > 0,
    };
    if (!c.ds4_metal_bootstrap_library(&source, &flags)) return false;
    if (!c.ds4_metal_bootstrap_base_pipelines()) return false;
    // This presence-only static choice is made after head-RMS/RoPE creation,
    // at the same catalogue boundary as C, not while compiling shader options.
    const schedule: *c.ds4_metal_scheduling = @ptrCast(context.schedule);
    schedule.commands.static_head_norm = preM5(std.mem.sliceTo(&s.device_name, 0)) and
        !present("DS4_METAL_DISABLE_PRE_M5_HEAD_RMS_ROPE_PIPELINE_STATIC");
    if (!c.ds4_metal_bootstrap_remaining_pipelines()) return false;
    s.initialized = 1;
    complete = true;
    return true;
}
/// Stop the thread outside the public cleanup autorelease scope, as in C.
pub fn beginCleanup(context: *const Context) bool {
    if (state(context).initialized == 0) return false;
    c.ds4_metal_bootstrap_stop_keepalive();
    return true;
}
pub fn cleanup(context: *const Context) void {
    c.ds4_metal_bootstrap_reset_fast_cache();
    Schedule.resetParallel(true);
    const schedule: *c.ds4_metal_scheduling = @ptrCast(context.schedule);
    const cache_state: *Cache.State = @ptrCast(context.cache);
    if (schedule.commands.batch) |command| {
        Schedule.closeEncoder();
        c.ds4_metal_bootstrap_commit_wait(command);
        c.ds4_metal_schedule_set_batch(null);
        cache_state.cache_done_seq = @max(cache_state.cache_done_seq, cache_state.cache_batch_seq);
        cache_state.cache_batch_seq = 0;
    }
    _ = Schedule.waitPending("cleanup");
    if (Cache.timingSummary() and !present("DS4_METAL_MEMORY_REPORT")) c.ds4_metal_bootstrap_memory_report();
    c.ds4_metal_bootstrap_retire_readback();
    const load: *c.ds4_metal_expert_load_state = @ptrCast(context.load);
    IO.shutdown(&load.pool);
    (Cache.Cache{ .state = cache_state, .streaming = context.streaming }).clearAll(true);
    c.ds4_metal_bootstrap_retire_addresses();
    c.ds4_metal_bootstrap_retire_pipelines();
    c.ds4_metal_bootstrap_retire_scratch();
    c.ds4_metal_bootstrap_retire_maps();
    c.ds4_metal_bootstrap_retire_bookkeeping();
    state(context).initialized = 0;
}
