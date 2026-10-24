//! Stateless ordinary Metal generation. Public session APIs compose the native
//! DS41/GLM/Flint owners. Qwen keeps its specialized final-head-only prefill;
//! numerical model work and graph allocation remain in the native model.
const std = @import("std");
const c = @import("generation_c");

pub const Output = struct {
    emit: c.ds4_token_emit_fn,
    done: c.ds4_generation_done_fn,
    user: ?*anyopaque,
    progress: c.ds4_session_progress_fn,
    progress_user: ?*anyopaque,

    pub fn token(self: Output, value: c_int) void {
        if (self.emit) |callback| callback(self.user, value);
    }
    pub fn finish(self: Output) void {
        if (self.done) |callback| callback(self.user);
    }
};
pub fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}
pub fn timing(prefix: [:0]const u8, input: c_int, generated: c_int, prefill: f64, decode: f64) void {
    c.ds4_log(c.__stderrp, c.DS4_LOG_TIMING, "ds4: %sprefill: %.2f t/s, generation: %.2f t/s\n", prefix.ptr, if (prefill > 0) @as(f64, @floatFromInt(input)) / prefill else @as(f64, 0), if (decode > 0) @as(f64, @floatFromInt(generated)) / decode else @as(f64, 0));
}
fn greedyTopOne() bool {
    const setting = c.getenv("DS4_CUDA_GREEDY_TOP1");
    return setting == null or setting[0] == 0 or c.strcmp(setting, "0") != 0;
}

pub fn generate(engine: *c.ds4_engine, core: *const c.ds4_engine_core, qwen: ?*const c.ds4_qwen4_config, optional_prompt: ?*const c.ds4_tokens, maximum: c_int, context: c_int, output: Output) c_int {
    const prompt = optional_prompt orelse {
        _ = c.fputs("ds4: prompt is empty or exceeds context size\n", c.__stderrp);
        return 1;
    };
    const flint = c.ds4_engine_is_flint(engine);
    const penalty = c.ds4_engine_overthinking_penalty_enabled(engine);
    if (!flint and qwen != null and !penalty) return @import("qwen4/generation.zig").generate(engine, core, qwen.?, prompt, maximum, context, output);
    const glm = c.ds4_engine_is_glm53(engine);
    if (prompt.len <= 0 or prompt.len > context or
        (prompt.len == context and !flint and !(glm and !penalty and maximum <= 0)))
    {
        _ = c.fputs("ds4: prompt is empty or exceeds context size\n", c.__stderrp);
        return 1;
    }
    if (glm and !penalty and maximum <= 0) {
        output.finish();
        return 0;
    }
    const created = now();
    var session: ?*c.ds4_session = null;
    if (c.ds4_session_create(&session, engine, context) != 0) return 1;
    defer c.ds4_session_free(session);
    c.ds4_session_set_progress(session, output.progress, output.progress_user);
    const prefill_start = if (c.ds4_engine_is_deepseek41(engine) and !penalty) created else now();
    var errors: [256]u8 = @splat(0);
    var status = c.ds4_session_sync(session, prompt, &errors, errors.len);
    c.ds4_session_set_progress(session, null, null);
    const prefill_end = now();
    if (status != 0 and !flint and !penalty) {
        _ = c.fprintf(c.__stderrp, "ds4: prefill failed: %s\n", &errors);
        return 1;
    }
    var generated: c_int = 0;
    const start = now();
    const top_one = c.ds4_engine_is_deepseek41(engine) and !penalty and greedyTopOne();
    var token = if (status == 0 and !flint) c.ds4_session_argmax(session) else -1;
    while (status == 0 and generated < maximum and c.ds4_session_pos(session) < context) {
        // Flint selects only when another packet fits the caller's budget.
        if (flint) token = c.ds4_session_argmax(session);
        if (token < 0) {
            status = 1;
            break;
        }
        if (c.ds4_token_is_stop(engine, token)) break;
        if (flint) {
            var accepted: [8]c_int = undefined;
            const count = c.ds4_session_eval_speculative_argmax(session, token, maximum - generated, c.ds4_token_eos(engine), &accepted, accepted.len, &errors, errors.len);
            if (count < 1) {
                status = 1;
                break;
            }
            for (accepted[0..@intCast(count)]) |value| output.token(value);
            generated += count;
            continue;
        }
        output.token(token);
        generated += 1;
        if (generated >= maximum or c.ds4_session_pos(session) + 1 >= context) break;
        if (top_one) {
            token = c.ds4_session_eval_argmax(session, token, &errors, errors.len);
            status = @intFromBool(token < 0);
        } else {
            status = c.ds4_session_eval(session, token, &errors, errors.len);
            if (status == 0) token = c.ds4_session_argmax(session);
        }
    }
    const end = now();
    if (status != 0) _ = c.fprintf(c.__stderrp, "ds4: greedy generation failed: %s\n", &errors);
    output.finish();
    timing(if (glm and !penalty) "GLM " else "", prompt.len, generated, prefill_end - prefill_start, end - start);
    return @intFromBool(status != 0);
}
