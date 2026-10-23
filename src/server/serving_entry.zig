//! Sole C facade for the complete serving transaction and existing policy tests.
const c = @import("server_serving_c");
const owner = @import("serving.zig");

export fn ds4_server_serving_model_id(engine: ?*c.ds4_engine) [*c]const u8 {
    return owner.modelId(engine);
}
export fn ds4_server_serving_known_model(id: [*c]const u8) bool {
    return owner.knownModel(id);
}
export fn ds4_server_serving_decode(engine: ?*c.ds4_engine, runtime: ?*c.server, endpoint: c.ds4_server_input_endpoint, body: [*c]const u8, default_tokens: c_int, context: c_int, out: *c.request, errors: [*c]u8, capacity: usize) bool {
    return owner.decodeInput(engine, runtime, endpoint, body, default_tokens, context, out, if (capacity == 0 and errors == null) &.{} else @as([*]u8, @ptrCast(errors))[0..capacity]);
}
export fn ds4_server_serving_exceeds_context(req: [*c]const c.request, context: c_int) bool {
    return owner.exceedsContext(req, context);
}
export fn ds4_server_serving_required_slot(runtime: ?*c.server, pending: ?*const c.job) c_int {
    return owner.requiredSlot(runtime, pending);
}
export fn ds4_server_serving_score(runtime: ?*c.server, slot: ?*c.server_slot, pending: ?*const c.job, required: c_int, now: c.time_t) c_int {
    return owner.score(runtime, slot, pending, required, now);
}
export fn ds4_server_serving_dispatch(runtime: ?*c.server) void {
    owner.dispatch(runtime);
}
export fn ds4_server_serving_enqueue(runtime: *c.server, pending: *c.job) bool {
    return owner.enqueue(runtime, pending);
}
export fn ds4_server_serving_worker(runtime: *c.server, stop: [*c]const volatile c.sig_atomic_t) void {
    owner.worker(runtime, stop);
}
export fn ds4_server_serving_slot_worker(slot: *c.server_slot, stop: [*c]const volatile c.sig_atomic_t) void {
    owner.slotWorker(slot, stop);
}
export fn ds4_server_serving_cancel(runtime: *c.server, pending: *c.job) void {
    owner.cancel(runtime, pending);
}
export fn ds4_server_serving_wait(runtime: *c.server, pending: *c.job) void {
    owner.waitForJob(runtime, pending);
}
export fn ds4_server_serving_stop(runtime: *c.server) void {
    owner.requestStop(runtime);
}
export fn ds4_server_serving_models_json(out: *c.buf, id: [*c]const u8, name: [*c]const u8, context: c_int, default_tokens: c_int) void {
    owner.modelValues(out, id, name, context, default_tokens);
}
export fn ds4_server_serving_client(runtime: *c.server, fd: c_int, stop: [*c]const volatile c.sig_atomic_t) void {
    owner.client(runtime, fd, stop);
}
