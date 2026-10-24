//! Synchronous model-runtime borrows shared by ordinary session operations.
//! Replay re-enters the native common session owner; no private Session layout
//! or C model traversal is needed by these captured callbacks.
const c = @import("gguf_c");
const View = c.ds4_session_view;

fn sync(user: ?*anyopaque, prompt: [*c]const c.ds4_tokens, errors: [*c]u8, capacity: usize) callconv(.c) c_int {
    const view: *View = @ptrCast(@alignCast(user orelse return 1));
    return c.ds4_native_session_sync(view, prompt, errors, capacity);
}

pub fn qwen(view: *View) c.ds4_qwen4_runtime {
    return .{
        .state = view.qwen,
        .frontier = view.frontier,
        .pending = view.pending,
        .control = view.control,
        .model = &view.core.*.model,
        .weights = &view.core.*.weights,
        .exact_sampling = view.core.*.dspark_exact_sampling,
        .sync = sync,
        .user = view,
    };
}

pub fn glm(view: *View) c.ds4_glm_runtime {
    return .{
        .state = view.glm,
        .frontier = view.frontier,
        .pending = view.pending,
        .control = view.control,
        .core = view.core,
        .ablate_mask = view.ablate_mask,
        .sync = sync,
        .user = view,
    };
}
