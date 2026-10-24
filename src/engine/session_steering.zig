//! Live steering publication over the actual selected model owner. Startup
//! remains responsible for loading vectors. Changes affect subsequent tokens;
//! already-computed KV keeps the influence of its earlier steering settings.
const std = @import("std");
const c = @import("gguf_c");
const Control = @import("session_control.zig");
const View = c.ds4_session_view;

pub fn scale(view: *const View) f32 {
    return switch (view.kind) {
        c.DS4_SESSION_DS41 => view.ds41.*.graph.steering_ffn_scale,
        c.DS4_SESSION_FLINT => c.flint_session_steering_ffn(view.flint.*),
        c.DS4_SESSION_QWEN => view.qwen.*.graph.steer_ffn_scale,
        c.DS4_SESSION_GLM53 => view.glm.*.graph.directional_steering_ffn_scale,
        else => view.core.*.directional_steering_ffn_scale,
    };
}
fn loaded(view: *const View) bool {
    if (view.kind == c.DS4_SESSION_DS41) return view.ds41.*.ready and view.ds41.*.graph.steering_dirs != null;
    // Flint's own setter checks its immutable engine bank; avoid mirroring it.
    if (view.kind == c.DS4_SESSION_FLINT) return true;
    if (view.kind == c.DS4_SESSION_QWEN) return view.qwen.*.ready and view.qwen.*.graph.steer_dirs != null;
    if (view.kind != c.DS4_SESSION_GLM53) return false;
    const graph = &view.glm.*.graph;
    if (graph.layer_start > graph.layer_end) return false;
    const tier = if (graph.placement == null) 0 else graph.placement[graph.layer_start + 1];
    return tier >= 0 and tier < c.DS4_MAX_GPUS and graph.directional_steering_dirs_by_tier[@intCast(tier)] != null;
}

pub fn set(view: *View, value: f32) c_int {
    if (!std.math.isFinite(value) or value < -100 or value > 100) return 1;
    if (view.transport.*.active) {
        _ = c.fputs("ds4: live steering changes require a local session\n", c.__stderrp);
        return 1;
    }
    if (view.kind == c.DS4_SESSION_FLINT and c.flint_session_set_steering_ffn(view.flint.*, value) != 0) return 1;
    if (value != 0 and !loaded(view)) {
        _ = c.fputs("ds4: live FFN steering needs a directional steering vector loaded at startup\n", c.__stderrp);
        return 1;
    }
    view.core.*.directional_steering_ffn_scale = value;
    switch (view.kind) {
        c.DS4_SESSION_DS41 => {
            view.ds41.*.graph.steering_ffn_scale = value;
            Control.clearCapture(view.control);
        },
        c.DS4_SESSION_QWEN => {
            view.qwen.*.graph.steer_ffn_scale = value;
            view.pending.*.have = 0;
        },
        c.DS4_SESSION_GLM53 => {
            view.glm.*.graph.directional_steering_ffn_scale = value;
            view.pending.*.have = 0;
            view.pending.*.rollback_valid = false;
        },
        else => Control.clearCapture(view.control),
    }
    view.frontier.*.mtp_draft_valid = false;
    return 0;
}
