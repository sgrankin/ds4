//! Admission policy for this whole prefill-load transaction. The real cache
//! supplies capacity; the driver snapshot contains only immutable entry facts.
const std = @import("std");
const c = @import("metal_c");
const Core = @import("expert_load.zig");
const present = Core.present;
pub fn addressRequested(view: c.ds4_metal_expert_load_view) bool {
    return view.streaming and
        (present("DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE") or
            present("DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR") or
            present("DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR") or
            view.prefill_building or view.glm_building or
            (present("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") and
                !present("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR")) or view.streaming) and
        !present("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE");
}
fn tokenLimit(name: [*:0]const u8, fallback: u32) u32 {
    const raw = c.getenv(name);
    if (raw != null and raw[0] != 0) {
        var end: [*c]u8 = null;
        const value = c.strtol(raw, &end, 10);
        // The original accepts a numeric prefix; a trailing suffix is allowed.
        if (end != raw) return if (value <= 0) 0 else @intCast(@min(@as(c_ulong, @intCast(value)), std.math.maxInt(u32)));
    }
    return fallback;
}
pub fn prefillEnabled(context: Core.Context, tokens: u32, total: u32, used: u32, gate_type: u32, down_type: u32) bool {
    if (!context.cache.streaming or tokens <= 1 or total == 0 or used != 6 or
        gate_type != c.DS4_METAL_TENSOR_IQ2_XXS or down_type != c.DS4_METAL_TENSOR_Q2_K or
        context.view.quality or present("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") or
        present("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") or present("DS4_METAL_MOE_WRITE_CLAMPED_ACT") or
        present("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION")) return false;
    // Every unique expert must fit at once. Other layers' slots can be reused
    // after the global cache fills; this is not a per-layer capacity budget.
    if (context.cache.configured() < total) return false;
    if (present("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR")) return true;
    const maximum = tokenLimit("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX", if (total == 384) 800 else if (total == 256) 760 else 0);
    const minimum = tokenLimit("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN", if (total == 384 or total == 256) 2 else 0);
    return maximum != 0 and tokens >= minimum and tokens <= maximum;
}
