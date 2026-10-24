//! CPU-only working-set budget arithmetic. Keep the proven large-host reserve,
//! resident 128 GiB GLM53 reserve and explicit wired-limit override intact.
const gib: u64 = 1024 * 1024 * 1024;
pub fn defaultReserve(base: u64, model: u64, glm53: bool) f64 {
    @setFloatMode(.optimized);
    const base_gib = @as(f64, @floatFromInt(base)) / gib;
    const model_gib = @as(f64, @floatFromInt(model)) / gib;
    if (base_gib >= 480 and base_gib <= 640 and model_gib >= base_gib * 0.80) return 24;
    if (glm53 and base_gib >= 108 and base_gib <= 160 and model_gib >= 70) return 18;
    return 32;
}
pub fn budgetBytes(base: u64, wired: u64, fraction: f64, reserve_gib: f64) u64 {
    @setFloatMode(.optimized);
    const fractional: u64 = @intFromFloat(@as(f64, @floatFromInt(base)) * fraction);
    const reserve: u64 = @intFromFloat(reserve_gib * gib);
    const after_reserve = base -| reserve;
    var bytes = if (reserve != 0) @min(fractional, after_reserve) else fractional;
    if (wired != 0) bytes = @max(bytes, if (wired > 2 * gib) wired - 2 * gib else wired);
    return bytes;
}
