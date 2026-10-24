//! Cold entries: neither platform device queries nor model/Engine code may
//! enter a host fixture that only checks admission arithmetic.
const math = @import("engine_admission_math.zig");
export fn ds4_native_glm_guard_default_reserve(base: u64, model: u64, glm53: bool) f64 {
    return math.defaultReserve(base, model, glm53);
}
export fn ds4_native_glm_guard_budget(base: u64, wired: u64, fraction: f64, reserve_gib: f64) u64 {
    return math.budgetBytes(base, wired, fraction, reserve_gib);
}
