//! One engine tensor per GDN layer contains all server-slot recurrent states.
//! Sessions own retained slot views; the engine keeps the parent handles alive.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");
const Pool = c.ds4_qwen4_state_pool;
const Config = dimensions.Config;

pub fn ensure(pool: *Pool, config: *const Config, slots: u32) bool {
    if (pool.slots != 0) return true;
    if (slots < 2 or slots > 64) return false;
    const state_bytes = dimensions.stateFloats(&config.shape) * @sizeOf(f32);
    const history_bytes = dimensions.historyFloats(&config.shape) * @sizeOf(f32);
    for (0..config.shape.n_layer) |layer| {
        if (!dimensions.isLinear(config, @intCast(layer))) continue;
        pool.state[layer] = c.ds4_gpu_tensor_alloc(state_bytes * slots);
        pool.history[layer] = c.ds4_gpu_tensor_alloc(history_bytes * slots);
        if (pool.state[layer] == null or pool.history[layer] == null) {
            free(pool);
            return false;
        }
    }
    pool.slots = slots;
    pool.used = 0;
    return true;
}

pub fn take(pool: *Pool) c_int {
    for (0..pool.slots) |slot| {
        const mask = @as(u64, 1) << @as(u6, @intCast(slot));
        if (pool.used & mask != 0) continue;
        pool.used |= mask;
        return @intCast(slot);
    }
    return -1;
}

pub fn release(pool: *Pool, slot: u32) void {
    pool.used &= ~(@as(u64, 1) << @as(u6, @intCast(slot)));
}

pub fn free(pool: *Pool) void {
    for (pool.state, pool.history) |state, history| {
        c.ds4_gpu_tensor_free(state);
        c.ds4_gpu_tensor_free(history);
    }
    pool.* = std.mem.zeroes(Pool);
}
