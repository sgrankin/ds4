//! CPU token-row staging for the graph's validated dense weight formats.
//! Quantized rows are decoded once, then copied to the four HC streams.
const c = @import("gguf_c");
const Model = c.ds4_model;
const Weight = c.ds4_tensor;

fn half(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub fn row(model: *const Model, weight: *const Weight, index: u64, output: []f32) bool {
    const width: usize = @intCast(weight.dim[0]);
    const base: [*]const u8 = @ptrCast(model.map + weight.abs_offset);
    switch (weight.type) {
        c.DS4_TENSOR_F32 => {
            const values: [*]align(1) const f32 = @ptrCast(base + index * width * @sizeOf(f32));
            for (output[0..width], values[0..width]) |*destination, value| destination.* = value;
        },
        c.DS4_TENSOR_F16, c.DS4_TENSOR_BF16 => {
            const values: [*]align(1) const u16 = @ptrCast(base + index * width * @sizeOf(u16));
            for (output[0..width], values[0..width]) |*destination, value| {
                destination.* = if (weight.type == c.DS4_TENSOR_F16) half(value) else @bitCast(@as(u32, value) << 16);
            }
        },
        c.DS4_TENSOR_Q8_0 => {
            const blocks = width / 32;
            const source = base + index * blocks * 34;
            for (0..blocks) |block| {
                const block_bytes = source + block * 34;
                const scale = half(@as(u16, block_bytes[0]) | (@as(u16, block_bytes[1]) << 8));
                for (0..32) |column| {
                    const code: i8 = @bitCast(block_bytes[2 + column]);
                    output[block * 32 + column] = scale * @as(f32, @floatFromInt(code));
                }
            }
        },
        c.DS4_TENSOR_Q4_0 => {
            const blocks = width / 32;
            const source = base + index * blocks * 18;
            for (0..blocks) |block| {
                const block_bytes = source + block * 18;
                const scale = half(@as(u16, block_bytes[0]) | (@as(u16, block_bytes[1]) << 8));
                for (0..16) |column| {
                    const low = @as(i32, block_bytes[2 + column] & 15) - 8;
                    const high = @as(i32, block_bytes[2 + column] >> 4) - 8;
                    output[block * 32 + column] = scale * @as(f32, @floatFromInt(low));
                    output[block * 32 + 16 + column] = scale * @as(f32, @floatFromInt(high));
                }
            }
        },
        else => return false,
    }
    return true;
}
