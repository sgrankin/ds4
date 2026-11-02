//! Exact narrow copies for scalar decode. GGUF widens checkpoint BF16/FP8
//! tensors; these immutable copies recover their original values, never a new
//! quantization. The engine owns the copies and session graphs only borrow.
const std = @import("std");
const c = @import("gguf_c");
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;

pub const Pair = struct {
    codes: ?*Tensor = null,
    scales: ?*Tensor = null,

    fn free(self: *Pair) void {
        c.ds4_gpu_tensor_free(self.codes);
        c.ds4_gpu_tensor_free(self.scales);
        self.* = .{};
    }
};

pub const Copies = struct {
    router: [64]?*Tensor = @splat(null),
    engram: [2]Pair = @splat(.{}),
    indexer: [64]Pair = @splat(.{}),
    bytes: u64 = 0,

    pub fn fromOpaque(pointer: ?*anyopaque) ?*Copies {
        return if (pointer) |value| @ptrCast(@alignCast(value)) else null;
    }

    pub fn create(model: *const c.ds4_model, weights: *const c.ds4_weights, layers: u32) ?*Copies {
        const self = std.heap.c_allocator.create(Copies) catch return null;
        self.* = .{};
        const router_enabled = c.getenv("DS4_METAL_DISABLE_V41_ROUTER_BF16") == null;
        const engram_enabled = c.getenv("DS4_METAL_DISABLE_V41_ENGRAM_E4M3") == null;
        const indexer_enabled = (c.ds4_gpu_v41_off() & c.DS4_GPU_V41_INDEXER_E4M3) == 0;
        var indexer_failed = false;
        for (weights.layer[0..@min(layers, self.router.len)], 0..) |layer, index| {
            if (router_enabled) self.router[index] = buildBf16(model, layer.ffn_gate_inp);
            if (engram_enabled and (index == 1 or index == 14))
                self.engram[if (index == 1) 0 else 1] = buildE4m3(model, layer.engram_kv);
            if (indexer_enabled and !indexer_failed and layer.indexer_attn_q_b != null) {
                self.indexer[index] = buildE4m3(model, layer.indexer_attn_q_b);
                if (self.indexer[index].codes == null) indexer_failed = true;
            }
        }
        // Match upstream's all-layer fallback: a checkpoint whose indexers do
        // not all narrow exactly keeps the original F16 reader throughout.
        if (indexer_failed) for (&self.indexer) |*pair| pair.free();
        for (self.router) |tensor| self.bytes += c.ds4_gpu_tensor_bytes(tensor);
        for (self.engram) |pair| self.bytes += c.ds4_gpu_tensor_bytes(pair.codes) + c.ds4_gpu_tensor_bytes(pair.scales);
        for (self.indexer) |pair| self.bytes += c.ds4_gpu_tensor_bytes(pair.codes) + c.ds4_gpu_tensor_bytes(pair.scales);
        return self;
    }

    pub fn destroy(self: *Copies) void {
        for (self.router) |tensor| c.ds4_gpu_tensor_free(tensor);
        for (&self.engram) |*pair| pair.free();
        for (&self.indexer) |*pair| pair.free();
        std.heap.c_allocator.destroy(self);
    }

    /// Pin the copies as one engine-owned residency set, then compile each
    /// reader on zero input. First-token work must not include this setup.
    pub fn prepare(self: *Copies, weights: *const c.ds4_weights, layers: u32) void {
        var tensors: [196]?*Tensor = undefined;
        var count: u32 = 0;
        var router_weight: [*c]const Weight = null;
        var engram_weight: [*c]const Weight = null;
        var indexer_weight: [*c]const Weight = null;
        var router: ?*Tensor = null;
        var engram: Pair = .{};
        var indexer: Pair = .{};
        for (weights.layer[0..@min(layers, self.router.len)], 0..) |layer, index| {
            if (self.router[index]) |tensor| {
                tensors[count] = tensor;
                count += 1;
                if (router == null) {
                    router = tensor;
                    router_weight = layer.ffn_gate_inp;
                }
            }
            if (self.indexer[index].codes != null) {
                tensors[count] = self.indexer[index].codes;
                tensors[count + 1] = self.indexer[index].scales;
                count += 2;
                if (indexer.codes == null) {
                    indexer = self.indexer[index];
                    indexer_weight = layer.indexer_attn_q_b;
                }
            }
            if (index == 1 or index == 14) {
                const pair = self.engram[if (index == 1) 0 else 1];
                if (pair.codes != null) {
                    tensors[count] = pair.codes;
                    tensors[count + 1] = pair.scales;
                    count += 2;
                    if (engram.codes == null) {
                        engram = pair;
                        engram_weight = layer.engram_kv;
                    }
                }
            }
        }
        if (count == 0) return;
        _ = c.ds4_gpu_weight_residency(&tensors, count);
        if (c.ds4_gpu_commands_active() != 0) return;
        var input_width: u64 = 0;
        var output_width: u64 = 0;
        for ([_][*c]const Weight{ router_weight, engram_weight, indexer_weight }) |weight| {
            if (weight != null) {
                input_width = @max(input_width, weight.*.dim[0]);
                output_width = @max(output_width, weight.*.dim[1]);
            }
        }
        const input = c.ds4_gpu_tensor_alloc(input_width * 4);
        const output = c.ds4_gpu_tensor_alloc(output_width * 4);
        defer {
            _ = c.ds4_gpu_synchronize();
            c.ds4_gpu_tensor_free(input);
            c.ds4_gpu_tensor_free(output);
        }
        if (input == null or output == null or c.ds4_gpu_tensor_fill_f32(input, 0, input_width) == 0) return;
        if (router_weight != null) _ = c.ds4_gpu_matmul_bf16_weights_tensor(output, router, router_weight.*.dim[0], router_weight.*.dim[1], input);
        if (engram_weight != null) _ = project(engram, output, input, engram_weight);
        if (indexer_weight != null) _ = project(indexer, output, input, indexer_weight);
    }
};

fn buildBf16(model: *const c.ds4_model, weight: [*c]const Weight) ?*Tensor {
    if (weight == null or weight.*.type != c.DS4_TENSOR_F32) return null;
    const count = weight.*.dim[0] * weight.*.dim[1];
    const source: [*]const u32 = @ptrCast(@alignCast(model.map + weight.*.abs_offset));
    for (source[0..@intCast(count)]) |bits| if (bits & 0xffff != 0) return null;
    const tensor = c.ds4_gpu_tensor_alloc_weights(count * 2) orelse return null;
    const contents = c.ds4_gpu_tensor_contents(tensor) orelse {
        c.ds4_gpu_tensor_free(tensor);
        return null;
    };
    const output: [*]u16 = @ptrCast(@alignCast(contents));
    for (source[0..@intCast(count)], 0..) |bits, index| output[index] = @truncate(bits >> 16);
    return tensor;
}

/// An E4M3 code only when the value is representable, including signed zero.
/// FP8 maxima/subnormals are checked before integer conversion.
fn e4m3Code(value: f32) ?u8 {
    const sign: u8 = @truncate((@as(u32, @bitCast(value)) >> 24) & 0x80);
    const magnitude = @abs(value);
    if (magnitude == 0) return sign;
    if (!(magnitude <= 448)) return null;
    var exponent: c_int = 0;
    const fraction = c.frexpf(magnitude, &exponent);
    if (exponent - 1 >= -6) {
        const mantissa = fraction * 16 - 8;
        if (mantissa != @floor(mantissa)) return null;
        return sign | (@as(u8, @intCast(exponent + 6)) << 3) | @as(u8, @intFromFloat(mantissa));
    }
    const mantissa = magnitude * 512;
    if (mantissa != @floor(mantissa) or mantissa < 1 or mantissa > 7) return null;
    return sign | @as(u8, @intFromFloat(mantissa));
}

const Job = struct {
    source: [*]const f16,
    columns: usize,
    codes: [*]u8,
    scales: [*]f32,
    failed: std.atomic.Value(bool) = .init(false),

    fn blockRow(raw: ?*anyopaque, block_row: usize) callconv(.c) void {
        const self: *Job = @ptrCast(@alignCast(raw.?));
        for (0..self.columns / 32) |block_column| {
            if (self.failed.load(.monotonic)) return;
            var maximum: f32 = 0;
            for (0..32) |row| for (0..32) |column| {
                maximum = @max(maximum, @abs(@as(f32, self.source[(block_row * 32 + row) * self.columns + block_column * 32 + column])));
            };
            if (!std.math.isFinite(maximum)) {
                self.failed.store(true, .monotonic);
                return;
            }
            const base: c_int = if (maximum > 0) @intFromFloat(@ceil(@log2(maximum / 448))) else 0;
            var fit = false;
            for ([_]c_int{ 0, 1, -1 }) |delta| {
                const exponent = base + delta;
                const inverse = c.ldexpf(1, -exponent);
                fit = true;
                block: for (0..32) |row| for (0..32) |column| {
                    const index = (block_row * 32 + row) * self.columns + block_column * 32 + column;
                    self.codes[index] = e4m3Code(@as(f32, self.source[index]) * inverse) orelse {
                        fit = false;
                        break :block;
                    };
                };
                if (fit) {
                    self.scales[block_row * (self.columns / 32) + block_column] = c.ldexpf(1, exponent);
                    break;
                }
            }
            if (!fit) {
                self.failed.store(true, .monotonic);
                return;
            }
        }
    }
};

fn buildE4m3(model: *const c.ds4_model, weight: [*c]const Weight) Pair {
    if (weight == null or weight.*.type != c.DS4_TENSOR_F16 or weight.*.dim[0] % 32 != 0 or weight.*.dim[1] % 32 != 0) return .{};
    const columns: usize = @intCast(weight.*.dim[0]);
    const rows: usize = @intCast(weight.*.dim[1]);
    var result = Pair{
        .codes = c.ds4_gpu_tensor_alloc_weights(rows * columns),
        .scales = c.ds4_gpu_tensor_alloc_weights((rows / 32) * (columns / 32) * 4),
    };
    const codes = c.ds4_gpu_tensor_contents(result.codes);
    const scales = c.ds4_gpu_tensor_contents(result.scales);
    if (codes == null or scales == null) {
        result.free();
        return .{};
    }
    var job = Job{
        .source = @ptrCast(@alignCast(model.map + weight.*.abs_offset)),
        .columns = columns,
        .codes = @ptrCast(codes.?),
        .scales = @ptrCast(@alignCast(scales.?)),
    };
    c.dispatch_apply_f(rows / 32, c.dispatch_get_global_queue(c.QOS_CLASS_USER_INITIATED, 0), &job, Job.blockRow);
    if (job.failed.load(.monotonic)) result.free();
    return result;
}

pub fn project(pair: Pair, output: ?*Tensor, input: ?*const Tensor, weight: [*c]const Weight) bool {
    return pair.codes != null and c.ds4_gpu_matmul_e4m3_bf16_tensor(output, pair.codes, pair.scales, weight.*.dim[0], weight.*.dim[1], input) != 0;
}
