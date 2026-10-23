//! Compressor scratch layout and exact reduction graph. All shader arithmetic
//! remains in the original MSL; host code only chooses packets and schedules.
const std = @import("std");
pub const c = @import("metal_c");
pub const api = @import("bridge.zig");
pub const Buffer = @import("attention_decode.zig").Buffer;
pub const Tensor = api.Tensor;
pub const View = c.ds4_metal_attention_view;
const Pipeline = c.ds4_metal_attention_pipeline;
const Launch = c.ds4_metal_compressor_launch;
const Args = c.ds4_metal_compressor_args;
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
pub const infinity: f32 = @bitCast(@as(u32, 0xff800000));
pub fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn errorLine(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
pub fn narrow(value: u64) i32 {
    return @bitCast(@as(u32, @truncate(value)));
}
pub fn at(view: View, offset: u64) View {
    var result = view;
    result.offset +%= offset;
    return result;
}
pub fn model(map: ?*const anyopaque, size: u64, offset: u64, bytes: u64) ?Buffer {
    var inner: u64 = 0;
    const pointer = api.retainModelRange(map, size, offset, bytes, &inner) orelse return null;
    return .{ .view = .{ .buffer = pointer, .offset = inner, .bytes = bytes } };
}
pub fn overlap(a: View, a_bytes: u64, b: View, b_bytes: u64) bool {
    if (a.buffer != b.buffer or a_bytes == 0 or b_bytes == 0) return false;
    return if (a.offset <= b.offset) a_bytes > b.offset - a.offset else b_bytes > a.offset - b.offset;
}
pub const Command = struct {
    token: ?*anyopaque = null,
    owned: c_int = 0,
    pub fn acquire(self: *Command) bool {
        self.drop();
        self.token = api.acquire(&self.owned);
        return self.token != null;
    }
    pub fn drop(self: *Command) void {
        if (self.token) |token| _ = api.release(token, self.owned, 0, "compressor");
        self.token = null;
    }
    pub fn finish(self: *Command, label: [*:0]const u8) bool {
        const token = self.token orelse return false;
        self.token = null;
        return api.release(token, self.owned, 1, label) != 0;
    }
};
pub const Ops = struct {
    command: *Command,
    scratch: [7]Buffer = @splat(.{ .view = std.mem.zeroes(View) }),
    pub fn deinit(self: *Ops) void {
        for (self.scratch) |buffer| buffer.deinit();
    }
    pub fn reserve(self: *Ops, kind: u32, bytes: u64) bool {
        if (self.scratch[kind].valid() and self.scratch[kind].view.bytes >= bytes) return true;
        const next = Buffer{ .view = c.ds4_metal_compressor_scratch(kind, bytes) };
        if (!next.valid()) return false;
        self.scratch[kind].deinit();
        self.scratch[kind] = next;
        return true;
    }
    pub fn view(self: *const Ops, kind: u32) View {
        return self.scratch[kind].view;
    }
    fn plan(kind: u32, packet: Args, pso: Pipeline) Launch {
        var result = std.mem.zeroes(Launch);
        result.kind = kind;
        result.args = packet;
        result.pipeline = pso;
        result.gx = 1;
        result.gy = 1;
        result.gz = 1;
        result.tx = 256;
        result.ty = 1;
        return result;
    }
    pub fn encode(self: *const Ops, dispatch: *const Launch) bool {
        return self.command.token != null and c.ds4_metal_compressor_encode(self.command.token.?, dispatch) != 0;
    }
    fn pipeline(kind: u32, half: bool) Pipeline {
        return c.ds4_metal_compressor_pipeline(kind, half);
    }
    fn copyThreads(width: u32, maximum: u64) u64 {
        var threads: u64 = 32;
        while (threads < width and threads < maximum) threads *= 2;
        return @max(1, @min(threads, @min(maximum, width)));
    }
    pub fn copy(self: *Ops, src: View, dst: View, cols: u32, rows: u32, planes: u32, src_col: u64, src_row: u64, src_plane: u64, dst_row: u64, dst_plane: u64) bool {
        if (src.buffer == null or dst.buffer == null or cols == 0 or rows == 0 or planes == 0) return false;
        const pso = pipeline(c.DS4_COMP_COPY, false);
        var packet = std.mem.zeroes(c.ds4_metal_copy_args);
        packet.nk0 = cols;
        packet.ne00 = cols;
        packet.ne01 = rows;
        packet.ne02 = planes;
        packet.ne03 = 1;
        packet.nb00 = src_col;
        packet.nb01 = src_row;
        packet.nb02 = src_plane;
        packet.nb03 = @as(u64, planes) *% src_plane;
        packet.ne0 = cols;
        packet.ne1 = rows;
        packet.ne2 = planes;
        packet.ne3 = 1;
        packet.nb0 = 4;
        packet.nb1 = dst_row;
        packet.nb2 = dst_plane;
        packet.nb3 = @as(u64, planes) *% dst_plane;
        var launch = plan(c.DS4_COMP_COPY, .{ .copy = packet }, pso);
        launch.kv = src;
        launch.out = dst;
        launch.tx = copyThreads(cols, pso.threads);
        launch.gx = ((cols + launch.tx - 1) / launch.tx) * rows;
        launch.gy = planes;
        return self.encode(&launch);
    }
    pub fn fill(self: *Ops, dst: View, width: u32, rows: u32, value: f32) bool {
        if (dst.buffer == null or width == 0 or rows == 0 or width & 3 != 0) return false;
        const pso = pipeline(c.DS4_COMP_FILL, false);
        const row_bytes = @as(u64, width) * 4;
        var packet = std.mem.zeroes(c.ds4_metal_attention_unary_args);
        packet.ne00 = narrow(width / 4);
        packet.ne01 = narrow(rows);
        packet.ne02 = 1;
        packet.ne03 = 1;
        packet.nb00 = 4;
        packet.nb01 = row_bytes;
        packet.nb02 = row_bytes;
        packet.nb03 = row_bytes;
        packet.ne0 = packet.ne00;
        packet.ne1 = packet.ne01;
        packet.ne2 = 1;
        packet.ne3 = 1;
        packet.nb0 = 4;
        packet.nb1 = row_bytes;
        packet.nb2 = row_bytes;
        packet.nb3 = row_bytes;
        packet.val = value;
        var launch = plan(c.DS4_COMP_FILL, .{ .fill = packet }, pso);
        launch.out = dst;
        launch.tx = @max(1, @min(width / 4, @min(pso.threads, 256)));
        launch.gx = ((width / 4 + launch.tx - 1) / launch.tx) * rows;
        return self.encode(&launch);
    }
    pub fn add(self: *Ops, a: View, b: View, dst: View, count: u32) bool {
        if (a.buffer == null or b.buffer == null or dst.buffer == null or count == 0) return false;
        const pso = pipeline(c.DS4_COMP_ADD, false);
        var launch = plan(c.DS4_COMP_ADD, .{ .add = .{ .n = count } }, pso);
        launch.kv = a;
        launch.score = b;
        launch.out = dst;
        launch.tx = @max(1, @min(count, @min(256, pso.threads)));
        launch.gx = (count + launch.tx - 1) / launch.tx;
        return self.encode(&launch);
    }
    pub fn setRows(self: *Ops, src: View, dst: View, indices: []const i32, dst_rows: u32, width: u32) bool {
        const count: u32 = @intCast(indices.len);
        const row_bytes = @as(u64, width) * 4;
        if (src.buffer == null or dst.buffer == null or count == 0 or width == 0 or dst.bytes < @as(u64, dst_rows) *% row_bytes or @as(u64, count) *% row_bytes > max64 - src.offset) return errorLine("ds4: Metal DS4 set_rows received invalid buffers");
        const pso = pipeline(c.DS4_COMP_SET_ROWS, false);
        var packet = std.mem.zeroes(c.ds4_gpu_set_rows_args);
        packet.nk0 = narrow(width);
        packet.ne01 = narrow(count);
        packet.nb01 = row_bytes;
        packet.nb02 = @as(u64, count) *% row_bytes;
        packet.nb03 = packet.nb02;
        packet.ne11 = 1;
        packet.ne12 = 1;
        packet.nb10 = 4;
        packet.nb11 = @as(u64, count) * 4;
        packet.nb12 = packet.nb11;
        packet.nb1 = row_bytes;
        packet.nb2 = @as(u64, dst_rows) *% row_bytes;
        packet.nb3 = packet.nb2;
        var launch = plan(c.DS4_COMP_SET_ROWS, .{ .set_rows = packet }, pso);
        const max_threads = if (pso.handle != null) pso.threads else 1024;
        var threads: u64 = 32;
        while (threads < width and threads < max_threads) threads *= 2;
        var per_group: u64 = 1;
        if (threads > width) {
            per_group = (threads + width - 1) / width;
            threads = width;
            if (per_group * threads > max_threads) per_group -= 1;
        }
        threads = @max(1, @min(threads, width));
        per_group = @max(1, per_group);
        var owned_indices = Buffer{ .view = std.mem.zeroes(View) };
        defer owned_indices.deinit();
        if (indices.len * 4 > 4096) {
            owned_indices.view = c.ds4_metal_output_scratch(c.DS4_OUT_IDS_SCRATCH, indices.len * 4, true, "ds4_set_rows_indices");
            if (!owned_indices.valid()) return false;
            const data = c.ds4_metal_attention_contents(owned_indices.view) orelse return false;
            @memcpy(@as([*]i32, @ptrCast(@alignCast(data)))[0..indices.len], indices);
            launch.indices = owned_indices.view;
        }
        launch.kv = src;
        launch.out = dst;
        launch.rows = indices.ptr;
        launch.row_count = count;
        launch.tx = threads;
        launch.ty = per_group;
        launch.gx = (count + per_group - 1) / per_group;
        return self.encode(&launch);
    }
    pub fn tileApe(self: *Ops, ape: View, dst: View, half: bool, width: u32, ratio: u32, pos: u32, count: u32) bool {
        const total = @as(u64, count) * width;
        if (ape.buffer == null or dst.buffer == null or width == 0 or ratio == 0 or count == 0 or total > max32) return false;
        const pso = pipeline(c.DS4_COMP_TILE_APE, half);
        var launch = plan(c.DS4_COMP_TILE_APE, .{ .tile = .{ .n_total = @intCast(total), .width = width, .ratio = ratio, .pos0 = pos } }, pso);
        launch.ape = ape;
        launch.out = dst;
        launch.gx = (total + 255) / 256;
        return self.encode(&launch);
    }
    pub fn addApe(self: *Ops, score: View, dst: View, ape: View, ape_type: u32, width: u32, ratio: u32, pos: u32, count: u32) bool {
        if (score.buffer == null or dst.buffer == null or ape.buffer == null or width == 0 or ratio == 0 or count == 0 or ape_type > 1) return false;
        const total = @as(u64, count) * width;
        if (total > max32) return errorLine("ds4: Metal compressor APE add received too many elements");
        const facts = c.ds4_metal_attention_get_facts().flags;
        if (facts & c.DS4_ATTN_M3 != 0 and !present("DS4_METAL_DISABLE_COMPRESSOR_APE_ADD")) {
            const pso = pipeline(c.DS4_COMP_SCORE_APE, ape_type == 1);
            if (pso.handle != null) {
                var launch = plan(c.DS4_COMP_SCORE_APE, .{ .score_ape = .{ .width = width, .ratio = ratio, .pos0 = pos, .n_tokens = count } }, pso);
                launch.score = score;
                launch.ape = ape;
                launch.out = dst;
                launch.tx = @min(256, pso.threads);
                if (launch.tx == 0) return false;
                launch.gx = (total + launch.tx - 1) / launch.tx;
                return self.encode(&launch);
            }
        }
        return self.reserve(c.DS4_COMP_APE, total * 4) and self.tileApe(ape, self.view(c.DS4_COMP_APE), ape_type == 1, width, ratio, pos, count) and self.add(score, self.view(c.DS4_COMP_APE), dst, @intCast(total));
    }
    pub fn projected(self: *Ops, kv: View, score: View, state_kv: View, state_score: View, ape: View, ape_type: u32, width: u32, ratio: u32, pos: u32, indices: []const i32, state_rows: u32) bool {
        if (indices.len == 0 or state_rows == 0 or width == 0) return false;
        const bytes = @as(u64, indices.len) *% width *% 4;
        return self.reserve(c.DS4_COMP_ADDED_SCORE, bytes) and
            self.addApe(score, self.view(c.DS4_COMP_ADDED_SCORE), ape, ape_type, width, ratio, pos, @intCast(indices.len)) and
            self.setRows(kv, state_kv, indices, state_rows, width) and self.setRows(self.view(c.DS4_COMP_ADDED_SCORE), state_score, indices, state_rows, width);
    }
    pub fn ggmlScratch(self: *Ops, rows_count: u32, width: u32) bool {
        const total = @as(u64, rows_count) * width;
        if (rows_count == 0 or width == 0 or total > max64 / 4) return false;
        return self.reserve(c.DS4_COMP_PRODUCT, total * 4) and self.reserve(c.DS4_COMP_SCORE_CONT, total * 4) and self.reserve(c.DS4_COMP_SCRATCH_SOFTMAX, total * 4);
    }
    fn softmax(self: *Ops, src: View, dst: View, width: u32, rows_count: u32, planes: u32) bool {
        if (src.buffer == null or dst.buffer == null or width == 0 or rows_count == 0 or planes == 0) return false;
        const pso = pipeline(c.DS4_COMP_SOFTMAX, width % 4 == 0);
        const row_bytes = @as(u64, width) * 4;
        const plane_bytes = @as(u64, rows_count) *% row_bytes;
        var packet = std.mem.zeroes(c.ds4_gpu_softmax_args);
        packet.ne00 = narrow(width);
        packet.ne01 = narrow(rows_count);
        packet.ne02 = narrow(planes);
        packet.nb01 = row_bytes;
        packet.nb02 = plane_bytes;
        packet.nb03 = @as(u64, planes) *% plane_bytes;
        packet.ne11 = narrow(width);
        packet.ne12 = narrow(rows_count);
        packet.ne13 = narrow(planes);
        packet.nb11 = row_bytes;
        packet.nb12 = plane_bytes;
        packet.nb13 = packet.nb03;
        packet.nb1 = row_bytes;
        packet.nb2 = plane_bytes;
        packet.nb3 = packet.nb03;
        packet.scale = 1;
        packet.n_head_log2 = 1;
        var launch = plan(c.DS4_COMP_SOFTMAX, .{ .softmax = packet }, pso);
        var threads: u64 = 32;
        const work = if (width % 4 == 0) width / 4 else width;
        while (threads < work and threads * rows_count * planes < 256) threads *= 2;
        launch.kv = src;
        launch.out = dst;
        launch.tx = @max(1, @min(threads, pso.threads));
        launch.gx = rows_count;
        launch.gy = planes;
        launch.shared = 128;
        return self.encode(&launch);
    }
    fn multiply(self: *Ops, width: u32, rows_count: u32) bool {
        const pso = pipeline(c.DS4_COMP_MUL, false);
        const row_bytes = @as(u64, width) * 4;
        const plane_bytes = @as(u64, rows_count) *% row_bytes;
        var packet = std.mem.zeroes(c.ds4_gpu_bin_args);
        packet.ne00 = narrow(width);
        packet.ne01 = narrow(rows_count);
        packet.ne02 = 1;
        packet.ne03 = 1;
        packet.nb00 = 4;
        packet.nb01 = row_bytes;
        packet.nb02 = plane_bytes;
        packet.nb03 = plane_bytes;
        packet.ne10 = packet.ne00;
        packet.ne11 = packet.ne01;
        packet.ne12 = 1;
        packet.ne13 = 1;
        packet.nb10 = 4;
        packet.nb11 = row_bytes;
        packet.nb12 = plane_bytes;
        packet.nb13 = plane_bytes;
        packet.ne0 = packet.ne00;
        packet.ne1 = packet.ne01;
        packet.ne2 = 1;
        packet.ne3 = 1;
        packet.nb0 = 4;
        packet.nb1 = row_bytes;
        packet.nb2 = plane_bytes;
        packet.nb3 = plane_bytes;
        var launch = plan(c.DS4_COMP_MUL, .{ .multiply = packet }, pso);
        var threads: u64 = 1;
        const maximum = @min(256, pso.threads);
        while (2 * threads < width and threads < maximum) threads *= 2;
        launch.kv = self.view(c.DS4_COMP_PRODUCT);
        launch.score = self.view(c.DS4_COMP_SCRATCH_SOFTMAX);
        launch.out = launch.kv;
        launch.tx = threads;
        launch.gx = rows_count;
        return self.encode(&launch);
    }
    fn sum(self: *Ops, dst: View, width: u32, rows_count: u32) bool {
        const pso = pipeline(c.DS4_COMP_SUM, false);
        const stride = @as(u64, width) * 4;
        var packet = std.mem.zeroes(c.ds4_gpu_kargs_sum_rows);
        packet.ne00 = width;
        packet.ne01 = rows_count;
        packet.ne02 = 1;
        packet.ne03 = 1;
        packet.nb00 = 4;
        packet.nb01 = stride;
        packet.nb02 = @as(u64, rows_count) *% stride;
        packet.nb03 = packet.nb02;
        packet.ne0 = 1;
        packet.ne1 = rows_count;
        packet.ne2 = 1;
        packet.ne3 = 1;
        packet.nb0 = 4;
        packet.nb1 = 4;
        packet.nb2 = @as(u64, rows_count) * 4;
        packet.nb3 = packet.nb2;
        var launch = plan(c.DS4_COMP_SUM, .{ .sum = packet }, pso);
        var threads: u64 = 32;
        while (threads < width and threads < pso.threads) threads *= 2;
        launch.kv = self.view(c.DS4_COMP_PRODUCT);
        launch.out = dst;
        launch.tx = @max(1, @min(threads, @min(pso.threads, width)));
        launch.gx = rows_count;
        launch.shared = 128;
        return self.encode(&launch);
    }
    pub fn reduce(self: *Ops, out: View, rows_count: u32, width: u32, decode: bool) bool {
        if (out.buffer == null or out.bytes < @as(u64, width) * 4 or !self.ggmlScratch(rows_count, width)) return false;
        const facts = c.ds4_metal_attention_get_facts().flags;
        const exact = decode and rows_count == 8 and (width == 128 or width == 512) and facts & c.DS4_ATTN_SSD == 0 and facts & (c.DS4_ATTN_PRE_M5 | c.DS4_ATTN_M5) != 0 and !present("DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_REDUCTION_FUSION");
        if (exact) {
            const pso = pipeline(c.DS4_COMP_EXACT_REDUCE, false);
            if (pso.handle != null and pso.simd_width == 32 and pso.threads >= 32) {
                if (!self.poison(rows_count, width)) return false;
                var launch = plan(c.DS4_COMP_EXACT_REDUCE, .{ .pack = .{ .head_dim = width, .n_comp = 1, .replay = 1, .n_threads = 32 } }, pso);
                launch.product = self.view(c.DS4_COMP_PRODUCT);
                launch.score_cont = self.view(c.DS4_COMP_SCORE_CONT);
                launch.softmax = self.view(c.DS4_COMP_SCRATCH_SOFTMAX);
                launch.gx = width;
                launch.tx = 32;
                launch.shared = 128;
                return self.encode(&launch) and self.sum(out, rows_count, width);
            }
        }
        // The exact graph has contiguous [head_dim,n_rows] scratch. Copy/pack
        // may fuse, but softmax → product → sum retain this reduction order.
        return self.softmax(self.view(c.DS4_COMP_SCORE_CONT), self.view(c.DS4_COMP_SCRATCH_SOFTMAX), rows_count, width, 1) and self.multiply(rows_count, width) and self.sum(out, rows_count, width);
    }
    fn poison(self: *Ops, rows_count: u32, width: u32) bool {
        return !present("DS4_METAL_TEST_POISON_COMPRESSOR_EXACT_REDUCTION_SCRATCH") or self.fill(self.view(c.DS4_COMP_SCRATCH_SOFTMAX), rows_count, width, @bitCast(@as(u32, 0x7fc01234)));
    }
    pub fn pool(self: *Ops, out: View, kv: View, score: View, kv_stride: [3]u64, score_stride: [3]u64, rows_count: u32, width: u32, count: u32) bool {
        if (out.buffer == null or kv.buffer == null or score.buffer == null or rows_count == 0 or width == 0 or count == 0 or out.bytes < @as(u64, width) *% count *% 4) return false;
        if (count == 1) {
            if (!self.ggmlScratch(rows_count, width)) return false;
            const row = @as(u64, rows_count) * 4;
            const plane = @as(u64, width) *% row;
            return self.copy(kv, self.view(c.DS4_COMP_PRODUCT), rows_count, width, 1, kv_stride[0], kv_stride[1], kv_stride[2], row, plane) and
                self.copy(score, self.view(c.DS4_COMP_SCORE_CONT), rows_count, width, 1, score_stride[0], score_stride[1], score_stride[2], row, plane) and self.reduce(out, rows_count, width, false);
        }
        const pso = pipeline(c.DS4_COMP_POOL, false);
        const packet = c.ds4_gpu_dsv4_softmax_pool_args{
            .ne00 = rows_count,
            .ne01 = width,
            .ne02 = count,
            .nb00 = kv_stride[0],
            .nb01 = kv_stride[1],
            .nb02 = kv_stride[2],
            .nb10 = score_stride[0],
            .nb11 = score_stride[1],
            .nb12 = score_stride[2],
            .ne0 = width,
            .ne1 = count,
            .nb0 = 4,
            .nb1 = @as(u64, width) * 4,
        };
        var launch = plan(c.DS4_COMP_POOL, .{ .pool = packet }, pso);
        launch.kv = kv;
        launch.score = score;
        launch.out = out;
        launch.gx = (@as(u64, width) * count + 255) / 256;
        return self.encode(&launch);
    }
    fn concat(self: *Ops, a: View, a_rows: u32, a_stride: u64, b: View, b_rows: u32, b_stride: u64, dst: View, width: u32, dst_stride: u64) bool {
        if (a.buffer == null or b.buffer == null or dst.buffer == null or width == 0 or a_rows == 0 or b_rows == 0) return false;
        const count = a_rows +% b_rows;
        var packet = std.mem.zeroes(c.ds4_gpu_concat_args);
        packet.ne00 = narrow(width);
        packet.ne01 = narrow(a_rows);
        packet.ne02 = 1;
        packet.ne03 = 1;
        packet.nb00 = 4;
        packet.nb01 = a_stride;
        packet.nb02 = @as(u64, a_rows) *% a_stride;
        packet.nb03 = packet.nb02;
        packet.ne10 = narrow(width);
        packet.ne11 = narrow(b_rows);
        packet.ne12 = 1;
        packet.ne13 = 1;
        packet.nb10 = 4;
        packet.nb11 = b_stride;
        packet.nb12 = @as(u64, b_rows) *% b_stride;
        packet.nb13 = packet.nb12;
        packet.ne0 = narrow(width);
        packet.ne1 = narrow(count);
        packet.ne2 = 1;
        packet.ne3 = 1;
        packet.nb0 = 4;
        packet.nb1 = dst_stride;
        packet.nb2 = @as(u64, count) *% dst_stride;
        packet.nb3 = packet.nb2;
        packet.dim = 1;
        const pso = pipeline(c.DS4_COMP_CONCAT, false);
        var launch = plan(c.DS4_COMP_CONCAT, .{ .concat = packet }, pso);
        launch.kv = a;
        launch.score = b;
        launch.out = dst;
        launch.gx = count;
        launch.tx = @max(1, @min(width, @min(1024, pso.threads)));
        return self.encode(&launch);
    }
    pub fn packMode(width: u32) bool {
        return (width == 128 or width == 512) and c.ds4_metal_attention_get_facts().flags & c.DS4_ATTN_M3 != 0 and !present("DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION") and c.ds4_metal_compressor_cached_pipelines() & (@as(u32, 1) << c.DS4_COMP_PACK) != 0;
    }
    pub fn directMode(width: u32, count: u32) bool {
        // A single row intentionally keeps the legacy GGML reduction graph.
        return count > 1 and (width == 128 or width == 512) and c.ds4_metal_attention_get_facts().flags & c.DS4_ATTN_M3 != 0 and !present("DS4_METAL_DISABLE_COMPRESSOR_RATIO4_DIRECT_POOL") and c.ds4_metal_compressor_cached_pipelines() & (@as(u32, 1) << c.DS4_COMP_DIRECT_POOL) != 0;
    }
    pub fn packRatio4(self: *Ops, kv: View, score: View, state_kv: View, state_score: View, width: u32, count: u32, replay: bool, out: ?View) bool {
        const total = @as(u64, count) *% 8 *% width;
        if (kv.buffer == null or score.buffer == null or state_kv.buffer == null or state_score.buffer == null or width == 0 or count == 0) return false;
        const direct = out != null;
        if (direct and (count <= 1 or @as(u64, width) * count > max32 or out.?.bytes < @as(u64, width) *% count *% 4)) return false;
        if (!direct and (width > max32 / 2 or count > max32 / 8 or total > max32 or self.view(c.DS4_COMP_KV).buffer == null or self.view(c.DS4_COMP_SCORE).buffer == null)) return false;
        const kind: u32 = if (direct) c.DS4_COMP_DIRECT_POOL else c.DS4_COMP_PACK;
        const pso = pipeline(kind, false);
        if (pso.handle == null) return false;
        const threads = @min(if (direct) 256 else @min(width, 256), pso.threads);
        if (threads == 0) return false;
        var launch = if (direct) plan(kind, .{ .direct_pool = .{ .n_rows = 8, .head_dim = width, .n_comp = count, .replay = @intFromBool(replay), .pad = 0 } }, pso) else plan(kind, .{ .pack = .{ .head_dim = width, .n_comp = count, .replay = @intFromBool(replay), .n_threads = @intCast(threads) } }, pso);
        launch.kv = kv;
        launch.score = score;
        launch.state_kv = state_kv;
        launch.state_score = state_score;
        launch.product = self.view(c.DS4_COMP_KV);
        launch.score_cont = self.view(c.DS4_COMP_SCORE);
        launch.tx = threads;
        if (out) |target| {
            launch.out = target;
            launch.gx = (@as(u64, width) * count + threads - 1) / threads;
        } else {
            launch.gx = count;
            launch.gy = 8;
        }
        return self.encode(&launch);
    }
    fn decodeMode(width: u32) bool {
        const facts = c.ds4_metal_attention_get_facts().flags;
        return (width == 128 or width == 512) and facts & (c.DS4_ATTN_PRE_M5 | c.DS4_ATTN_M5) != 0 and !present("DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION") and c.ds4_metal_compressor_cached_pipelines() & (@as(u32, 1) << c.DS4_COMP_DECODE_PACK) != 0;
    }
    fn exactPoolEnabled() bool {
        const facts = c.ds4_metal_attention_get_facts().flags;
        if (present("DS4_METAL_DISABLE_M5_COMPRESSOR_EXACT_POOL_RATIO4")) return false;
        if (facts & c.DS4_ATTN_PRE_M5 != 0 and (present("DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") or present("DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_POOL_RATIO4"))) return false;
        return facts & (c.DS4_ATTN_PRE_M5 | c.DS4_ATTN_M5) != 0;
    }
    fn decodePack(self: *Ops, out: View, state_kv: View, state_score: View, width: u32) bool {
        if (out.buffer == null or state_kv.buffer == null or state_score.buffer == null or width == 0 or !self.ggmlScratch(8, width)) return false;
        if ((width == 128 or width == 512) and exactPoolEnabled()) {
            const pso = pipeline(c.DS4_COMP_EXACT_POOL, false);
            const state_bytes = @as(u64, width) * 16 * 4;
            if (pso.handle != null and !overlap(out, @as(u64, width) * 4, state_kv, state_bytes) and !overlap(out, @as(u64, width) * 4, state_score, state_bytes) and pso.simd_width == 32 and pso.threads >= 32) {
                if (!self.poison(8, width)) return false;
                var launch = plan(c.DS4_COMP_EXACT_POOL, .{ .pack = .{ .head_dim = width, .n_comp = 1, .replay = 1, .n_threads = 32 } }, pso);
                launch.state_kv = state_kv;
                launch.state_score = state_score;
                launch.softmax = self.view(c.DS4_COMP_SCRATCH_SOFTMAX);
                launch.product = self.view(c.DS4_COMP_PRODUCT);
                launch.out = out;
                launch.gx = width;
                launch.tx = 32;
                launch.shared = 128;
                return self.encode(&launch);
            }
        }
        const pso = pipeline(c.DS4_COMP_DECODE_PACK, false);
        const threads = @min(width, @min(256, pso.threads));
        if (pso.handle == null or threads == 0) return false;
        var launch = plan(c.DS4_COMP_DECODE_PACK, .{ .pack = .{ .head_dim = width, .n_comp = 1, .replay = 1, .n_threads = @intCast(threads) } }, pso);
        launch.state_kv = state_kv;
        launch.state_score = state_score;
        launch.product = self.view(c.DS4_COMP_PRODUCT);
        launch.score_cont = self.view(c.DS4_COMP_SCORE_CONT);
        launch.gx = 8;
        launch.tx = threads;
        return self.encode(&launch) and self.reduce(out, 8, width, true);
    }
    pub fn statePool(self: *Ops, out: View, state_kv: View, state_score: View, width: u32, ratio: u32, decode: bool) bool {
        if (state_kv.buffer == null or state_score.buffer == null or width == 0 or ratio == 0) return false;
        const coff: u32 = if (ratio == 4) 2 else 1;
        const state_width = coff *% width;
        const state_rows = coff *% ratio;
        const bytes = @as(u64, state_width) *% state_rows *% 4;
        if (state_kv.bytes < bytes or state_score.bytes < bytes) return false;
        const state_stride = @as(u64, state_width) * 4;
        if (ratio != 4) return self.pool(out, state_kv, state_score, .{ state_stride, 4, @as(u64, state_rows) *% state_stride }, .{ state_stride, 4, @as(u64, state_rows) *% state_stride }, ratio, width, 1);
        if (decode and decodeMode(width)) return self.decodePack(out, state_kv, state_score, width);
        const packed_bytes = @as(u64, width) * 8 * 4;
        const row_stride = @as(u64, width) * 4;
        if (!self.reserve(c.DS4_COMP_KV, packed_bytes) or !self.reserve(c.DS4_COMP_SCORE, packed_bytes)) return false;
        const current = 4 * state_stride + row_stride;
        return self.concat(state_kv, 4, state_stride, at(state_kv, current), 4, state_stride, self.view(c.DS4_COMP_KV), width, row_stride) and
            self.concat(state_score, 4, state_stride, at(state_score, current), 4, state_stride, self.view(c.DS4_COMP_SCORE), width, row_stride) and
            self.pool(out, self.view(c.DS4_COMP_KV), self.view(c.DS4_COMP_SCORE), .{ row_stride, 4, packed_bytes }, .{ row_stride, 4, packed_bytes }, 8, width, 1);
    }
    pub fn shift(self: *Ops, state_kv: View, state_score: View, width: u32) bool {
        const pso = pipeline(c.DS4_COMP_SHIFT, false);
        if (state_kv.buffer == null or state_score.buffer == null or pso.handle == null or width == 0) return false;
        var launch = plan(c.DS4_COMP_SHIFT, .{ .shift = .{ .width = width } }, pso);
        launch.state_kv = state_kv;
        launch.state_score = state_score;
        launch.gx = (@as(u64, 4 *% width) + 255) / 256;
        return self.encode(&launch);
    }
};
