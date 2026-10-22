//! Exact router primitive packet and launch preparation. Actual records are
//! imported once; the ARC leaf only binds the selected packet and named views.
const std = @import("std");
pub const c = @import("metal_c");
pub const View = c.ds4_metal_attention_view;
pub const Plan = c.ds4_metal_router_launch;
pub const Pso = c.ds4_metal_router_pipeline;
pub const Buffer = @import("attention_decode.zig").Buffer;
pub fn bits(value: u32) i32 {
    return @bitCast(value);
}
fn unsigned(value: i32) u64 {
    return @bitCast(@as(i64, value));
}
pub fn pso(kind: u32, hot: bool) Pso {
    return c.ds4_metal_router_pipeline_get(kind, hot);
}
pub fn plan(kind: u32, pipeline: Pso) Plan {
    var result = std.mem.zeroes(Plan);
    result.kind = kind;
    result.pipeline = pipeline;
    result.groups_x = 1;
    result.groups_y = 1;
    result.threads = 1;
    return result;
}
pub const Ops = struct {
    command: ?*anyopaque,
    pub fn encode(self: Ops, launch: *const Plan) bool {
        return self.command != null and launch.pipeline.handle != null and
            c.ds4_metal_router_encode(self.command, launch) != 0;
    }
    pub fn unary(self: Ops, pipeline: Pso, src: View, dst: View, width: u32, rows: u32, four: bool, low: f32, high: f32) bool {
        if (src.buffer == null or dst.buffer == null or width == 0 or rows == 0 or (four and width & 3 != 0)) return false;
        const count = if (four) width / 4 else width;
        const row = @as(u64, width) * 4;
        var p = plan(c.DS4_ROUTER_UNARY, pipeline);
        p.args.unary = .{
            .ne00 = bits(count),
            .ne01 = bits(rows),
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = 4,
            .nb01 = row,
            .nb02 = row,
            .nb03 = row,
            .ne0 = bits(count),
            .ne1 = bits(rows),
            .ne2 = 1,
            .ne3 = 1,
            .nb0 = 4,
            .nb1 = row,
            .nb2 = row,
            .nb3 = row,
            .slope = 0,
            .scale = 0,
            .bias = 0,
            .val = 0,
            .min = low,
            .max = high,
        };
        p.threads = @max(1, @min(unsigned(p.args.unary.ne00), @min(pipeline.threads, 256)));
        p.groups_x = ((unsigned(p.args.unary.ne00) +% p.threads -% 1) / p.threads) *% unsigned(p.args.unary.ne01);
        p.source = src;
        p.target = dst;
        return self.encode(&p);
    }
    pub fn binary(self: Ops, pipeline: Pso, args: c.ds4_gpu_bin_args, a: View, b: View, dst: View, scalar: ?f32) bool {
        if (a.buffer == null or dst.buffer == null or (scalar == null and b.buffer == null) or args.ne0 <= 0 or args.ne1 <= 0) return false;
        var p = plan(if (scalar != null) c.DS4_ROUTER_SCALE else c.DS4_ROUTER_BINARY, pipeline);
        p.args.binary = args;
        p.source = a;
        p.other = b;
        p.target = dst;
        if (scalar) |value| p.scale = value;
        // Preserve the old power-of-two loop, including its strict '< width'.
        const cap = @min(pipeline.threads, 256);
        var threads: u64 = 1;
        while (2 * threads < @as(u64, @intCast(args.ne0)) and threads < cap) threads *= 2;
        p.threads = threads;
        p.groups_x = unsigned(args.ne1);
        p.groups_y = unsigned(args.ne2);
        return self.encode(&p);
    }
    pub fn hashRows(self: Ops, table: View, tokens: View, token: ?u32, dst: View, hash_rows: u32, columns: u32, rows: u32) bool {
        if (table.buffer == null or dst.buffer == null or (tokens.buffer == null and token == null) or hash_rows == 0 or columns == 0 or rows == 0) return false;
        var p = plan(c.DS4_ROUTER_GATHER, pso(c.DS4_ROUTER_GATHER_I32, false));
        const row = @as(u64, columns) * 4;
        const token_bytes = @as(u64, rows) * 4;
        p.args.gather = .{
            .ne00t = bits(columns),
            .ne00 = bits(columns),
            .nb01 = row,
            .nb02 = @as(u64, hash_rows) *% row,
            .nb03 = @as(u64, hash_rows) *% row,
            .ne10 = bits(rows),
            .nb10 = 4,
            .nb11 = token_bytes,
            .nb12 = token_bytes,
            .nb1 = row,
            .nb2 = @as(u64, rows) *% row,
            .nb3 = @as(u64, rows) *% row,
        };
        p.source = table;
        p.other = tokens;
        p.target = dst;
        p.inline_token = token != null;
        p.token = bits(token orelse 0);
        p.threads = @max(1, @min(columns, p.pipeline.threads));
        p.groups_x = ((@as(u64, columns) + p.threads - 1) / p.threads) * rows;
        return self.encode(&p);
    }
    pub fn gatherWeights(self: Ops, probs: View, selected: View, dst: View, experts: u32, used: u32, rows: u32) bool {
        if (probs.buffer == null or selected.buffer == null or dst.buffer == null or experts == 0 or used == 0 or rows == 0) return false;
        var p = plan(c.DS4_ROUTER_GATHER, pso(c.DS4_ROUTER_GATHER_F32, false));
        const prob_row = @as(u64, experts) * 4;
        const selected_row = @as(u64, used) * 4;
        p.args.gather = .{
            .ne00t = 1,
            .ne00 = 1,
            .nb01 = 4,
            .nb02 = prob_row,
            .nb03 = @as(u64, rows) *% prob_row,
            .ne10 = bits(used),
            .nb10 = 4,
            .nb11 = selected_row,
            .nb12 = @as(u64, rows) *% selected_row,
            .nb1 = 4,
            .nb2 = selected_row,
            .nb3 = @as(u64, rows) *% selected_row,
        };
        p.source = probs;
        p.other = selected;
        p.target = dst;
        p.groups_x = used;
        p.groups_y = rows;
        return self.encode(&p);
    }
    pub fn sum(self: Ops, src: View, dst: View, width: u32, rows: u32) bool {
        if (src.buffer == null or dst.buffer == null or width == 0 or rows == 0) return false;
        var p = plan(c.DS4_ROUTER_SUM, pso(c.DS4_ROUTER_SUM_PSO, false));
        const row = @as(u64, width) * 4;
        const bytes = @as(u64, rows) *% row;
        p.args.sum = .{
            .ne00 = width,
            .ne01 = rows,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = 4,
            .nb01 = row,
            .nb02 = bytes,
            .nb03 = bytes,
            .ne0 = 1,
            .ne1 = rows,
            .ne2 = 1,
            .ne3 = 1,
            .nb0 = 4,
            .nb1 = 4,
            .nb2 = @as(u64, rows) * 4,
            .nb3 = @as(u64, rows) * 4,
        };
        var threads: u64 = 32;
        while (threads < width and threads < p.pipeline.threads) threads *= 2;
        p.threads = @max(1, @min(@min(threads, p.pipeline.threads), width));
        p.groups_x = rows;
        p.shared = 128;
        p.source = src;
        p.target = dst;
        return self.encode(&p);
    }
};
pub fn binaryArgs(width: u32, rows: u32, rhs_width: u32, rowwise: bool) c.ds4_gpu_bin_args {
    const left_row = @as(u64, width) * 4;
    const right_row = @as(u64, rhs_width) * 4;
    return .{
        .ne00 = bits(width),
        .ne01 = bits(rows),
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = 4,
        .nb01 = left_row,
        .nb02 = if (rowwise) @as(u64, rows) *% left_row else left_row,
        .nb03 = if (rowwise) @as(u64, rows) *% left_row else left_row,
        .ne10 = bits(rhs_width),
        .ne11 = if (rowwise) bits(rows) else 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = 4,
        .nb11 = right_row,
        .nb12 = if (rowwise) @as(u64, rows) *% right_row else right_row,
        .nb13 = if (rowwise) @as(u64, rows) *% right_row else right_row,
        .ne0 = bits(width),
        .ne1 = bits(rows),
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = 4,
        .nb1 = left_row,
        .nb2 = if (rowwise) @as(u64, rows) *% left_row else left_row,
        .nb3 = if (rowwise) @as(u64, rows) *% left_row else left_row,
        .offs = 0,
        .o1 = @splat(0),
    };
}
