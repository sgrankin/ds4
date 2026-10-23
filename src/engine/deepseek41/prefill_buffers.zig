//! Borrowed prefill row views and carry storage. Every returned view is its
//! own retained GPU handle; backing allocations remain owned by the graph.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const layout = @import("layout.zig");
const Graph = c.ds41_gpu_graph;
const Row = c.ds41_prefill_row;

pub const Views = struct {
    row: Row = std.mem.zeroes(Row),

    pub fn init(config: *const Config, graph: *const Graph, count: u32) ?Views {
        var result = Views{};
        var complete = false;
        defer if (!complete) result.deinit();
        const plan = layout.Layout.fromRecord(config, graph);
        inline for (comptime std.meta.fieldNames(Row)) |name| {
            const width = plan.rowWidth(@field(layout.RowField, name));
            @field(result.row, name) = c.ds4_gpu_tensor_view(@field(graph.batch, name), 0, @as(u64, count) * width * 4) orelse return null;
        }
        complete = true;
        return result;
    }

    pub fn deinit(self: *Views) void {
        inline for (comptime std.meta.fieldNames(Row)) |name| c.ds4_gpu_tensor_free(@field(self.row, name));
        self.row = std.mem.zeroes(Row);
    }
};

pub fn selectRow(graph: *Graph, row: *const Row) void {
    // This is a shallow execution binding. Do not free its graph-shaped copy.
    inline for (comptime std.meta.fieldNames(Row)) |name| @field(graph, name) = @field(row, name);
}

pub fn carryCopy(config: *const Config, graph: *Graph, offset: u32, count: u32, store: bool) bool {
    if (offset > graph.carry_cap or count > graph.carry_cap - offset) return false;
    const plan = layout.Layout.fromRecord(config, graph);
    inline for (layout.carry_fields) |field| {
        const name = @tagName(field);
        const width = plan.rowWidth(field);
        const format: u32 = switch (field) {
            .residual => c.DS4_V41_CARRY_BF16,
            .block_mask => c.DS4_V41_CARRY_MASK,
            else => c.DS4_V41_CARRY_F32,
        };
        if (graph.compact_carry and format != c.DS4_V41_CARRY_F32) {
            if (c.ds4_gpu_dsv41_carry_copy(@field(graph.carry, name), offset, @field(graph.batch, name), @intCast(width), count, format, store) == 0) return false;
        } else {
            const bytes = width * 4;
            const ok = if (store)
                c.ds4_gpu_tensor_copy(@field(graph.carry, name), @as(u64, offset) * bytes, @field(graph.batch, name), 0, @as(u64, count) * bytes)
            else
                c.ds4_gpu_tensor_copy(@field(graph.batch, name), 0, @field(graph.carry, name), @as(u64, offset) * bytes, @as(u64, count) * bytes);
            if (ok == 0) return false;
        }
    }
    return true;
}
