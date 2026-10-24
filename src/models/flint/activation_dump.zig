//! Opt-in direction-building capture. A prefill residual dispatch writes one
//! final-token residual row per layer; CPU export happens after the full prompt.
//! Nothing is allocated, dispatched or read back when the prefix is absent.
const std = @import("std");
const c = @import("flint_c");
const decode = @import("decode.zig");
const prefill = @import("prefill.zig");
const a = std.heap.c_allocator;

pub const Dump = struct {
    prefix: [:0]u8,
    component: prefill.Component,
    buffer: *anyopaque,

    pub fn init(gpu: *c.flint_gpu) !?Dump {
        const prefix = c.getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
        if (prefix == null or prefix[0] == 0) return null;
        const name = c.getenv("DS4_METAL_GRAPH_DUMP_NAME");
        if (name == null) return error.ActivationDumpNeedsComponent;
        const component: prefill.Component = if (std.mem.eql(u8, std.mem.span(name), "residual"))
            .ffn
        else if (std.mem.eql(u8, std.mem.span(name), "attn_residual"))
            .attention
        else
            return error.UnsupportedActivationDumpComponent;
        const position = c.getenv("DS4_METAL_GRAPH_DUMP_POS");
        if (position != null and !std.mem.eql(u8, std.mem.span(position), "0")) return error.UnsupportedActivationDumpPosition;
        const owned_prefix = try a.dupeSentinel(u8, std.mem.span(prefix), 0);
        errdefer a.free(owned_prefix);
        const buffer = c.flint_gpu_buffer(gpu, decode.layers * 2048 * 4) orelse return error.GpuAllocation;
        return .{ .prefix = owned_prefix, .component = component, .buffer = buffer };
    }

    pub fn deinit(self: *Dump) void {
        c.flint_gpu_buffer_free(self.buffer);
        a.free(self.prefix);
    }

    pub fn view(self: *const Dump) prefill.Capture {
        return .{ .component = self.component, .output = .{ .handle = self.buffer } };
    }

    pub fn write(self: *const Dump) !void {
        const values = decode.contents(f32, self.buffer, decode.layers * 2048);
        const component = if (self.component == .ffn) "residual" else "attn_residual";
        for (0..decode.layers) |layer| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}_{s}-{d}_pos0.bin", .{ self.prefix, component, layer }, 0);
            defer a.free(path);
            const file = c.fopen(path, "wb") orelse return error.OpenActivationDump;
            const written = c.fwrite(values[layer * 2048 ..][0..2048].ptr, 4, 2048, file);
            const closed = c.fclose(file);
            if (written != 2048 or closed != 0) return error.WriteActivationDump;
        }
    }
};
