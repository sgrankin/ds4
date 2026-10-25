//! Per-request uploads used by both dense Clef and the Omni thinker. Their
//! lifetime ends only after the inference command has completed.
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const a = std.heap.c_allocator;
const View = b.View;
pub const Injection = struct {
    rows: []const u32,
    features: []const f32,
    deepstack: []const []const f32 = &.{},
};
const Entry = struct { count: u32, rows: View, features: View, deepstack: [3]?View = @splat(null) };
const Scatter = extern struct { rows: u32, width: u32, additive: u32, pad: u32 = 0 };
pub const Prepared = struct {
    gpu: *c.flint_gpu,
    buffers: std.ArrayList(b.Buffer) = .empty,
    entries: []Entry,

    pub fn init(gpu: *c.flint_gpu, items: []const Injection, tokens: usize, width: u32) !Prepared {
        var self = Prepared{ .gpu = gpu, .entries = try a.alloc(Entry, items.len) };
        errdefer self.deinit();
        for (items, self.entries) |item, *entry| {
            if (item.features.len != item.rows.len * width or (item.deepstack.len != 0 and item.deepstack.len != 3)) return error.InvalidMediaFeatures;
            for (item.rows) |row| if (row >= tokens) return error.InvalidMediaRow;
            entry.* = .{ .count = @intCast(item.rows.len), .rows = try self.upload(u32, item.rows), .features = try self.upload(f32, item.features) };
            for (item.deepstack, 0..) |features, i| {
                if (features.len != item.features.len) return error.InvalidDeepStack;
                entry.deepstack[i] = try self.upload(f32, features);
            }
        }
        return self;
    }
    pub fn deinit(self: *Prepared) void {
        for (self.buffers.items) |handle| c.flint_gpu_buffer_free(handle);
        self.buffers.deinit(a);
        a.free(self.entries);
    }
    fn upload(self: *Prepared, comptime T: type, values: []const T) !View {
        const buffer = c.flint_gpu_buffer(self.gpu, values.len * @sizeOf(T)) orelse return error.GpuAllocation;
        self.buffers.append(a, buffer) catch |err| {
            c.flint_gpu_buffer_free(buffer);
            return err;
        };
        const view = View{ .buffer = buffer };
        @memcpy(b.contents(T, view, values.len), values);
        return view;
    }
    pub fn inject(self: *const Prepared, e: b.Encoder, hidden: View, width: u32, layer: ?u32) !void {
        for (self.entries) |entry| {
            if (entry.count == 0) continue;
            const features = if (layer) |i| entry.deepstack[i] orelse continue else entry.features;
            try e.dispatch("clef_media_scatter", Scatter{ .rows = entry.count, .width = width, .additive = @intFromBool(layer != null) }, &.{ entry.rows.bind(1), features.bind(2), hidden.bind(3) }, .{ (entry.count * width + 255) / 256, 1, 1 }, 256);
        }
    }
};
