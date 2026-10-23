//! One bounded Engram prefix reader. The caller lends table/IDs/output until
//! join, including cancellation and GPU failures. The object must remain at a
//! stable address after start: its pthread borrows this private owner directly.
const std = @import("std");
const c = @import("gguf_c");
const Graph = c.ds41_gpu_graph;
const cols = c.DS4_ENGRAM_COLS;
const dim = c.DS4_ENGRAM_DIM;

pub const Prefetch = struct {
    thread: c.pthread_t = undefined,
    table: *const c.ds4_engram_table = undefined,
    ids: [*]const u32 = undefined,
    out: [*]f32 = undefined,
    count: u32 = 0,
    ready: std.atomic.Value(u32) = .init(0),
    active: bool = false,
    ok: bool = false,
    cancel: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),

    fn read(raw: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Prefetch = @ptrCast(@alignCast(raw.?));
        self.ok = true;
        var offset: u32 = 0;
        while (offset < self.count) : (offset += 2048) {
            const rows: u32 = @min(self.count - offset, 2048);
            if (self.cancel.load(.monotonic) or !c.ds4_engram_read_batch(self.table, self.ids + @as(usize, offset) * 2 * cols, rows, 2 * cols, self.out + @as(usize, offset) * cols * dim)) {
                self.ok = false;
                break;
            }
            // Publish completed rows only. Later pread batches write disjoint
            // output ranges, so the GPU may consume this finished prefix.
            self.ready.store(offset + rows, .release);
        }
        self.done.store(true, .release);
        return null;
    }

    pub fn start(self: *Prefetch, graph: *Graph, table: u32, count: u32) bool {
        // The previous reader must already be joined before reusing its state
        // or shared output buffer for the other trained Engram layer.
        self.* = .{
            .table = &graph.table[table],
            .ids = @ptrCast(&graph.prefill_ids[0][table][0]),
            .count = count,
        };
        const output = c.ds4_gpu_tensor_contents(graph.engram_prefetch) orelse return false;
        self.out = @ptrCast(@alignCast(output));
        if (c.pthread_create(&self.thread, null, read, self) != 0) return false;
        self.active = true;
        return true;
    }

    pub fn wait(self: *Prefetch, rows: u32, cancel: c.ds4_session_cancel_fn, userdata: ?*anyopaque) bool {
        if (!self.active or rows > self.count) return false;
        while (self.ready.load(.acquire) < rows) {
            if (self.done.load(.acquire)) return self.ready.load(.acquire) >= rows;
            if (cancel) |check| if (check(userdata)) return false;
            const pause = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 1000000 };
            _ = c.nanosleep(&pause, null);
        }
        return true;
    }

    pub fn join(self: *Prefetch, cancel: bool) bool {
        if (!self.active) return true;
        if (cancel) self.cancel.store(true, .monotonic);
        if (c.pthread_join(self.thread, null) != 0)
            c.ds4_engine_model_die("cannot join V4.1 Engram reader safely");
        self.active = false;
        return self.ok;
    }
};
