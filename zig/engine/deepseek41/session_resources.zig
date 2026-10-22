//! Complete V4.1 Session device/host-resource admission and lifetime.
//! Session shell and transport registration stay outside this owner. Config,
//! Model/Weights and Engine TP state are borrowed; graph and host buffers are
//! published only after construction succeeds and are charged exactly once.
const std = @import("std");
const c = @import("gguf_c");
const Memory = @import("memory.zig").Memory;
const Graph = @import("resources.zig").Graph;
const graph_resources = @import("resources.zig");
const Drafter = @import("dspark.zig");
const Layout = @import("layout.zig").Layout;

pub const Options = struct {
    context: u32,
    chunk: u32,
    quality: bool,
    streaming: bool,
    drafter: ?*const c.ds4_dspark_weights,
};

fn floats(count: u32) [*c]f32 {
    // Keep existing host-allocation admission and fatal OOM diagnostics. The
    // numerical Sampling boundary already exposes this exact engine action.
    const allocation = c.ds4_sampling_alloc(@as(usize, count) * @sizeOf(f32)) orelse unreachable;
    return @ptrCast(@alignCast(allocation));
}

pub const SessionResources = struct {
    device: *c.ds4_deepseek41_session_device,
    frontier: *c.ds4_session_frontier,
    memory: Memory,
    model: *const c.ds4_model,
    path: [*:0]const u8,
    transport: *const c.ds4_engine_tp_state,

    pub fn init(self: *const SessionResources, options: Options) bool {
        if (options.context > 1048576) return false;
        const config = self.memory.config;
        const planned = Layout.init(config, options.context, options.chunk).bytes();
        const total = self.memory.state.ds41_session_bytes +| planned;
        // Additional sessions cannot silently shrink the configured cache.
        if (!self.memory.admit(total, false)) return false;
        var graph = Graph.init(config.*, self.model, self.memory.weights, self.path, options.context, options.chunk, options.streaming) catch return false;
        defer graph.deinit();
        const record = &graph.record;
        record.quality = options.quality;
        if (options.drafter) |weights| {
            record.dspark = Drafter.allocate(config, weights) orelse return false;
            record.allocation_bytes +%= @as(u64, 64) << 20;
        }

        const tp = self.transport;
        if (tp.active) {
            record.tp_world = 2;
            record.tp_rank = @bitCast(tp.rank);
            record.tp_out = tp.out_views;
            record.tp_in = tp.in_views;
            record.tp_batch_out = tp.batch_out_views;
            record.tp_batch_in = tp.batch_in_views;
            if (tp.vocab_split) {
                const half_bytes = @as(u64, config.shape.n_vocab) / 2 * @sizeOf(f32);
                const rank: u64 = @bitCast(@as(i64, tp.rank));
                record.tp_logits_half = c.ds4_gpu_tensor_view(record.logits, rank *% half_bytes, half_bytes) orelse return false;
                if (c.ds4_gpu_tensor_fill_f32(record.logits, 0, config.shape.n_vocab) == 0) return false;
            }
        }

        const logits = floats(config.shape.n_vocab);
        const probabilities = floats(config.shape.n_vocab);
        self.frontier.ctx_size = @intCast(options.context);
        self.frontier.prefill_cap = record.prefill_cap;
        self.frontier.logits = logits;
        self.frontier.sample_probs = probabilities;
        self.device.graph = graph.take();
        self.device.ready = true;
        self.memory.state.ds41_session_bytes +%= self.device.graph.allocation_bytes;
        return true;
    }
};

pub fn deinit(device: *c.ds4_deepseek41_session_device, frontier: *c.ds4_session_frontier, memory: *c.ds4_engine_memory) void {
    if (device.ready) {
        // Decharge the current allocation, including demand-grown workspace and
        // the optional drafter reserve. Graph releases all async/Engram/tensor
        // resources and retained aliases through its existing native owner.
        memory.ds41_session_bytes -%= device.graph.allocation_bytes;
        graph_resources.freeRecord(&device.graph);
        device.ready = false;
    }
    c.ds4_tokens_free(&frontier.checkpoint);
    c.free(frontier.checkpoint_images);
    c.free(frontier.logits);
    c.free(frontier.sampling_logits);
    c.free(frontier.sample_probs);
    // sync_images is borrowed. Clearing the dying frontier must not free it.
    // The remaining C shell cleanup can safely free these now-empty owners.
    frontier.* = std.mem.zeroes(c.ds4_session_frontier);
}
