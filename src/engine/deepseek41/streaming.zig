//! Model-view selection and bounded file-cache preparation for V4.1 Metal.
//! Spans/workers are native-owned; tensor descriptors and mapped bytes are
//! borrowed. Keep this owner at a stable address while prepare jobs are live,
//! and join them before its Model/Weights owners retire.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const experts = @import("expert_cache.zig");
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Weight = c.ds4_tensor;
const allocator = std.heap.c_allocator;
const max_u64 = std.math.maxInt(u64);

fn outOfMemory() noreturn {
    c.ds4_engine_model_die("out of memory");
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn nowSeconds() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    const seconds: f64 = @floatFromInt(time.tv_sec);
    const nanoseconds: f64 = @floatFromInt(time.tv_nsec);
    return seconds + nanoseconds * 1.0e-9;
}
fn groupViews() u32 {
    const value = c.getenv("DS4_METAL_Q4_PRO_MAP_GROUPS");
    if (value == null or value[0] == 0) return 1;
    var end: [*c]u8 = null;
    const groups = c.strtoul(value, &end, 10);
    return if (end != value and end[0] == 0 and groups > 0 and groups <= 384 and 384 % groups == 0) @intCast(groups) else 1;
}

const Span = struct {
    first: u64,
    end: u64,
    isolated: bool,

    fn before(_: void, a: Span, b: Span) bool {
        return a.first < b.first or (a.first == b.first and a.end < b.end);
    }
};

pub const Spans = struct {
    items: std.ArrayList(Span) = .empty,
    max_tensor_bytes: u64 = 0,

    pub fn deinit(self: *Spans) void {
        self.items.deinit(allocator);
    }
    pub fn append(self: *Spans, first: u64, end: u64, isolated: bool) void {
        if (first == max_u64 or end <= first) return;
        self.items.append(allocator, .{ .first = first, .end = end, .isolated = isolated }) catch outOfMemory();
    }
    pub fn include(self: *Spans, pointer: [*c]const Weight) void {
        if (pointer == null or pointer.*.bytes == 0) return;
        const weight: *const Weight = @ptrCast(pointer);
        const isolation_threshold = 2 * 1024 * 1024 * 1024;
        const groups = groupViews();
        if (weight.type == c.DS4_TENSOR_Q4_K and weight.ndim == 3 and weight.dim[2] == 384 and
            weight.bytes >= isolation_threshold and weight.bytes % groups == 0)
        {
            // Large PRO Q4 expert blobs must not hide inside broad layer
            // views. Optional stable groups avoid stacking giant aliases on
            // the ordinary selected-expert path.
            const bytes = weight.bytes / groups;
            self.max_tensor_bytes = @max(self.max_tensor_bytes, bytes);
            for (0..groups) |group| {
                const first = weight.abs_offset +% (@as(u64, group) *% bytes);
                self.append(first, first +% bytes, true);
            }
            return;
        }
        self.max_tensor_bytes = @max(self.max_tensor_bytes, weight.bytes);
        const isolated = (weight.type == c.DS4_TENSOR_Q4_K or weight.type == c.DS4_TENSOR_MXFP4) and
            weight.bytes >= isolation_threshold;
        self.append(weight.abs_offset, weight.abs_offset +% weight.bytes, isolated);
    }
    pub fn includeLayer(self: *Spans, layer: *const Layer, decode: bool, size_class: ?u64) void {
        // Every actual layer field is a borrowed tensor descriptor. The C
        // collection order follows this same record; other-family fields are
        // NULL in the validated V4.1 table.
        inline for (comptime std.meta.fieldNames(Layer)) |name| {
            const routed = comptime std.mem.eql(u8, name, "ffn_gate_exps") or
                std.mem.eql(u8, name, "ffn_up_exps") or std.mem.eql(u8, name, "ffn_down_exps");
            if (!decode or !routed) self.include(@field(layer, name));
        }
        if (decode) {
            // A boosted/outlier layer cannot use the cache's dominant slab
            // size class. Include its full experts for direct mapped reads.
            const bytes = experts.layerBytes(layer);
            if (bytes != null and size_class != null and bytes.? != size_class.?) {
                self.include(layer.ffn_gate_exps);
                self.include(layer.ffn_up_exps);
                self.include(layer.ffn_down_exps);
            }
        }
    }
    pub fn includeOutput(self: *Spans, weights: *const Weights) void {
        const tensors = [_][*c]const Weight{
            weights.output_hc_base, weights.output_hc_fn, weights.output_hc_scale,
            weights.output_norm,    weights.output,       weights.output_hc_norm,
            weights.output_hc_down, weights.output_hc_up,
        };
        for (tensors) |tensor| self.include(tensor);
        // N-gram rows remain on disk and never enter GPU weight views.
    }
    pub fn finish(self: *Spans) bool {
        if (self.items.items.len == 0 or self.max_tensor_bytes == 0) return false;
        std.mem.sortUnstable(Span, self.items.items, {}, Span.before);
        var count: usize = 0;
        for (self.items.items) |span| {
            if (count == 0 or span.first > self.items.items[count - 1].end or
                span.isolated or self.items.items[count - 1].isolated)
            {
                self.items.items[count] = span;
                count += 1;
            } else if (span.end > self.items.items[count - 1].end) {
                self.items.items[count - 1].end = span.end;
            }
        }
        self.items.items.len = count;
        return count != 0;
    }
};

const Range = struct { first: u64, bytes: u64 };

fn prepareThreads() u32 {
    var value = c.getenv("DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS");
    if (value == null or value[0] == 0) value = c.getenv("DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_THREADS");
    if (value == null or value[0] == 0) return 8;
    var end: [*c]u8 = null;
    const count = c.strtoul(value, &end, 10);
    if (end == value or end[0] != 0 or count == 0) return 1;
    return @intCast(@min(count, 16));
}

fn readRange(model: *const Model, range: Range, read_bytes: *u64, sink: *u8) bool {
    return c.ds4_native_pagein_pread(model, range.first, range.bytes, read_bytes, sink);
}

const Worker = struct {
    job: *Job,
    first: usize,
    stride: usize,
    thread: c.pthread_t = undefined,
    touched: u64 = 0,
    milliseconds: f64 = 0,
    sink: u8 = 0,
    ok: bool = true,

    fn run(argument: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Worker = @ptrCast(@alignCast(argument orelse return null));
        const started = if (self.job.profile) nowSeconds() else 0.0;
        var index = self.first;
        while (index < self.job.ranges.items.len) : (index += self.stride) {
            if (!readRange(self.job.model, self.job.ranges.items[index], &self.touched, &self.sink)) {
                self.ok = false;
                break;
            }
        }
        if (self.job.profile) self.milliseconds = (nowSeconds() - started) * 1000.0;
        return null;
    }
};

const Job = struct {
    model: *const Model = undefined,
    ranges: std.ArrayList(Range) = .empty,
    workers: []Worker = &.{},
    layer: u32 = 0,
    tokens: u32 = 0,
    bytes: u64 = 0,
    profile: bool = false,
    started: bool = false,

    fn start(self: *Job, model: *const Model, layer: u32, tokens: u32, spans: *const Spans) bool {
        self.* = .{ .model = model, .layer = layer, .tokens = tokens };
        self.profile = present("DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE") or
            present("DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE") or
            present("DS4_METAL_STREAMING_PREFILL_LAYER_MADVISE_PROFILE") or
            present("DS4_METAL_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE");
        const threads = prepareThreads();
        const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
        for (spans.items.items) |span| {
            const bytes = span.end - span.first;
            var consumed: u64 = 0;
            var chunk = bytes / threads;
            if (chunk > page) chunk = chunk / page * page;
            if (chunk == 0) chunk = bytes;
            for (0..threads) |thread| {
                if (consumed == bytes) break;
                const take = if (thread + 1 == threads or bytes - consumed <= chunk) bytes - consumed else chunk;
                self.ranges.append(allocator, .{ .first = span.first + consumed, .bytes = take }) catch outOfMemory();
                consumed += take;
            }
            self.bytes +|= bytes;
        }
        self.workers = allocator.alloc(Worker, threads) catch outOfMemory();
        for (self.workers, 0..) |*worker, index| {
            worker.* = .{ .job = self, .first = index, .stride = threads };
            const rc = c.pthread_create(&worker.thread, null, Worker.run, worker);
            if (rc != 0) {
                const message: [*:0]const u8 = if (threads == 1)
                    "ds4: Metal streaming prefill layer page-in thread failed: %s\n"
                else
                    "ds4: Metal streaming prefill layer page-in worker failed: %s\n";
                _ = c.fprintf(c.__stderrp, message, c.strerror(rc));
                for (self.workers[0..index]) |started_worker| _ = c.pthread_join(started_worker.thread, null);
                self.release();
                return false;
            }
        }
        self.started = true;
        return true;
    }

    fn join(self: *Job) bool {
        if (!self.started) return true;
        const started = if (self.profile) nowSeconds() else 0.0;
        var first_error: c_int = 0;
        var ok = true;
        var touched: u64 = 0;
        var milliseconds: f64 = 0;
        var sink: u8 = 0;
        for (self.workers) |*worker| {
            const rc = c.pthread_join(worker.thread, null);
            if (rc != 0 and first_error == 0) first_error = rc;
            ok = ok and rc == 0 and worker.ok;
            touched +|= worker.touched;
            milliseconds = @max(milliseconds, worker.milliseconds);
            sink ^= worker.sink;
        }
        const wait_ms = if (self.profile) (nowSeconds() - started) * 1000.0 else 0.0;
        if (self.profile) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill layer pread layer=%u tokens=%u threads=%u ranges=%u bytes=%.2f GiB wait=%.3f ms thread=%.3f ms read=%.2f GiB ok=%d\n", self.layer, self.tokens, @as(c_uint, @intCast(self.workers.len)), @as(c_uint, @intCast(self.ranges.items.len)), @as(f64, @floatFromInt(self.bytes)) / 1073741824.0, wait_ms, milliseconds, @as(f64, @floatFromInt(touched)) / 1073741824.0, @as(c_int, @intFromBool(ok)));
        }
        if (first_error != 0)
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill layer page-in join failed: %s\n", c.strerror(first_error));
        self.release();
        return ok;
    }
    fn release(self: *Job) void {
        allocator.free(self.workers);
        self.ranges.deinit(allocator);
        self.* = .{};
    }
};

const Slot = struct { job: Job = .{}, layer: u32 = 0, active: bool = false };

pub const Streaming = struct {
    model: *const Model,
    weights: *const Weights,
    config: *const Config,
    prepare_count: u32,
    size_class: ?u64,
    slots: [2]Slot = @splat(.{}),

    // The V4.1 executor chooses one or two bounded lookahead slots. No model
    // data or mapped views are duplicated by constructing this owner.
    pub fn init(model: *const Model, weights: *const Weights, config: *const Config, prepare_count: u32) Streaming {
        return .{ .model = model, .weights = weights, .config = config, .prepare_count = prepare_count, .size_class = experts.sizeClass(config, weights) };
    }

    fn layerSpans(self: *const Streaming, spans: *Spans, index: u32, decode: bool) bool {
        if (index >= self.config.shape.n_layer) return false;
        if (!decode and index == 0) spans.include(self.weights.token_embd);
        spans.includeLayer(&self.weights.layer[index], decode, self.size_class);
        return spans.finish();
    }
    fn install(self: *const Streaming, spans: *const Spans, label: [*:0]const u8) bool {
        const offsets = allocator.alloc(u64, spans.items.items.len) catch outOfMemory();
        defer allocator.free(offsets);
        const sizes = allocator.alloc(u64, spans.items.items.len) catch outOfMemory();
        defer allocator.free(sizes);
        for (spans.items.items, offsets, sizes) |span, *offset, *bytes| {
            offset.* = span.first;
            bytes.* = span.end - span.first;
        }
        const ok = c.ds4_gpu_set_model_map_spans(self.model.map, self.model.size, offsets.ptr, sizes.ptr, @intCast(offsets.len), spans.max_tensor_bytes) != 0;
        if (!ok) _ = c.fprintf(c.__stderrp, "ds4: Metal SSD streaming failed to map %s model spans\n", label);
        return ok;
    }
    pub fn mapToken(self: *const Streaming) bool {
        var spans = Spans{};
        defer spans.deinit();
        spans.include(self.weights.token_embd);
        if (!spans.finish()) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal SSD streaming could not build token embedding span\n");
            return false;
        }
        return self.install(&spans, "token embedding");
    }
    pub fn mapLayer(self: *const Streaming, index: u32) bool {
        return self.mapOneLayer(index, false);
    }
    pub fn mapLayerDecode(self: *const Streaming, index: u32) bool {
        return self.mapOneLayer(index, true);
    }
    fn mapOneLayer(self: *const Streaming, index: u32, decode: bool) bool {
        var spans = Spans{};
        defer spans.deinit();
        if (!self.layerSpans(&spans, index, decode)) {
            const message: [*:0]const u8 = if (decode) "ds4: Metal SSD streaming could not build decode layer %u spans\n" else "ds4: Metal SSD streaming could not build layer %u spans\n";
            _ = c.fprintf(c.__stderrp, message, index);
            return false;
        }
        return self.install(&spans, if (decode) "decode layer" else "layer");
    }
    pub fn mapStatic(self: *const Streaming) bool {
        var spans = Spans{};
        defer spans.deinit();
        spans.include(self.weights.token_embd);
        for (self.weights.layer[0..self.config.shape.n_layer]) |*layer| spans.includeLayer(layer, true, self.size_class);
        spans.includeOutput(self.weights);
        if (!spans.finish()) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal SSD streaming could not build static decode spans\n");
            return false;
        }
        return self.install(&spans, "static decode");
    }
    pub fn mapOutput(self: *const Streaming) bool {
        var spans = Spans{};
        defer spans.deinit();
        spans.includeOutput(self.weights);
        if (!spans.finish()) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal SSD streaming could not build output head spans\n");
            return false;
        }
        return self.install(&spans, "output head");
    }

    fn start(self: *Streaming, job: *Job, index: u32, rows: u32) bool {
        // DS41 always requests NULL-graph, pread-only layer preparation.
        // Other generic page-in/madvise modes belong to their C callers.
        if (present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") or
            present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE")) return true;
        var spans = Spans{};
        defer spans.deinit();
        return self.layerSpans(&spans, index, false) and job.start(self.model, index, rows, &spans);
    }
    fn find(self: *Streaming, index: u32) ?*Slot {
        for (self.slots[0..self.prepare_count]) |*slot| {
            if (slot.active and slot.layer == index) return slot;
        }
        return null;
    }
    pub fn prepareStart(self: *Streaming, index: u32, rows: u32) bool {
        if (index >= self.config.shape.n_layer or self.find(index) != null) return true;
        for (self.slots[0..self.prepare_count]) |*slot| {
            if (slot.active) continue;
            slot.* = .{ .layer = index };
            if (!self.start(&slot.job, index, rows)) {
                slot.* = .{};
                return false;
            }
            slot.active = slot.job.started;
            return true;
        }
        _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill prepare queue is full before layer %u\n", index);
        return false;
    }
    pub fn prepareJoin(self: *Streaming, index: u32, rows: u32) bool {
        if (self.find(index)) |slot| {
            const ok = slot.job.join();
            slot.* = .{};
            return ok;
        }
        var job = Job{};
        return self.start(&job, index, rows) and job.join();
    }
    pub fn joinAll(self: *Streaming) bool {
        var ok = true;
        for (self.slots[0..self.prepare_count]) |*slot| {
            if (!slot.active) continue;
            const joined = slot.job.join();
            ok = joined and ok;
            slot.* = .{};
        }
        return ok;
    }
};

/// Admission counts the guard-only static set, excluding ALL routed experts.
/// Runtime static maps deliberately include outlier expert layouts; using that
/// larger set here would charge those blobs twice (fixed weights + cache).
/// The bound-weight predicates select token/output ownership at the C facade.
pub fn staticBytes(config: *const Config, weights: *const Weights, include_token: bool, include_output: bool) ?u64 {
    var spans = Spans{};
    defer spans.deinit();
    if (include_token) spans.include(weights.token_embd);
    for (weights.layer[0..config.shape.n_layer]) |*layer| spans.includeLayer(layer, true, null);
    if (include_output) spans.includeOutput(weights);
    if (!spans.finish()) return null;
    var total: u64 = 0;
    for (spans.items.items) |span| total +|= span.end - span.first;
    return total;
}
