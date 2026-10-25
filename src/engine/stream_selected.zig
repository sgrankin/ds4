//! One persistent selected-expert I/O worker, shared by the Metal model graphs.
//! Finish consumes the completed mailbox before its tensor/model/layer borrows
//! may retire. IDs survive a failed cache load for the caller's sync retry.
const std = @import("std");
const c = @import("stream_selected_c");
pub const Job = c.metal_graph_selected_async_load;
const Tensor = c.ds4_gpu_tensor;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;

const Counts = struct { experts: u32, selected: u32 };

const Service = struct {
    mutex: c.pthread_mutex_t = c.ds4_stream_selected_mutex_init,
    ready: c.pthread_cond_t = c.ds4_stream_selected_cond_init,
    completed: c.pthread_cond_t = c.ds4_stream_selected_cond_init,
    thread: c.pthread_t = std.mem.zeroes(c.pthread_t),
    started: bool = false,
    has_job: bool = false,
    done: bool = false,
    job: Job = std.mem.zeroes(Job),
    counts: Counts = .{ .experts = 0, .selected = 0 },
    metal_context: @TypeOf(c.ds4_metal_context_selected()) = null,
    host_context: @TypeOf(c.ds4_host_context_selected()) = null,

    fn worker(context: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Service = @ptrCast(@alignCast(context.?));
        // Cache paths must not wait for command buffers here while the main
        // thread is encoding. Such waits become load failures, retried by the
        // caller synchronously after finish instead of deadlocking the encoder.
        while (true) {
            _ = c.pthread_mutex_lock(&self.mutex);
            while (!self.has_job) _ = c.pthread_cond_wait(&self.ready, &self.mutex);
            var job = self.job;
            const counts = self.counts;
            const metal_context = self.metal_context;
            const host_context = self.host_context;
            _ = c.pthread_mutex_unlock(&self.mutex);

            const previous_host = c.ds4_host_context_select(host_context);
            const previous = c.ds4_metal_context_select(metal_context);
            c.ds4_gpu_stream_expert_cache_note_service_thread();
            run(&job, counts);
            _ = c.ds4_metal_context_select(previous);
            _ = c.ds4_host_context_select(previous_host);

            _ = c.pthread_mutex_lock(&self.mutex);
            self.job = job;
            self.has_job = false;
            self.done = true;
            _ = c.pthread_cond_signal(&self.completed);
            _ = c.pthread_mutex_unlock(&self.mutex);
        }
    }

    fn ensureWorker(self: *Service) bool {
        _ = c.pthread_mutex_lock(&self.mutex);
        if (self.started) {
            _ = c.pthread_mutex_unlock(&self.mutex);
            return true;
        }
        const result = c.pthread_create(&self.thread, null, worker, self);
        if (result != 0) {
            _ = c.pthread_mutex_unlock(&self.mutex);
            _ = c.fprintf(c.__stderrp, "ds4: failed to start Metal streaming async selected load worker: %s\n", c.strerror(result));
            return false;
        }
        self.started = true;
        _ = c.pthread_mutex_unlock(&self.mutex);
        return true;
    }
};

// This replaces the old C globals. There is one mailbox and one worker for
// the process, with the existing explicit start/finish admission policy.
var service: Service = .{};

fn table(job: *const Job, counts: Counts) c.ds4_gpu_stream_expert_table {
    const layer = job.layer;
    return .{
        .model_map = job.model.*.map,
        .model_size = job.model.*.size,
        .layer = job.il,
        .n_total_expert = counts.experts,
        .gate_offset = if (layer.*.ffn_gate_exps != null) layer.*.ffn_gate_exps.*.abs_offset else 0,
        .up_offset = if (layer.*.ffn_up_exps != null) layer.*.ffn_up_exps.*.abs_offset else 0,
        .down_offset = if (layer.*.ffn_down_exps != null) layer.*.ffn_down_exps.*.abs_offset else 0,
        .gate_expert_bytes = job.gate_expert_bytes,
        .down_expert_bytes = job.down_expert_bytes,
    };
}

fn run(job: *Job, counts: Counts) void {
    job.ok = false;
    if (job.router_selected == null or job.model == null or job.layer == null or
        counts.experts == 0 or counts.experts > c.DS4_MAX_EXPERT or
        counts.selected == 0 or counts.selected > c.DS4_MAX_EXPERT_USED) return;
    if (job.event_value != 0) {
        if (c.ds4_gpu_wait_selected_readback_ready(job.event_value, "selected-id async expert load") == 0) return;
        if (c.ds4_gpu_tensor_read(job.router_selected, 0, &job.selected_ids, @as(u64, counts.selected) * @sizeOf(i32)) == 0) return;
    }
    for (job.selected_ids[0..counts.selected]) |id| {
        if (id < 0 or @as(u32, @intCast(id)) >= counts.experts) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming async selected expert id %d is outside 0..%u at layer %u\n", id, counts.experts, job.il);
            return;
        }
    }
    job.ids_ok = true;
    const experts = table(job, counts);
    if (c.ds4_gpu_stream_expert_cache_begin_selected_load(&experts, &job.selected_ids, counts.selected) == 0) return;
    job.ok = true;
}

pub fn start(maybe_job: ?*Job, selected: ?*Tensor, model: [*c]const Model, layer: [*c]const Layer, index: u32, event: u64, gate_bytes: u64, down_bytes: u64, expert_count: u32, selected_count: u32) bool {
    const job = maybe_job orelse return false;
    if (selected == null or event == 0) return false;
    if (!service.ensureWorker()) return false;
    job.* = std.mem.zeroes(Job);
    job.router_selected = selected;
    job.model = model;
    job.layer = layer;
    job.il = index;
    job.event_value = event;
    job.gate_expert_bytes = gate_bytes;
    job.down_expert_bytes = down_bytes;

    _ = c.pthread_mutex_lock(&service.mutex);
    // A completed job still owns the mailbox until its caller consumes it.
    if (service.has_job or service.done) {
        _ = c.pthread_mutex_unlock(&service.mutex);
        return false;
    }
    service.job = job.*;
    service.job.ok = false;
    service.counts = .{ .experts = expert_count, .selected = selected_count };
    service.metal_context = c.ds4_metal_context_selected();
    service.host_context = c.ds4_host_context_selected();
    service.has_job = true;
    _ = c.pthread_cond_signal(&service.ready);
    _ = c.pthread_mutex_unlock(&service.mutex);
    job.active = true;
    return true;
}

pub fn finish(maybe_job: ?*Job) bool {
    const job = maybe_job orelse return false;
    if (!job.active) return false;
    _ = c.pthread_mutex_lock(&service.mutex);
    while (!service.done) _ = c.pthread_cond_wait(&service.completed, &service.mutex);
    job.* = service.job;
    const selected = service.counts.selected;
    service.done = false;
    _ = c.pthread_mutex_unlock(&service.mutex);
    job.active = false;
    if (!job.ok) return false;
    return c.ds4_gpu_routed_moe_set_selected_override(&job.selected_ids, selected) != 0;
}
