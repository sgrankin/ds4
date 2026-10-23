//! Worker/UI synchronization. UI writes remain outside the mutex; the worker
//! publishes owned batches and wakes through a level-triggered pipe byte.
//! This owner borrows the ONE actual Worker record shared with the C fallback.
const std = @import("std");
const c = @import("agent_runtime_c");
const allocator = std.heap.c_allocator;

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn duplicate(value: [*c]const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, text(value), 0) catch {
        c.perror("ds4-agent: malloc");
        c.exit(1);
    }).ptr;
}
fn message(errors: []u8, value: [*c]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", value);
}
fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1e-9;
}

pub const Communications = struct {
    worker: *c.agent_worker,

    fn lock(self: Communications) void {
        _ = c.pthread_mutex_lock(&self.worker.mu);
    }
    fn unlock(self: Communications) void {
        _ = c.pthread_mutex_unlock(&self.worker.mu);
    }
    fn signal(self: Communications) void {
        _ = c.pthread_cond_signal(&self.worker.cond);
    }
    fn contextSize(self: Communications) c_int {
        const worker = self.worker;
        var current = if (worker.session != null) c.ds4_session_ctx(worker.session) else 0;
        if (current <= 0 and worker.cfg != null) current = worker.cfg.*.gen.ctx_size;
        return current;
    }
    /// Caller holds mu. Bursts of sampled tokens must not flood the pipe.
    pub fn wakeLocked(self: Communications) void {
        const worker = self.worker;
        if (worker.wake_pending) return;
        worker.wake_pending = true;
        _ = c.write(worker.wake_fd[1], "x", 1);
    }
    /// Queue rendered bytes for the UI; the worker never writes the terminal.
    /// Failed growth leaves the old allocation/capacity/output unchanged. Keep
    /// the existing doubling policy and recoverable OOM rather than a fatal
    /// collection append; consume() later transfers this allocation verbatim.
    pub fn publish(self: Communications, value: []const u8) void {
        if (value.len == 0) return;
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        const needed = worker.out_len +% value.len +% 1;
        if (needed > worker.out_cap) {
            var capacity = if (worker.out_cap != 0) worker.out_cap *% 2 else 4096;
            while (capacity < needed) capacity *%= 2;
            const storage = c.realloc(worker.out, capacity) orelse return;
            worker.out = @ptrCast(storage);
            worker.out_cap = capacity;
        }
        @memcpy(worker.out[worker.out_len..][0..value.len], value);
        worker.out_len +%= value.len;
        worker.out[worker.out_len] = 0;
        self.wakeLocked();
    }
    pub fn setStatus(self: Communications, state: c.agent_worker_state) void {
        self.lock();
        defer self.unlock();
        self.worker.status.state = state;
        if (state != c.AGENT_WORKER_PREFILL) self.worker.status.prefill_tps = 0;
        if (state != c.AGENT_WORKER_GENERATING) self.worker.status.greedy_sampling = false;
        self.wakeLocked();
    }
    pub fn setError(self: Communications, value: [*c]const u8) void {
        self.lock();
        defer self.unlock();
        const status = &self.worker.status;
        status.state = c.AGENT_WORKER_ERROR;
        status.prefill_tps = 0;
        status.greedy_sampling = false;
        _ = c.snprintf(&status.@"error", status.@"error".len, "%s", @as([*c]const u8, if (value != null) value else "unknown error"));
        self.wakeLocked();
    }
    pub fn progress(self: Communications, event: [*c]const u8, current: c_int) void {
        if (event == null) return;
        self.applyPendingPower();
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        var done: c_int = 0;
        if (std.mem.eql(u8, text(event), "prefill_work")) {
            // GLM work is relative display progress. Checkpoint chunk events
            // still carry absolute positions after a completed chunk.
            done = current;
        } else if (std.mem.eql(u8, text(event), "prefill_chunk") or std.mem.eql(u8, text(event), "prefill_display")) {
            if (worker.progress_direct) return;
            done = current - worker.progress_base;
        } else return;
        const total = @max(0, worker.status.prefill_total);
        done = @max(0, @min(done, total));
        worker.status.prefill_total = total;
        worker.status.prefill_done = done;
        const elapsed = now() - worker.progress_started_at;
        worker.status.prefill_tps = if (done > 0 and elapsed > 0) @as(f64, @floatFromInt(done)) / elapsed else 0;
        self.wakeLocked();
    }
    pub fn interrupted(self: Communications) bool {
        self.lock();
        defer self.unlock();
        return self.worker.interrupt or self.worker.stop;
    }
    /// Ctrl+C is latched until the worker reaches a stable append-only boundary
    /// and acknowledges it before IDLE; otherwise UI sees a stale interrupt.
    pub fn clearInterrupt(self: Communications) void {
        self.lock();
        defer self.unlock();
        self.worker.interrupt = false;
    }
    pub fn confirmWeb(self: Communications, prompt: [*c]const u8, errors: []u8) bool {
        const worker = self.worker;
        self.lock();
        defer self.unlock();
        worker.web_approval_pending = true;
        worker.web_approval_answered = false;
        worker.web_approval_result = false;
        worker.web_approval_error[0] = 0;
        _ = c.snprintf(&worker.web_approval_message, worker.web_approval_message.len, "%s", @as([*c]const u8, if (prompt != null) prompt else "Start visible Chrome browser? (y/n) "));
        self.wakeLocked();
        while (!worker.stop and !worker.interrupt and !worker.web_approval_answered)
            _ = c.pthread_cond_wait(&worker.cond, &worker.mu);
        var accepted = worker.web_approval_result;
        if (!worker.web_approval_answered and (worker.stop or worker.interrupt)) {
            accepted = false;
            worker.web_approval_pending = false;
            _ = c.snprintf(&worker.web_approval_error, worker.web_approval_error.len, "interrupted");
        }
        if (!accepted) message(errors, if (worker.web_approval_error[0] != 0)
            &worker.web_approval_error
        else
            "user denied Chrome browser start");
        return accepted;
    }
    pub fn takeWeb(self: Communications, prompt: []u8) bool {
        self.lock();
        defer self.unlock();
        const pending = self.worker.web_approval_pending;
        if (pending) {
            message(prompt, &self.worker.web_approval_message);
            self.worker.web_approval_pending = false;
        }
        return pending;
    }
    pub fn answerWeb(self: Communications, accepted: bool, reason: [*c]const u8) void {
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        worker.web_approval_result = accepted;
        worker.web_approval_answered = true;
        if (!accepted) _ = c.snprintf(&worker.web_approval_error, worker.web_approval_error.len, "%s", @as([*c]const u8, if (text(reason).len != 0) reason else "user denied Chrome browser start"));
        self.signal();
        self.wakeLocked();
    }
    /// Queued user input must not preempt a tool. Ask UI only after publishing
    /// the tool result so the next turn sees both result and pending correction.
    pub fn requestDrain(self: Communications) [*c]u8 {
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        worker.queued_user_drain_pending = true;
        worker.queued_user_drain_answered = false;
        c.free(worker.queued_user_drain_text);
        worker.queued_user_drain_text = null;
        self.wakeLocked();
        self.signal();
        while (!worker.stop and !worker.queued_user_drain_answered) _ = c.pthread_cond_wait(&worker.cond, &worker.mu);
        const value = worker.queued_user_drain_text;
        worker.queued_user_drain_text = null;
        worker.queued_user_drain_pending = false;
        worker.queued_user_drain_answered = false;
        return value;
    }
    pub fn takeDrain(self: Communications) bool {
        self.lock();
        defer self.unlock();
        const pending = self.worker.queued_user_drain_pending;
        if (pending) self.worker.queued_user_drain_pending = false;
        return pending;
    }
    pub fn answerDrain(self: Communications, owned: [*c]u8) void {
        self.lock();
        defer self.unlock();
        c.free(self.worker.queued_user_drain_text);
        self.worker.queued_user_drain_text = owned;
        self.worker.queued_user_drain_answered = true;
        self.signal();
        self.wakeLocked();
    }
    pub fn requestSave(self: Communications) void {
        self.lock();
        defer self.unlock();
        self.worker.save_requested = true;
        self.signal();
        self.wakeLocked();
    }
    pub fn requestCompact(self: Communications) void {
        self.lock();
        defer self.unlock();
        self.worker.compact_requested = true;
        self.signal();
        self.wakeLocked();
    }
    pub fn requestPower(self: Communications, percent: c_int) void {
        self.lock();
        defer self.unlock();
        self.worker.requested_power = percent;
        self.worker.power_requested = true;
        self.worker.status.power_percent = percent;
        self.signal();
        self.wakeLocked();
    }
    pub fn takeSave(self: Communications) bool {
        self.lock();
        defer self.unlock();
        const requested = self.worker.save_requested;
        self.worker.save_requested = false;
        return requested;
    }
    pub fn takeCompact(self: Communications) bool {
        self.lock();
        defer self.unlock();
        const requested = self.worker.compact_requested;
        self.worker.compact_requested = false;
        return requested;
    }
    fn takePower(self: Communications, output: [*c]c_int) bool {
        self.lock();
        defer self.unlock();
        const requested = self.worker.power_requested;
        if (requested) {
            if (output != null) output.* = self.worker.requested_power;
            self.worker.power_requested = false;
        }
        return requested;
    }
    pub fn applyPendingPower(self: Communications) void {
        var percent: c_int = 0;
        if (!self.takePower(&percent)) return;
        if (c.ds4_session_set_power(self.worker.session, percent) != 0) {
            self.publish("\npower change failed\n");
            return;
        }
        self.lock();
        defer self.unlock();
        self.worker.cfg.*.engine.power_percent = percent;
        self.worker.status.power_percent = percent;
        self.wakeLocked();
    }
    pub fn takeRawRestore(self: Communications) bool {
        self.lock();
        defer self.unlock();
        const restore = self.worker.raw_mode_needs_restore;
        if (restore) self.worker.raw_mode_needs_restore = false;
        return restore;
    }
    /// Busy input stays editable in the UI. On acceptance publish PREFILL
    /// before handing off the command; noninteractive EOF must not race the
    /// worker's actual prefill start and mistake this window for idle.
    pub fn submit(self: Communications, value: [*c]const u8) bool {
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        const accepted = worker.initialized and worker.status.state == c.AGENT_WORKER_IDLE and
            worker.cmd_text == null and !worker.think_requested;
        if (accepted) {
            worker.cmd_text = duplicate(value);
            worker.status.state = c.AGENT_WORKER_PREFILL;
            worker.progress_direct = false;
            worker.status.prefill_done = 0;
            worker.status.prefill_total = 0;
            worker.status.prefill_label = c.ds4_agent_comms_next_label();
            worker.status.prefill_tps = 0;
            worker.status.generated = 0;
            worker.status.gen_tps = 0;
            worker.status.greedy_sampling = false;
            self.signal();
        }
        return accepted;
    }
    /// Caller holds mu; pending power takes precedence over current preference.
    fn powerLocked(self: Communications) c_int {
        if (self.worker.power_requested) return self.worker.requested_power;
        const percent = self.worker.cfg.*.engine.power_percent;
        return if (percent > 0) percent else 100;
    }
    pub fn interrupt(self: Communications) void {
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        worker.interrupt = true;
        if (worker.cfg != null and worker.cfg.*.engine.distributed.role == c.DS4_DISTRIBUTED_COORDINATOR and
            (worker.status.state == c.AGENT_WORKER_PREFILL or worker.status.state == c.AGENT_WORKER_GENERATING or
                worker.status.state == c.AGENT_WORKER_COMPACTING))
        {
            worker.status.state = c.AGENT_WORKER_DRAINING;
            worker.status.prefill_tps = 0;
            worker.status.greedy_sampling = false;
            self.wakeLocked();
        }
    }
    pub fn stop(self: Communications) void {
        self.lock();
        defer self.unlock();
        self.worker.stop = true;
        self.signal();
    }
    fn refreshStatusLocked(self: Communications) void {
        self.worker.status.ctx_used = self.worker.transcript.len;
        self.worker.status.ctx_size = self.contextSize();
        self.worker.status.power_percent = self.powerLocked();
    }
    /// Move the output allocation under mu; terminal writes happen after unlock.
    pub fn consume(self: Communications, output: [*c][*c]u8, length: [*c]usize, status: [*c]c.agent_status) void {
        self.lock();
        defer self.unlock();
        const worker = self.worker;
        if (output != null) {
            output.* = worker.out;
            length.* = worker.out_len;
            worker.out = null;
            worker.out_len = 0;
            worker.out_cap = 0;
        }
        self.refreshStatusLocked();
        if (status != null) status.* = worker.status;
        worker.wake_pending = false;
    }
    pub fn getStatus(self: Communications, output: *c.agent_status) void {
        self.lock();
        defer self.unlock();
        self.refreshStatusLocked();
        output.* = self.worker.status;
    }
    pub fn idle(self: Communications) bool {
        self.lock();
        defer self.unlock();
        return self.worker.initialized and !self.worker.think_requested and
            (self.worker.status.state == c.AGENT_WORKER_IDLE or self.worker.status.state == c.AGENT_WORKER_ERROR);
    }
    pub fn initialized(self: Communications, output: [*c]c.agent_status) bool {
        self.lock();
        defer self.unlock();
        self.refreshStatusLocked();
        if (output != null) output.* = self.worker.status;
        return self.worker.initialized;
    }
};

pub fn drainFileDescriptor(descriptor: c_int) void {
    var buffer: [128]u8 = undefined;
    while (true) {
        const count = c.read(descriptor, &buffer, buffer.len);
        if (count > 0) continue;
        if (count < 0 and c.__error().* == c.EINTR) continue;
        return;
    }
}
