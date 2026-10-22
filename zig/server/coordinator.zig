//! Model execution admission and batched decode for persistent server slots.
//! The actual shared runtime records remain the sole state: no Engine/Session
//! layouts, snapshots of counters or per-field callbacks. C owns accept/job
//! routing and thread startup; this owner controls model admission until every
//! in-flight backend call has released its borrowed job/session.
const std = @import("std");
const c = @import("server_runtime_c");
const allocator = std.heap.c_allocator;
const Server = c.server;
const Slot = c.server_slot;
const Stop = *const volatile c.sig_atomic_t;

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn lock(mutex: *c.pthread_mutex_t) void {
    _ = c.pthread_mutex_lock(mutex);
}
fn unlock(mutex: *c.pthread_mutex_t) void {
    _ = c.pthread_mutex_unlock(mutex);
}
fn wake(state: *Server) void {
    _ = c.pthread_cond_broadcast(&state.model_cv);
}
fn cancelled(slot: *const Slot) bool {
    if (slot.running == null) return false;
    const job: *c.job = @ptrCast(slot.running);
    lock(&job.mu);
    defer unlock(&job.mu);
    return job.cancelled;
}
fn errorMessage(destination: []u8, value: []const u8) void {
    if (destination.len == 0) return;
    const count = @min(destination.len - 1, value.len);
    @memcpy(destination[0..count], value[0..count]);
    destination[count] = 0;
}
fn nowSeconds() f64 {
    @setFloatMode(.optimized);
    var value: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &value);
    return @as(f64, @floatFromInt(value.tv_sec)) + @as(f64, @floatFromInt(value.tv_nsec)) * 1e-9;
}
fn addMicroseconds(time: *c.struct_timespec, microseconds: c_long) void {
    if (microseconds <= 0) return;
    time.tv_nsec += @rem(microseconds, 1_000_000) * 1000;
    time.tv_sec += @divTrunc(microseconds, 1_000_000) + @divTrunc(time.tv_nsec, 1_000_000_000);
    time.tv_nsec = @rem(time.tv_nsec, 1_000_000_000);
}
pub fn coalesceMicroseconds() c_long {
    const environment = c.getenv("DS4_SERVER_DECODE_COALESCE_US");
    if (environment != null and environment[0] != 0) {
        var end: [*c]u8 = null;
        const value = c.strtol(environment, &end, 10);
        if (end != environment and end[0] == 0 and value >= 0 and value <= 100_000) return value;
    }
    return 2000;
}

pub const Coordinator = struct {
    state: *Server,

    /// model_mu is held. Fairness is round-robin among waiting prefill slots;
    /// any queued decode has priority before the next bounded prefill quantum.
    pub fn nextPrefill(self: Coordinator) c_int {
        const state = self.state;
        if (state.slot_count <= 0) return -1;
        var offset: c_int = 1;
        while (offset <= state.slot_count) : (offset += 1) {
            const id = @rem(state.last_prefill_slot + offset, state.slot_count);
            if (state.slots[@intCast(id)].prefill_waiting) return id;
        }
        return -1;
    }

    pub fn enterPrefill(self: Coordinator, slot: *Slot, stop_requested: Stop) bool {
        const state = self.state;
        if (cancelled(slot)) return false;
        if (!state.batched_mode) {
            lock(&state.inference_mu);
            if (cancelled(slot)) {
                unlock(&state.inference_mu);
                return false;
            }
            return true;
        }
        lock(&state.model_mu);
        slot.prefill_waiting = true;
        wake(state);
        while (stop_requested.* == 0 and !cancelled(slot) and
            (state.model_busy or state.decode_pending > 0 or self.nextPrefill() != slot.id))
        {
            _ = c.pthread_cond_wait(&state.model_cv, &state.model_mu);
        }
        if (stop_requested.* != 0 or cancelled(slot)) {
            slot.prefill_waiting = false;
            wake(state);
            unlock(&state.model_mu);
            return false;
        }
        slot.prefill_waiting = false;
        state.last_prefill_slot = slot.id;
        state.model_busy = true;
        unlock(&state.model_mu);
        lock(&state.inference_mu);
        return true;
    }
    pub fn leavePrefill(self: Coordinator) void {
        const state = self.state;
        unlock(&state.inference_mu);
        if (!state.batched_mode) return;
        lock(&state.model_mu);
        state.model_busy = false;
        wake(state);
        unlock(&state.model_mu);
    }
    pub fn quantumFor(self: Coordinator, generation_active: bool) c_int {
        const state = self.state;
        var budget: c_int = if (generation_active) state.mixed_prefill_quantum else 2048;
        if (generation_active and budget < 1024 and state.engine != null and c.ds4_engine_is_glm53(state.engine)) budget = 1024;
        return budget;
    }
    pub fn quantum(self: Coordinator) c_int {
        lock(&self.state.model_mu);
        const active = self.state.active_generations > 0;
        unlock(&self.state.model_mu);
        return self.quantumFor(active);
    }
    pub fn generationEnter(self: Coordinator) void {
        if (!self.state.batched_mode) return;
        lock(&self.state.model_mu);
        self.state.active_generations += 1;
        wake(self.state);
        unlock(&self.state.model_mu);
    }
    pub fn generationLeave(self: Coordinator) void {
        if (!self.state.batched_mode) return;
        lock(&self.state.model_mu);
        if (self.state.active_generations > 0) self.state.active_generations -= 1;
        wake(self.state);
        unlock(&self.state.model_mu);
    }

    /// model_mu is held. Pending work can be withdrawn without touching the
    /// backend. An in-flight call still owns the session and stack-owned job.
    pub fn cancelPending(self: Coordinator, slot: *Slot) bool {
        if (!slot.decode_pending or slot.decode_in_flight) return false;
        slot.decode_pending = false;
        self.state.decode_pending -= 1;
        slot.decode_rc = c.DS4_SESSION_SYNC_INTERRUPTED;
        errorMessage(&slot.decode_err, "client disconnected");
        slot.decode_done = true;
        wake(self.state);
        return true;
    }

    pub fn eval(self: Coordinator, slot: *Slot, token: c_int, speculative: bool, accepted: [*c]c_int, accepted_count: *c_int, errors: []u8, stop_requested: Stop) c_int {
        const state = self.state;
        accepted_count.* = 0;
        if (!state.batched_mode) {
            if (stop_requested.* != 0 or cancelled(slot)) {
                errorMessage(errors, if (stop_requested.* != 0) "shutdown requested" else "client disconnected");
                return c.DS4_SESSION_SYNC_INTERRUPTED;
            }
            lock(&state.inference_mu);
            const rc = c.ds4_session_eval(slot.session, token, if (errors.len != 0) errors.ptr else null, errors.len);
            unlock(&state.inference_mu);
            if (rc == 0) {
                accepted[0] = token;
                accepted_count.* = 1;
            }
            return rc;
        }
        lock(&state.model_mu);
        defer unlock(&state.model_mu);
        if (stop_requested.* != 0 or cancelled(slot)) {
            errorMessage(errors, if (stop_requested.* != 0) "shutdown requested" else "client disconnected");
            return c.DS4_SESSION_SYNC_INTERRUPTED;
        }
        if (slot.decode_pending or slot.decode_in_flight) {
            errorMessage(errors, "session already has a decode in flight");
            return 1;
        }
        slot.decode_token = token;
        slot.decode_speculative = speculative;
        slot.decode_accepted_count = 0;
        slot.decode_rc = 1;
        slot.decode_err[0] = 0;
        slot.decode_done = false;
        slot.decode_pending = true;
        state.decode_pending += 1;
        wake(state);
        while (!slot.decode_done) {
            const client_cancelled = cancelled(slot);
            if ((client_cancelled or stop_requested.* != 0) and self.cancelPending(slot)) {
                if (!client_cancelled) errorMessage(&slot.decode_err, "shutdown requested");
                break;
            }
            // Never return while an in-flight backend call can still invoke
            // the stack job's cancellation callback or access this session.
            _ = c.pthread_cond_wait(&state.model_cv, &state.model_mu);
        }
        var rc = slot.decode_rc;
        if (stop_requested.* != 0 and rc == 0) rc = c.DS4_SESSION_SYNC_INTERRUPTED;
        if (rc != 0) errorMessage(errors, if (stop_requested.* != 0) "shutdown requested" else if (slot.decode_err[0] != 0) std.mem.span(@as([*:0]const u8, @ptrCast(&slot.decode_err))) else "decode interrupted");
        slot.decode_done = false;
        if (rc == 0) {
            accepted_count.* = slot.decode_accepted_count;
            @memcpy(accepted[0..@intCast(accepted_count.*)], slot.decode_accepted[0..@intCast(accepted_count.*)]);
        }
        return rc;
    }

    /// Runs in the one decode coordinator thread. Group plain/exact sampling
    /// before greedy speculative requests; both share one model admission turn.
    pub fn run(self: Coordinator) void {
        const state = self.state;
        const capacity: usize = @intCast(state.slot_count);
        const items = allocator.alloc(c.ds4_decode_item, capacity) catch fatal();
        defer allocator.free(items);
        const members = allocator.alloc(*Slot, capacity) catch fatal();
        defer allocator.free(members);
        const accepted = allocator.alloc([2]c_int, capacity) catch fatal();
        defer allocator.free(accepted);
        const accepted_counts = allocator.alloc(c_int, capacity) catch fatal();
        defer allocator.free(accepted_counts);
        const coalesce = coalesceMicroseconds();
        const log_batches = c.getenv("DS4_SERVER_BATCH_LOG") != null;
        lock(&state.model_mu);
        defer unlock(&state.model_mu);
        while (true) {
            while (state.decode_pending == 0 and !state.model_stopping) _ = c.pthread_cond_wait(&state.model_cv, &state.model_mu);
            if (state.decode_pending == 0 and state.model_stopping) break;
            if (coalesce > 0 and state.decode_pending < state.slot_count) {
                var deadline: c.struct_timespec = undefined;
                _ = c.clock_gettime(c.CLOCK_REALTIME, &deadline);
                addMicroseconds(&deadline, coalesce);
                var observed = state.decode_pending;
                while (!state.model_stopping and state.decode_pending < state.slot_count and state.decode_pending < state.active_generations) {
                    const rc = c.pthread_cond_timedwait(&state.model_cv, &state.model_mu, &deadline);
                    if (rc == c.ETIMEDOUT) break;
                    if (state.decode_pending == observed and rc != 0) break;
                    observed = state.decode_pending;
                }
            }
            while (state.model_busy) _ = c.pthread_cond_wait(&state.model_cv, &state.model_mu);
            if (state.model_stopping and state.decode_pending == 0) break;
            var count: usize = 0;
            var plain_count: usize = 0;
            for (0..2) |group| {
                for (0..capacity) |index| {
                    const slot: *Slot = @ptrCast(&state.slots[index]);
                    if (!slot.decode_pending or slot.decode_speculative != (group != 0)) continue;
                    slot.decode_pending = false;
                    slot.decode_in_flight = true;
                    state.decode_pending -= 1;
                    members[count] = slot;
                    items[count] = .{ .session = slot.session, .token = slot.decode_token };
                    count += 1;
                }
                if (group == 0) plain_count = count;
            }
            if (count == 0) continue;
            state.model_busy = true;
            unlock(&state.model_mu);
            var errors: [160]u8 = @splat(0);
            const started = if (log_batches) nowSeconds() else 0;
            lock(&state.inference_mu);
            var rc: c_int = if (plain_count != 0) c.ds4_sessions_eval_batch(items.ptr, @intCast(plain_count), &errors, errors.len) else 0;
            for (0..plain_count) |index| {
                accepted[index][0] = items[index].token;
                accepted_counts[index] = 1;
            }
            if (rc == 0 and count > plain_count)
                rc = c.ds4_sessions_eval_batch_speculative_argmax(items.ptr + plain_count, @intCast(count - plain_count), accepted.ptr + plain_count, accepted_counts.ptr + plain_count, &errors, errors.len);
            if (rc != 0) for (items[0..count]) |item| c.ds4_session_invalidate(item.session);
            unlock(&state.inference_mu);
            if (log_batches) c.ds4_server_coordinator_log_batch(@intCast(count), (nowSeconds() - started) * 1000, rc);
            lock(&state.model_mu);
            state.model_busy = false;
            for (members[0..count], 0..) |slot, index| {
                slot.decode_in_flight = false;
                slot.decode_rc = rc;
                if (rc == 0) {
                    slot.decode_accepted_count = accepted_counts[index];
                    @memcpy(slot.decode_accepted[0..@intCast(accepted_counts[index])], accepted[index][0..@intCast(accepted_counts[index])]);
                } else errorMessage(&slot.decode_err, if (errors[0] != 0) std.mem.span(@as([*:0]const u8, @ptrCast(&errors))) else "batched decode failed");
                slot.decode_done = true;
            }
            wake(state);
        }
    }
    pub fn stop(self: Coordinator) void {
        if (!self.state.batched_mode) return;
        lock(&self.state.model_mu);
        self.state.model_stopping = true;
        wake(self.state);
        unlock(&self.state.model_mu);
    }
};

// C interoperability facade; nullability follows the old private entrypoints.
export fn ds4_server_coordinator_next_prefill(state: ?*const Server) c_int {
    return (Coordinator{ .state = @constCast(state orelse return -1) }).nextPrefill();
}
export fn ds4_server_coordinator_prefill_enter(state: ?*Server, slot: ?*Slot, stop: Stop) bool {
    return (Coordinator{ .state = state orelse return false }).enterPrefill(slot orelse return false, stop);
}
export fn ds4_server_coordinator_prefill_leave(state: ?*Server) void {
    (Coordinator{ .state = state orelse return }).leavePrefill();
}
export fn ds4_server_coordinator_quantum_for(state: *const Server, active: bool) c_int {
    return (Coordinator{ .state = @constCast(state) }).quantumFor(active);
}
export fn ds4_server_coordinator_quantum(state: *Server) c_int {
    return (Coordinator{ .state = state }).quantum();
}
export fn ds4_server_coordinator_generation_enter(state: ?*Server) void {
    (Coordinator{ .state = state orelse return }).generationEnter();
}
export fn ds4_server_coordinator_generation_leave(state: ?*Server) void {
    (Coordinator{ .state = state orelse return }).generationLeave();
}
export fn ds4_server_coordinator_cancel_pending(state: ?*Server, slot: ?*Slot) bool {
    return (Coordinator{ .state = state orelse return false }).cancelPending(slot orelse return false);
}
export fn ds4_server_coordinator_eval(state: ?*Server, slot: ?*Slot, token: c_int, speculative: bool, accepted: [*c]c_int, count: *c_int, errors: [*c]u8, error_length: usize, stop: Stop) c_int {
    count.* = 0;
    const buffer: []u8 = if (errors == null or error_length == 0) &.{} else errors[0..error_length];
    return (Coordinator{ .state = state orelse return 1 }).eval(slot orelse return 1, token, speculative, accepted, count, buffer, stop);
}
export fn ds4_server_coordinator_coalesce_us() c_long {
    return coalesceMicroseconds();
}
export fn ds4_server_coordinator_run(state: *Server) void {
    (Coordinator{ .state = state }).run();
}
export fn ds4_server_coordinator_stop(state: *Server) void {
    (Coordinator{ .state = state }).stop();
}
