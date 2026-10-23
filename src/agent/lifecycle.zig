//! Worker startup/shutdown and the sole Session-mutating command thread.
//! UI submits plain text and requests under the actual Worker mutex; direct
//! native owners perform turns, persistence, compaction and preference changes.
const std = @import("std");
const c = @import("agent_runtime_c");
const sessions = @import("worker_sessions.zig");
const Sessions = sessions.Sessions;
const Summary = @import("summary.zig").Summary;
const Turn = @import("turn.zig").Turn;
const Trace = @import("trace.zig").Trace;

fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
pub fn setNonblocking(fd: c_int, enabled: bool, old: [*c]c_int) c_int {
    const flags = c.fcntl(fd, c.F_GETFL, @as(c_int, 0));
    if (flags < 0) return -1;
    if (old != null) old.* = flags;
    return c.fcntl(fd, c.F_SETFL, if (enabled) flags | c.O_NONBLOCK else flags & ~@as(c_int, c.O_NONBLOCK));
}
fn defaultCacheDirectory() [*c]u8 {
    const override = c.getenv("DS4_AGENT_CACHE_DIR");
    if (sessions.text(override).len != 0) return sessions.duplicate(sessions.text(override));
    const home = c.getenv("HOME");
    return c.ds4_kvstore_path_join(if (sessions.text(home).len != 0) home else @as([*c]const u8, "."), ".ds4/kvcache");
}
fn webLog(context: ?*anyopaque, message: [*c]const u8) callconv(.c) void {
    if (context == null or sessions.text(message).len == 0) return;
    (Trace{ .worker = @ptrCast(@alignCast(context.?)) }).log("web: %s", .{message});
}

pub const Lifecycle = struct {
    worker: *c.agent_worker,

    fn trace(self: Lifecycle) Trace {
        return .{ .worker = self.worker };
    }
    fn sessionActions(self: Lifecycle) Sessions {
        return .{ .worker = self.worker };
    }
    fn shouldStop(self: Lifecycle) bool {
        _ = c.pthread_mutex_lock(&self.worker.mu);
        defer _ = c.pthread_mutex_unlock(&self.worker.mu);
        return self.worker.stop;
    }
    pub fn waitRoute(self: Lifecycle, errors: []u8) bool {
        const worker = self.worker;
        if (worker.cfg == null or worker.cfg.*.engine.distributed.role != c.DS4_DISTRIBUTED_COORDINATOR) return true;
        var previous: [160]u8 = @splat(0);
        var ticks: c_uint = 0;
        const delay = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 250000000 };
        while (true) {
            const ready = c.ds4_session_distributed_route_ready(worker.session, errors.ptr, errors.len);
            if (ready > 0) {
                if (ticks != 0) {
                    if (worker.cfg.*.non_interactive) _ = c.fprintf(c.__stderrp, "ds4-agent: distributed route ready\n") else c.ds4_agent_turn_system_status(worker, "Distributed route ready.");
                }
                if (errors.len != 0) errors[0] = 0;
                return true;
            }
            if (ready < 0) return false;
            const reason: [*c]const u8 = if (errors.len != 0 and errors[0] != 0) errors.ptr else "route incomplete";
            if (!std.mem.eql(u8, sessions.text(&previous), sessions.text(reason)) or ticks % 20 == 0) {
                if (worker.cfg.*.non_interactive) _ = c.fprintf(c.__stderrp, "ds4-agent: waiting for distributed route: %s\n", reason) else {
                    var message: [224]u8 = undefined;
                    _ = c.snprintf(&message, message.len, "Waiting for distributed route: %s", reason);
                    c.ds4_agent_turn_system_status(worker, &message);
                }
                _ = c.snprintf(&previous, previous.len, "%s", reason);
            }
            if (self.shouldStop()) {
                sessions.report(errors, "agent stopped while waiting for distributed route", .{});
                return false;
            }
            _ = c.nanosleep(&delay, null);
            ticks +%= 1;
        }
    }
    pub fn deferredSave(self: Lifecycle) void {
        const worker = self.worker;
        if (!c.ds4_agent_comms_take_save(worker)) return;
        c.ds4_agent_comms_status(worker, c.AGENT_WORKER_SAVING);
        var errors: [160]u8 = @splat(0);
        var sha: [41]u8 = undefined;
        var count: c_int = 0;
        if (self.sessionActions().saveNow(&sha, &count, &errors)) sessions.print(worker, "\nsaved session %.8s (%d tokens)\n", .{ &sha, count }) else sessions.print(worker, "\nsave failed: %s\n", .{if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "unknown error")});
        c.ds4_agent_comms_status(worker, c.AGENT_WORKER_IDLE);
    }
    pub fn deferredCompact(self: Lifecycle) void {
        const worker = self.worker;
        if (!c.ds4_agent_comms_take_compact(worker)) return;
        if (!self.sessionActions().hasUser()) {
            sessions.emit(worker, "\ncompact skipped: nothing to compact\n");
            return;
        }
        const before = worker.transcript.len;
        var errors: [160]u8 = @splat(0);
        if ((Summary{ .worker = worker }).compactAndExport("user requested compaction", &errors)) {
            if (worker.transcript.len != before) {
                _ = c.pthread_mutex_lock(&worker.mu);
                worker.session_dirty = true;
                c.ds4_agent_comms_wake_locked(worker);
                _ = c.pthread_mutex_unlock(&worker.mu);
            } else sessions.emit(worker, "\ncompact skipped: nothing to compact\n");
            c.ds4_agent_comms_status(worker, c.AGENT_WORKER_IDLE);
        } else if (std.mem.eql(u8, sessions.text(&errors), "interrupted")) {
            c.ds4_agent_comms_clear_interrupt(worker);
            c.ds4_agent_comms_status(worker, c.AGENT_WORKER_IDLE);
        } else c.ds4_agent_comms_error(worker, if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "context compaction failed"));
    }
    /// UI requests are consumed in their original priority order. The command
    /// allocation transfers out of Worker under the lock and is freed only
    /// after its whole turn returns; deferred actions then see a stable frontier.
    pub fn run(self: Lifecycle) void {
        const worker = self.worker;
        const config: *c.agent_config = @ptrCast(worker.cfg);
        self.trace().log("agent worker start ctx=%d backend=%s model=%s trace=%s", .{ sessions.contextSize(worker), c.ds4_backend_name(config.engine.backend), if (config.engine.model_path != null) config.engine.model_path else @as([*c]const u8, ""), if (config.gen.trace_path != null) config.gen.trace_path else @as([*c]const u8, "") });
        var errors: [160]u8 = @splat(0);
        var initialized = self.waitRoute(&errors);
        if (initialized and !config.gen.raw_prompt) initialized = self.sessionActions().reset(&errors);
        if (!initialized) c.ds4_agent_comms_error(worker, if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "failed to initialize system prompt"));
        if (config.gen.raw_prompt) self.trace().log("raw one-shot prompt mode: skipping agent system prompt", .{}) else self.trace().tokens("initial_system_prompt", &worker.transcript, 0);
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.initialized = true;
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        while (true) {
            _ = c.pthread_mutex_lock(&worker.mu);
            while (!worker.stop and worker.cmd_text == null and !worker.save_requested and !worker.compact_requested and !worker.power_requested and !worker.think_requested)
                _ = c.pthread_cond_wait(&worker.cond, &worker.mu);
            if (worker.stop) {
                _ = c.pthread_mutex_unlock(&worker.mu);
                break;
            }
            if (worker.power_requested) {
                _ = c.pthread_mutex_unlock(&worker.mu);
                c.ds4_agent_comms_apply_power(worker);
                continue;
            }
            if (worker.think_requested) {
                worker.status.state = c.AGENT_WORKER_PREFILL;
                _ = c.pthread_mutex_unlock(&worker.mu);
                self.sessionActions().applyThinking();
                _ = c.pthread_mutex_lock(&worker.mu);
                worker.status.state = if (worker.cmd_text != null) c.AGENT_WORKER_PREFILL else c.AGENT_WORKER_IDLE;
                c.ds4_agent_comms_wake_locked(worker);
                _ = c.pthread_mutex_unlock(&worker.mu);
                continue;
            }
            if (worker.cmd_text == null and worker.save_requested) {
                _ = c.pthread_mutex_unlock(&worker.mu);
                self.deferredSave();
                continue;
            }
            if (worker.cmd_text == null and worker.compact_requested) {
                _ = c.pthread_mutex_unlock(&worker.mu);
                self.deferredCompact();
                continue;
            }
            const command = worker.cmd_text;
            worker.cmd_text = null;
            _ = c.pthread_mutex_unlock(&worker.mu);
            if (config.gen.raw_prompt) _ = (Turn{ .worker = worker }).raw(command) else _ = (Turn{ .worker = worker }).run(command);
            c.free(command);
            c.ds4_agent_comms_apply_power(worker);
            self.deferredCompact();
            self.deferredSave();
        }
        c.ds4_agent_comms_status(worker, c.AGENT_WORKER_STOPPED);
    }
    /// Initialize synchronization/cache/Web/session resources before launching
    /// the single mutating thread. Preserve the existing failure return policy;
    /// successful owners are released by deinit after joining the thread.
    pub fn init(worker: *c.agent_worker, engine: ?*c.ds4_engine, config: *c.agent_config) c_int {
        worker.* = zero(c.agent_worker);
        worker.engine = engine;
        worker.cfg = config;
        worker.wake_fd = .{ -1, -1 };
        _ = c.pthread_mutex_init(&worker.mu, null);
        _ = c.pthread_cond_init(&worker.cond, null);
        worker.status.state = c.AGENT_WORKER_IDLE;
        if (c.pipe(&worker.wake_fd) != 0) return -1;
        var previous_flags: c_int = undefined;
        _ = setNonblocking(worker.wake_fd[0], true, &previous_flags);
        _ = setNonblocking(worker.wake_fd[1], true, &previous_flags);
        if (c.ds4_session_create(&worker.session, engine, config.gen.ctx_size) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: session backend is required\n");
            return -1;
        }
        worker.cache_dir = defaultCacheDirectory();
        if (!sessions.makeDirectory(worker.cache_dir)) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: failed to create %s: %s\n", worker.cache_dir, c.strerror(c.__error().*));
            return -1;
        }
        const web = c.ds4_web_config{
            .home_dir = c.getenv("HOME"),
            .port = 9333,
            .confirm = c.ds4_agent_comms_web_confirm,
            .confirm_privdata = worker,
            .log = webLog,
            .log_privdata = worker,
            .cancel = c.ds4_agent_comms_cancel,
            .cancel_privdata = worker,
        };
        worker.tools.web = c.ds4_web_create(&web);
        worker.sysprompt_path = c.ds4_kvstore_path_join(worker.cache_dir, "sysprompt.kv");
        if (sessions.text(config.gen.trace_path).len != 0) {
            worker.trace = c.fopen(config.gen.trace_path, "ab");
            if (worker.trace == null) {
                _ = c.fprintf(c.__stderrp, "ds4-agent: failed to open trace %s: %s\n", config.gen.trace_path, c.strerror(c.__error().*));
                return -1;
            }
        }
        var cache_errors: [160]u8 = @splat(0);
        worker.state_cache = c.ds4_state_cache_runtime_open(engine, &config.engine, sessions.contextSize(worker), &worker.state_cache_profile, &cache_errors, cache_errors.len);
        if (cache_errors[0] != 0) _ = c.fprintf(c.__stderrp, "ds4-agent: state cache disabled: %s\n", &cache_errors);
        // Native Debug prefill frames exceed Darwin's small default pthread
        // stack. Use an explicit inference stack; actual graph ownership and
        // algorithms are unchanged. Retire attributes after either outcome.
        var attributes: c.pthread_attr_t = undefined;
        if (c.pthread_attr_init(&attributes) != 0) return -1;
        defer _ = c.pthread_attr_destroy(&attributes);
        if (c.pthread_attr_setstacksize(&attributes, 8 * 1024 * 1024) != 0) return -1;
        if (c.pthread_create(&worker.thread, &attributes, thread, worker) != 0) {
            _ = c.ds4_state_cache_close(worker.state_cache, null, 0);
            worker.state_cache = null;
            return -1;
        }
        return 0;
    }
    /// Join before freeing any model/session/tool/trace owner reachable by the
    /// command thread or its process monitors. Stop is the existing wake action.
    pub fn deinit(self: Lifecycle) void {
        const worker = self.worker;
        c.ds4_agent_comms_stop(worker);
        if (worker.thread != null) _ = c.pthread_join(worker.thread, null);
        c.ds4_agent_executor_jobs_free(&worker.tools);
        c.ds4_web_free(worker.tools.web);
        c.ds4_agent_runtime_frontiers_clear(worker);
        _ = c.ds4_state_cache_close(worker.state_cache, null, 0);
        worker.state_cache = null;
        c.ds4_session_free(worker.session);
        c.ds4_agent_runtime_context_clear(worker);
        c.ds4_tokens_free(&worker.transcript);
        c.ds4_agent_runtime_images_clear(worker);
        c.free(worker.cache_dir);
        c.free(worker.sysprompt_path);
        c.free(worker.identity.title);
        c.free(worker.identity.legacy_path);
        c.free(worker.queued_user_drain_text);
        for (worker.wake_fd) |fd| if (fd >= 0) {
            _ = c.close(fd);
        };
        if (worker.trace != null) _ = c.fclose(worker.trace);
        c.free(worker.cmd_text);
        c.free(worker.out);
        _ = c.pthread_cond_destroy(&worker.cond);
        _ = c.pthread_mutex_destroy(&worker.mu);
    }
};

pub fn thread(context: ?*anyopaque) callconv(.c) ?*anyopaque {
    (Lifecycle{ .worker = @ptrCast(@alignCast(context.?)) }).run();
    return null;
}
