//! Interactive and headless fronts for the SAME Worker. The UI owns terminal
//! polling and strings; the worker thread remains the only Session mutator.
const std = @import("std");
const c = @import("agent_runtime_c");
const Ui = @import("ui_text.zig");
const Terminal = @import("terminal.zig");
const lifecycle = @import("lifecycle.zig");
const Lifecycle = lifecycle.Lifecycle;
const Sessions = @import("worker_sessions.zig").Sessions;
const duplicate = @import("worker_sessions.zig").duplicate;
const communications = @import("communications.zig");
const Communications = communications.Communications;
const Commands = @import("commands.zig");
const Presentation = @import("presentation.zig");

fn bytes(pointer: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
fn marker(message: []const u8) void {
    Terminal.write(c.STDERR_FILENO, message);
    Terminal.write(c.STDERR_FILENO, "\n");
}
fn readAvailable(input: *c.agent_input_buf, eof: *bool) c_int {
    var buffer: [4096]u8 = undefined;
    while (true) {
        const count = c.read(c.STDIN_FILENO, &buffer, buffer.len);
        if (count > 0) {
            Ui.append(input, buffer[0..@intCast(count)]);
            continue;
        }
        if (count == 0) {
            eof.* = true;
            return 0;
        }
        const failure = c.__error().*;
        if (failure == c.EINTR) continue;
        if (failure == c.EAGAIN or failure == c.EWOULDBLOCK) return 0;
        c.perror("ds4-agent: read stdin");
        return -1;
    }
}

/// With -p this is one-shot. Otherwise readiness is announced on stderr and
/// bytes arriving within 200 ms form one prompt; later input can queue during
/// generation. This remains the existing small protocol, with no slash parser.
pub fn headless(engine: ?*c.ds4_engine, config: *c.agent_config) c_int {
    var worker: c.agent_worker = undefined;
    if (Lifecycle.init(&worker, engine, config) != 0) return 1;
    defer (Lifecycle{ .worker = &worker }).deinit();
    const comms = Communications{ .worker = &worker };
    const one_shot = config.gen.prompt != null;
    var submitted = false;
    var eof = false;
    var waiting = false;
    var old_flags: c_int = 0;
    var input = std.mem.zeroes(c.agent_input_buf);
    var queue_state = std.mem.zeroes(c.agent_prompt_queue);
    const queue = Ui.Queue{ .state = &queue_state };
    defer {
        Ui.free(&input);
        queue.deinit();
    }
    var quiet_deadline: f64 = 0;
    var result: c_int = 0;
    if (!one_shot and lifecycle.setNonblocking(c.STDIN_FILENO, true, &old_flags) != 0) {
        c.perror("ds4-agent: nonblocking stdin");
        return 1;
    }
    defer if (!one_shot) {
        _ = c.fcntl(c.STDIN_FILENO, c.F_SETFL, old_flags);
    };
    while (true) {
        const initialized = comms.initialized(null);
        var idle = comms.idle();
        if (one_shot and !submitted and initialized) {
            if (comms.submit(config.gen.prompt)) submitted = true;
            idle = false;
        }
        if (!one_shot and queue_state.len != 0 and idle) {
            var queued = queue.takeAll();
            if (comms.submit(queued)) idle = false else {
                queue.pushFront(queued);
                queued = null;
            }
            c.free(queued);
        }
        if (!one_shot and initialized and idle and queue_state.len == 0 and input.len == 0 and !eof and !waiting) {
            marker("+DWARFSTAR_WAITING");
            waiting = true;
        }
        var timeout_ms: c_int = -1;
        if (!one_shot and input.len > 0) {
            const remaining = quiet_deadline - Commands.now();
            timeout_ms = if (remaining <= 0) 0 else @as(c_int, @intFromFloat(remaining * 1000)) + 1;
        }
        var descriptors: [2]c.struct_pollfd = undefined;
        descriptors[0] = .{ .fd = worker.wake_fd[0], .events = c.POLLIN, .revents = 0 };
        const read_stdin = !one_shot and initialized and !eof;
        if (read_stdin) descriptors[1] = .{ .fd = c.STDIN_FILENO, .events = c.POLLIN, .revents = 0 };
        const polled = c.poll(&descriptors, if (read_stdin) 2 else 1, timeout_ms);
        if (polled < 0) {
            if (c.__error().* == c.EINTR) continue;
            c.perror("ds4-agent: poll");
            result = 1;
            break;
        }
        if (descriptors[0].revents & c.POLLIN != 0) communications.drainFileDescriptor(worker.wake_fd[0]);
        if (read_stdin and descriptors[1].revents & (c.POLLIN | c.POLLHUP) != 0) {
            const previous_length = input.len;
            if (readAvailable(&input, &eof) != 0) {
                result = 1;
                break;
            }
            if (input.len != previous_length) {
                quiet_deadline = Commands.now() + 0.200;
                waiting = false;
            }
        }
        var output: [*c]u8 = null;
        var output_length: usize = 0;
        var status = std.mem.zeroes(c.agent_status);
        comms.consume(&output, &output_length, &status);
        if (output != null and output_length != 0) {
            Terminal.write(c.STDOUT_FILENO, bytes(output, output_length));
            _ = c.fflush(c.__stdoutp);
        }
        c.free(output);
        if (comms.takeDrain()) comms.answerDrain(queue.takeAll());
        if (status.state == c.AGENT_WORKER_ERROR) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: %s\n", if (status.@"error"[0] != 0) @as([*c]const u8, &status.@"error") else @as([*c]const u8, "worker error"));
            result = 1;
            break;
        }
        if (!one_shot and input.len != 0 and (eof or Commands.now() >= quiet_deadline)) {
            const prompt = Ui.take(&input);
            defer c.free(prompt);
            if (comms.idle() and queue_state.len == 0) {
                if (!comms.submit(prompt)) {
                    queue.push(prompt);
                    marker("+DWARFSTAR_QUEUED");
                }
            } else {
                queue.push(prompt);
                marker("+DWARFSTAR_QUEUED");
            }
            waiting = false;
        }
        if (one_shot and submitted and comms.idle()) break;
        if (!one_shot and eof and input.len == 0 and queue_state.len == 0 and comms.idle()) break;
    }
    // Drain publication between the final state transition and loop exit.
    var output: [*c]u8 = null;
    var output_length: usize = 0;
    comms.consume(&output, &output_length, null);
    if (output != null and output_length != 0) {
        Terminal.write(c.STDOUT_FILENO, bytes(output, output_length));
        _ = c.fflush(c.__stdoutp);
    }
    c.free(output);
    return result;
}

fn terminalState(status: c.agent_worker_state) bool {
    return status == c.AGENT_WORKER_IDLE or status == c.AGENT_WORKER_ERROR or status == c.AGENT_WORKER_STOPPED;
}
fn same(value: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, Ui.text(value), expected);
}
// /power, /steer and file-prefix commands historically admit space/tab only;
// known-command classification uses locale isspace. Keep that distinction.
fn narrowCommand(value: []const u8, name: []const u8) bool {
    return std.mem.startsWith(u8, value, name) and (value.len == name.len or value[name.len] == ' ' or value[name.len] == '\t');
}
fn argument(command: [*c]u8, name: []const u8) [*c]u8 {
    var value = command + name.len;
    while (value[0] == ' ' or value[0] == '\t') value += 1;
    return value;
}
fn firstWord(value: [*c]u8) [*c]u8 {
    var end = value;
    while (end[0] != 0 and end[0] != ' ' and end[0] != '\t') end += 1;
    end[0] = 0;
    return value;
}

/// The process-wide callback/interrupt storage belongs to the sole facade.
/// This owner only borrows it while its stack Worker and linenoise are alive.
pub const Frontend = struct {
    completion_slot: *?*c.agent_worker,
    completion_callback: ?*const c.linenoiseCompletionCallback,
    interrupt: *volatile c.sig_atomic_t,

    pub fn interactive(self: Frontend, engine: ?*c.ds4_engine, config: *c.agent_config) c_int {
        var worker: c.agent_worker = undefined;
        if (Lifecycle.init(&worker, engine, config) != 0) return 1;
        var state = Interactive{ .worker = &worker, .config = config };
        const home = c.getenv("HOME");
        _ = c.snprintf(&state.history, state.history.len, "%s/.ds4_agent_history", if (Ui.text(home).len != 0) home else @as([*c]const u8, "."));
        // Protected ANSI scroll rows keep streamed output above the real
        // prompt/footer; unsupported terminals use prompt-below-output mode.
        c.linenoiseSetMultiLine(1);
        _ = c.linenoiseHistorySetMaxLen(512);
        _ = c.linenoiseHistoryLoad(&state.history);
        self.completion_slot.* = &worker;
        c.linenoiseSetCompletionCallback(self.completion_callback);
        state.refresh(null, 80);
        if (state.editor().start(&state.prompt, &state.footer, null) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-agent: failed to start line editor\n");
            (Lifecycle{ .worker = &worker }).deinit();
            return 1;
        }
        state.welcome();
        state.initial_pending = if (Ui.text(config.gen.prompt).len != 0) duplicate(Ui.text(config.gen.prompt)) else null;
        while (state.running) {
            const comms = state.comms();
            if (comms.takeRawRestore()) c.linenoiseRestoreRawMode();
            var descriptors = [_]c.struct_pollfd{
                .{ .fd = c.STDIN_FILENO, .events = c.POLLIN, .revents = 0 },
                .{ .fd = worker.wake_fd[0], .events = c.POLLIN, .revents = 0 },
            };
            const queued_input = !state.edit.paste_open and !state.edit.paste_start_pending and c.linenoiseEditQueuedInput(&state.edit.edit) > 0;
            const result = c.poll(&descriptors, descriptors.len, if (queued_input) 0 else 100);
            if (result < 0 and c.__error().* != c.EINTR) break;
            if (self.interrupt.* != 0) {
                self.interrupt.* = 0;
                state.interruptOrCancel();
            }
            if (result > 0 and descriptors[0].revents & c.POLLIN != 0) state.editor().readStdin();
            // Raw Ctrl+C is byte 3. Handle it before draining a potentially
            // large output backlog so a busy stream cannot delay interruption.
            if (state.editor().takeByte(3)) state.interruptOrCancel();
            if (result > 0 and descriptors[1].revents & c.POLLIN != 0) communications.drainFileDescriptor(worker.wake_fd[0]);
            const footer_columns = state.consumeOutput();
            if (comms.takeDrain()) {
                const echo = state.queue().echoAll();
                const queued = state.queue().takeAll();
                if (echo != null) {
                    state.refreshFooter(footer_columns);
                    state.editor().writeAsync(Ui.text(echo), &state.prompt, &state.footer, true);
                    c.free(echo);
                }
                comms.answerDrain(queued);
                continue;
            }
            var approval: [256]u8 = undefined;
            if (comms.takeWeb(&approval)) {
                state.approveWeb(&approval);
                continue;
            }
            if (state.initial_pending != null and comms.idle() and comms.submit(state.initial_pending)) {
                c.free(state.initial_pending);
                state.initial_pending = null;
            }
            if (state.initial_pending == null and state.pending.len != 0 and comms.idle()) state.submitQueued(footer_columns);
            if (state.pending.len != 0 and state.editor().takeByte(24)) { // Ctrl+X
                const queued = state.queue().pop();
                state.editor().replaceInput(queued);
                state.refresh(&state.pending, state.terminalColumns());
                state.editor().setPromptStatus(&state.prompt, &state.footer);
                c.free(queued);
            }
            if (state.pending.len != 0 and !comms.idle() and state.editor().takeEscape()) comms.interrupt();
            if (!state.edit.paste_open and !state.edit.paste_start_pending and c.linenoiseEditQueuedInput(&state.edit.edit) > 0) state.feedLine();
        }
        c.free(state.initial_pending);
        c.free(state.restore_line);
        state.queue().deinit();
        state.editor().stop();
        state.editor().restoreLayout();
        c.linenoiseSetCompletionCallback(null);
        self.completion_slot.* = null;
        if (!state.exit_save_handled and Commands.beforeExiting(&worker) == .now) c.exit(0);
        (Lifecycle{ .worker = &worker }).deinit();
        return 0;
    }
};

const Interactive = struct {
    worker: *c.agent_worker,
    config: *c.agent_config,
    edit: c.agent_editor = std.mem.zeroes(c.agent_editor),
    pending: c.agent_prompt_queue = std.mem.zeroes(c.agent_prompt_queue),
    status: c.agent_status = undefined,
    history: [c.PATH_MAX]u8 = undefined,
    prompt: [160]u8 = undefined,
    footer: [4096]u8 = undefined,
    initial_pending: [*c]u8 = null,
    restore_line: [*c]u8 = null,
    running: bool = true,
    exit_save_handled: bool = false,
    welcome_after_restart: bool = false,
    redraw_after_restart: bool = false,

    fn editor(self: *Interactive) Terminal.Editor {
        return .{ .state = &self.edit };
    }
    fn queue(self: *Interactive) Ui.Queue {
        return .{ .state = &self.pending };
    }
    fn comms(self: *Interactive) Communications {
        return .{ .worker = self.worker };
    }
    fn sessions(self: *Interactive) Sessions {
        return .{ .worker = self.worker };
    }
    fn terminalColumns(self: *Interactive) c_int {
        return if (self.edit.edit.cols > 0) @intCast(self.edit.edit.cols) else 80;
    }
    fn refresh(self: *Interactive, queue_state: ?*const c.agent_prompt_queue, columns: c_int) void {
        self.comms().getStatus(&self.status);
        Ui.copy(&self.prompt, "ds4-agent> ");
        Ui.footer(&self.status, queue_state, columns, &self.footer);
    }
    fn refreshFooter(self: *Interactive, columns: c_int) void {
        Ui.footer(&self.status, &self.pending, columns, &self.footer);
    }
    fn welcome(self: *Interactive) void {
        var banner: [256]u8 = undefined;
        Ui.welcome(self.config, &banner);
        self.editor().writeAsync(Ui.text(&banner), &self.prompt, &self.footer, true);
    }
    fn interruptOrCancel(self: *Interactive) void {
        if (self.comms().idle()) self.editor().cancelInput(&self.prompt, &self.footer) else self.comms().interrupt();
    }
    fn consumeOutput(self: *Interactive) c_int {
        var output: [*c]u8 = null;
        var length: usize = 0;
        self.comms().consume(&output, &length, &self.status);
        defer c.free(output);
        Ui.copy(&self.prompt, "ds4-agent> ");
        const footer_columns = self.terminalColumns();
        self.refreshFooter(footer_columns);
        const terminal = terminalState(self.status.state);
        if ((output != null and length != 0) or self.edit.deferred_output.len != 0) self.editor().writeAsync(bytes(output, length), &self.prompt, &self.footer, terminal) else {
            self.editor().setPromptStatus(&self.prompt, &self.footer);
            self.editor().flush(terminal);
            if (self.edit.hidden and terminal) self.editor().show();
        }
        if (self.status.state == c.AGENT_WORKER_ERROR and self.status.@"error"[0] != 0) {
            var message: [320]u8 = undefined;
            const count = c.snprintf(&message, message.len, "\nds4-agent: %s\n", &self.status.@"error");
            self.editor().writeAsync(message[0..if (count > 0) @intCast(count) else 0], &self.prompt, &self.footer, true);
            _ = c.pthread_mutex_lock(&self.worker.mu);
            self.worker.status.state = c.AGENT_WORKER_IDLE;
            self.worker.status.prefill_tps = 0;
            self.worker.status.greedy_sampling = false;
            self.worker.status.@"error"[0] = 0;
            _ = c.pthread_mutex_unlock(&self.worker.mu);
        }
        return footer_columns;
    }
    fn approveWeb(self: *Interactive, message: [*c]const u8) void {
        const saved = if (self.edit.active and self.edit.edit.buf != null and self.edit.edit.len != 0) duplicate(bytes(self.edit.edit.buf, self.edit.edit.len)) else null;
        defer c.free(saved);
        self.editor().stop();
        self.editor().restoreLayout();
        var timed_out = false;
        const accepted = Commands.yesNo(message, .{ .timeout_seconds = 30, .automatic = .no }, &timed_out);
        self.comms().answerWeb(accepted, if (timed_out) "Chrome browser start approval timed out" else null);
        self.refresh(&self.pending, self.terminalColumns());
        _ = self.editor().start(&self.prompt, &self.footer, saved);
    }
    fn submitQueued(self: *Interactive, footer_columns: c_int) void {
        const echo = self.queue().echoAll();
        defer c.free(echo);
        var queued = self.queue().takeAll();
        defer c.free(queued);
        if (self.comms().submit(queued)) {
            _ = c.linenoiseHistoryAdd(queued);
            _ = c.linenoiseHistorySave(&self.history);
            self.refreshFooter(footer_columns);
            if (echo != null) self.editor().writeAsync(Ui.text(echo), &self.prompt, &self.footer, true);
        } else {
            self.queue().pushFront(queued);
            queued = null;
        }
    }
    fn feedLine(self: *Interactive) void {
        if (self.edit.hidden) {
            // A real key during a partial generated line must expose a prompt
            // on a new line rather than repainting over the generated text.
            self.editor().show();
        }
        c.__error().* = 0;
        const line = c.linenoiseEditFeed(&self.edit.edit);
        if (line == c.linenoiseEditMore) return;
        if (line == null) {
            if (c.__error().* == c.EAGAIN) self.interruptOrCancel() else self.running = false;
            return;
        }
        defer c.linenoiseFree(line);
        const trimmed = std.mem.trim(u8, Ui.text(line), " \t\r\n");
        // linenoise owns the allocation; only trim the borrowed command view.
        const command: [*c]u8 = @constCast(trimmed.ptr);
        command[trimmed.len] = 0;
        const was_below = self.edit.prompt_below_output;
        const had_open_line = self.edit.output_line_open;
        const previous_column = self.edit.output_col;
        self.editor().stop();
        self.dispatch(command, !self.comms().idle());
        if (self.running) {
            self.refresh(&self.pending, self.terminalColumns());
            _ = self.editor().start(&self.prompt, &self.footer, self.restore_line);
            if (!self.edit.scroll_region and was_below) {
                self.edit.output_line_open = had_open_line;
                self.edit.prompt_below_output = was_below;
                self.edit.output_col = previous_column;
            }
            if (self.welcome_after_restart) {
                self.welcome();
                self.welcome_after_restart = false;
            }
            if (self.redraw_after_restart) {
                self.editor().writeAsync("", &self.prompt, &self.footer, true);
                self.redraw_after_restart = false;
            }
            c.free(self.restore_line);
            self.restore_line = null;
        }
    }
    fn dispatch(self: *Interactive, command: [*c]u8, busy: bool) void {
        const value = Ui.text(command);
        if (value.len == 0) return;
        if (same(command, "/help")) {
            Commands.help();
        } else if (same(command, "/save")) {
            if (busy) {
                self.comms().requestSave();
                _ = c.printf("save scheduled at next safe point\n");
            } else {
                var errors: [160]u8 = @splat(0);
                if (!self.sessions().save(&errors)) _ = c.printf("save failed: %s\n", &errors);
            }
        } else if (same(command, "/compact")) {
            self.comms().requestCompact();
            if (busy) _ = c.printf("compaction scheduled at next safe point\n");
        } else if (same(command, "/list")) {
            c.ds4_agent_sessions_list(self.worker.cache_dir, @intCast(c.ds4_engine_model_id(self.worker.engine)), Presentation.columns());
        } else if (narrowCommand(value, "/power")) {
            const arg = argument(command, "/power");
            var percent: c_int = 0;
            if (arg[0] == 0 or !Commands.power(arg, &percent)) _ = c.printf("usage: /power <1..100>\n") else self.comms().requestPower(percent);
        } else if (Commands.withArguments(value, "/think")) {
            const arg = argument(command, "/think");
            var mode: c.ds4_think_mode = c.DS4_THINK_HIGH;
            if (!c.ds4_engine_is_deepseek41(self.worker.engine) or (arg[0] != 0 and !c.ds4_think_mode_parse_level(arg, &mode))) {
                _ = c.printf("usage: /think [0..100] (V4.1 only)\n");
            } else {
                _ = c.pthread_mutex_lock(&self.worker.mu);
                self.worker.requested_think = mode;
                self.worker.think_requested = true;
                _ = c.pthread_cond_signal(&self.worker.cond);
                self.comms().wakeLocked();
                _ = c.pthread_mutex_unlock(&self.worker.mu);
                if (busy) _ = c.printf("thinking change scheduled after this turn\n");
            }
        } else if (Commands.withArguments(value, "/hints")) {
            var enabled: bool = undefined;
            if (!Commands.hints(argument(command, "/hints"), &enabled)) _ = c.printf("usage: /hints on|off\n") else {
                _ = c.pthread_mutex_lock(&self.worker.mu);
                self.worker.hints.enabled = enabled;
                _ = c.pthread_mutex_unlock(&self.worker.mu);
                _ = c.printf("hints %s (applies at the next conversation boundary)\n", @as([*:0]const u8, if (enabled) "on" else "off"));
            }
        } else if (narrowCommand(value, "/steer")) {
            if (busy) _ = c.printf("command requires the model to be idle: %s\n", command) else {
                const arg = argument(command, "/steer");
                if (arg[0] == 0) _ = c.printf("Steering FFN: %g.\n", @as(f64, c.ds4_session_directional_steering_ffn(self.worker.session))) else {
                    var scale: f32 = 0;
                    if (!Commands.steering(arg, &scale)) _ = c.printf("usage: /steer <-100..100>\n") else if (c.ds4_session_set_directional_steering_ffn(self.worker.session, scale) == 0) {
                        self.worker.cfg.*.engine.directional_steering_ffn = scale;
                        _ = c.printf("Steering FFN: %g.\n", @as(f64, scale));
                    }
                }
            }
        } else if (command[0] == '/' and !Commands.known(value)) {
            _ = c.write(c.STDOUT_FILENO, "\x07", 1);
            self.restore_line = duplicate(value);
        } else if (command[0] == '/' and busy) {
            _ = c.printf("command requires the model to be idle: %s\n", command);
        } else if (same(command, "/quit") or same(command, "/exit")) {
            // Raw mode, nonblocking stdin and scroll margins must be restored
            // before the save question; .now deliberately exits immediately.
            self.editor().stop();
            self.editor().restoreLayout();
            switch (Commands.beforeExiting(self.worker)) {
                .now => c.exit(0),
                .clean => {
                    self.exit_save_handled = true;
                    self.running = false;
                },
                .cancel => {
                    _ = self.editor().start(&self.prompt, &self.footer, null);
                },
            }
        } else if (same(command, "/new")) {
            self.editor().restoreLayout();
            if (Commands.beforeLeaving(self.worker)) {
                var errors: [160]u8 = @splat(0);
                if (!self.sessions().reset(&errors)) _ = c.printf("new session failed: %s\n", &errors) else self.welcome_after_restart = true;
            }
        } else if (narrowCommand(value, "/switch")) {
            const arg = argument(command, "/switch");
            if (arg[0] == 0) _ = c.printf("usage: /switch <sha-prefix>\n") else {
                self.editor().restoreLayout();
                if (Commands.beforeLeaving(self.worker)) {
                    var errors: [160]u8 = @splat(0);
                    if (!self.sessions().switchTo(firstWord(arg), c.DS4_AGENT_HISTORY_DEFAULT_TURNS, &errors)) _ = c.printf("switch failed: %s\n", &errors) else self.redraw_after_restart = true;
                }
            }
        } else if (narrowCommand(value, "/del")) {
            const arg = argument(command, "/del");
            if (arg[0] == 0) _ = c.printf("usage: /del <sha-prefix>\n") else {
                var sha: [41]u8 = @splat(0);
                var errors: [160]u8 = @splat(0);
                if (c.ds4_agent_sessions_delete(self.worker.cache_dir, @intCast(c.ds4_engine_model_id(self.worker.engine)), firstWord(arg), &sha, &errors, errors.len)) _ = c.printf("deleted session %.8s\n", &sha) else _ = c.printf("delete failed: %s\n", &errors);
            }
        } else if (narrowCommand(value, "/strip")) {
            const arg = argument(command, "/strip");
            if (arg[0] == 0) _ = c.printf("usage: /strip <sha-prefix>\n") else {
                var sha: [41]u8 = @splat(0);
                var errors: [160]u8 = @splat(0);
                var tokens: u32 = 0;
                if (c.ds4_agent_sessions_strip(self.worker.cache_dir, @intCast(c.ds4_engine_model_id(self.worker.engine)), self.worker.engine, firstWord(arg), &sha, &tokens, &errors, errors.len)) _ = c.printf("stripped session %.8s (%u tokens)\n", &sha, tokens) else _ = c.printf("strip failed: %s\n", &errors);
            }
        } else if (narrowCommand(value, "/history")) {
            const arg = argument(command, "/history");
            const turns = if (arg[0] == 0) c.DS4_AGENT_HISTORY_DEFAULT_TURNS else Commands.integerDefault(arg, c.DS4_AGENT_HISTORY_DEFAULT_TURNS, 1, c.DS4_AGENT_HISTORY_MAX_TURNS);
            var errors: [160]u8 = @splat(0);
            if (!self.sessions().showHistory(turns, &errors)) _ = c.printf("history failed: %s\n", &errors);
        } else if (busy) {
            self.queue().push(command);
        } else {
            _ = c.linenoiseHistoryAdd(command);
            _ = c.linenoiseHistorySave(&self.history);
            if (self.comms().submit(command)) {
                const echo = Ui.echo(command);
                defer c.free(echo);
                _ = c.printf("%s", echo);
                _ = c.fflush(c.__stdoutp);
            } else self.restore_line = duplicate(value);
        }
    }
};
