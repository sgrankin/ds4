//! The reader never draws: it only queues navigation under a mutex. Rendering
//! stays on the main thread. Keep ISIG so Ctrl-C can repair the alternate screen.
const std = @import("std");
const c = @import("eval_c");
pub const Keys = struct { move: c_int, enter: bool, pause: bool, quit: bool };
pub const Input = struct {
    enabled: bool = false,
    raw_mode: bool = false,
    thread_started: bool = false,
    initialized: bool = false,
    running: c.sig_atomic_t = 0,
    thread: c.pthread_t = undefined,
    mutex: c.pthread_mutex_t = undefined,
    original: c.struct_termios = undefined,
    pending: Keys = .{ .move = 0, .enter = false, .pause = false, .quit = false },

    fn setRunning(self: *Input, value: c.sig_atomic_t) void {
        @as(*volatile c.sig_atomic_t, &self.running).* = value;
    }
    fn isRunning(self: *Input) bool {
        return @as(*volatile c.sig_atomic_t, &self.running).* != 0;
    }
    fn queue(self: *Input, key: u8) void {
        _ = c.pthread_mutex_lock(&self.mutex);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        switch (key) {
            'A' => self.pending.move -= 1,
            'B' => self.pending.move += 1,
            '\r', '\n' => self.pending.enter = true,
            'p', 'P' => self.pending.pause = true,
            'q', 'Q' => self.pending.quit = true,
            else => {},
        }
    }
    fn reader(context: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Input = @ptrCast(@alignCast(context.?));
        var escape: u2 = 0;
        var bytes: [64]u8 = undefined;
        while (self.isRunning()) {
            const count = c.read(c.STDIN_FILENO, &bytes, bytes.len);
            if (count <= 0) {
                _ = c.usleep(5000);
                continue;
            }
            for (bytes[0..@intCast(count)]) |byte| {
                switch (escape) {
                    0 => if (byte == 27) {
                        escape = 1;
                    } else if (byte == 'p' or byte == 'P' or byte == 'q' or byte == 'Q' or byte == '\r' or byte == '\n') self.queue(byte),
                    1 => escape = if (byte == '[' or byte == 'O') 2 else 0,
                    else => {
                        if (byte == 'A' or byte == 'B') self.queue(byte);
                        escape = 0;
                    },
                }
            }
        }
        return null;
    }
    pub fn start(self: *Input) void {
        if (c.isatty(c.STDIN_FILENO) == 0) return;
        if (c.tcgetattr(c.STDIN_FILENO, &self.original) != 0) return;
        if (!self.initialized) {
            _ = c.pthread_mutex_init(&self.mutex, null);
            self.initialized = true;
        }
        var raw = self.original;
        raw.c_lflag &= ~@as(c.tcflag_t, c.ICANON | c.ECHO);
        raw.c_cc[c.VMIN] = 0;
        raw.c_cc[c.VTIME] = 1;
        if (c.tcsetattr(c.STDIN_FILENO, c.TCSANOW, &raw) != 0) return;
        _ = c.pthread_mutex_lock(&self.mutex);
        self.pending = .{ .move = 0, .enter = false, .pause = false, .quit = false };
        _ = c.pthread_mutex_unlock(&self.mutex);
        self.raw_mode = true;
        self.enabled = true;
        self.setRunning(1);
        if (c.pthread_create(&self.thread, null, reader, self) == 0) {
            self.thread_started = true;
        } else {
            self.setRunning(0);
            self.enabled = false;
            _ = c.tcsetattr(c.STDIN_FILENO, c.TCSANOW, &self.original);
            self.raw_mode = false;
        }
    }
    pub fn stop(self: *Input) void {
        if (self.thread_started) {
            self.setRunning(0);
            _ = c.pthread_join(self.thread, null);
            self.thread_started = false;
        }
        if (self.raw_mode) {
            _ = c.tcsetattr(c.STDIN_FILENO, c.TCSANOW, &self.original);
            self.raw_mode = false;
        }
        self.enabled = false;
    }
    /// Signal path cannot join a thread or use stdio. The caller writes the
    /// terminal repair bytes and restores the default signal disposition.
    pub fn signalStop(self: *Input) void {
        self.setRunning(0);
        if (self.raw_mode) {
            _ = c.tcsetattr(c.STDIN_FILENO, c.TCSANOW, &self.original);
            self.raw_mode = false;
        }
    }
    pub fn take(self: *Input) Keys {
        _ = c.pthread_mutex_lock(&self.mutex);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        const keys = self.pending;
        self.pending = .{ .move = 0, .enter = false, .pause = false, .quit = false };
        return keys;
    }
};
