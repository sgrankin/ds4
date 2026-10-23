//! Optional visible-browser library. One actual private Web record owns its
//! profile/configuration; request sockets and disposable tabs are scoped values.
//! Retirement never kills Chrome: its profile/window is user-visible state.
const std = @import("std");
const common = @import("web_common.zig");
const c = common.c;
const allocator = common.allocator;
const Buffer = common.Buffer;
const Errors = common.Errors;
const Connection = @import("web_transport.zig").Connection;
const http = @import("web_transport.zig").http;
const Web = c.ds4_web;
const Tab = c.web_tab;

pub fn create(options: ?*const c.ds4_web_config) *Web {
    const state = allocator.create(Web) catch common.fatal("ds4_web: malloc");
    state.* = std.mem.zeroes(Web);
    var home: [*c]const u8 = c.getenv("HOME");
    if (options) |cfg| if (cfg.home_dir != null and cfg.home_dir[0] != 0) {
        home = cfg.home_dir;
    };
    if (home == null or home[0] == 0) home = ".";
    _ = c.snprintf(&state.home, state.home.len, "%s", home);
    _ = c.snprintf(&state.profile_dir, state.profile_dir.len, "%s/.ds4/browser", home);
    state.port = if (options != null and options.?.port > 0) options.?.port else c.DS4_WEB_DEFAULT_PORT;
    state.next_cdp_id = 1;
    if (options) |cfg| {
        state.confirm = cfg.confirm;
        state.confirm_privdata = cfg.confirm_privdata;
        state.log = cfg.log;
        state.log_privdata = cfg.log_privdata;
        state.cancel = cfg.cancel;
        state.cancel_privdata = cfg.cancel_privdata;
    }
    return state;
}
pub fn free(state: ?*Web) void {
    // Do not kill Chrome. The browser profile is user-visible state; leaving
    // it alive keeps repeated web calls cheaper and avoids disturbing it.
    if (state) |owner| allocator.destroy(owner);
}
pub fn freeTab(tab: *Tab) void {
    c.free(tab.id);
    c.free(tab.ws_url);
    tab.id = null;
    tab.ws_url = null;
}
fn mkdir(path: [*:0]const u8) bool {
    if (path[0] == 0) return false;
    var copy: [c.PATH_MAX]u8 = undefined;
    _ = c.snprintf(&copy, copy.len, "%s", path);
    const length = common.text(&copy).len;
    for (1..length) |index| if (copy[index] == '/') {
        copy[index] = 0;
        if (c.mkdir(&copy, 0o700) != 0 and c.__error().* != c.EEXIST) return false;
        copy[index] = '/';
    };
    return c.mkdir(&copy, 0o700) == 0 or c.__error().* == c.EEXIST;
}
fn chrome() [:0]u8 {
    const configured = c.getenv("DS4_CHROME");
    if (configured != null and configured[0] != 0) return common.duplicate(common.text(configured));
    const paths = [_][*:0]const u8{
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
        "/opt/google/chrome/chrome",
    };
    for (paths) |path| if (c.access(path, c.X_OK) == 0) {
        return common.duplicate(common.text(path));
    };
    const path = c.getenv("PATH");
    if (path != null) {
        var directories = std.mem.tokenizeScalar(u8, common.text(path), ':');
        while (directories.next()) |directory| {
            inline for (.{ "google-chrome", "google-chrome-stable", "chromium", "chromium-browser" }) |name| {
                var candidate: [c.PATH_MAX]u8 = undefined;
                _ = c.snprintf(&candidate, candidate.len, "%.*s/%s", @as(c_int, @intCast(directory.len)), directory.ptr, @as([*:0]const u8, name));
                if (c.access(&candidate, c.X_OK) == 0) return common.duplicate(common.text(&candidate));
            }
        }
    }
    return common.duplicate("google-chrome");
}
fn application() ?[*:0]const u8 {
    if (c.getenv("DS4_CHROME") != null) return null;
    if (c.access("/Applications/Google Chrome.app", c.F_OK) == 0) return "Google Chrome";
    if (c.access("/Applications/Chromium.app", c.F_OK) == 0) return "Chromium";
    return null;
}
pub const Browser = struct {
    state: *Web,
    fn alive(self: Browser) bool {
        var errors: [160]u8 = @splat(0);
        const response = http("GET", self.state.port, "/json/version", .{ .bytes = &errors }) orelse return false;
        defer allocator.free(response);
        return std.mem.find(u8, common.visible(response), "webSocketDebuggerUrl") != null;
    }
    fn spawn(self: Browser, errors: Errors) bool {
        if (!mkdir(@ptrCast(&self.state.profile_dir))) {
            errors.set("failed to create Chrome profile dir %s: %s", .{ @as([*c]const u8, &self.state.profile_dir), c.strerror(c.__error().*) });
            return false;
        }
        const executable = chrome();
        defer allocator.free(executable);
        const app = application();
        const via_open = app != null and c.access("/usr/bin/open", c.X_OK) == 0;
        var port_arg: [64]u8 = undefined;
        var profile_arg: [c.PATH_MAX + 64]u8 = undefined;
        _ = c.snprintf(&port_arg, port_arg.len, "--remote-debugging-port=%d", self.state.port);
        _ = c.snprintf(&profile_arg, profile_arg.len, "--user-data-dir=%s", @as([*c]const u8, &self.state.profile_dir));
        const pid = c.fork();
        if (pid < 0) {
            errors.set("failed to fork Chrome: %s", .{c.strerror(c.__error().*)});
            return false;
        }
        if (pid == 0) {
            const null_fd = c.open("/dev/null", c.O_RDWR);
            if (null_fd >= 0) {
                _ = c.dup2(null_fd, c.STDOUT_FILENO);
                _ = c.dup2(null_fd, c.STDERR_FILENO);
                if (null_fd > 2) _ = c.close(null_fd);
            }
            // Only the original libc descriptor/exec actions run after fork.
            // All arguments and owners were prepared by the parent beforehand.
            if (via_open) {
                _ = c.execlp("/usr/bin/open", "open", "-g", "-na", app.?, "--args", @as([*c]const u8, &port_arg), "--remote-allow-origins=*", @as([*c]const u8, &profile_arg), "--no-first-run", "--no-default-browser-check", "--disable-sync", "--use-mock-keychain", "--password-store=basic", "--mute-audio", "about:blank", @as([*c]u8, null));
            } else {
                _ = c.execlp(executable.ptr, executable.ptr, @as([*c]const u8, &port_arg), "--remote-allow-origins=*", @as([*c]const u8, &profile_arg), "--no-first-run", "--no-default-browser-check", "--disable-sync", "--use-mock-keychain", "--password-store=basic", "--mute-audio", "about:blank", @as([*c]u8, null));
            }
            c._exit(127);
        }
        self.state.chrome_pid = pid;
        for (0..80) |_| {
            if (common.cancel(self.state, errors)) return false;
            if (self.alive()) {
                common.log(self.state, "Chrome browser session is ready");
                return true;
            }
            var status: c_int = 0;
            if (c.waitpid(pid, &status, c.WNOHANG) == pid) {
                self.state.chrome_pid = 0;
                if (via_open) continue;
                errors.set("Chrome exited before CDP became ready", .{});
                return false;
            }
            if (!common.sleep(self.state, 250)) {
                errors.set("interrupted", .{});
                return false;
            }
        }
        errors.set("Chrome did not expose CDP on port %d", .{self.state.port});
        return false;
    }
    fn ensure(self: Browser, errors: Errors) bool {
        if (self.alive()) return true;
        if (self.state.chrome_pid > 0) {
            var status: c_int = 0;
            _ = c.waitpid(self.state.chrome_pid, &status, c.WNOHANG);
            self.state.chrome_pid = 0;
        }
        if (!self.state.browser_allowed) {
            const confirm = self.state.confirm orelse {
                errors.set("starting a visible Chrome browser requires interactive approval", .{});
                return false;
            };
            if (confirm(self.state.confirm_privdata, "The web tool wants to start a visible Chrome browser. Allow? (y/n) ", if (errors.bytes.len == 0) null else errors.bytes.ptr, errors.bytes.len) == 0) {
                if (errors.empty()) errors.set("user denied Chrome browser start", .{});
                return false;
            }
            self.state.browser_allowed = true;
        }
        return self.spawn(errors);
    }
    fn browserUrl(self: Browser, errors: Errors) ?[:0]u8 {
        const response = http("GET", self.state.port, "/json/version", errors) orelse return null;
        defer allocator.free(response);
        const url = common.string(common.visible(response), "webSocketDebuggerUrl");
        if (url == null) errors.set("Chrome did not return a browser WebSocket URL", .{});
        return url;
    }
    fn pageCount(self: Browser) c_int {
        var errors: [160]u8 = @splat(0);
        const response = http("GET", self.state.port, "/json/list", .{ .bytes = &errors }) orelse return -1;
        defer allocator.free(response);
        const source = common.visible(response);
        var position: usize = 0;
        var count: c_int = 0;
        while (std.mem.findPos(u8, source, position, "\"type\"")) |found| {
            position = found + 6;
            while (position < source.len and std.mem.indexOfScalar(u8, " \t\r\n", source[position]) != null) position += 1;
            const colon = position;
            position = @min(position + 1, source.len);
            if (colon == source.len or source[colon] != ':') continue;
            while (position < source.len and std.mem.indexOfScalar(u8, " \t\r\n", source[position]) != null) position += 1;
            if (std.mem.startsWith(u8, source[position..], "\"page\"")) count += 1;
        }
        return count;
    }
    fn httpTarget(self: Browser, url: []const u8, errors: Errors) ?[:0]u8 {
        const encoded = common.urlEncode(url);
        defer allocator.free(encoded);
        var path = Buffer{};
        defer path.deinit();
        path.add("/json/new?");
        path.add(if (encoded.len != 0) encoded else "about%3Ablank");
        const target = path.take();
        defer allocator.free(target);
        // Modern Chrome requires PUT; it also works without a browser window.
        const response = http("PUT", self.state.port, target.ptr, errors) orelse return null;
        defer allocator.free(response);
        const id = common.string(common.visible(response), "id");
        if (id == null) errors.set("PUT /json/new returned no target id", .{});
        return id;
    }
    pub fn openTab(self: Browser, url: []const u8, tab: *Tab, errors: Errors) bool {
        tab.* = std.mem.zeroes(Tab);
        const browser_url = self.browserUrl(errors) orelse return false;
        const response = creation: {
            defer allocator.free(browser_url);
            var connection = Connection{ .web = self.state };
            if (!connection.connect(browser_url, errors)) return false;
            defer connection.close();
            const quoted = common.quote(url);
            defer allocator.free(quoted);
            var parameters = Buffer{};
            defer parameters.deinit();
            parameters.add("{\"url\":");
            parameters.add(quoted);
            parameters.add(",\"background\":true,\"newWindow\":false}");
            break :creation connection.call("Target.createTarget", parameters.bytes(), errors);
        };
        if (response) |value| {
            defer allocator.free(value);
            if (common.string(common.visible(value), "targetId")) |id| tab.id = id.ptr else {
                const message = common.string(common.visible(value), "message");
                common.log(self.state, if (message) |why| why.ptr else "Target.createTarget returned no target id");
                if (message) |why| allocator.free(why);
            }
        }
        // A windowless Chrome rejects newWindow:false. Recover through its
        // HTTP action instead of abandoning this and subsequent tool calls.
        if (tab.id == null) {
            var detail: [160]u8 = @splat(0);
            if (self.httpTarget(url, .{ .bytes = &detail })) |id| tab.id = id.ptr else {
                freeTab(tab);
                if (detail[0] != 0) errors.set("Chrome did not return a page target id (%s)", .{@as([*c]const u8, &detail)}) else errors.set("Chrome did not return a page target id", .{});
                return false;
            }
            common.log(self.state, "recovered Chrome target creation over HTTP");
        }
        var page_url: [c.PATH_MAX + 128]u8 = undefined;
        _ = c.snprintf(&page_url, page_url.len, "ws://127.0.0.1:%d/devtools/page/%s", self.state.port, tab.id);
        tab.ws_url = common.duplicate(common.text(&page_url)).ptr;
        return true;
    }
    pub fn closeTab(self: Browser, optional_tab: ?*const Tab) void {
        const tab = optional_tab orelse return;
        if (tab.id == null or tab.id[0] == 0) return;
        // Never leave Chrome without a page/window; that state breaks the next
        // background Target.createTarget. One retained tab holds it open.
        const remaining = self.pageCount();
        if (remaining >= 0 and remaining <= 1) {
            common.log(self.state, "keeping last Chrome page target open");
            return;
        }
        const encoded = common.urlEncode(common.text(tab.id));
        defer allocator.free(encoded);
        var path = Buffer{};
        defer path.deinit();
        path.add("/json/close/");
        path.add(encoded);
        const target = path.take();
        defer allocator.free(target);
        var errors: [160]u8 = @splat(0);
        if (http("GET", self.state.port, target.ptr, .{ .bytes = &errors })) |body| allocator.free(body) else if (errors[0] != 0) common.log(self.state, &errors);
    }
    fn run(self: Browser, url: []const u8, javascript: []const u8, dynamic: bool, errors: Errors) ?[:0]u8 {
        if (!self.ensure(errors)) return null;
        var tab = std.mem.zeroes(Tab);
        if (!self.openTab("about:blank", &tab, errors)) return null;
        defer freeTab(&tab);
        defer self.closeTab(&tab);
        var connection = Connection{ .web = self.state };
        if (!connection.connect(common.text(tab.ws_url), errors)) return null;
        defer connection.close();
        if (!prepare(&connection, errors) or !navigate(&connection, url, errors) or !waitNavigated(&connection, errors)) return null;
        if (connection.evaluate(common.text(c.ds4_web_click_google_consent_js), errors)) |clicked| {
            defer allocator.free(clicked);
            if (common.visible(clicked).len != 0) {
                common.log(self.state, clicked.ptr);
                if (!common.sleep(self.state, 1500)) {
                    errors.set("interrupted", .{});
                    return null;
                }
                var wait_errors: [160]u8 = @splat(0);
                const local = Errors{ .bytes = &wait_errors };
                if (!waitNavigated(&connection, local) and local.interrupted()) {
                    errors.set("interrupted", .{});
                    return null;
                }
            }
        }
        if (dynamic and !scroll(&connection, errors)) return null;
        return connection.evaluate(javascript, errors);
    }
};

fn waitReady(connection: *Connection, errors: Errors) bool {
    for (0..80) |_| {
        if (common.cancel(connection.web, errors)) return false;
        const ready = check: {
            const state = connection.evaluate("document.readyState", errors) orelse break :check false;
            defer allocator.free(state);
            const value = common.visible(state);
            break :check std.mem.eql(u8, value, "complete") or std.mem.eql(u8, value, "interactive");
        };
        if (ready) {
            if (common.sleep(connection.web, 800)) return true;
            errors.set("interrupted", .{});
            return false;
        }
        if (!common.sleep(connection.web, 250)) {
            errors.set("interrupted", .{});
            return false;
        }
    }
    return true;
}
fn navigate(connection: *Connection, url: []const u8, errors: Errors) bool {
    const quoted = common.quote(url);
    defer allocator.free(quoted);
    var parameters = Buffer{};
    defer parameters.deinit();
    parameters.add("{\"url\":");
    parameters.add(quoted);
    parameters.byte('}');
    const response = connection.call("Page.navigate", parameters.bytes(), errors) orelse return false;
    allocator.free(response);
    return true;
}
fn waitNavigated(connection: *Connection, errors: Errors) bool {
    var last_length: c_long = -1;
    var stable: c_int = 0;
    var saw_real_url = false;
    for (0..100) |iteration| {
        if (common.cancel(connection.web, errors)) return false;
        const Probe = struct { real: bool, ready: bool, length: c_long };
        const probe: ?Probe = observed: {
            const result = connection.evaluate(common.text(c.ds4_web_probe_js), errors) orelse break :observed null;
            defer allocator.free(result);
            const bytes = common.visible(result);
            const first = std.mem.indexOfScalar(u8, bytes, '\n');
            const second = if (first) |index| std.mem.findScalarPos(u8, bytes, index + 1, '\n') else null;
            if (first == null or second == null) {
                errors.set("page readiness probe returned malformed data", .{});
                break :observed null;
            }
            const href = bytes[0..first.?];
            const ready = bytes[first.? + 1 .. second.?];
            break :observed .{ .real = href.len != 0 and !std.mem.eql(u8, href, "about:blank") and !std.mem.startsWith(u8, href, "chrome://"), .ready = std.mem.eql(u8, ready, "complete") or std.mem.eql(u8, ready, "interactive"), .length = c.strtol(result.ptr + second.? + 1, null, 10) };
        };
        if (probe) |value| {
            saw_real_url = saw_real_url or value.real;
            stable = if (value.length > 0 and value.length == last_length) stable + 1 else 0;
            last_length = value.length;
            if (saw_real_url and value.ready and value.length > 0 and stable >= 2) {
                if (common.sleep(connection.web, 500)) return true;
                errors.set("interrupted", .{});
                return false;
            }
            if (saw_real_url and value.ready and iteration >= 24) return true;
        }
        if (!common.sleep(connection.web, 250)) {
            errors.set("interrupted", .{});
            return false;
        }
    }
    return true;
}
fn prepare(connection: *Connection, errors: Errors) bool {
    for ([_][]const u8{ "Page.enable", "Runtime.enable" }) |method| {
        const response = connection.call(method, "{}", errors) orelse return false;
        allocator.free(response);
    }
    connection.optional("Emulation.setFocusEmulationEnabled", "{\"enabled\":true}");
    connection.optional("Emulation.setDeviceMetricsOverride", "{\"width\":1365,\"height\":900,\"deviceScaleFactor\":1,\"mobile\":false}");
    return waitReady(connection, errors);
}
fn scroll(connection: *Connection, errors: Errors) bool {
    if (common.cancel(connection.web, errors)) return false;
    var local: [160]u8 = @splat(0);
    const detail = Errors{ .bytes = &local };
    const result = connection.evaluate(common.text(c.ds4_web_scroll_js), detail);
    if (result == null and detail.interrupted()) {
        errors.set("interrupted", .{});
        return false;
    }
    if (result) |value| allocator.free(value);
    return !common.cancel(connection.web, errors);
}
pub fn search(optional_web: ?*Web, query: [*c]const u8, errors: Errors) ?[:0]u8 {
    const web = optional_web orelse {
        errors.set("web subsystem is not initialized", .{});
        return null;
    };
    const value = common.text(query);
    if (value.len == 0) {
        errors.set("google_search requires query", .{});
        return null;
    }
    const encoded = common.urlEncode(value);
    defer allocator.free(encoded);
    var url = Buffer{};
    defer url.deinit();
    url.add("https://www.google.com/search?q=");
    url.add(encoded);
    return (Browser{ .state = web }).run(url.bytes(), common.text(c.ds4_web_extract_search_js), false, errors);
}
pub fn visit(optional_web: ?*Web, url: [*c]const u8, errors: Errors) ?[:0]u8 {
    const web = optional_web orelse {
        errors.set("web subsystem is not initialized", .{});
        return null;
    };
    const value = common.text(url);
    if (value.len == 0) {
        errors.set("visit_page requires url", .{});
        return null;
    }
    return (Browser{ .state = web }).run(value, common.text(c.ds4_web_extract_page_js), true, errors);
}
