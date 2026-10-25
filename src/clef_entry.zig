//! Narrow ABI for existing startup/HTTP owners; model execution stays native.
const std = @import("std");
const c = @import("clef_runtime_c");
const Engine = @import("models/clef/engine.zig").Engine;
const Wire = @import("server/decisions.zig");
const a = std.heap.c_allocator;
const Registry = struct { engines: std.ArrayList(*Engine) = .empty };

fn report(buffer: [*c]u8, capacity: usize, message: []const u8) void {
    if (buffer == null or capacity == 0) return;
    const n = @min(capacity - 1, message.len);
    @memcpy(buffer[0..n], message[0..n]);
    buffer[n] = 0;
}
fn engine(pointer: *c.ds4_decision_engine) *Engine {
    return @ptrCast(@alignCast(pointer));
}

export fn ds4_decision_engine_detect(directory: [*c]const u8) bool {
    if (directory == null) return false;
    const path = std.fmt.allocPrintSentinel(a, "{s}/joint_head_config.json", .{std.mem.span(directory)}, 0) catch return false;
    defer a.free(path);
    return c.access(path, c.R_OK) == 0;
}
export fn ds4_decision_engine_open(directory: [*c]const u8, max_tokens: usize, message: [*c]u8, capacity: usize) ?*c.ds4_decision_engine {
    if (directory == null) return null;
    const self = Engine.open(std.mem.span(directory), max_tokens) catch |err| {
        report(message, capacity, @errorName(err));
        return null;
    };
    return @ptrCast(self);
}
export fn ds4_decision_engine_close(pointer: ?*c.ds4_decision_engine) void {
    if (pointer) |self| engine(self).close();
}
export fn ds4_decision_engine_name(pointer: *c.ds4_decision_engine) [*:0]const u8 {
    return engine(pointer).name().ptr;
}
export fn ds4_decision_engine_context(pointer: *c.ds4_decision_engine) usize {
    return engine(pointer).max_tokens;
}
export fn ds4_decision_engine_request(pointer: *c.ds4_decision_engine, body: [*c]const u8, openai: bool, output: [*c][*c]u8, message: [*c]u8, capacity: usize, cancel: c.ds4_decision_cancel_fn, cancel_context: ?*anyopaque) c_int {
    output[0] = null;
    if (body == null) {
        report(message, capacity, "missing request body");
        return 400;
    }
    if (cancel) |check| if (check(cancel_context)) return 499;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const allocator = arena.allocator();
    const protocol: Wire.Protocol = if (openai) .openai else .systemone;
    const parsed = Wire.parse(allocator, std.mem.span(body), protocol) catch |err| {
        report(message, capacity, @errorName(err));
        return 400;
    };
    return execute(engine(pointer), allocator, parsed, protocol, output, message, capacity, cancel, cancel_context);
}

fn execute(self: *Engine, allocator: std.mem.Allocator, parsed: Wire.Parsed, protocol: Wire.Protocol, output: [*c][*c]u8, message: [*c]u8, capacity: usize, cancel: c.ds4_decision_cancel_fn, cancel_context: ?*anyopaque) c_int {
    const result = self.decideCancelable(allocator, parsed.request, cancel, cancel_context) catch |err| {
        report(message, capacity, @errorName(err));
        if (err == error.Cancelled) return 499;
        if (err == error.ContextTooLong or err == error.SchemaTooLong) {
            report(message, capacity, "too many tokens: maximum context length exceeded");
            return 422;
        }
        return switch (err) {
            error.MediaTooLarge, error.ImageTooLarge, error.AudioTooLarge => 413,
            error.MediaFetchFailed, error.MediaHttpStatus => 502,
            error.MediaNotSupported,
            error.UnsupportedAudio,
            error.UnsupportedImage,
            error.UnsupportedMediaSource,
            error.UnsupportedSampleRate,
            error.InvalidMediaSource,
            error.InvalidCharacter,
            error.InvalidPadding,
            error.InvalidImage,
            error.InvalidPng,
            error.InvalidJpeg,
            error.InvalidImageGeometry,
            error.ImageAspectRatio,
            error.InvalidEncodedMedia,
            error.EmptyAudio,
            error.AudioTooShort,
            error.InvalidAudioSamples,
            error.NonFiniteMediaValue,
            error.NonfiniteAudio,
            error.AudioPowerOverflow,
            error.InvalidAudioOffset,
            error.EmptyVideo,
            error.InvalidVideoFrames,
            error.InvalidVideoTimestamps,
            error.InvalidVideoTimestamp,
            error.InvalidVideoRate,
            error.InvalidVideoGeometry,
            error.ExpectedNumber,
            => 400,
            else => 500,
        };
    };
    if (cancel) |check| if (check(cancel_context)) return 499;
    const json = Wire.serialize(allocator, parsed, result, protocol) catch |err| {
        report(message, capacity, @errorName(err));
        return 500;
    };
    const owned = a.dupeSentinel(u8, json, 0) catch {
        report(message, capacity, "out of memory");
        return 500;
    };
    output[0] = owned.ptr;
    return 200;
}

fn registry(pointer: *c.ds4_decision_registry) *Registry {
    return @ptrCast(@alignCast(pointer));
}
export fn ds4_decision_registry_create() ?*c.ds4_decision_registry {
    const self = a.create(Registry) catch return null;
    self.* = .{};
    return @ptrCast(self);
}
export fn ds4_decision_registry_free(pointer: ?*c.ds4_decision_registry) void {
    if (pointer) |raw| {
        const self = registry(raw);
        for (self.engines.items) |item| item.close();
        self.engines.deinit(a);
        a.destroy(self);
    }
}
export fn ds4_decision_registry_add(pointer: *c.ds4_decision_registry, directory: [*c]const u8, max_tokens: usize, message: [*c]u8, capacity: usize) bool {
    const self = registry(pointer);
    const item = Engine.open(std.mem.span(directory), max_tokens) catch |err| {
        report(message, capacity, @errorName(err));
        return false;
    };
    for (self.engines.items) |existing| if (std.mem.eql(u8, existing.name(), item.name())) {
        item.close();
        report(message, capacity, "decision model is already loaded");
        return false;
    };
    self.engines.append(a, item) catch {
        item.close();
        report(message, capacity, "out of memory");
        return false;
    };
    return true;
}
export fn ds4_decision_registry_count(pointer: *c.ds4_decision_registry) usize {
    return registry(pointer).engines.items.len;
}
const Model = struct {
    id: []const u8,
    object: []const u8 = "model",
    created: u64 = 0,
    owned_by: []const u8 = "local",
    context_length: usize,
    capabilities: []const []const u8 = &.{"decisions"},
    input_modalities: []const []const u8 = &.{"text"},
};
export fn ds4_decision_registry_models(pointer: *c.ds4_decision_registry, id: [*c]const u8) [*c]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const allocator = arena.allocator();
    var entries: std.ArrayList(Model) = .empty;
    for (registry(pointer).engines.items) |item| {
        if (id != null and !std.mem.eql(u8, item.name(), std.mem.span(id))) continue;
        entries.append(allocator, .{
            .id = item.name(),
            .context_length = item.max_tokens,
            .input_modalities = if (item.backend.config.family == .omni)
                &.{ "text", "image", "audio", "video" }
            else
                &.{ "text", "image", "video" },
        }) catch return null;
    }
    if (id != null and entries.items.len == 0) return null;
    const json = if (id != null)
        std.json.Stringify.valueAlloc(allocator, entries.items[0], .{}) catch return null
    else
        std.json.Stringify.valueAlloc(allocator, entries.items, .{}) catch return null;
    return (a.dupeSentinel(u8, json, 0) catch return null).ptr;
}
export fn ds4_decision_registry_request(pointer: *c.ds4_decision_registry, body: [*c]const u8, openai: bool, output: [*c][*c]u8, message: [*c]u8, capacity: usize, cancel: c.ds4_decision_cancel_fn, cancel_context: ?*anyopaque) c_int {
    output[0] = null;
    if (body == null) {
        report(message, capacity, "missing request body");
        return 400;
    }
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const allocator = arena.allocator();
    const protocol: Wire.Protocol = if (openai) .openai else .systemone;
    const parsed = Wire.parse(allocator, std.mem.span(body), protocol) catch |err| {
        report(message, capacity, @errorName(err));
        return 400;
    };
    for (registry(pointer).engines.items) |item| {
        if (std.mem.eql(u8, parsed.model, item.name())) return execute(item, allocator, parsed, protocol, output, message, capacity, cancel, cancel_context);
    }
    report(message, capacity, "requested decision model is not loaded");
    return 404;
}
