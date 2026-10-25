//! Encoded media sources. Standard base64 and libcurl provide the wire codecs;
//! image/audio/video preparation stays in its own model-specific components.
const std = @import("std");
const c = @import("clef_media_c");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
pub const Cancel = ?*const fn (?*anyopaque) callconv(.c) bool;
const max_bytes = 64 * 1024 * 1024;
var once: c.pthread_once_t = c.ds4_clef_media_once_init;
var initialized = false;

fn initialize() callconv(.c) void {
    initialized = c.curl_global_init(c.CURL_GLOBAL_DEFAULT) == c.CURLE_OK;
}

pub fn decodeBase64(a: Allocator, source: []const u8) ![]u8 {
    const decoder = if (source.len % 4 == 0) std.base64.standard.Decoder else std.base64.standard_no_pad.Decoder;
    const size = try decoder.calcSizeForSlice(source);
    if (size > max_bytes) return error.MediaTooLarge;
    const bytes = try a.alloc(u8, size);
    errdefer a.free(bytes);
    try decoder.decode(bytes, source);
    return bytes;
}

pub fn load(a: Allocator, source: Value) ![]u8 {
    return loadCancelable(a, source, null, null);
}

pub fn loadCancelable(a: Allocator, source: Value, cancel: Cancel, context: ?*anyopaque) ![]u8 {
    if (cancel) |check| if (check(context)) return error.Cancelled;
    if (source == .object) {
        for ([_][]const u8{ "base64", "data_base64" }) |key| {
            if (source.object.get(key)) |value| {
                if (value != .string) return error.InvalidMediaSource;
                return decodeBase64(a, value.string);
            }
        }
        if (source.object.get("url")) |value| return loadCancelable(a, value, cancel, context);
        return error.InvalidMediaSource;
    }
    if (source != .string) return error.InvalidMediaSource;
    const uri = source.string;
    if (std.mem.startsWith(u8, uri, "data:")) {
        const comma = std.mem.indexOfScalar(u8, uri, ',') orelse return error.InvalidMediaSource;
        if (!std.mem.endsWith(u8, uri[0..comma], ";base64")) return error.InvalidMediaSource;
        return decodeBase64(a, uri[comma + 1 ..]);
    }
    if (!std.mem.startsWith(u8, uri, "https://") and !std.mem.startsWith(u8, uri, "http://")) return error.UnsupportedMediaSource;
    if (c.pthread_once(&once, initialize) != 0 or !initialized) return error.MediaFetchInitialization;
    const handle = c.curl_easy_init() orelse return error.MediaFetchInitialization;
    defer c.curl_easy_cleanup(handle);
    const url = try a.dupeSentinel(u8, uri, 0);
    defer a.free(url);
    var sink = Sink{ .allocator = a, .cancel = cancel, .context = context };
    defer sink.bytes.deinit(a);
    try option(handle, c.CURLOPT_URL, url.ptr);
    try option(handle, c.CURLOPT_PROTOCOLS_STR, @as([*:0]const u8, "http,https"));
    try option(handle, c.CURLOPT_REDIR_PROTOCOLS_STR, @as([*:0]const u8, "http,https"));
    try option(handle, c.CURLOPT_FOLLOWLOCATION, @as(c_long, 1));
    try option(handle, c.CURLOPT_MAXREDIRS, @as(c_long, 5));
    try option(handle, c.CURLOPT_NOSIGNAL, @as(c_long, 1));
    try option(handle, c.CURLOPT_CONNECTTIMEOUT, @as(c_long, 10));
    try option(handle, c.CURLOPT_TIMEOUT, @as(c_long, 30));
    try option(handle, c.CURLOPT_MAXFILESIZE_LARGE, @as(c.curl_off_t, max_bytes));
    try option(handle, c.CURLOPT_WRITEFUNCTION, @as(c.curl_write_callback, receive));
    try option(handle, c.CURLOPT_WRITEDATA, @as(?*anyopaque, @ptrCast(&sink)));
    try option(handle, c.CURLOPT_NOPROGRESS, @as(c_long, 0));
    try option(handle, c.CURLOPT_XFERINFOFUNCTION, @as(c.curl_xferinfo_callback, progress));
    try option(handle, c.CURLOPT_XFERINFODATA, @as(?*anyopaque, @ptrCast(&sink)));
    const result = c.curl_easy_perform(handle);
    if (sink.failure) |failure| return failure;
    if (cancel) |check| if (check(context)) return error.Cancelled;
    if (result != c.CURLE_OK) return error.MediaFetchFailed;
    var status: c_long = 0;
    if (c.curl_easy_getinfo(handle, c.CURLINFO_RESPONSE_CODE, &status) != c.CURLE_OK or status < 200 or status >= 300) return error.MediaHttpStatus;
    return sink.bytes.toOwnedSlice(a);
}

fn option(handle: *c.CURL, name: c.CURLoption, value: anytype) !void {
    if (c.curl_easy_setopt(handle, name, value) != c.CURLE_OK) return error.MediaFetchConfiguration;
}

const Sink = struct {
    allocator: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    failure: ?anyerror = null,
    cancel: Cancel,
    context: ?*anyopaque,
};
fn receive(data: [*c]u8, size: usize, count: usize, payload: ?*anyopaque) callconv(.c) usize {
    const sink: *Sink = @ptrCast(@alignCast(payload.?));
    const n = std.math.mul(usize, size, count) catch return 0;
    if (n > max_bytes - sink.bytes.items.len) {
        sink.failure = error.MediaTooLarge;
        return 0;
    }
    sink.bytes.appendSlice(sink.allocator, data[0..n]) catch |err| {
        sink.failure = err;
        return 0;
    };
    return n;
}
fn progress(payload: ?*anyopaque, _: c.curl_off_t, _: c.curl_off_t, _: c.curl_off_t, _: c.curl_off_t) callconv(.c) c_int {
    const sink: *Sink = @ptrCast(@alignCast(payload.?));
    return if (sink.cancel) |check| @intFromBool(check(sink.context)) else 0;
}

test "data URLs and model-native base64 objects decode with standard codecs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("abc", try load(a, .{ .string = "data:image/png;base64,YWJj" }));
    const source = try std.json.parseFromSliceLeaky(Value, a, "{\"data_base64\":\"YWI\"}", .{});
    try std.testing.expectEqualStrings("ab", try load(a, source));
}
