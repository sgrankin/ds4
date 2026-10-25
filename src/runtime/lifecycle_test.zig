//! Small actual-Metal ownership fixture; no model checkpoint is loaded.
const std = @import("std");
const c = @import("runtime_c");
const source =
    "#include <metal_stdlib>\nusing namespace metal;\n" ++
    "kernel void runtime_value(constant uint &value [[buffer(0)]], device uint *out [[buffer(1)]]) { out[0] = value; }\n";

const Instance = struct {
    gpu: *c.flint_gpu,
    output: *anyopaque,
    fn open() !Instance {
        const gpu = c.flint_gpu_create() orelse return error.CreateContext;
        errdefer c.flint_gpu_free(gpu);
        var message: [1024]u8 = @splat(0);
        if (!c.flint_gpu_compile(gpu, source, source.len, &message, message.len)) return error.CompileFixture;
        const output = c.flint_gpu_buffer(gpu, @sizeOf(u32)) orelse return error.CreateBuffer;
        return .{ .gpu = gpu, .output = output };
    }
    fn close(self: Instance) void {
        c.flint_gpu_buffer_free(self.output);
        c.flint_gpu_free(self.gpu);
    }
    fn submit(self: Instance, value: u32, wait: bool) !void {
        const command = c.flint_gpu_command_begin(self.gpu) orelse return error.CreateCommand;
        const bindings = [_]c.flint_gpu_binding{.{ .handle = self.output, .offset = 0, .index = 1 }};
        const shape = [_]u32{ 1, 1, 1 };
        const ok = c.flint_gpu_dispatch(self.gpu, command, "runtime_value", &value, @sizeOf(u32), &bindings, bindings.len, &shape, &shape);
        const done = c.flint_gpu_command_end(self.gpu, command, wait);
        if (!ok or !done) return error.ExecuteFixture;
    }
    fn read(self: Instance) u32 {
        return @as(*u32, @ptrCast(@alignCast(c.flint_gpu_contents(self.output)))).*;
    }
};
fn check(value: bool) !void {
    if (!value) return error.LifetimeInvariant;
}
const Primitive = struct {
    input: *c.ds4_gpu_tensor,
    output: *c.ds4_gpu_tensor,
    fn open() !Primitive {
        const input = c.ds4_gpu_tensor_alloc(16) orelse return error.CreateTensor;
        const output = c.ds4_gpu_tensor_alloc(16) orelse return error.CreateTensor;
        return .{ .input = input, .output = output };
    }
    fn close(self: Primitive) void {
        c.ds4_gpu_tensor_free(self.input);
        c.ds4_gpu_tensor_free(self.output);
    }
    fn add(self: Primitive, value: f32) !void {
        try check(c.ds4_gpu_tensor_fill_f32(self.input, value, 4) != 0);
        try check(c.ds4_gpu_add_tensor(self.output, self.input, self.input, 4) != 0);
        var rows: [4]f32 = undefined;
        try check(c.ds4_gpu_tensor_read(self.output, 0, &rows, 16) != 0);
        for (rows) |row| try check(row == value * 2);
    }
};
fn run() !void {
    const base = c.ds4_runtime_metal_users();
    // Queries and repeated empty cleanup must not manufacture state that an
    // uninitialized context's GPU cleanup would never retire.
    const empty = c.ds4_metal_context_create() orelse return error.CreateContext;
    const previous_empty = c.ds4_metal_context_select(empty);
    const empty_stats = c.ds4_native_metal_tensor_stats();
    try check(empty_stats.live_bytes == 0 and empty_stats.peak_bytes == 0);
    c.ds4_native_metal_tensor_tracking_reset();
    c.ds4_metal_exact_cache_cleanup();
    c.ds4_metal_exact_cache_clear("empty-context");
    c.ds4_metal_exact_cache_report();
    try check(c.ds4_metal_resource_state_slot().* == null and c.ds4_metal_exact_cache_state_slot().* == null);
    _ = c.ds4_metal_context_select(previous_empty);
    c.ds4_metal_context_destroy(empty);
    const first = try Instance.open();
    var first_open = true;
    defer if (first_open) first.close();
    const second = try Instance.open();
    defer second.close();
    try check(c.ds4_runtime_metal_users() == base + 2);
    var device: ?*anyopaque = null;
    var queue: ?*anyopaque = null;
    try check(c.ds4_runtime_metal_acquire(&device, &queue));
    c.ds4_runtime_metal_release();
    try first.submit(11, true);
    try second.submit(22, false);
    try first.submit(33, true);
    try check(first.read() == 33 and second.read() == 22);
    first.close();
    first_open = false;
    try check(c.ds4_runtime_metal_users() == base + 1);
    try second.submit(44, true);
    try check(second.read() == 44);
    // These are real separate cache/map/schedule records, not two handles
    // pointing to the former globals. No second model/PSO lifecycle is implied.
    const owner = c.ds4_metal_context_create() orelse return error.CreateContext;
    const other = c.ds4_metal_context_create() orelse return error.CreateContext;
    const previous = c.ds4_metal_context_select(owner);
    const own_state = c.ds4_metal_bootstrap_driver();
    own_state.cache.*.cache_hits = 91;
    _ = c.ds4_metal_context_select(other);
    const other_state = c.ds4_metal_bootstrap_driver();
    try check(own_state.cache != other_state.cache and own_state.schedule != other_state.schedule and own_state.load != other_state.load);
    try check(other_state.cache.*.cache_hits == 0);
    _ = c.ds4_metal_context_select(owner);
    try check(c.ds4_metal_bootstrap_driver().cache.*.cache_hits == 91);
    // DS4's model driver acquires its own borrow and retires only its resources.
    try check(c.ds4_gpu_init() != 0);
    try check(c.ds4_runtime_metal_users() == base + 2);
    const first_primitive = try Primitive.open();
    try first_primitive.add(3);
    _ = c.ds4_metal_context_select(other);
    try check(c.ds4_gpu_init() != 0);
    const survivor = try Primitive.open();
    try survivor.add(5);
    _ = c.ds4_metal_context_select(owner);
    try first_primitive.add(7);
    first_primitive.close();
    c.ds4_gpu_cleanup();
    _ = c.ds4_metal_context_select(other);
    c.ds4_metal_context_destroy(owner);
    try check(c.ds4_runtime_metal_users() == base + 2);
    // PSOs, tracked tensor handles and scratch survive the other owner's close.
    try survivor.add(11);
    survivor.close();
    c.ds4_gpu_cleanup();
    _ = c.ds4_metal_context_select(previous);
    c.ds4_metal_context_destroy(other);
    try check(c.ds4_runtime_metal_users() == base + 1);
    try second.submit(55, true);
    try check(second.read() == 55);
    var same_device: ?*anyopaque = null;
    var same_queue: ?*anyopaque = null;
    try check(c.ds4_runtime_metal_acquire(&same_device, &same_queue));
    c.ds4_runtime_metal_release();
    try check(device == same_device and queue == same_queue);
    // A failed third engine compilation cannot invalidate the survivor.
    const failed = c.flint_gpu_create() orelse return error.CreateContext;
    var message: [1024]u8 = @splat(0);
    const bad_source = "this is not Metal source";
    const accepted = c.flint_gpu_compile(failed, bad_source, bad_source.len, &message, message.len);
    c.flint_gpu_free(failed);
    try check(!accepted and c.ds4_runtime_metal_users() == base + 1);
    try second.submit(66, true);
    try check(second.read() == 66);
    std.debug.print("Runtime Metal: A→B→A, two initialized DS4 owners, survivor GPU primitive, async completion, engine close, shared queue identity and failed compilation pass.\n", .{});
}
export fn main() c_int {
    run() catch |err| {
        std.debug.print("Runtime Metal: {s}\n", .{@errorName(err)});
        return 1;
    };
    if (c.ds4_runtime_metal_users() != 0) return 1;
    return 0;
}
