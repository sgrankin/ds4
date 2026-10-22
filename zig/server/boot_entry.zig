//! Sole facade: C option fixtures and production startup use these same bodies.
const c = @import("server_boot_c");
const owner = @import("boot.zig");

export fn ds4_server_boot_parse(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend) c.server_config {
    return owner.parse(argc, argv, default_backend);
}
export fn ds4_server_boot_close(runtime: *c.server) void {
    owner.close(runtime);
}
export fn ds4_server_boot_main(argc: c_int, argv: [*c][*c]u8, stop: [*c]volatile c.sig_atomic_t, listen_fd: [*c]volatile c.sig_atomic_t, default_backend: c.ds4_backend) c_int {
    return owner.run(argc, argv, stop, listen_fd, default_backend);
}
