//! Native program entry. The bootstrap owns startup/serving; the two signal
//! flags are defined once by its Zig facade and shared with residual C I/O.
const c = @import("server_boot_c");

export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    return c.ds4_server_boot_main(argc, argv, &c.ds4_server_stop_requested, &c.ds4_server_listen_fd, c.DS4_BACKEND_METAL);
}
