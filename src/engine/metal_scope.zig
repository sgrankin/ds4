//! Explicit native ABI entry scope. Jobs/workers capture their own owner;
//! selecting on the caller does not propagate through pthread_create.
pub fn Scope(comptime c: type) type {
    return struct {
        previous: @TypeOf(c.ds4_metal_context_selected()),
        selected: @TypeOf(c.ds4_metal_context_selected()),
        previous_host: @TypeOf(c.ds4_host_context_selected()),
        selected_host: @TypeOf(c.ds4_host_context_selected()),
        const Self = @This();

        pub fn enter(core: *const c.ds4_engine_core) Self {
            return .{
                .previous = c.ds4_metal_context_select(core.metal_context),
                .selected = core.metal_context,
                .previous_host = c.ds4_host_context_select(core.host_context),
                .selected_host = core.host_context,
            };
        }
        pub fn leave(self: Self) void {
            // Core retirement clears its selection before releasing the owner.
            // A nested same-owner entry must not restore that retired pointer.
            const retired = self.previous == self.selected and c.ds4_metal_context_selected() == null;
            const host_retired = self.previous_host == self.selected_host and c.ds4_host_context_selected() == null;
            _ = c.ds4_host_context_select(if (host_retired) null else self.previous_host);
            _ = c.ds4_metal_context_select(if (retired) null else self.previous);
        }
    };
}
