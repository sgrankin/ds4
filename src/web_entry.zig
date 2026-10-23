//! Sole opaque browser ABI and existing local recovery-fixture actions.
const common = @import("web_common.zig");
const c = common.c;
const owner = @import("web.zig");
fn errors(pointer: [*c]u8, length: usize) common.Errors {
    return .{ .bytes = if (pointer == null or length == 0) &.{} else pointer[0..length] };
}
export fn ds4_web_create(options: ?*const c.ds4_web_config) *c.ds4_web {
    return owner.create(options);
}
export fn ds4_web_free(web: ?*c.ds4_web) void {
    owner.free(web);
}
export fn ds4_web_google_search(web: ?*c.ds4_web, query: [*c]const u8, message: [*c]u8, length: usize) [*c]u8 {
    return if (owner.search(web, query, errors(message, length))) |result| result.ptr else null;
}
export fn ds4_web_visit_page(web: ?*c.ds4_web, url: [*c]const u8, message: [*c]u8, length: usize) [*c]u8 {
    return if (owner.visit(web, url, errors(message, length))) |result| result.ptr else null;
}
export fn ds4_web_open_tab(web: *c.ds4_web, url: [*c]const u8, tab: *c.web_tab, message: [*c]u8, length: usize) bool {
    return (owner.Browser{ .state = web }).openTab(common.text(url), tab, errors(message, length));
}
export fn ds4_web_close_tab(web: ?*c.ds4_web, tab: ?*const c.web_tab) void {
    if (web) |state| (owner.Browser{ .state = state }).closeTab(tab);
}
export fn ds4_web_tab_free(tab: ?*c.web_tab) void {
    if (tab) |value| owner.freeTab(value);
}
