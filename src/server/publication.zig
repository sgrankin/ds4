//! Commit completed conversation items before advertising the response ID.
//! Durable response files and connection-local continuations share this policy;
//! Conversations owns its version check and the store owns atomic file writes.
const std = @import("std");
const c = @import("server_output_c");
const allocator = std.heap.c_allocator;

fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}

fn published(request: *c.request, id: [*c]const u8) void {
    _ = c.snprintf(&request.responses_published_id, request.responses_published_id.len, "%s", id);
}

pub fn publish(request: *c.request, id: [*c]const u8, public_json: [*c]const u8, errors: []u8) bool {
    if (request.conversation_id != null) {
        const status = c.ds4_server_response_string_field(public_json, "status");
        defer c.free(status);
        if (std.mem.eql(u8, text(status), "completed")) {
            const output = c.ds4_conversation_json_field(public_json, "output");
            defer c.free(output);
            const result = c.ds4_conversation_publish(request.responses_store_dir, request.conversation_id, request.conversation_version, request.conversation_input, output, request.responses_replay, errors.ptr, errors.len);
            if (result != 0) {
                request.responses_store_error_status = result;
                return false;
            }
        }
    }

    if (!request.responses_store) {
        // An ephemeral ID identifies the connection's live model frontier but
        // deliberately creates no durable response record.
        if (request.response_sink != null) published(request, id);
        return true;
    }
    const record = std.fmt.allocPrint(allocator, "{{\"response\":{s},\"replay\":{s}}}\n", .{
        text(public_json), if (request.responses_replay == null) "[]" else text(request.responses_replay),
    }) catch {
        _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
        c.exit(1);
    };
    defer allocator.free(record);
    if (c.ds4_response_store_write(request.responses_store_dir, id, record.ptr, record.len, errors.ptr, errors.len) != 0) return false;
    published(request, id);
    return true;
}
