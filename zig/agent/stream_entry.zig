//! Sole C facade. Presentation and Stream use the SAME generated C module.
const owner = @import("stream.zig");
const Record = owner.Record;

export fn ds4_agent_stream_text(record: *Record, bytes: [*c]const u8, length: usize, finish: bool) void {
    const span: []const u8 = if (length == 0) &.{} else bytes[0..length];
    (owner.Stream{ .record = record }).writeText(span, finish);
}
export fn ds4_agent_stream_greedy(record: ?*const Record) bool {
    return owner.wantsGreedy(record);
}
export fn ds4_agent_stream_lookahead(record: *const Record) bool {
    return owner.needsLookahead(record);
}
