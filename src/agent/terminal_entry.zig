//! Sole C seam for existing UI callers and the same production terminal tests.
const c = @import("agent_runtime_c");
const Ui = @import("ui_text.zig");
const Terminal = @import("terminal.zig");
const Editor = Terminal.Editor;
fn bytes(value: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else value[0..length];
}
fn output(value: [*c]u8, length: usize) []u8 {
    return if (length == 0) &.{} else value[0..length];
}
export fn ds4_agent_ui_write(fd: c_int, value: [*c]const u8, length: usize) void {
    Terminal.write(fd, bytes(value, length));
}
export fn ds4_agent_ui_input_append(buffer: *c.agent_input_buf, value: [*c]const u8, length: usize) void {
    Ui.append(buffer, bytes(value, length));
}
export fn ds4_agent_ui_input_take(buffer: *c.agent_input_buf) [*c]u8 {
    return Ui.take(buffer);
}
export fn ds4_agent_ui_input_free(buffer: *c.agent_input_buf) void {
    Ui.free(buffer);
}
export fn ds4_agent_ui_tty() bool {
    return Ui.tty();
}
export fn ds4_agent_ui_echo_text(value: [*c]const u8) [*c]u8 {
    return Ui.echo(value);
}
export fn ds4_agent_ui_echo(value: [*c]const u8) void {
    const rendered = Ui.echo(value);
    defer c.free(rendered);
    _ = c.printf("%s", rendered);
    _ = c.fflush(c.__stdoutp);
}
export fn ds4_agent_ui_context_size(value: c_int, buffer: [*c]u8, length: usize) void {
    Ui.contextSize(value, output(buffer, length));
}
export fn ds4_agent_ui_progress(done: c_int, total: c_int, rate: f64, buffer: [*c]u8, length: usize, color: bool) void {
    Ui.progressBar(done, total, rate, output(buffer, length), color);
}
export fn ds4_agent_ui_prompt(_: [*c]const c.agent_status, buffer: [*c]u8, length: usize) void {
    Ui.copy(output(buffer, length), "ds4-agent> ");
}
export fn ds4_agent_ui_status(status: *const c.agent_status, buffer: [*c]u8, length: usize) void {
    Ui.statusText(status, output(buffer, length));
}
export fn ds4_agent_ui_footer(status: *const c.agent_status, queue: ?*const c.agent_prompt_queue, columns: c_int, buffer: [*c]u8, length: usize) void {
    Ui.footer(status, queue, columns, output(buffer, length));
}
export fn ds4_agent_ui_queue_push(queue: *c.agent_prompt_queue, value: [*c]const u8) void {
    (Ui.Queue{ .state = queue }).push(value);
}
export fn ds4_agent_ui_queue_pop(queue: *c.agent_prompt_queue) [*c]u8 {
    return (Ui.Queue{ .state = queue }).pop();
}
export fn ds4_agent_ui_queue_front(queue: *c.agent_prompt_queue, value: [*c]u8) void {
    (Ui.Queue{ .state = queue }).pushFront(value);
}
export fn ds4_agent_ui_queue_take(queue: *c.agent_prompt_queue) [*c]u8 {
    return (Ui.Queue{ .state = queue }).takeAll();
}
export fn ds4_agent_ui_queue_echo(queue: *c.agent_prompt_queue) [*c]u8 {
    return (Ui.Queue{ .state = queue }).echoAll();
}
export fn ds4_agent_ui_queue_peek(queue: *const c.agent_prompt_queue) [*c]const u8 {
    return if (queue.len == 0) null else queue.v[0];
}
export fn ds4_agent_ui_queue_free(queue: *c.agent_prompt_queue) void {
    (Ui.Queue{ .state = queue }).deinit();
}
export fn ds4_agent_ui_size(rows: *c_int, columns: *c_int) bool {
    return Terminal.terminalSize(rows, columns);
}
export fn ds4_agent_ui_read(editor: *c.agent_editor) void {
    (Editor{ .state = editor }).readStdin();
}
export fn ds4_agent_ui_take_byte(editor: *c.agent_editor, byte: u8) bool {
    return (Editor{ .state = editor }).takeByte(byte);
}
export fn ds4_agent_ui_take_escape(editor: *c.agent_editor) bool {
    return (Editor{ .state = editor }).takeEscape();
}
export fn ds4_agent_ui_replace(editor: *c.agent_editor, value: [*c]const u8) void {
    (Editor{ .state = editor }).replaceInput(value);
}
export fn ds4_agent_ui_note(editor: *c.agent_editor, value: [*c]const u8, length: usize) void {
    (Editor{ .state = editor }).noteOutput(bytes(value, length));
}
export fn ds4_agent_ui_start(editor: *c.agent_editor, prompt: [*c]const u8, status: [*c]const u8, initial: [*c]const u8) c_int {
    return (Editor{ .state = editor }).start(prompt, status, initial);
}
export fn ds4_agent_ui_stop(editor: *c.agent_editor) void {
    (Editor{ .state = editor }).stop();
}
export fn ds4_agent_ui_hide(editor: *c.agent_editor) void {
    (Editor{ .state = editor }).hide();
}
export fn ds4_agent_ui_show(editor: *c.agent_editor) void {
    (Editor{ .state = editor }).show();
}
export fn ds4_agent_ui_prompt_status(editor: *c.agent_editor, prompt: [*c]const u8, status: [*c]const u8) void {
    (Editor{ .state = editor }).setPromptStatus(prompt, status);
}
export fn ds4_agent_ui_flush(editor: *c.agent_editor, force: bool) void {
    (Editor{ .state = editor }).flush(force);
}
export fn ds4_agent_ui_scroll(editor: *c.agent_editor, value: [*c]const u8, length: usize, settle: bool) bool {
    return (Editor{ .state = editor }).writeScroll(bytes(value, length), settle);
}
export fn ds4_agent_ui_async(editor: *c.agent_editor, value: [*c]const u8, length: usize, prompt: [*c]const u8, status: [*c]const u8, force: bool) void {
    (Editor{ .state = editor }).writeAsync(bytes(value, length), prompt, status, force);
}
export fn ds4_agent_ui_cancel(editor: *c.agent_editor, prompt: [*c]const u8, status: [*c]const u8) void {
    (Editor{ .state = editor }).cancelInput(prompt, status);
}
export fn ds4_agent_ui_restore(editor: *c.agent_editor) void {
    (Editor{ .state = editor }).restoreLayout();
}
export fn ds4_agent_ui_welcome(config: *const c.agent_config, buffer: [*c]u8, length: usize) void {
    Ui.welcome(config, output(buffer, length));
}
export fn ds4_agent_ui_write_welcome(editor: *c.agent_editor, config: *const c.agent_config, prompt: [*c]const u8, status: [*c]const u8) void {
    var banner: [256]u8 = undefined;
    Ui.welcome(config, &banner);
    (Editor{ .state = editor }).writeAsync(Ui.text(&banner), prompt, status, true);
}
