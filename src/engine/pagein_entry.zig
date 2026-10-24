//! Thin C entry points over the native I/O/job owners and actual shared records.
const c = @import("pagein_c");
const pagein = @import("pagein.zig");
const io = @import("pagein_io.zig");

fn slots(pointer: [*c]pagein.Slot, count: u32) []pagein.Slot {
    return if (count == 0) &.{} else pointer[0..count];
}
export fn ds4_native_pagein_install_spans(model: [*c]const c.ds4_model, spans: [*c]const c.ds4_model_map_span_vec, label: [*c]const u8) bool {
    return pagein.install(model, spans, label);
}
export fn ds4_native_pagein_touch(model: [*c]const c.ds4_model, offset: u64, bytes: u64, touched: ?*u64, sink: ?*u8) bool {
    return io.touch(model, offset, bytes, touched, sink);
}
export fn ds4_native_pagein_pread(model: [*c]const c.ds4_model, offset: u64, bytes: u64, touched: ?*u64, sink: ?*u8) bool {
    return io.pread(model, offset, bytes, touched, sink);
}
export fn ds4_native_pagein_madvise(model: [*c]const c.ds4_model, offset: u64, bytes: u64, enabled: bool, touched: ?*u64) bool {
    return io.advise(model, offset, bytes, enabled, touched);
}
export fn ds4_native_pagein_readahead(model: [*c]const c.ds4_model, offset: u64, bytes: u64, enabled: bool) void {
    io.readAhead(model, offset, bytes, enabled);
}
export fn ds4_native_pagein_layer_start(model: [*c]const c.ds4_model, weights: [*c]const c.ds4_weights, layer: u32, rows: u32, flags: u32, have_graph: bool, streaming: bool, layers: u32, job: ?*pagein.Job) bool {
    return pagein.layerStart(model, weights, layer, rows, flags, have_graph, streaming, layers, job);
}
export fn ds4_native_pagein_selected_start(model: [*c]const c.ds4_model, weights: [*c]const c.ds4_layer_weights, selected: ?*c.ds4_gpu_tensor, layer: u32, rows: u32, gate_bytes: u64, down_bytes: u64, have_graph: bool, streaming: bool, experts: u32, used: u32, job: ?*pagein.Job) bool {
    return pagein.selectedStart(model, weights, selected, layer, rows, gate_bytes, down_bytes, have_graph, streaming, experts, used, job);
}
export fn ds4_native_pagein_join(job: ?*pagein.Job, selected: bool) bool {
    return pagein.join(job, selected);
}
export fn ds4_native_pagein_prepare_start(model: [*c]const c.ds4_model, weights: [*c]const c.ds4_weights, layer: u32, rows: u32, flags: u32, have_graph: bool, streaming: bool, layers: u32, values: [*c]pagein.Slot, count: u32) bool {
    return pagein.prepareStart(model, weights, layer, rows, flags, have_graph, streaming, layers, slots(values, count));
}
export fn ds4_native_pagein_prepare_join_layer(model: [*c]const c.ds4_model, weights: [*c]const c.ds4_weights, layer: u32, rows: u32, flags: u32, have_graph: bool, streaming: bool, layers: u32, values: [*c]pagein.Slot, count: u32) bool {
    return pagein.prepareJoin(model, weights, layer, rows, flags, have_graph, streaming, layers, slots(values, count));
}
export fn ds4_native_pagein_prepare_join_all(values: [*c]pagein.Slot, count: u32) bool {
    return pagein.prepareJoinAll(slots(values, count));
}
