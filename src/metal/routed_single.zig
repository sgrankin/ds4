//! Complete single-token routed-MoE transaction. The selected expert cache and
//! table libraries retain their actual state; this owner controls admission,
//! resource preparation, GPU order and command completion.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Batch = @import("routed_batch.zig");
const Selection = @import("routed_single_selected.zig");
const Dispatch = @import("routed_single_dispatch.zig");
const Policy = @import("routed_single_policy.zig");
const Request = c.ds4_metal_routed_request;
const Facts = c.ds4_metal_single_facts;
const Execution = Dispatch.Execution;
const present = Batch.present;
const cached = Batch.cached;
const max = std.math.maxInt(u64);

fn groupError(count: u32, offset: bool) bool {
    _ = c.fprintf(c.__stderrp, "ds4: Metal routed MoE Q4 group%u %s overflow\n", count, @as([*:0]const u8, if (offset) "offset" else "byte size"));
    return false;
}
fn prepareGroups(e: *Execution, gate_bytes: u64, down_bytes: u64) bool {
    const q = e.q;
    const p = e.policy;
    const count: u32 = if (p.group8) 8 else if (p.group24) 24 else 6;
    const size: u32 = if (p.group8) 48 else if (p.group24) 16 else 64;
    if (size > max / q.gate_expert_bytes or size > max / q.down_expert_bytes) return groupError(count, false);
    const gate_group = @as(u64, size) * q.gate_expert_bytes;
    const down_group = @as(u64, size) * q.down_expert_bytes;
    var slot: u32 = 0;
    while (slot < count) : (slot += 1) {
        // These relative products intentionally retain C's unsigned wrapping.
        const gate = @as(u64, slot) *% gate_group;
        const down = @as(u64, slot) *% down_group;
        if (gate > gate_bytes or gate_group > gate_bytes - gate or down > down_bytes or down_group > down_bytes - down or gate > max - q.gate_offset or gate > max - q.up_offset or down > max - q.down_offset) return groupError(count, true);
        if (!c.ds4_metal_single_wrap(e.resource, 7, q.gate_offset + gate, q.up_offset + gate, q.down_offset + down, gate_group, down_group, @intFromBool(p.group24_exact), slot)) return Dispatch.silent(if (p.group24) 32794 else if (p.group8) 32727 else 32643);
    }
    if (p.gather) {
        if (q.gate_expert_bytes > max / 6 or q.down_expert_bytes > max / 6) {
            _ = Batch.errorLine("ds4: Metal routed MoE Q4 gather scratch byte size overflow");
            return false;
        }
        if (!c.ds4_metal_single_gather_storage(e.resource, 6 * q.gate_expert_bytes, 6 * q.down_expert_bytes)) return Dispatch.silent(32670);
    }
    return true;
}
fn prepare(e: *Execution, selected: *Selection.State, gate_bytes: u64, down_bytes: u64) bool {
    const q = e.q;
    const p = e.policy;
    if (p.address or p.table) {
        return c.ds4_metal_single_tables(e.resource, p.address, if (p.address) p.address_auto else p.table_auto) != 0 or Dispatch.silent(if (p.address) 32546 else 32573);
    }
    if (p.exact) {
        return c.ds4_metal_single_wrap(e.resource, 7, q.gate_offset, q.up_offset, q.down_offset, gate_bytes, down_bytes, 1, 0xffffffff) or Dispatch.silent(32601);
    }
    if (p.group6 or p.group8 or p.group24 or p.gather) return prepareGroups(e, gate_bytes, down_bytes);
    if (p.slots()) return selected.prepare(e);
    if (!p.grouped) return c.ds4_metal_single_wrap(e.resource, 7, q.gate_offset, q.up_offset, q.down_offset, gate_bytes, down_bytes, 0, 0xffffffff) or Dispatch.silent(33326);
    return true;
}
fn ordinaryPair(e: *const Execution) ?*const c.ds4_metal_hc_pipeline {
    if (Batch.has(&e.facts.batch, c.DS4_ROUTED_QUALITY)) return null;
    return switch (e.q.gate_type) {
        c.DS4_METAL_TENSOR_IQ2_XXS => cached(&e.facts.batch, c.DS4_ROUTED_IQ2_PAIR),
        c.DS4_METAL_TENSOR_Q4_K => cached(&e.facts.batch, c.DS4_ROUTED_Q4_PAIR),
        else => null,
    };
}
fn path(e: *const Execution, s: Selection.State) [*:0]const u8 {
    const p = e.policy;
    return if (p.grouped) "q4_grouped_pair_swiglu" else if (p.group6) "q4_group6_pair_swiglu" else if (p.group8) "q4_group8_pair_swiglu" else if (p.group24) "q4_group24_split_gate_up" else if (p.exact) "q4_exact_pair_swiglu" else if (p.address) "q4_addr_pair_swiglu" else if (p.table) "q4_table_pair_swiglu" else if (p.gather) "q4_gather_slots6_pair_swiglu" else if (s.split_deferred) "iq2_stream_split_pair_swiglu" else if (s.masked) "iq2_stream_addr_mask_pair_swiglu" else if (s.address) "iq2_stream_addr_pair_swiglu" else if (p.slots_iq2) "iq2_slots6_pair_swiglu" else if (p.slots_mx) "mxfp4_slots6_pair_swiglu" else if (p.slots_q4) "q4_slots6_pair_swiglu" else if (p.fused) "pair_swiglu" else if (ordinaryPair(e) != null) "pair" else "single";
}
fn windows(e: *Execution, down: bool) bool {
    const q = e.q;
    var first: u32 = 0;
    while (first < q.total_experts) : (first +%= e.policy.group_size) {
        const count = @min(e.policy.group_size, q.total_experts - first);
        if (!e.window(down, first, count, down and first != 0)) return false;
    }
    return true;
}
fn gateUp(e: *Execution, selected: *Selection.State) bool {
    const p = e.policy;
    const f = &e.facts.batch;
    if (selected.staged_gate) return true;
    if (p.grouped) return windows(e, false);
    if (p.address) return e.pair(c.DS4_SINGLE_Q4_ADDRESS_PAIR, cached(f, c.DS4_ROUTED_ADDR_Q4_SWI), 0);
    if (p.table) return e.ops.mv(c.DS4_ROUTED_TABLE_PAIR, cached(f, c.DS4_ROUTED_TABLE_Q4_SWI), e.gate, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
    if (p.group6 or p.group8) return e.pair(c.DS4_SINGLE_GROUP_ARRAY_PAIR, cached(f, if (p.group8) c.DS4_ROUTED_GROUP8_SWI else c.DS4_ROUTED_GROUP6_SWI), if (p.group8) 8 else 6);
    if (p.group24) {
        const program = cached(f, c.DS4_ROUTED_GROUP24);
        const group = std.mem.zeroes(c.ds4_gpu_moe_expert_group_args);
        const split = std.mem.zeroes(c.ds4_gpu_stream_expert_split_args);
        return e.fixed(c.DS4_SINGLE_GROUP24_MV, program, e.gate, 0, 24, group, split) and e.fixed(c.DS4_SINGLE_GROUP24_MV, program, e.gate, 1, 24, group, split);
    }
    if (p.gather or p.slots()) {
        if (selected.address) {
            if (selected.masked) return if (selected.split_deferred) selected.split(e) else e.masked(false, 0x3f);
            return e.ops.mv(c.DS4_ROUTED_ADDR_PAIR, cached(f, c.DS4_ROUTED_ADDR_IQ2_SWI), e.gate, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
        }
        return (!selected.cache or c.ds4_metal_single_mark(e.resource, 0)) and e.pair(c.DS4_SINGLE_ARRAY_PAIR, p.slotPair(e.facts), 6);
    }
    if (p.fused) return e.ops.mv(c.DS4_ROUTED_PAIR_SWI, p.pair_pipeline, e.gate, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, p.pair_nsg, false);
    if (ordinaryPair(e)) |program| return e.ops.mv(c.DS4_ROUTED_PAIR, program, e.gate, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
    const program = Batch.mvPipeline(f, e.q.gate_type);
    const nsg = Batch.simdgroups(e.q.gate_type);
    const own_rows = e.q.gate_type == c.DS4_METAL_TENSOR_Q8_0;
    return e.ops.mv(c.DS4_ROUTED_MV, program, e.gate, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, nsg, own_rows) and
        e.ops.mv(c.DS4_ROUTED_MV, program, e.gate, 1, c.DS4_ROUTED_X, c.DS4_ROUTED_UP, nsg, own_rows);
}
fn projectDown(e: *Execution, selected: *Selection.State) bool {
    const p = e.policy;
    const f = &e.facts.batch;
    if (selected.split_completed) return true;
    if (p.grouped) return windows(e, true);
    if (p.address) return e.sum(c.DS4_SINGLE_Q4_ADDRESS_SUM, cached(f, c.DS4_ROUTED_ADDR_Q4_SUM), 0);
    if (p.table) return e.ops.mv(c.DS4_ROUTED_TABLE_SUM, cached(f, c.DS4_ROUTED_TABLE_Q4_SUM), e.down, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_OUT, 2, false);
    if (p.group6 or p.group8) return e.sum(c.DS4_SINGLE_GROUP_ARRAY_SUM, cached(f, if (p.group8) c.DS4_ROUTED_GROUP8_SUM else c.DS4_ROUTED_GROUP6_SUM), if (p.group8) 8 else 6);
    if (p.group24) return e.sum(c.DS4_SINGLE_GROUP24_SUM, cached(f, c.DS4_ROUTED_GROUP24_SUM), 24);
    if (p.gather or p.slots()) {
        if (selected.address) {
            if (e.q.down_type == c.DS4_METAL_TENSOR_IQ2_XXS) return e.ops.mv(c.DS4_ROUTED_ADDR_MV, cached(f, c.DS4_ROUTED_ADDR_IQ2), e.down, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_DOWN, 2, false);
            if (selected.masked) return e.masked(true, 0x3f);
            return e.ops.mv(c.DS4_ROUTED_ADDR_SUM, cached(f, c.DS4_ROUTED_ADDR_Q2_SUM), e.down, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_OUT, 2, false);
        }
        return (!selected.cache or c.ds4_metal_single_mark(e.resource, 0)) and e.sum(c.DS4_SINGLE_ARRAY_SUM, p.slotSum(e.facts), 6);
    }
    if (p.direct) return e.ops.mv(c.DS4_ROUTED_DOWN_SUM, p.sum_pipeline, e.down, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_OUT, p.sum_nsg, false);
    return e.ops.mv(c.DS4_ROUTED_MV, Batch.mvPipeline(f, e.q.down_type), e.down, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_DOWN, Batch.simdgroups(e.q.down_type), e.q.down_type == c.DS4_METAL_TENSOR_Q8_0);
}

pub fn run(request: *const Request, add_in: ?*const c.ds4_gpu_tensor) c_int {
    var armed = c.ds4_metal_single_scope_begin();
    defer c.ds4_metal_single_scope_end(armed);
    if (api.init() == 0) return 0;
    var q = request.*;
    var facts = std.mem.zeroes(Facts);
    c.ds4_metal_single_get_facts(q.total_experts, &facts);
    // Bind the TP-owned expert range and let shaders rebase selected IDs.
    q.gate_offset +%= @as(u64, facts.batch.first_expert) *% q.gate_expert_bytes;
    q.up_offset +%= @as(u64, facts.batch.first_expert) *% q.gate_expert_bytes;
    q.down_offset +%= @as(u64, facts.batch.first_expert) *% q.down_expert_bytes;
    if (q.out == null or q.gate == null or q.up == null or q.mid == null or q.x == null or q.model == null or q.selected == null or q.weights == null or q.total_experts == 0 or q.used_experts == 0 or q.used_experts > 8 or q.gate_expert_bytes == 0 or q.down_expert_bytes == 0 or q.gate_row_bytes == 0 or q.down_row_bytes == 0) return @intFromBool(Dispatch.silent(32044));
    if (q.input_width % 256 != 0 or q.middle_width % 256 != 0) return @intFromBool(Dispatch.silent(32047));
    c.ds4_metal_single_note_token(q.layer);
    const middle = @as(u64, q.used_experts) *% q.middle_width *% 4;
    const tensors = .{ q.x, q.gate, q.up, q.mid, q.out, q.selected, q.weights };
    const sizes = .{ @as(u64, q.input_width) * 4, middle, middle, middle, @as(u64, q.output_width) * 4, @as(u64, q.used_experts) * 4, @as(u64, q.used_experts) * 4 };
    inline for (tensors, sizes) |tensor, required| {
        if (api.tensorHasBuffer(tensor) == 0 or api.tensorBytes(tensor) < required) return Batch.errorLine("ds4: Metal routed tensor MoE received undersized activation buffers");
    }
    if (q.used_experts > 1 and (api.tensorHasBuffer(q.experts) == 0 or api.tensorBytes(q.experts) < @as(u64, q.used_experts) *% q.output_width *% 4)) return Batch.errorLine("ds4: Metal routed tensor MoE received undersized expert output buffer");
    if (q.total_experts > max / q.gate_expert_bytes or q.total_experts > max / q.down_expert_bytes) return Batch.errorLine("ds4: Metal routed MoE tensor byte size overflow");
    const gate_bytes = @as(u64, facts.batch.bound_experts) * q.gate_expert_bytes;
    const down_bytes = @as(u64, facts.batch.bound_experts) * q.down_expert_bytes;
    if (Batch.nr0(q.gate_type) == 0 or Batch.nr0(q.down_type) == 0 or Batch.mvPipeline(&facts.batch, q.gate_type) == null or Batch.mvPipeline(&facts.batch, q.down_type) == null) {
        _ = c.fprintf(c.__stderrp, "ds4: unsupported Metal routed MoE quant types gate=%u down=%u\n", q.gate_type, q.down_type);
        return @intFromBool(Dispatch.silent(32158));
    }
    const gate = Batch.mvArgs(&q, &facts.batch, false);
    var down_args = Batch.mvArgs(&q, &facts.batch, true);
    down_args.tp_addend = @intFromBool(add_in != null);
    const policy = Policy.choose(&q, &facts, add_in != null, gate, down_args, gate_bytes, down_bytes) orelse return 0;
    if (!policy.slots()) Selection.clearOverride();
    const resource = c.ds4_metal_single_open(&q, add_in) orelse return 0;
    defer c.ds4_metal_single_close(resource);
    var execution = Execution.init(&q, &facts, policy, resource, gate, down_args);
    execution.ops.command.label = "routed tensor MoE";
    defer execution.deinit();
    const boundary = policy.boundary() and c.ds4_gpu_commands_active() != 0;
    var selected = Selection.State{};
    if (!prepare(&execution, &selected, gate_bytes, down_bytes)) return 0;
    if (boundary and (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0)) return @intFromBool(Dispatch.silent(33305));
    execution.path = path(&execution, selected);
    if (!execution.begin()) return 0;
    var ok = true;
    if (!selected.staged_gate and policy.gather) {
        ok = execution.gather(0) and execution.gather(1) and execution.gather(2);
        execution.stage("q4_gather", &ok);
    }
    if (!ok) return @intFromBool(Dispatch.silent(33455));
    ok = gateUp(&execution, &selected);
    execution.stage("gate_up", &ok);
    if (ok and (!policy.fused or policy.group24)) ok = execution.ops.activate(false);
    execution.stage("activation_weight", &ok);
    if (ok) {
        var parallel: c.ds4_metal_single_parallel = undefined;
        ok = c.ds4_metal_single_parallel_pending(resource, execution.ops.command.token.?, &parallel);
    }
    if (ok) ok = projectDown(&execution, &selected);
    execution.stage("down", &ok);
    if (ok and q.used_experts > 1 and !policy.direct and !selected.split_completed) ok = execution.ops.sum();
    execution.stage("sum", &ok);
    if (ok) ok = c.ds4_metal_single_parallel_second(execution.ops.command.token.?);
    if (!ok) return @intFromBool(Dispatch.silent(34395));
    if (!execution.ops.command.finish()) return @intFromBool(Dispatch.silent(34397));
    if (boundary and (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0)) return @intFromBool(Dispatch.silent(34372));
    armed = false;
    return 1;
}
