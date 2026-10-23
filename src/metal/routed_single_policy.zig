//! Single-token routed-MoE path selection. It consumes the real invocation and
//! one immutable GPU/resource-fact snapshot; it owns no cache or GPU handles.
const c = @import("metal_c");
const Batch = @import("routed_batch.zig");
const Request = c.ds4_metal_routed_request;
const Facts = c.ds4_metal_single_facts;
const Pipeline = c.ds4_metal_hc_pipeline;
const Mv = c.ds4_gpu_mul_mv_id_args;
const iq2 = c.DS4_METAL_TENSOR_IQ2_XXS;
const q2 = c.DS4_METAL_TENSOR_Q2_K;
const q4 = c.DS4_METAL_TENSOR_Q4_K;
const mx = c.DS4_METAL_TENSOR_MXFP4;
const present = Batch.present;
const absent = Batch.absent;
const cached = Batch.cached;
const has = Batch.has;
fn extra(f: *const Facts, key: usize) ?*const Pipeline {
    return f.cached[key];
}
fn extraPair(f: *const Facts, a: usize, b: usize) bool {
    return extra(f, a) != null and extra(f, b) != null;
}
fn commonPair(f: *const Facts, a: usize, b: usize) bool {
    return Batch.pairPresent(&f.batch, a, b);
}

pub const Policy = struct {
    pair_pipeline: ?*const Pipeline = null,
    sum_pipeline: ?*const Pipeline = null,
    pair_nsg: u32 = 2,
    sum_nsg: u32 = 2,
    fused: bool = false,
    direct: bool = false,
    grouped: bool = false,
    group_size: u32 = 0,
    group6: bool = false,
    group8: bool = false,
    group24: bool = false,
    group24_exact: bool = false,
    exact: bool = false,
    address: bool = false,
    table: bool = false,
    table_auto: bool = false,
    address_auto: bool = false,
    queue: bool = false,
    gather: bool = false,
    slots_q4: bool = false,
    slots_iq2: bool = false,
    slots_mx: bool = false,
    stream_iq2: bool = false,
    selected_profile: bool = false,
    shared_event: bool = false,
    base_views: bool = false,
    transient_views: bool = false,

    pub fn slots(self: Policy) bool {
        return self.slots_q4 or self.slots_iq2 or self.slots_mx or self.stream_iq2;
    }
    pub fn viewMode(self: Policy) [*:0]const u8 {
        return if (self.base_views) "base" else if (self.transient_views) "transient" else "cached";
    }
    pub fn slotPair(self: Policy, f: *const Facts) ?*const Pipeline {
        return if (self.slots_iq2) extra(f, c.DS4_SINGLE_SLOTS_IQ2_PAIR) else cached(&f.batch, if (self.slots_mx) c.DS4_ROUTED_SLOTS_MX_SWI else c.DS4_ROUTED_SLOTS_Q4_SWI);
    }
    pub fn slotSum(self: Policy, f: *const Facts) ?*const Pipeline {
        return if (self.slots_iq2) extra(f, c.DS4_SINGLE_SLOTS_Q2_SUM) else cached(&f.batch, if (self.slots_mx) c.DS4_ROUTED_SLOTS_MX_SUM else c.DS4_ROUTED_SLOTS_Q4_SUM);
    }
    pub fn selectedPath(self: Policy) [*:0]const u8 {
        return if (self.stream_iq2) "iq2/iq2" else if (self.slots_iq2) "iq2/q2" else if (self.slots_mx) "mxfp4/mxfp4" else "q4/q4";
    }
    pub fn boundary(self: Policy) bool {
        return (self.grouped and
            absent("DS4_METAL_DISABLE_Q4_GROUPED_BOUNDARY")) or (self.exact and
            absent("DS4_METAL_DISABLE_Q4_EXACT_BOUNDARY")) or (self.table and
            present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") and
            !self.queue and
            absent("DS4_METAL_DISABLE_Q4_TABLE_BOUNDARY"));
    }
};

/// The nsg=1 MXFP4 variants retain the lane accumulation order; only the
/// row-to-SIMD-group mapping changes. Evaluate env settings on each dispatch.
fn specialize(q: *const Request, f: *const Facts, add_in: bool, gate: Mv, down: Mv) Policy {
    const b = &f.batch;
    const quality = has(b, c.DS4_ROUTED_QUALITY);
    const streaming = has(b, c.DS4_ROUTED_SSD);
    const pre = has(b, c.DS4_ROUTED_PRE_M5);
    const m5 = has(b, c.DS4_ROUTED_M5);
    const write_clamped = present("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    const resident = q.force_resident or !streaming;
    const mx_pair = q.gate_type == mx and q.down_type == mx;
    const tg_multiple = pre and
        absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_TG_MULTIPLE") and
        mx_pair and
        q.used_experts == 6 and
        b.world == 1 and
        !add_in and
        !quality and
        !write_clamped and
        resident;
    const fixed_pair = tg_multiple and
        absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_FIXED_ROUTE_PAIR") and
        q.input_width == 4096 and
        q.middle_width == 2048 and
        q.total_experts == 256 and
        q.gate_row_bytes == 2176 and
        gate.nei0 == 6 and
        gate.nei1 == 1 and
        gate.ne11 == 1 and
        gate.tp_world == 1 and
        gate.tp_expert_base == 0 and
        extra(f, c.DS4_SINGLE_MX_FIXED_PAIR) != null;
    const fixed_sum = fixed_pair and
        absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_FIXED_ROUTE_SUM6") and
        q.output_width == 4096 and
        q.down_row_bytes == 1088 and
        q.down_expert_bytes == 4456448 and
        down.nei0 == 6 and
        down.nei1 == 1 and
        down.ne11 == 6 and
        down.tp_world == 1 and
        down.tp_expert_base == 0 and
        down.tp_addend == 0 and
        extra(f, c.DS4_SINGLE_MX_FIXED_SUM) != null;
    const full_sum = fixed_sum and
        absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_SUM6_FULL_ROWS") and
        down.ne00 == 2048 and
        down.ne01 == 4096 and
        down.ne0 == 4096 and
        down.nr0 == 2 and
        extra(f, c.DS4_SINGLE_MX_FULL_SUM) != null;
    // Every specialized trip count is proved by these real packet/shape
    // checks. The per-lane K walk and accumulation order stay unchanged.
    const static_pair = fixed_pair and
        absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_STATIC_TRIP") and
        gate.ne00 == 4096 and
        gate.nb01 == 2176 and
        extra(f, c.DS4_SINGLE_MX_STATIC_PAIR) != null;
    const static_sum = full_sum and static_pair and down.nb01 == 1088 and down.nei0 == 6 and extra(f, c.DS4_SINGLE_MX_STATIC_SUM) != null;
    const tp_static = m5 and
        absent("DS4_METAL_DISABLE_M5_TP_MXFP4_STATIC") and
        mx_pair and
        q.used_experts == 6 and
        q.total_experts == 256 and
        q.input_width == 4096 and
        q.middle_width == 2048 and
        q.output_width == 4096 and
        q.gate_row_bytes == 2176 and
        q.gate_expert_bytes == 4456448 and
        q.down_row_bytes == 1088 and
        q.down_expert_bytes == 4456448 and
        gate.nei0 == 6 and
        gate.nei1 == 1 and
        gate.ne11 == 1 and
        down.nei0 == 6 and
        down.nei1 == 1 and
        down.ne11 == 6 and
        gate.tp_world == 2 and
        down.tp_world == 2 and
        ((b.rank == 0 and
            b.first_expert == 0) or (b.rank == 1 and
            b.first_expert +% b.bound_experts == q.total_experts)) and
        !quality and
        !write_clamped and
        resident and
        commonPair(f, c.DS4_ROUTED_MX_TP_SWI, c.DS4_ROUTED_MX_TP_SUM);
    var p = Policy{};
    if (q.gate_type == iq2) p.pair_pipeline = cached(b, c.DS4_ROUTED_IQ2_SWI) else if (q.gate_type == q4) p.pair_pipeline = cached(b, c.DS4_ROUTED_Q4_SWI) else if (q.gate_type == mx) {
        p.pair_pipeline = cached(b, c.DS4_ROUTED_MX_SWI);
        if (tp_static) {
            p.pair_pipeline = cached(b, c.DS4_ROUTED_MX_TP_SWI);
            p.pair_nsg = 1;
        } else if (pre and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_NSG1")) {
            p.pair_pipeline = extra(f, if (static_pair) c.DS4_SINGLE_MX_STATIC_PAIR else if (fixed_pair) c.DS4_SINGLE_MX_FIXED_PAIR else if (tg_multiple) c.DS4_SINGLE_MX_TG_PAIR else c.DS4_SINGLE_MX_NSG1_PAIR);
            p.pair_nsg = 1;
        }
    }
    const iq2_pack2 = q.gate_type == iq2 and
        q.used_experts == 6 and
        q.total_experts == 256 and
        q.input_width == 4096 and
        q.middle_width == 2048 and
        absent("DS4_METAL_DISABLE_M5_IQ2_PAIR_PACK2") and
        m5;
    if (iq2_pack2 and extra(f, c.DS4_SINGLE_IQ2_PACK2) != null) {
        p.pair_pipeline = extra(f, c.DS4_SINGLE_IQ2_PACK2);
        p.pair_nsg = 4;
    }
    p.fused = !quality and !write_clamped and absent("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") and p.pair_pipeline != null;
    if (q.down_type == q2) p.sum_pipeline = cached(b, c.DS4_ROUTED_Q2_SUM) else if (q.down_type == q4) p.sum_pipeline = cached(b, c.DS4_ROUTED_Q4_SUM) else if (q.down_type == mx) {
        p.sum_pipeline = cached(b, c.DS4_ROUTED_MX_SUM);
        if (tp_static) {
            p.sum_pipeline = cached(b, c.DS4_ROUTED_MX_TP_SUM);
            p.sum_nsg = 1;
        } else if (pre and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_NSG1")) {
            p.sum_pipeline = extra(f, if (static_sum) c.DS4_SINGLE_MX_STATIC_SUM else if (full_sum) c.DS4_SINGLE_MX_FULL_SUM else if (fixed_sum) c.DS4_SINGLE_MX_FIXED_SUM else if (tg_multiple) c.DS4_SINGLE_MX_TG_SUM else c.DS4_SINGLE_MX_NSG1_SUM);
            p.sum_nsg = 1;
        }
    } else if (q.down_type == iq2 and b.world == 2) {
        // Streaming experts are not at model-map offsets: only the resident
        // GLM TP split may use this ordinary IQ2 down-sum.
        p.sum_pipeline = extra(f, c.DS4_SINGLE_IQ2_SUM);
    }
    p.direct = !quality and (q.used_experts == 6 or (q.used_experts == 8 and b.world == 2)) and p.sum_pipeline != null;
    return p;
}
fn parallelValid(q: *const Request, f: *const Facts, add_in: bool, p: Policy) bool {
    const b = &f.batch;
    if (!f.parallel_pending) return true;
    // A concurrent encoder destroys implicit dependencies. Only the resident
    // fused producer -> direct-sum consumer pair has explicit level barriers.
    const common = f.parallel_mode == 2 and
        q.used_experts == 6 and
        q.total_experts == 256 and
        q.input_width == 4096 and
        q.middle_width == 2048 and
        q.output_width == 4096 and
        !add_in and
        (q.force_resident or !has(b, c.DS4_ROUTED_SSD)) and
        absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and
        p.fused and
        p.direct;
    const iq2_route = q.gate_type == iq2 and
        q.down_type == q2 and
        q.gate_row_bytes == 1056 and
        q.gate_expert_bytes == 2162688 and
        q.down_row_bytes == 672 and
        q.down_expert_bytes == 2752512 and
        b.first_expert == 0 and
        b.bound_experts == q.total_experts and
        b.rank == 0 and
        b.world == 1;
    const mx_route = q.gate_type == mx and
        q.down_type == mx and
        q.gate_row_bytes == 2176 and
        q.gate_expert_bytes == 4456448 and
        q.down_row_bytes == 1088 and
        q.down_expert_bytes == 4456448 and
        ((b.rank == 0 and
            b.first_expert == 0) or (b.rank == 1 and
            b.first_expert +% b.bound_experts == q.total_experts)) and
        b.world == 2;
    return common and (iq2_route or mx_route);
}
fn groupSize(total: u32) u32 {
    var size: u32 = 32;
    const text = c.getenv("DS4_METAL_Q4_EXPERT_GROUP_SIZE");
    if (text != null and text[0] != 0) {
        var end: [*c]u8 = null;
        const amount = c.strtoul(text, &end, 10);
        if (end != text and end.* == 0 and amount > 0 and amount <= 0xffffffff) size = @intCast(amount);
    }
    if (size == 0) size = 1;
    return @min(size, total);
}

fn selectedProfile() ?[*:0]const u8 {
    const raw = c.getenv("DS4_METAL_SELECTED_PROFILE");
    const fallback = c.getenv("DS4_METAL_Q4_SELECTED_PROFILE");
    return if (raw != null) @ptrCast(raw) else if (fallback != null) @ptrCast(fallback) else null;
}
fn selectedLayer(layer: u32) bool {
    const raw = c.getenv("DS4_METAL_SELECTED_PROFILE_LAYER");
    const fallback = c.getenv("DS4_METAL_Q4_SELECTED_PROFILE_LAYER");
    const value = if (raw != null) raw else fallback;
    if (value == null or value[0] == 0) return true;
    var end: [*c]u8 = null;
    const amount = c.strtol(value, &end, 10);
    return end != null and end.* == 0 and amount >= 0 and @as(u32, @truncate(@as(c_ulong, @intCast(amount)))) == layer;
}
pub fn choose(q: *const Request, f: *const Facts, add_in: bool, gate: Mv, down: Mv, gate_bytes: u64, down_bytes: u64) ?Policy {
    const b = &f.batch;
    const streaming = has(b, c.DS4_ROUTED_SSD);
    var p = specialize(q, f, add_in, gate, down);
    if (!parallelValid(q, f, add_in, p)) {
        _ = Batch.errorLine("ds4: concurrent FFN requires a supported resident fused pair-SwiGLU + direct sum6 route");
        return null;
    }
    // Only fused ID pair+sum owns the TP expert split. Other variants would
    // compute complete sums on both ranks and double the eventual combine.
    if ((b.world > 1 or add_in) and !(p.fused and p.direct)) {
        _ = Batch.errorLine("ds4: tensor-parallel routed MoE requires the fused pair+sum6 decode path");
        return null;
    }
    const large = gate_bytes >= 2 << 30 and down_bytes >= 2 << 30;
    const q4_pair = q.gate_type == q4 and q.down_type == q4;
    // Grouped Q4 walks every expert window and was much slower on PRO than
    // active selected slots. Retain its explicit profiling opt-in.
    p.grouped = q4_pair and
        q.used_experts == 6 and
        q.total_experts >= 128 and
        large and
        p.fused and
        p.direct and
        commonPair(f, c.DS4_ROUTED_GROUP_Q4_SWI, c.DS4_ROUTED_GROUP_Q4_SUM) and
        !streaming and
        present("DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS") and
        absent("DS4_METAL_DISABLE_Q4_GROUPED_EXPERTS");
    p.group_size = if (p.grouped) groupSize(q.total_experts) else 0;
    const q4_small = !streaming and !p.grouped and q4_pair and q.used_experts == 6 and q.total_experts == 384 and large and p.direct;
    p.group6 = q4_small and
        p.fused and
        commonPair(f, c.DS4_ROUTED_GROUP6_SWI, c.DS4_ROUTED_GROUP6_SUM) and
        present("DS4_METAL_ENABLE_Q4_GROUP6_EXPERT_TABLE") and
        absent("DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE");
    p.group8 = q4_small and
        !p.group6 and
        p.fused and
        commonPair(f, c.DS4_ROUTED_GROUP8_SWI, c.DS4_ROUTED_GROUP8_SUM) and
        present("DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE") and
        absent("DS4_METAL_DISABLE_Q4_GROUP8_EXPERT_TABLE");
    p.group24 = q4_small and
        !p.group6 and
        !p.group8 and
        commonPair(f, c.DS4_ROUTED_GROUP24, c.DS4_ROUTED_GROUP24_SUM) and
        present("DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE") and
        absent("DS4_METAL_DISABLE_Q4_GROUP24_EXPERT_TABLE");
    p.group24_exact = p.group24 and present("DS4_METAL_Q4_GROUP24_EXACT_VIEWS") and absent("DS4_METAL_Q4_GROUP24_BASE_VIEWS");
    const no_group = !p.grouped and !p.group6 and !p.group8 and !p.group24;
    // The full-tensor ID path is an arithmetic analogue of IQ2, but multi-GiB
    // PRO tensors remain opt-in: exact selected slices measured much faster.
    p.exact = no_group and
        q4_pair and
        q.used_experts == 6 and
        q.total_experts == 384 and
        large and
        b.max_buffer_bytes != 0 and
        gate_bytes <= b.max_buffer_bytes and
        down_bytes <= b.max_buffer_bytes and
        p.fused and
        p.direct and
        !streaming and
        present("DS4_METAL_ENABLE_Q4_EXACT_TENSOR_ID") and
        absent("DS4_METAL_DISABLE_Q4_EXACT_TENSOR_ID");
    p.table_auto = q4_pair and Batch.tableAuto(q, b, gate_bytes, down_bytes);
    p.address_auto = q4_pair and
        !streaming and
        present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") and
        Batch.q4SelectedAllowed(b) and
        q.total_experts == 384 and
        q.used_experts == 6 and
        large and
        commonPair(f, c.DS4_ROUTED_ADDR_Q4_SWI, c.DS4_ROUTED_ADDR_Q4_SUM) and
        absent("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") and
        absent("DS4_METAL_DISABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") and
        absent("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE") and
        has(b, c.DS4_ROUTED_QUEUE_RESIDENCY);
    p.queue = (p.table_auto or p.address_auto or present("DS4_METAL_Q4_TABLE_QUEUE_RESIDENCY_SET")) and has(b, c.DS4_ROUTED_QUEUE_RESIDENCY);
    const indirect = !streaming and no_group and !p.exact and q4_pair and q.used_experts == 6 and q.total_experts == 384 and large and p.fused and p.direct;
    const residency = present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") or p.queue or present("DS4_METAL_Q4_TABLE_MODEL_RESIDENCY_SET") or present("DS4_METAL_USE_QUEUE_RESIDENCY_SET");
    p.address = indirect and
        commonPair(f, c.DS4_ROUTED_ADDR_Q4_SWI, c.DS4_ROUTED_ADDR_Q4_SUM) and
        (present("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") or p.address_auto) and
        (present("DS4_METAL_Q4_ADDR_USE_RESOURCES") or residency) and
        absent("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE");
    p.table = indirect and
        !p.address and
        commonPair(f, c.DS4_ROUTED_TABLE_Q4_SWI, c.DS4_ROUTED_TABLE_Q4_SUM) and
        b.table_encoders and
        (present("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") or p.table_auto) and
        (present("DS4_METAL_Q4_TABLE_USE_RESOURCES") or residency) and
        absent("DS4_METAL_DISABLE_Q4_EXPERT_TABLE");
    const selectable = no_group and !p.exact and !p.address and !p.table;
    p.gather = selectable and
        q4_pair and
        q.used_experts == 6 and
        q.total_experts == 384 and
        large and
        p.fused and
        p.direct and
        extra(f, c.DS4_SINGLE_Q4_GATHER) != null and
        commonPair(f, c.DS4_ROUTED_SLOTS_Q4_SWI, c.DS4_ROUTED_SLOTS_Q4_SUM) and
        !streaming and
        present("DS4_METAL_ENABLE_Q4_GATHER_SLOTS") and
        absent("DS4_METAL_DISABLE_Q4_GATHER_SLOTS");
    p.slots_q4 = !q.force_resident and
        selectable and
        !p.gather and
        q4_pair and
        q.used_experts == 6 and
        q.total_experts >= 128 and
        (streaming or large) and
        p.fused and
        p.direct and
        Batch.q4SelectedAllowed(b) and
        commonPair(f, c.DS4_ROUTED_SLOTS_Q4_SWI, c.DS4_ROUTED_SLOTS_Q4_SUM) and
        absent("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS");
    p.slots_iq2 = !q.force_resident and
        streaming and
        q.gate_type == iq2 and
        q.down_type == q2 and
        q.used_experts == 6 and
        p.fused and
        p.direct and
        extraPair(f, c.DS4_SINGLE_SLOTS_IQ2_PAIR, c.DS4_SINGLE_SLOTS_Q2_SUM) and
        absent("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
    p.slots_mx = !q.force_resident and
        streaming and
        q.gate_type == mx and
        q.down_type == mx and
        q.used_experts == 6 and
        q.total_experts >= 128 and
        p.fused and
        p.direct and
        commonPair(f, c.DS4_ROUTED_SLOTS_MX_SWI, c.DS4_ROUTED_SLOTS_MX_SUM) and
        absent("DS4_METAL_DISABLE_MXFP4_SELECTED_EXPERT_VIEWS");
    p.stream_iq2 = !q.force_resident and
        streaming and
        q.gate_type == iq2 and
        q.down_type == iq2 and
        q.used_experts <= 8 and
        p.fused and
        commonPair(f, c.DS4_ROUTED_ADDR_IQ2_SWI, c.DS4_ROUTED_ADDR_IQ2) and
        absent("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE");
    p.selected_profile = p.slots() and selectedProfile() != null and selectedLayer(q.layer);
    p.shared_event = p.slots_q4 and present("DS4_METAL_Q4_SELECTED_SHARED_EVENT");
    p.base_views = p.slots_q4 and present("DS4_METAL_Q4_SELECTED_USE_BASE_VIEWS") and absent("DS4_METAL_Q4_SELECTED_EXACT_VIEWS");
    p.transient_views = p.slots_q4 and !p.base_views and present("DS4_METAL_Q4_SELECTED_TRANSIENT_VIEWS");
    return p;
}
