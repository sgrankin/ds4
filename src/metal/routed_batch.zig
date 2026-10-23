//! Batched routed-MoE policy and command transaction. Shader math is unchanged.
//! Cache/table resources and the separate one-row controller are explicit
//! temporary library dependencies. No cache or Engine/Session state is mirrored.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Request = c.ds4_metal_routed_request;
const Facts = c.ds4_metal_routed_facts;
const Pipeline = c.ds4_metal_hc_pipeline;
const Launch = c.ds4_metal_routed_launch;
const Mv = c.ds4_gpu_mul_mv_id_args;
const Mm = c.ds4_gpu_mul_mm_id_args;
const Act = c.ds4_gpu_dsv4_moe_swiglu_weight_args;
const iq2 = c.DS4_METAL_TENSOR_IQ2_XXS;
const q2 = c.DS4_METAL_TENSOR_Q2_K;
const q4 = c.DS4_METAL_TENSOR_Q4_K;
const mx = c.DS4_METAL_TENSOR_MXFP4;
const q8 = c.DS4_METAL_TENSOR_Q8_0;
const q8k = c.DS4_METAL_TENSOR_Q8_K;
var mxfp4_mpp: ?bool = null;
var q4_mpp: ?bool = null;
fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
pub fn present(comptime name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn absent(comptime name: [*:0]const u8) bool {
    return !present(name);
}
pub fn has(f: *const Facts, mask: u32) bool {
    return f.flags & mask != 0;
}
fn narrow(value: u32) i32 {
    return @bitCast(value);
}
fn bytes(value: [*c]const u8) []const u8 {
    return std.mem.span(value);
}
pub fn errorLine(comptime line: [*:0]const u8) c_int {
    return api.reportError(line);
}
pub fn cached(f: *const Facts, key: usize) ?*const Pipeline {
    return f.cached[key];
}
pub fn pairPresent(f: *const Facts, a: usize, b: usize) bool {
    return cached(f, a) != null and cached(f, b) != null;
}
fn dynamic(kind: u32, name: [*:0]const u8, nsg: i16) ?*const Pipeline {
    return c.ds4_metal_routed_pipeline(kind, name, nsg);
}
pub fn typeName(kind: u32) [*:0]const u8 {
    return switch (kind) {
        iq2 => "iq2_xxs",
        q2 => "q2_k",
        q4 => "q4_k",
        c.DS4_METAL_TENSOR_Q5_K => "q5_k",
        c.DS4_METAL_TENSOR_Q6_K => "q6_k",
        mx => "mxfp4",
        else => "unknown",
    };
}
pub fn nr0(kind: u32) u32 {
    return switch (kind) {
        q8, q8k, q4, mx => 2,
        iq2, q2 => 4,
        else => 0,
    };
}
pub fn shared(kind: u32) u64 {
    return switch (kind) {
        q8 => 256,
        iq2 => 2176,
        mx => 128,
        else => 0,
    };
}
pub fn simdgroups(kind: u32) u64 {
    return if (kind == q8) 4 else 2;
}
pub fn mvPipeline(f: *const Facts, kind: u32) ?*const Pipeline {
    return switch (kind) {
        q8 => dynamic(1, "kernel_mul_mv_id_q8_0_f32", 4),
        q8k => dynamic(1, "kernel_mul_mv_id_q8_K_f32", 2),
        iq2 => cached(f, c.DS4_ROUTED_IQ2),
        q2 => cached(f, c.DS4_ROUTED_Q2),
        q4 => cached(f, c.DS4_ROUTED_Q4),
        mx => cached(f, c.DS4_ROUTED_MX),
        else => null,
    };
}
fn mmPipeline(kind: u32, half: bool) ?*const Pipeline {
    const name: [*:0]const u8 = switch (kind) {
        q8 => if (half) "kernel_mul_mm_id_q8_0_f16" else "kernel_mul_mm_id_q8_0_f32",
        q8k => if (half) "kernel_mul_mm_id_q8_K_f16" else "kernel_mul_mm_id_q8_K_f32",
        iq2 => if (half) "kernel_mul_mm_id_iq2_xxs_f16" else "kernel_mul_mm_id_iq2_xxs_f32",
        q2 => if (half) "kernel_mul_mm_id_q2_K_f16" else "kernel_mul_mm_id_q2_K_f32",
        q4 => if (half) "kernel_mul_mm_id_q4_K_f16" else "kernel_mul_mm_id_q4_K_f32",
        c.DS4_METAL_TENSOR_Q5_K => if (half) "kernel_mul_mm_id_q5_K_f16" else "kernel_mul_mm_id_q5_K_f32",
        c.DS4_METAL_TENSOR_Q6_K => if (half) "kernel_mul_mm_id_q6_K_f16" else "kernel_mul_mm_id_q6_K_f32",
        mx => if (half) "kernel_mul_mm_id_mxfp4_f16" else "kernel_mul_mm_id_mxfp4_f32",
        else => return null,
    };
    return dynamic(2, name, 0);
}
fn mapName(used: u32) ?[*:0]const u8 {
    return switch (used) {
        1 => "kernel_mul_mm_id_map0_ne20_1",
        2 => "kernel_mul_mm_id_map0_ne20_2",
        4 => "kernel_mul_mm_id_map0_ne20_4",
        5 => "kernel_mul_mm_id_map0_ne20_5",
        6 => "kernel_mul_mm_id_map0_ne20_6",
        8 => "kernel_mul_mm_id_map0_ne20_8",
        10 => "kernel_mul_mm_id_map0_ne20_10",
        16 => "kernel_mul_mm_id_map0_ne20_16",
        22 => "kernel_mul_mm_id_map0_ne20_22",
        else => null,
    };
}
fn autoBound(comptime name: [*:0]const u8, fallback: u32) u32 {
    const value = c.getenv(name);
    if (value != null and value[0] != 0) {
        var end: [*c]u8 = null;
        const amount = c.strtol(value, &end, 10);
        if (end != value) {
            if (amount <= 0) return 0;
            if (@as(c_ulong, @intCast(amount)) > std.math.maxInt(u32)) return std.math.maxInt(u32);
            return @intCast(amount);
        }
    }
    return fallback;
}
fn selectedAddress(q: *const Request, f: *const Facts) bool {
    if (!has(f, c.DS4_ROUTED_SSD) or q.rows <= 1 or q.total_experts == 0 or q.used_experts != 6 or q.gate_type != iq2 or q.down_type != q2 or has(f, c.DS4_ROUTED_QUALITY) or present("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") or present("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") or present("DS4_METAL_MOE_WRITE_CLAMPED_ACT") or present("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") or f.cache_budget < q.total_experts) return false;
    if (present("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR")) return true;
    const high = autoBound("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX", if (q.total_experts == 384) 800 else if (q.total_experts == 256) 760 else 0);
    const low = autoBound("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN", if (q.total_experts == 384 or q.total_experts == 256) 2 else 0);
    return high != 0 and q.rows >= low and q.rows <= high;
}
pub fn q4SelectedAllowed(f: *const Facts) bool {
    if (has(f, c.DS4_ROUTED_SSD)) return true;
    if (has(f, c.DS4_ROUTED_GLM)) return false;
    return present("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS") or present("DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS") or present("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") or present("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") or present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") or present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO");
}
pub fn tableAuto(q: *const Request, f: *const Facts, gate_bytes: u64, down_bytes: u64) bool {
    return q.total_experts == 384 and q.used_experts == 6 and gate_bytes >= 2 << 30 and down_bytes >= 2 << 30 and !has(f, c.DS4_ROUTED_SSD) and present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") and absent("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") and absent("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") and has(f, c.DS4_ROUTED_QUEUE_RESIDENCY);
}
fn single(q: *const Request, f: *const Facts, gate_bytes: u64, down_bytes: u64) bool {
    const table = present("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") or tableAuto(q, f, gate_bytes, down_bytes);
    const grouped = present("DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS") and absent("DS4_METAL_DISABLE_Q4_GROUPED_EXPERTS") and pairPresent(f, c.DS4_ROUTED_GROUP_Q4_SWI, c.DS4_ROUTED_GROUP_Q4_SUM);
    const slots = q4SelectedAllowed(f) and absent("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS") and pairPresent(f, c.DS4_ROUTED_SLOTS_Q4_SWI, c.DS4_ROUTED_SLOTS_Q4_SUM);
    const group6 = q.total_experts == 384 and present("DS4_METAL_ENABLE_Q4_GROUP6_EXPERT_TABLE") and absent("DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE") and pairPresent(f, c.DS4_ROUTED_GROUP6_SWI, c.DS4_ROUTED_GROUP6_SUM);
    const group8 = q.total_experts == 384 and present("DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE") and absent("DS4_METAL_DISABLE_Q4_GROUP8_EXPERT_TABLE") and pairPresent(f, c.DS4_ROUTED_GROUP8_SWI, c.DS4_ROUTED_GROUP8_SUM);
    const group24 = q.total_experts == 384 and present("DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE") and absent("DS4_METAL_DISABLE_Q4_GROUP24_EXPERT_TABLE") and pairPresent(f, c.DS4_ROUTED_GROUP24, c.DS4_ROUTED_GROUP24_SUM);
    const exact = q.total_experts == 384 and f.max_buffer_bytes != 0 and gate_bytes <= f.max_buffer_bytes and down_bytes <= f.max_buffer_bytes and present("DS4_METAL_ENABLE_Q4_EXACT_TENSOR_ID") and absent("DS4_METAL_DISABLE_Q4_EXACT_TENSOR_ID");
    const tables = q.total_experts == 384 and table and absent("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") and pairPresent(f, c.DS4_ROUTED_TABLE_Q4_SWI, c.DS4_ROUTED_TABLE_Q4_SUM) and f.table_encoders;
    const address = q.total_experts == 384 and present("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") and absent("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE") and pairPresent(f, c.DS4_ROUTED_ADDR_Q4_SWI, c.DS4_ROUTED_ADDR_Q4_SUM);
    const common = q.rows == 1 and q.used_experts == 6 and q.total_experts >= 128 and !has(f, c.DS4_ROUTED_QUALITY) and absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and absent("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION");
    const use_q4 = q.gate_type == q4 and q.down_type == q4 and (has(f, c.DS4_ROUTED_SSD) or (gate_bytes >= 2 << 30 and down_bytes >= 2 << 30)) and (grouped or group6 or group8 or group24 or exact or address or tables or slots);
    const use_mx = q.gate_type == mx and q.down_type == mx and pairPresent(f, c.DS4_ROUTED_MX_SWI, c.DS4_ROUTED_MX_SUM) and (!has(f, c.DS4_ROUTED_SSD) or pairPresent(f, c.DS4_ROUTED_SLOTS_MX_SWI, c.DS4_ROUTED_SLOTS_MX_SUM));
    return common and (use_q4 or use_mx);
}
pub const Policy = struct {
    selected_addr: bool = false,
    cached_iq2: bool = false,
    table: bool = false,
    automatic: bool = false,
    queue: bool = false,
    mm: bool = false,
    packed_rhs: bool = false,
    packed_m32: bool = false,
    tiny_pair: bool = false,
    tiny_swi: bool = false,
    tp_static: bool = false,
    v41_decode: bool = false,
    half: bool = false,
    pair_mm: bool = false,
    compact: bool = false,
    scatter: bool = false,
    pair_tail: bool = false,
    down_tail: bool = false,
    down_lut: bool = false,
    packed_limit: u64 = 0,
    fn cached(self: Policy) bool {
        return self.selected_addr or self.cached_iq2;
    }
    fn path(self: Policy) [*:0]const u8 {
        return if (self.table) "q4_table_pair_swiglu" else if (self.selected_addr) "iq2_batch_stream_addr" else if (self.cached_iq2) (if (self.mm) "iq2_cached_mm" else "iq2_cached_mv") else if (self.pair_mm) "mm_id_pair_swiglu" else if (self.mm) "mm_id" else if (self.tiny_swi) "tiny_pair_swiglu" else if (self.tiny_pair) "tiny_pair_mv" else "mv";
    }
};
fn policy(q: *const Request, f: *const Facts, gate_bytes: u64, down_bytes: u64) Policy {
    const quality = has(f, c.DS4_ROUTED_QUALITY);
    const streaming = has(f, c.DS4_ROUTED_SSD);
    const mpp = has(f, c.DS4_ROUTED_MPP);
    const pre = has(f, c.DS4_ROUTED_PRE_M5);
    var p = Policy{};
    // Every unique expert in this layer must fit simultaneously: address
    // kernels consume all of them in one dispatch, with global cache reuse.
    p.selected_addr =
        !q.force_resident and
        selectedAddress(q, f) and
        q.rows > 1 and
        q.total_experts <= 384 and
        !quality and
        absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and
        absent("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") and
        pairPresent(f, c.DS4_ROUTED_ADDR_IQ2_SWI, c.DS4_ROUTED_ADDR_Q2_SUM);
    // Small full-GLM appends reuse cached bytes with the same MV/MM math;
    // large prefills retain sequential whole-layer reads.
    p.cached_iq2 =
        streaming and
        !q.force_resident and
        f.world == 1 and
        !has(f, c.DS4_ROUTED_FULL_LAYER) and
        absent("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER") and
        q.gate_type == iq2 and
        q.down_type == iq2 and
        q.used_experts == 8 and
        q.rows < 256 and
        q.total_experts <= 384 and
        c.ds4_metal_routed_cache_admit(q.gate_expert_bytes, q.down_expert_bytes) >= q.total_experts and
        absent("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE") and
        absent("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE") and
        absent("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") and
        absent("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") and
        absent("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION") and
        absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and
        absent("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");
    // Indirect tables retain every expert. Automatic selection requires a
    // queue residency set and cannot coexist with a bounded SSD cache.
    p.automatic = q.gate_type == q4 and q.down_type == q4 and tableAuto(q, f, gate_bytes, down_bytes);
    p.queue = (p.automatic or present("DS4_METAL_Q4_TABLE_QUEUE_RESIDENCY_SET")) and has(f, c.DS4_ROUTED_QUEUE_RESIDENCY);
    p.table =
        !streaming and
        q.gate_type == q4 and
        q.down_type == q4 and
        q.used_experts == 6 and
        q.rows > 1 and
        q.total_experts == 384 and
        gate_bytes >= 2 << 30 and
        down_bytes >= 2 << 30 and
        !quality and
        absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and
        absent("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") and
        absent("DS4_METAL_DISABLE_Q4_BATCH_EXPERT_TABLE") and
        absent("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") and
        pairPresent(f, c.DS4_ROUTED_TABLE_Q4_SWI, c.DS4_ROUTED_TABLE_Q4_SUM) and
        f.table_encoders and
        (present("DS4_METAL_ENABLE_Q4_BATCH_EXPERT_TABLE") or present("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") or p.automatic) and
        (present("DS4_METAL_Q4_TABLE_USE_RESOURCES") or present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") or p.queue or present("DS4_METAL_Q4_TABLE_MODEL_RESIDENCY_SET"));
    p.mm = !p.table and !p.selected_addr and q.rows >= 32 and mapName(q.used_experts) != null;
    // V4.1's 8k tile has a 512 MiB packed budget; other shapes keep 256 MiB.
    const v41 = q.total_experts == 384 and q.input_width == 5120 and q.middle_width == 2304;
    p.packed_limit = @as(u64, if (v41) 512 else 256) << 20;
    const packed_header = (@as(u64, q.total_experts) + 31) / 32 * 32 * 4;
    const packed_rows = @as(u64, q.rows) * q.used_experts;
    const packed_width = @max(q.input_width, q.middle_width);
    p.packed_rhs =
        p.mm and
        !quality and
        (!streaming or q.force_resident) and
        q.rows >= 512 and
        q.rows <= (if (v41) @as(u32, 8192) else 4096) and
        q.used_experts == 6 and
        q.total_experts <= 384 and
        q.input_width <= 5120 and
        q.middle_width <= 4096 and
        packed_header +% packed_rows *% packed_width *% 2 <= p.packed_limit and
        mpp and
        ((q.gate_type == iq2 and q.down_type == q2) or (f.world == 2 and q.gate_type == mx and q.down_type == mx and absent("DS4_METAL_DISABLE_MXFP4_MM_ID_MPP"))) and
        absent("DS4_METAL_DISABLE_ROUTED_MPP_PACKED");
    p.packed_m32 = p.packed_rhs and q.rows >= 8192 and v41 and q.gate_type == iq2 and q.down_type == q2 and absent("DS4_METAL_DISABLE_V41_PACKED_M32N128");
    // Through eight rows V4.1 must keep scalar reduction order before BF16/router.
    p.v41_decode =
        q.rows <= 8 and
        v41 and
        q.used_experts == 6 and
        q.output_width == 5120 and
        ((q.gate_type == iq2 and q.down_type == q2) or (q.gate_type == q4 and q.down_type == q4));
    p.tiny_pair =
        !quality and
        (q.rows <= 5 or p.v41_decode or (q.rows == 6 and has(f, c.DS4_ROUTED_M5) and (q.gate_type == iq2 or q.gate_type == q4 or (f.world == 2 and q.gate_type == mx)))) and
        !p.table and
        !p.mm and
        ((q.gate_type == iq2 and cached(f, c.DS4_ROUTED_IQ2_PAIR) != null) or (q.gate_type == q4 and cached(f, c.DS4_ROUTED_Q4_PAIR) != null) or (q.gate_type == mx and cached(f, c.DS4_ROUTED_MX_SWI) != null and absent("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION") and absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT")));
    p.tp_static =
        p.tiny_pair and
        has(f, c.DS4_ROUTED_M5) and
        f.world == 2 and
        q.gate_type == mx and
        q.down_type == mx and
        q.used_experts == 6 and
        q.total_experts == 256 and
        q.input_width == 4096 and
        q.middle_width == 2048 and
        q.output_width == 4096 and
        q.gate_row_bytes == 2176 and
        q.gate_expert_bytes == 4456448 and
        q.down_row_bytes == 1088 and
        q.down_expert_bytes == 4456448 and
        !streaming and
        absent("DS4_METAL_DISABLE_M5_TP_MXFP4_STATIC") and
        pairPresent(f, c.DS4_ROUTED_MX_TP_SWI, c.DS4_ROUTED_MX_TP_SUM);
    const tiny = if (q.gate_type == iq2) cached(f, c.DS4_ROUTED_IQ2_SWI) else if (q.gate_type == q4) cached(f, c.DS4_ROUTED_Q4_SWI) else if (q.gate_type == mx) cached(f, if (p.tp_static) c.DS4_ROUTED_MX_TP_SWI else c.DS4_ROUTED_MX_SWI) else null;
    p.tiny_swi = p.tiny_pair and tiny != null and absent("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION") and absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    // Grouped MMA already narrows activations to half. Keep its weighted
    // intermediate in that precision to avoid an F32 write/read; quality
    // retains the reference F32 intermediate.
    p.half = !quality and !p.table and !p.selected_addr;
    // Fused gate/up uses the same compact map and accumulation/epilogue as
    // separate GEMMs. Its kernel has no TP expert-ownership support.
    p.pair_mm =
        p.mm and
        !(q.gate_type == iq2 and mpp) and
        f.world != 2 and
        p.half and
        q.used_experts == 6 and
        ((q.gate_type == iq2 and q.down_type == q2) or (q.gate_type == q4 and q.down_type == q4) or (q.gate_type == mx and q.down_type == mx)) and
        absent("DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU") and
        absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and
        absent("DS4_METAL_GRAPH_DUMP_PREFIX");
    // The resident pre-M5 compact tile uses two SIMD groups / 8 KiB, and
    // culls the empty final half-tile while keeping staging/barriers intact.
    // Token-centric scatter emits the established work_count/uint2 trailer.
    p.compact =
        p.pair_mm and
        q.gate_type == mx and
        q.down_type == mx and
        f.world == 1 and
        ((pre and !streaming and q.rows >= 2048 and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_SWIGLU_COMPACT_TILE")) or f.tests & c.DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE != 0);
    p.scatter =
        p.compact and
        q.gate_type == mx and
        q.down_type == mx and
        q.used_experts == 6 and
        !streaming and
        f.world == 1 and
        ((pre and q.rows >= 2048 and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_MAP_SCATTER")) or f.tests & c.DS4_GPU_TEST_MXFP4_MAP_SCATTER != 0);
    p.pair_tail =
        p.pair_mm and
        q.gate_type == mx and
        q.down_type == mx and
        f.world == 1 and
        ((pre and !streaming and q.rows >= 2048 and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL")) or f.tests & c.DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL != 0);
    p.down_tail =
        p.mm and
        p.half and
        q.gate_type == mx and
        q.down_type == mx and
        f.world == 1 and
        ((pre and !streaming and q.rows >= 2048 and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL")) or f.tests & c.DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL != 0);
    // The half-result dequant table retains exact E8M0/E2M1 results for the
    // guarded resident down projection; it changes no selected experts.
    p.down_lut =
        p.mm and
        p.half and
        q.gate_type == mx and
        q.down_type == mx and
        q.used_experts == 6 and
        !streaming and
        f.world == 1 and
        ((pre and q.rows >= 2048 and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_HALF_LUT")) or f.tests & c.DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT != 0);
    return p;
}
pub fn mvArgs(q: *const Request, f: *const Facts, down: bool) Mv {
    const width = if (down) q.middle_width else q.input_width;
    const outputs = if (down) q.output_width else q.middle_width;
    const row = if (down) q.down_row_bytes else q.gate_row_bytes;
    const expert = if (down) q.down_expert_bytes else q.gate_expert_bytes;
    const used = if (down) q.used_experts else 1;
    var a = zero(Mv);
    a.nei0 = narrow(q.used_experts);
    a.nei1 = narrow(q.rows);
    a.nbi1 = @as(u64, q.used_experts) * 4;
    a.ne00 = narrow(width);
    a.ne01 = narrow(outputs);
    a.ne02 = narrow(q.total_experts);
    a.nb00 = if (width / 256 != 0) row / (width / 256) else 1;
    a.nb01 = row;
    a.nb02 = expert;
    a.ne10 = narrow(width);
    a.ne11 = narrow(used);
    a.ne12 = narrow(q.rows);
    a.ne13 = 1;
    a.nb10 = 4;
    a.nb11 = @as(u64, width) * 4;
    a.nb12 = @as(u64, used) *% a.nb11;
    a.ne0 = narrow(outputs);
    a.ne1 = narrow(q.used_experts);
    a.nb1 = @as(u64, outputs) * 4;
    a.nr0 = narrow(nr0(if (down) q.down_type else q.gate_type));
    a.tp_rank = f.rank;
    a.tp_world = f.world;
    a.tp_expert_base = narrow(f.first_expert);
    return a;
}
fn mmArgs(q: *const Request, f: *const Facts, down: bool, half: bool) Mm {
    const width = if (down) q.middle_width else q.input_width;
    const used = if (down) q.used_experts else 1;
    const row = if (down) q.down_row_bytes else q.gate_row_bytes;
    const expert = if (down) q.down_expert_bytes else q.gate_expert_bytes;
    var a = zero(Mm);
    a.ne00 = narrow(width);
    a.ne02 = narrow(q.total_experts);
    a.nb01 = row;
    a.nb02 = expert;
    a.nb03 = @as(u64, q.total_experts) *% expert;
    a.ne11 = narrow(used);
    a.nb10 = if (half) 2 else 4;
    a.nb11 = @as(u64, width) * a.nb10;
    a.nb12 = @as(u64, used) *% a.nb11;
    a.nb13 = @as(u64, q.rows) *% a.nb12;
    a.ne20 = narrow(q.used_experts);
    a.ne21 = narrow(q.rows);
    a.ne0 = narrow(if (down) q.output_width else q.middle_width);
    a.ne1 = narrow(q.used_experts);
    a.r2 = 1;
    a.r3 = 1;
    a.tp_rank = f.rank;
    a.tp_world = f.world;
    a.tp_expert_base = narrow(f.first_expert);
    return a;
}
pub fn activation(q: *const Request, half: bool, clamped: bool) Act {
    return .{ .width = q.middle_width, .rows = q.rows *% q.used_experts, .gate_row_stride = @as(u64, q.middle_width) * 4, .up_row_stride = @as(u64, q.middle_width) * 4, .mid_row_stride = @as(u64, q.middle_width) * @as(u64, if (half) 2 else 4), .weight_stride = 4, .write_clamped = @intFromBool(clamped), .clamp_value = q.clamp };
}

const Programs = struct {
    gate_mv: ?*const Pipeline,
    down_mv: ?*const Pipeline,
    map: ?*const Pipeline = null,
    gate_mm: ?*const Pipeline = null,
    up_mm: ?*const Pipeline = null,
    down_mm: ?*const Pipeline = null,
    pair_mm: ?*const Pipeline = null,
    mm_mpp: bool = false,
};
fn programs(q: *const Request, f: *const Facts, p: Policy, gate_mv: *const Pipeline, down_mv: *const Pipeline) ?Programs {
    var out = Programs{ .gate_mv = gate_mv, .down_mv = down_mv };
    if (!p.mm) return out;
    out.map = dynamic(0, if (p.scatter) "kernel_mul_mm_id_map_scatter_work_ne20_6" else mapName(q.used_experts).?, 0);
    out.gate_mm = mmPipeline(q.gate_type, false);
    out.up_mm = mmPipeline(q.gate_type, false);
    out.down_mm = if (p.down_lut) dynamic(2, if (p.down_tail) "kernel_mul_mm_id_mxfp4_f16_half_lut_tail_cull" else "kernel_mul_mm_id_mxfp4_f16_half_lut", 0) else if (p.down_tail) dynamic(2, "kernel_mul_mm_id_mxfp4_f16_tail_cull", 0) else mmPipeline(q.down_type, p.half);
    const mpp = has(f, c.DS4_ROUTED_MPP);
    if (mpp and q.gate_type == iq2) {
        if (dynamic(2, "kernel_mul_mm_id_iq2_xxs_f32_mpp", 0)) |pipeline| {
            out.gate_mm = pipeline;
            out.up_mm = pipeline;
        }
    }
    // Preserve these two per-process settings' original first-use boundary.
    if (mxfp4_mpp == null) mxfp4_mpp = absent("DS4_METAL_DISABLE_MXFP4_MM_ID_MPP");
    if (mpp and mxfp4_mpp.? and q.gate_type == mx) {
        if (dynamic(2, "kernel_mul_mm_id_mxfp4_f32_mpp", 0)) |pipeline| {
            out.gate_mm = pipeline;
            out.up_mm = pipeline;
        }
    }
    if (q4_mpp == null) q4_mpp = absent("DS4_METAL_DISABLE_Q4_K_MM_ID_MPP");
    if (mpp and q4_mpp.? and q.gate_type == q4) {
        if (dynamic(2, "kernel_mul_mm_id_q4_K_f32_dbuf_mpp", 0)) |pipeline| {
            out.gate_mm = pipeline;
            out.up_mm = pipeline;
            out.mm_mpp = true;
        }
    }
    if (mpp and mxfp4_mpp.? and p.half and q.down_type == mx) {
        if (dynamic(2, "kernel_mul_mm_id_mxfp4_f16_mpp", 0)) |pipeline| out.down_mm = pipeline;
    }
    if (mpp and q4_mpp.? and p.half and q.down_type == q4) {
        if (dynamic(2, "kernel_mul_mm_id_q4_K_f16_dbuf_mpp", 0)) |pipeline| {
            out.down_mm = pipeline;
            out.mm_mpp = true;
        }
    }
    if (mpp and p.half and (q.down_type == q2 or q.down_type == iq2)) {
        if (dynamic(2, if (q.down_type == q2) "kernel_mul_mm_id_q2_K_f16_mpp" else "kernel_mul_mm_id_iq2_xxs_f16_mpp", 0)) |pipeline| out.down_mm = pipeline;
    }
    if (p.packed_rhs) {
        out.gate_mm = dynamic(0, if (q.gate_type == mx) "kernel_mul_mm_id_mxfp4_mpp_packed" else if (p.packed_m32) "kernel_mul_mm_id_iq2_xxs_mpp_packed_m32n128" else "kernel_mul_mm_id_iq2_xxs_mpp_packed", 0);
        out.up_mm = out.gate_mm;
        out.down_mm = dynamic(0, if (q.down_type == mx) "kernel_mul_mm_id_mxfp4_mpp_packed" else if (p.packed_m32) "kernel_mul_mm_id_q2_K_mpp_packed_m32n128" else "kernel_mul_mm_id_q2_K_mpp_packed", 0);
        out.mm_mpp = false;
    }
    if (p.pair_mm) {
        // Exact half-domain MXFP4 scaling keeps the selected accumulation.
        const half_scale = q.gate_type == mx and has(f, c.DS4_ROUTED_PRE_M5) and absent("DS4_METAL_DISABLE_PRE_M5_MXFP4_MM_ID_PAIR_HALF_SCALE") and !has(f, c.DS4_ROUTED_QUALITY) and !has(f, c.DS4_ROUTED_SSD) and f.world == 1;
        const name: [*:0]const u8 = if (q.gate_type == q4) "kernel_mul_mm_id_q4_K_pair_swiglu_f16" else if (q.gate_type == mx) (if (p.compact) (if (half_scale) "kernel_mul_mm_id_mxfp4_pair_swiglu_f16_compact_tail_cull_half_scale" else "kernel_mul_mm_id_mxfp4_pair_swiglu_f16_compact_tail_cull") else if (p.pair_tail) (if (half_scale) "kernel_mul_mm_id_mxfp4_pair_swiglu_f16_tail_cull_half_scale" else "kernel_mul_mm_id_mxfp4_pair_swiglu_f16_tail_cull") else (if (half_scale) "kernel_mul_mm_id_mxfp4_pair_swiglu_f16_half_scale" else "kernel_mul_mm_id_mxfp4_pair_swiglu_f16")) else "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16";
        out.pair_mm = dynamic(0, name, 0);
    }
    if (out.map == null or out.gate_mm == null or out.up_mm == null or out.down_mm == null or (p.pair_mm and out.pair_mm == null)) return null;
    if (p.cached_iq2) {
        out.gate_mm = dynamic(2, if (mpp) "kernel_mul_mm_id_iq2_xxs_cached_f32_mpp" else "kernel_mul_mm_id_iq2_xxs_cached_f32", 0);
        out.up_mm = out.gate_mm;
        out.down_mm = dynamic(2, if (p.half) (if (mpp) "kernel_mul_mm_id_iq2_xxs_cached_f16_mpp" else "kernel_mul_mm_id_iq2_xxs_cached_f16") else "kernel_mul_mm_id_iq2_xxs_cached_f32", 0);
        if (out.gate_mm == null or out.down_mm == null) return null;
    }
    return out;
}
const Work = struct { counts: u64, offset: u64, capacity: u64, bytes: u64 };
fn work(args: Mm) ?Work {
    if (args.ne20 <= 0 or args.ne21 <= 0 or args.ne02 <= 0) return null;
    const experts: u64 = @intCast(args.ne02);
    const rows: u64 = @intCast(args.ne21);
    const used: u64 = @intCast(args.ne20);
    const counts = experts * 4;
    const ids = experts *% rows *% 4;
    if (counts > std.math.maxInt(u64) - ids) return null;
    const offset = (counts +% ids +% 7) & ~@as(u64, 7);
    const capacity = (used * rows + 31 * experts + 31) / 32;
    if (capacity > (std.math.maxInt(u64) - 8) / 8 or offset > std.math.maxInt(u64) - 8 - capacity * 8) return null;
    return .{ .counts = counts, .offset = offset, .capacity = capacity, .bytes = offset + 8 + capacity * 8 };
}
fn threads(pipeline: *const Pipeline, width: u64) u64 {
    return @max(1, @min(width, @min(256, c.ds4_metal_routed_max_threads(pipeline))));
}
fn binary(width: u32, rows: u32, lhs: u64, rhs: u64, dst: u64, scalar: bool) c.ds4_gpu_bin_args {
    var a = zero(c.ds4_gpu_bin_args);
    a.ne00 = narrow(width);
    a.ne01 = narrow(rows);
    a.ne02 = 1;
    a.ne03 = 1;
    a.nb00 = 4;
    a.nb01 = lhs;
    a.nb02 = @as(u64, rows) *% lhs;
    a.nb03 = a.nb02;
    a.ne10 = if (scalar) 1 else narrow(width);
    a.ne11 = narrow(rows);
    a.ne12 = 1;
    a.ne13 = 1;
    a.nb10 = 4;
    a.nb11 = rhs;
    a.nb12 = @as(u64, rows) *% rhs;
    a.nb13 = a.nb12;
    a.ne0 = narrow(width);
    a.ne1 = narrow(rows);
    a.ne2 = 1;
    a.ne3 = 1;
    a.nb0 = 4;
    a.nb1 = dst;
    a.nb2 = @as(u64, rows) *% dst;
    a.nb3 = a.nb2;
    return a;
}
pub const Command = struct {
    token: ?*anyopaque = null,
    owned: c_int = 0,
    label: [*:0]const u8 = "routed batch MoE",
    pub fn acquire(self: *Command) bool {
        self.drop();
        self.token = api.acquire(&self.owned);
        return self.token != null;
    }
    pub fn drop(self: *Command) void {
        if (self.token) |token| _ = api.release(token, self.owned, 0, self.label);
        self.token = null;
    }
    pub fn finish(self: *Command) bool {
        const token = self.token orelse return false;
        self.token = null;
        return api.release(token, self.owned, 1, self.label) != 0;
    }
};
fn trim(value: []const u8) []const u8 {
    var first: usize = 0;
    var last = value.len;
    while (first < last and c.isspace(value[first]) != 0) first += 1;
    while (last > first and c.isspace(value[last - 1]) != 0) last -= 1;
    return value[first..last];
}
fn equalInsensitive(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}
fn layerMatches(value: [*c]const u8, layer: u32) bool {
    if (value == null) return true;
    const range = trim(bytes(value));
    if (range.len == 0 or equalInsensitive(range, "all")) return true;
    var offset: usize = 0;
    while (offset < range.len) {
        while (offset < range.len and (range[offset] == ' ' or range[offset] == '\t' or range[offset] == ',')) offset += 1;
        if (offset >= range.len) break;
        var end: [*c]u8 = null;
        const first = c.strtoul(range.ptr + offset, &end, 10);
        const finish = @intFromPtr(end) - @intFromPtr(range.ptr);
        if (finish == offset or finish > range.len or first > std.math.maxInt(u32)) return false;
        offset = finish;
        var last = first;
        if (offset < range.len and range[offset] == '-') {
            offset += 1;
            last = c.strtoul(range.ptr + offset, &end, 10);
            const ending = @intFromPtr(end) - @intFromPtr(range.ptr);
            if (ending == offset or ending > range.len or last > std.math.maxInt(u32)) return false;
            offset = ending;
        }
        if (first <= layer and layer <= last) return true;
        while (offset < range.len and (range[offset] == ' ' or range[offset] == '\t')) offset += 1;
        if (offset < range.len and range[offset] != ',') return false;
    }
    return false;
}
pub fn profileEnabled(comptime flag_env: [*:0]const u8, comptime layer_env: [*:0]const u8, layer: u32) bool {
    const raw = c.getenv(flag_env);
    if (raw == null) return false;
    const flag = trim(bytes(raw));
    const layer_raw = c.getenv(layer_env);
    const layer_text = if (layer_raw == null) "" else trim(bytes(layer_raw));
    if (flag.len != 0) {
        if (equalInsensitive(flag, "0") or equalInsensitive(flag, "false") or equalInsensitive(flag, "no") or equalInsensitive(flag, "off")) return false;
        if (layer_text.len == 0 and !equalInsensitive(flag, "1") and !equalInsensitive(flag, "true") and !equalInsensitive(flag, "yes") and !equalInsensitive(flag, "on") and !equalInsensitive(flag, "all")) return layerMatches(raw, layer);
    }
    return layerMatches(layer_raw, layer);
}
fn profileLayer(layer: u32) bool {
    return profileEnabled("DS4_METAL_MOE_STAGE_PROFILE", "DS4_METAL_MOE_STAGE_PROFILE_LAYER", layer);
}
pub const Execution = struct {
    q: *const Request,
    f: *const Facts,
    p: Policy,
    resource: *c.ds4_metal_routed_resources,
    command: Command = .{},
    profile: bool = false,
    stage_start: f64 = 0,
    fn launch(self: *Execution, kind: u32, pipeline: ?*const Pipeline, args: c.ds4_metal_routed_args, grid: Grid, options: Options) bool {
        const program = pipeline orelse return false;
        var l = zero(Launch);
        l.kind = kind;
        l.pipeline = program;
        l.args = args;
        l.activation = activation(self.q, kind == c.DS4_ROUTED_MM_PAIR, false);
        l.groups_x = grid.x;
        l.groups_y = grid.y;
        l.groups_z = grid.z;
        l.threads_x = grid.threads;
        l.threads_y = grid.simd;
        l.shared_bytes = grid.shared;
        l.source = options.source;
        l.second = options.second;
        l.destination = options.destination;
        l.weight = options.weight;
        l.source_extra = options.source_extra;
        l.second_extra = options.second_extra;
        l.destination_extra = options.destination_extra;
        l.counts_bytes = options.counts;
        l.work_offset = options.work;
        l.cached = options.cached;
        l.queue_residency = self.p.queue;
        l.use_resources = present("DS4_METAL_MOE_MM_ID_USE_RESOURCES") and absent("DS4_METAL_DISABLE_MOE_MM_ID_USE_RESOURCES");
        return c.ds4_metal_routed_encode(self.resource, self.command.token.?, &l);
    }
    fn stage(self: *Execution, name: [*:0]const u8, ok: *bool) void {
        @setFloatMode(.optimized);
        if (!ok.* or !self.profile) return;
        if (c.ds4_gpu_end_commands() == 0) {
            ok.* = false;
            return;
        }
        const now = c.ds4_metal_routed_now_ms();
        const filter = c.getenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
        if (filter == null or filter[0] == 0 or std.mem.find(u8, bytes(name), bytes(filter)) != null) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal routed MoE stage layer=%u tokens=%u pairs=%u experts=%u gate=%s down=%s path=%s mid=%s %s=%.3f ms\n", self.q.layer, self.q.rows, self.q.rows *% self.q.used_experts, self.q.used_experts, typeName(self.q.gate_type), typeName(self.q.down_type), self.p.path(), @as([*:0]const u8, if (self.p.half) "f16" else "f32"), name, now - self.stage_start);
        }
        self.stage_start = now;
        if (c.ds4_gpu_begin_commands() == 0 or !self.command.acquire()) ok.* = false;
    }
    pub fn mv(self: *Execution, kind: u32, pipeline: ?*const Pipeline, args: Mv, weight: u32, source: u32, destination: u32, nsg: u64, own_rows: bool) bool {
        if (args.ne00 <= 0 or args.ne01 <= 0 or args.nei0 <= 0 or args.nei1 <= 0) return false;
        const rows_per_group = @as(u64, @intCast(args.nr0)) * if (own_rows) @as(u64, 1) else nsg;
        const is_sum = kind == c.DS4_ROUTED_DOWN_SUM or kind == c.DS4_ROUTED_ADDR_SUM or kind == c.DS4_ROUTED_TABLE_SUM;
        return self.launch(kind, pipeline, .{ .mv = args }, .{ .x = (@as(u64, @intCast(args.ne01)) + rows_per_group - 1) / rows_per_group, .y = if (is_sum) @intCast(args.nei1) else 1, .z = if (is_sum) 1 else @as(u64, @intCast(args.nei0)) * @as(u64, @intCast(args.nei1)), .simd = nsg, .shared = shared(if (weight == 2) self.q.down_type else self.q.gate_type) }, .{ .weight = weight, .source = source, .destination = destination });
    }
    fn mm(self: *Execution, pipeline: ?*const Pipeline, args: Mm, weight: u32, source: u32, destination: u32, programs_used: Programs) bool {
        if (args.ne00 <= 0 or args.ne0 <= 0) return false;
        const w = work(args) orelse return false;
        const tile: u64 = if (self.p.packed_m32) 32 else 64;
        return self.launch(c.DS4_ROUTED_MM, pipeline, .{ .mm = args }, .{ .x = w.capacity, .y = (@as(u64, @intCast(args.ne0)) + tile - 1) / tile, .threads = 128, .shared = if (self.p.packed_m32) 16384 else if (programs_used.mm_mpp) 12288 else 8192 }, .{ .source = source, .destination = destination, .weight = weight, .counts = w.counts, .work = w.offset, .cached = self.p.cached_iq2 });
    }
    fn pack(self: *Execution, args: Mm, half: bool) bool {
        if (args.ne00 <= 0) return false;
        const w = work(args) orelse return false;
        const rows = @as(u64, @intCast(args.ne20)) * @as(u64, @intCast(args.ne21));
        const header = (@as(u64, @intCast(args.ne02)) + 31) / 32 * 32 * 4;
        const row_bytes = @as(u64, @intCast(args.ne00)) * 2;
        if (rows > (std.math.maxInt(u64) - header) / row_bytes) return false;
        const size = header + rows * row_bytes;
        if (size > self.p.packed_limit or !c.ds4_metal_routed_scratch(self.resource, 2, size)) return false;
        const program = dynamic(0, if (half) "kernel_moe_pack_rhs_f16" else "kernel_moe_pack_rhs_f32", 0) orelse return false;
        const offsets = dynamic(0, "kernel_moe_packed_offsets", 0) orelse return false;
        return self.launch(c.DS4_ROUTED_PACK_OFFSETS, offsets, .{ .count = @intCast(args.ne02) }, .{ .x = 1 }, .{}) and self.launch(c.DS4_ROUTED_PACK_RHS, program, .{ .mm = args }, .{ .x = w.capacity * 32, .threads = 128 }, .{ .source = if (half) c.DS4_ROUTED_MID else c.DS4_ROUTED_X, .counts = w.counts, .work = w.offset });
    }
    fn bin(self: *Execution, pipeline: ?*const Pipeline, args: c.ds4_gpu_bin_args, options: Options) bool {
        const program = pipeline orelse return false;
        if (args.ne0 <= 0 or args.ne1 <= 0) return false;
        const maximum = @min(256, c.ds4_metal_routed_max_threads(program));
        var nth: u64 = 1;
        while (2 * nth < @as(u64, @intCast(args.ne0)) and nth < maximum) nth *= 2;
        return self.launch(c.DS4_ROUTED_BIN, program, .{ .binary = args }, .{ .x = @intCast(args.ne1), .y = @intCast(args.ne2), .z = @intCast(args.ne3), .threads = nth }, options);
    }
    fn unary(self: *Execution, key: usize, source: u32, destination: u32, c4: bool, minimum: f32, maximum: f32) bool {
        const width = self.q.middle_width;
        const rows = self.q.rows *% self.q.used_experts;
        if (width == 0 or rows == 0 or (c4 and width & 3 != 0)) return false;
        const program = cached(self.f, key) orelse return false;
        const kernel = if (c4) width / 4 else width;
        var a = zero(c.ds4_metal_attention_unary_args);
        a.ne00 = narrow(kernel);
        a.ne01 = narrow(rows);
        a.ne02 = 1;
        a.ne03 = 1;
        a.nb00 = 4;
        a.nb01 = @as(u64, width) * 4;
        a.nb02 = a.nb01;
        a.nb03 = a.nb01;
        a.ne0 = narrow(kernel);
        a.ne1 = narrow(rows);
        a.ne2 = 1;
        a.ne3 = 1;
        a.nb0 = 4;
        a.nb1 = a.nb01;
        a.nb2 = a.nb01;
        a.nb3 = a.nb01;
        a.min = minimum;
        a.max = maximum;
        const nth = threads(program, kernel);
        return self.launch(c.DS4_ROUTED_UNARY, program, .{ .unary = a }, .{ .x = ((@as(u64, kernel) + nth - 1) / nth) * rows, .threads = nth }, .{ .source = source, .destination = destination });
    }
    pub fn activate(self: *Execution, half: bool) bool {
        const q = self.q;
        const rows = q.rows *% q.used_experts;
        if (q.middle_width == 0 or rows == 0) return false;
        const program = dynamic(0, if (half) "kernel_dsv4_moe_swiglu_weight_f16" else "kernel_dsv4_moe_swiglu_weight", 0) orelse return false;
        return self.launch(c.DS4_ROUTED_ACTIVATION, program, .{ .activation = activation(q, half, present("DS4_METAL_MOE_WRITE_CLAMPED_ACT")) }, .{ .x = rows, .threads = threads(program, q.middle_width) }, .{});
    }
    fn flat(self: *Execution) bool {
        const n: u32 = @truncate(@as(u64, self.q.rows *% self.q.used_experts) * self.q.middle_width);
        if (n == 0) return false;
        const program = cached(self.f, c.DS4_ROUTED_SWIGLU) orelse return false;
        const a = c.ds4_gpu_glu_args{ .ne00 = narrow(n), .nb01 = @as(u64, n) * 4, .ne10 = narrow(n), .nb11 = @as(u64, n) * 4, .ne0 = narrow(n), .nb1 = @as(u64, n) * 4, .i00 = 0, .i10 = 0, .alpha = 1, .limit = 0 };
        const nth = threads(program, n);
        return self.launch(c.DS4_ROUTED_FLAT, program, .{ .glu = a }, .{ .x = (@as(u64, n) + nth - 1) / nth, .threads = nth }, .{});
    }
    pub fn sum(self: *Execution) bool {
        const q = self.q;
        if (q.output_width == 0 or q.rows == 0 or q.used_experts < 2) return false;
        const row = @as(u64, q.output_width) * 4;
        const stride = row * q.used_experts;
        const program = if (q.used_experts == 6) cached(self.f, c.DS4_ROUTED_SUM6) else if (q.used_experts == 8) cached(self.f, c.DS4_ROUTED_SUM8) else null;
        if (program) |pipeline| {
            const args = c.ds4_gpu_dsv4_moe_sum6_args{ .width = q.output_width, .tokens = q.rows, .src_token_stride = stride, .dst_token_stride = row };
            if (self.launch(c.DS4_ROUTED_SUM, pipeline, .{ .sum = args }, .{ .x = q.rows, .threads = threads(pipeline, q.output_width) }, .{})) return true;
        }
        if (!self.bin(cached(self.f, c.DS4_ROUTED_ADD), binary(q.output_width, q.rows, stride, stride, row, false), .{ .source = c.DS4_ROUTED_DOWN, .second = c.DS4_ROUTED_DOWN, .second_extra = row, .destination = c.DS4_ROUTED_OUT })) return false;
        var slot: u32 = 2;
        while (slot < q.used_experts) : (slot += 1) {
            if (!self.bin(cached(self.f, c.DS4_ROUTED_ADD), binary(q.output_width, q.rows, row, stride, row, false), .{ .source = c.DS4_ROUTED_OUT, .second = c.DS4_ROUTED_DOWN, .second_extra = @as(u64, slot) * row, .destination = c.DS4_ROUTED_OUT })) return false;
        }
        return true;
    }
};
const Grid = struct { x: u64, y: u64 = 1, z: u64 = 1, threads: u64 = 32, simd: u64 = 1, shared: u64 = 0 };
const Options = struct { source: u32 = 0, second: u32 = 0, destination: u32 = 0, weight: u32 = 0, source_extra: u64 = 0, second_extra: u64 = 0, destination_extra: u64 = 0, counts: u64 = 0, work: u64 = 0, cached: bool = false };

/// Preserve the existing public transaction: one selected-ID preparation, one
/// command owner, then gate/up → activation/weight → down → expert sum.
pub fn run(request: *const Request) c_int {
    if (api.init() == 0) return 0;
    var q = request.*;
    var facts = zero(Facts);
    c.ds4_metal_routed_get_facts(q.total_experts, &facts);
    // Bind the TP-owned expert range and let shaders rebase selected IDs.
    q.gate_offset +%= @as(u64, facts.first_expert) *% q.gate_expert_bytes;
    q.up_offset +%= @as(u64, facts.first_expert) *% q.gate_expert_bytes;
    q.down_offset +%= @as(u64, facts.first_expert) *% q.down_expert_bytes;
    if (q.out == null or q.gate == null or q.up == null or q.mid == null or q.x == null or q.model == null or q.selected == null or q.weights == null or q.rows == 0 or q.total_experts == 0 or q.used_experts == 0 or q.used_experts > 8) return 0;
    if (q.gate_expert_bytes == 0 or q.down_expert_bytes == 0 or q.gate_row_bytes == 0 or q.down_row_bytes == 0) return 0;
    if (q.input_width % 256 != 0 or q.middle_width % 256 != 0) return 0;
    if (q.total_experts > std.math.maxInt(u64) / q.gate_expert_bytes or q.total_experts > std.math.maxInt(u64) / q.down_expert_bytes) return errorLine("ds4: Metal routed batch MoE tensor byte size overflow");
    const gate_bytes = @as(u64, facts.bound_experts) * q.gate_expert_bytes;
    const down_bytes = @as(u64, facts.bound_experts) * q.down_expert_bytes;
    // Multi-GiB PRO Q4 one-row appends retain the existing single-token action.
    // It owns selected-expert views and exact tensor-ID fallback separately.
    if (single(&q, &facts, gate_bytes, down_bytes)) {
        if (q.mid_is_f16 != null) q.mid_is_f16.* = false;
        return c.ds4_gpu_routed_moe_one_tensor(q.out, q.gate, q.up, q.mid, q.experts, q.model, q.model_bytes, q.gate_offset, q.up_offset, q.down_offset, q.gate_type, q.down_type, q.gate_expert_bytes, q.gate_row_bytes, q.down_expert_bytes, q.down_row_bytes, q.input_width, q.middle_width, q.output_width, q.selected, q.weights, q.total_experts, q.used_experts, q.clamp, q.x, null, q.layer, q.force_resident);
    }
    const pair_rows = q.rows *% q.used_experts;
    const input_bytes = @as(u64, q.rows) *% q.input_width *% 4;
    const middle_bytes = @as(u64, q.rows) *% q.used_experts *% q.middle_width *% 4;
    const output_bytes = @as(u64, q.rows) *% q.output_width *% 4;
    const selected_bytes = @as(u64, q.rows) *% q.used_experts *% 4;
    const buffers = .{ q.x, q.gate, q.up, q.mid, q.out, q.selected, q.weights };
    const sizes = .{ input_bytes, middle_bytes, middle_bytes, middle_bytes, output_bytes, selected_bytes, selected_bytes };
    inline for (buffers, sizes) |tensor, required| {
        if (api.tensorHasBuffer(tensor) == 0 or api.tensorBytes(tensor) < required) return errorLine("ds4: Metal routed batch MoE received undersized activation buffers");
    }
    const gate_mv = mvPipeline(&facts, q.gate_type);
    const down_mv = mvPipeline(&facts, q.down_type);
    if (nr0(q.gate_type) == 0 or nr0(q.down_type) == 0 or gate_mv == null or down_mv == null) {
        _ = c.fprintf(c.__stderrp, "ds4: unsupported Metal routed batch MoE quant types gate=%u down=%u\n", q.gate_type, q.down_type);
        return 0;
    }
    const p = policy(&q, &facts, gate_bytes, down_bytes);
    const gate_args = mvArgs(&q, &facts, false);
    const down_args = mvArgs(&q, &facts, true);
    const gate_mm_args = mmArgs(&q, &facts, false, false);
    const down_mm_args = mmArgs(&q, &facts, true, p.half);
    const programs_used = programs(&q, &facts, p, gate_mv.?, down_mv.?) orelse return 0;
    const resource = c.ds4_metal_routed_open(&q) orelse return 0;
    defer c.ds4_metal_routed_close(resource);
    var unique: u32 = 0;
    if (p.cached()) {
        const had_batch = c.ds4_gpu_commands_active() != 0;
        const ready = c.ds4_metal_routed_take_ready(q.selected);
        const overlap = p.selected_addr and ready;
        // Router IDs are complete; only shared-expert work may still be queued.
        if (had_batch and (if (overlap) c.ds4_gpu_flush_commands() else c.ds4_gpu_end_commands()) == 0) return 0;
        var bad: u32 = std.math.maxInt(u32);
        var count: u32 = 0;
        if (!c.ds4_metal_routed_cache_prepare(resource, &unique, &bad, &count)) return 0;
        if (unique == 0) {
            c.ds4_metal_routed_cache_clear(q.layer);
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u produced no resident experts\n", q.layer);
            return 0;
        }
        if (bad != std.math.maxInt(u32)) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u lost resident expert resource %u/%u during preparation\n", q.layer, bad, count);
            c.ds4_metal_routed_cache_clear(q.layer);
            return 0;
        }
        if (had_batch and !overlap and c.ds4_gpu_begin_commands() == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u failed to reopen command batch after preparation\n", q.layer);
            c.ds4_metal_routed_cache_clear(q.layer);
            return 0;
        }
    }
    if (p.table) {
        const tables = c.ds4_metal_routed_tables(resource, p.automatic);
        if (tables == 0) return 0;
        if (tables == 2 and (p.queue or present("DS4_METAL_Q4_TABLE_RESIDENCY_SET")) and absent("DS4_METAL_Q4_TABLE_MODEL_RESIDENCY_SET")) return errorLine("ds4: Metal Q4 batch expert table residency set is not available");
    } else {
        const scratch_bytes = @as(u64, pair_rows) *% q.output_width *% 4;
        if (q.used_experts > 1 and (api.tensorHasBuffer(q.experts) == 0 or api.tensorBytes(q.experts) < scratch_bytes)) {
            if (!c.ds4_metal_routed_scratch(resource, 0, scratch_bytes)) {
                if (p.cached()) c.ds4_metal_routed_cache_clear(q.layer);
                return 0;
            }
        }
        if (!p.cached()) {
            if (!c.ds4_metal_routed_wrap(resource, gate_bytes, down_bytes)) return 0;
            if (present("DS4_GLM_TP_DEBUG") and q.layer == 3) c.ds4_metal_routed_debug_bind(resource, &gate_args, gate_bytes);
        }
    }
    const table_boundary = p.table and c.ds4_gpu_commands_active() != 0 and present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") and !p.queue and absent("DS4_METAL_DISABLE_Q4_TABLE_BOUNDARY");
    if (table_boundary and (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0)) return 0;
    var execution = Execution{ .q = &q, .f = &facts, .p = p, .resource = resource };
    defer execution.command.drop();
    if (!execution.command.acquire()) return 0;
    if (!c.ds4_metal_routed_residency(resource, execution.command.token.?, p.table, p.queue)) return 0;
    execution.profile = c.ds4_gpu_commands_active() != 0 and profileLayer(q.layer);
    if (execution.profile) {
        if (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0 or !execution.command.acquire()) return 0;
        execution.stage_start = c.ds4_metal_routed_now_ms();
    }
    if (p.cached_iq2 and !c.ds4_metal_routed_mark(resource)) return 0;
    const sum_pipeline = if (q.down_type == q2) cached(&facts, c.DS4_ROUTED_Q2_SUM) else if (q.down_type == q4) cached(&facts, c.DS4_ROUTED_Q4_SUM) else if (q.down_type == mx) cached(&facts, if (p.tp_static) c.DS4_ROUTED_MX_TP_SUM else c.DS4_ROUTED_MX_SUM) else null;
    const direct_sum = !has(&facts, c.DS4_ROUTED_QUALITY) and !p.table and !p.mm and q.used_experts == 6 and (q.rows <= 4 or p.v41_decode or p.tp_static) and sum_pipeline != null;
    var ok: bool = false;
    if (p.selected_addr) {
        ok = execution.mv(c.DS4_ROUTED_ADDR_PAIR, cached(&facts, c.DS4_ROUTED_ADDR_IQ2_SWI), gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
        if (!ok) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u failed to encode gate/up path tokens=%u unique=%u\n", q.layer, q.rows, unique);
    } else if (p.table) {
        ok = execution.mv(c.DS4_ROUTED_TABLE_PAIR, cached(&facts, c.DS4_ROUTED_TABLE_Q4_SWI), gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
    } else if (p.mm) {
        // One expert-major map serves all three matmuls for these routed IDs.
        const w = work(gate_mm_args) orelse return 0;
        var map_args = zero(c.ds4_gpu_mul_mm_id_map_args);
        map_args.ne02 = narrow(q.total_experts);
        map_args.ne10 = narrow(q.input_width);
        map_args.ne11 = 1;
        map_args.nb11 = @as(u64, q.input_width) * 4;
        map_args.nb12 = map_args.nb11;
        map_args.ne21 = narrow(q.rows);
        map_args.ne20 = narrow(q.used_experts);
        map_args.nb21 = @as(u64, q.used_experts) * 4;
        ok = c.ds4_metal_routed_scratch(resource, 1, w.bytes) and execution.launch(c.DS4_ROUTED_MAP, programs_used.map, .{ .map = map_args }, .{ .x = 1, .threads = q.total_experts, .shared = @as(u64, q.total_experts) * q.used_experts * 2 }, .{ .counts = w.counts, .work = w.offset });
        if (ok and p.packed_rhs) ok = execution.pack(gate_mm_args, false);
        execution.stage("map", &ok);
        if (ok and p.pair_mm) {
            ok = gate_mm_args.ne00 > 0 and gate_mm_args.ne0 > 0 and execution.launch(c.DS4_ROUTED_MM_PAIR, programs_used.pair_mm, .{ .mm = gate_mm_args }, .{ .x = w.capacity, .y = (@as(u64, q.middle_width) + (if (p.compact) @as(u64, 31) else 63)) / (if (p.compact) @as(u64, 32) else 64), .threads = if (p.compact) 64 else 128, .shared = if (p.compact) 8192 else 16384 }, .{ .counts = w.counts, .work = w.offset });
            execution.stage("gate_up_fused", &ok);
        } else if (ok) {
            ok = execution.mm(programs_used.gate_mm, gate_mm_args, 0, if (p.packed_rhs) c.DS4_ROUTED_PACKED else c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, programs_used);
            execution.stage("gate", &ok);
        }
        if (ok and !p.pair_mm) {
            ok = execution.mm(programs_used.up_mm, gate_mm_args, 1, if (p.packed_rhs) c.DS4_ROUTED_PACKED else c.DS4_ROUTED_X, c.DS4_ROUTED_UP, programs_used);
            execution.stage("up", &ok);
        }
    } else if (p.cached_iq2 and p.tiny_swi) {
        ok = execution.mv(c.DS4_ROUTED_ADDR_PAIR, cached(&facts, c.DS4_ROUTED_ADDR_IQ2_SWI), gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
    } else if (p.cached_iq2) {
        ok = execution.mv(c.DS4_ROUTED_ADDR_MV, cached(&facts, c.DS4_ROUTED_ADDR_IQ2), gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, simdgroups(q.gate_type), q.gate_type == q8) and execution.mv(c.DS4_ROUTED_ADDR_MV, cached(&facts, c.DS4_ROUTED_ADDR_IQ2), gate_args, 1, c.DS4_ROUTED_X, c.DS4_ROUTED_UP, simdgroups(q.gate_type), q.gate_type == q8);
    } else if (p.tiny_swi) {
        const pipeline = cached(&facts, if (q.gate_type == iq2) c.DS4_ROUTED_IQ2_SWI else if (q.gate_type == q4) c.DS4_ROUTED_Q4_SWI else if (p.tp_static) c.DS4_ROUTED_MX_TP_SWI else c.DS4_ROUTED_MX_SWI);
        ok = execution.mv(c.DS4_ROUTED_PAIR_SWI, pipeline, gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, if (p.tp_static) 1 else 2, false);
    } else if (p.tiny_pair) {
        ok = execution.mv(c.DS4_ROUTED_PAIR, cached(&facts, if (q.gate_type == iq2) c.DS4_ROUTED_IQ2_PAIR else c.DS4_ROUTED_Q4_PAIR), gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, 2, false);
    } else {
        ok = execution.mv(c.DS4_ROUTED_MV, gate_mv, gate_args, 0, c.DS4_ROUTED_X, c.DS4_ROUTED_GATE, simdgroups(q.gate_type), q.gate_type == q8) and execution.mv(c.DS4_ROUTED_MV, gate_mv, gate_args, 1, c.DS4_ROUTED_X, c.DS4_ROUTED_UP, simdgroups(q.gate_type), q.gate_type == q8);
    }
    execution.stage("gate_up", &ok);
    const fused_activation = !has(&facts, c.DS4_ROUTED_QUALITY) and !p.table;
    const half_mid = p.mm and fused_activation and p.half;
    if (q.mid_is_f16 != null) q.mid_is_f16.* = half_mid;
    if (ok and (p.selected_addr or p.table or p.pair_mm or p.tiny_swi)) {
        // These selected kernels already wrote weighted SwiGLU rows into mid.
    } else if (ok and fused_activation) {
        ok = execution.activate(half_mid);
    } else if (ok and q.clamp > 1.0e-6) {
        ok = execution.unary(c.DS4_ROUTED_CLAMP, c.DS4_ROUTED_GATE, c.DS4_ROUTED_GATE, false, -std.math.floatMax(f32), q.clamp) and execution.unary(c.DS4_ROUTED_SILU, c.DS4_ROUTED_GATE, c.DS4_ROUTED_MID, true, 0, 0) and execution.unary(c.DS4_ROUTED_CLAMP, c.DS4_ROUTED_UP, c.DS4_ROUTED_UP, false, -q.clamp, q.clamp);
        if (ok) ok = execution.bin(cached(&facts, c.DS4_ROUTED_MUL), binary(q.middle_width, pair_rows, @as(u64, q.middle_width) * 4, @as(u64, q.middle_width) * 4, @as(u64, q.middle_width) * 4, false), .{ .source = c.DS4_ROUTED_MID, .second = c.DS4_ROUTED_UP, .destination = c.DS4_ROUTED_MID });
    } else if (ok) {
        ok = execution.flat();
    }
    if (ok and !fused_activation and !p.table) ok = execution.bin(cached(&facts, c.DS4_ROUTED_WEIGHT), binary(q.middle_width, pair_rows, @as(u64, q.middle_width) * 4, 4, @as(u64, q.middle_width) * 4, true), .{ .source = c.DS4_ROUTED_MID, .second = c.DS4_ROUTED_WEIGHTS, .destination = c.DS4_ROUTED_MID });
    execution.stage("activation_weight", &ok);
    if (ok) {
        const destination: u32 = if (q.used_experts == 1) c.DS4_ROUTED_OUT else c.DS4_ROUTED_DOWN;
        if (p.selected_addr) {
            ok = execution.mv(c.DS4_ROUTED_ADDR_SUM, cached(&facts, c.DS4_ROUTED_ADDR_Q2_SUM), down_args, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_OUT, 2, false);
            if (!ok) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u failed to encode down path tokens=%u unique=%u\n", q.layer, q.rows, unique);
        } else if (p.table) {
            ok = execution.mv(c.DS4_ROUTED_TABLE_SUM, cached(&facts, c.DS4_ROUTED_TABLE_Q4_SUM), down_args, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_OUT, 2, false);
        } else if (direct_sum) {
            ok = execution.mv(c.DS4_ROUTED_DOWN_SUM, sum_pipeline, down_args, 2, c.DS4_ROUTED_MID, c.DS4_ROUTED_OUT, if (p.tp_static) 1 else 2, false);
        } else if (p.mm) {
            if (p.packed_rhs) ok = execution.pack(down_mm_args, true);
            if (ok) ok = execution.mm(programs_used.down_mm, down_mm_args, 2, if (p.packed_rhs) c.DS4_ROUTED_PACKED else c.DS4_ROUTED_MID, destination, programs_used);
        } else if (p.cached_iq2) {
            ok = execution.mv(c.DS4_ROUTED_ADDR_MV, cached(&facts, c.DS4_ROUTED_ADDR_IQ2), down_args, 2, c.DS4_ROUTED_MID, destination, simdgroups(q.down_type), q.down_type == q8);
        } else {
            ok = execution.mv(c.DS4_ROUTED_MV, down_mv, down_args, 2, c.DS4_ROUTED_MID, destination, simdgroups(q.down_type), q.down_type == q8);
        }
    }
    execution.stage("down", &ok);
    if (ok and q.used_experts > 1 and !direct_sum and !p.table and !p.selected_addr) ok = execution.sum();
    execution.stage("sum", &ok);
    if (!ok) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal routed batch MoE failed before submit layer=%u tokens=%u gate=%s down=%s path=%s\n", q.layer, q.rows, typeName(q.gate_type), typeName(q.down_type), p.path());
        return 0;
    }
    if (!execution.command.finish()) {
        if (p.cached()) c.ds4_metal_routed_cache_clear(q.layer);
        return 0;
    }
    if (p.cached() and execution.command.owned == 0) {
        if (c.ds4_gpu_end_commands() == 0) {
            c.ds4_metal_routed_cache_clear(q.layer);
            return 0;
        }
        if (c.ds4_gpu_begin_commands() == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u failed to reopen command batch after execution\n", q.layer);
            return 0;
        }
    }
    if (table_boundary and (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0)) return 0;
    return 1;
}
