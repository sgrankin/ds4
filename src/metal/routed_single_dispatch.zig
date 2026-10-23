//! Fixed single-token dispatch preparation; all geometry and shader packets
//! are native. ARC only owns actual buffers and performs Metal encoder calls.
const std = @import("std");
const c = @import("metal_c");
const Batch = @import("routed_batch.zig");
const Policy = @import("routed_single_policy.zig").Policy;
const Request = c.ds4_metal_routed_request;
const Facts = c.ds4_metal_single_facts;
const Mv = c.ds4_gpu_mul_mv_id_args;
const Pipeline = c.ds4_metal_hc_pipeline;
pub fn silent(line: c_int) bool {
    if (Batch.present("DS4_GLM_TP_DEBUG")) _ = c.fprintf(c.__stderrp, "ds4: routed_moe_one silent return at line %d\n", line);
    return false;
}
pub const Execution = struct {
    q: *const Request,
    facts: *const Facts,
    policy: Policy,
    resource: *c.ds4_metal_single_resources,
    ops: Batch.Execution,
    gate: Mv,
    down: Mv,
    profile: bool = false,
    started: f64 = 0,
    path: [*:0]const u8 = "single",

    pub fn init(q: *const Request, facts: *const Facts, policy: Policy, resource: *c.ds4_metal_single_resources, gate: Mv, down: Mv) Execution {
        return .{ .q = q, .facts = facts, .policy = policy, .resource = resource, .gate = gate, .down = down, .ops = .{ .q = q, .f = &facts.batch, .p = .{ .queue = policy.queue }, .resource = c.ds4_metal_single_base(resource).? } };
    }
    pub fn deinit(self: *Execution) void {
        self.ops.command.drop();
    }
    pub fn begin(self: *Execution) bool {
        if (!self.ops.command.acquire()) return silent(33337);
        if (!c.ds4_metal_routed_residency(self.ops.resource, self.ops.command.token.?, self.policy.address or self.policy.table, self.policy.queue)) return silent(33314);
        self.profile = c.ds4_gpu_commands_active() != 0 and Batch.profileEnabled("DS4_METAL_MOE_ONE_STAGE_PROFILE", "DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER", self.q.layer);
        if (self.profile) {
            if (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0) return silent(33349);
            if (!self.ops.command.acquire()) return silent(33380);
            self.started = c.ds4_metal_routed_now_ms();
        }
        return true;
    }
    pub fn stage(self: *Execution, label: [*:0]const u8, ok: *bool) void {
        @setFloatMode(.optimized);
        if (!ok.* or !self.profile) return;
        if (c.ds4_gpu_end_commands() == 0) {
            ok.* = false;
            return;
        }
        const now = c.ds4_metal_routed_now_ms();
        const filter = c.getenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
        if (filter == null or filter[0] == 0 or std.mem.find(u8, std.mem.span(label), std.mem.span(filter)) != null) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal routed MoE one stage layer=%u pairs=%u experts=%u gate=%s down=%s path=%s %s=%.3f ms\n", self.q.layer, self.q.used_experts, self.q.used_experts, Batch.typeName(self.q.gate_type), Batch.typeName(self.q.down_type), self.path, label, now - self.started);
        }
        self.started = now;
        if (c.ds4_gpu_begin_commands() == 0 or !self.ops.command.acquire()) ok.* = false;
    }
    pub fn flush(self: *Execution) bool {
        return c.ds4_gpu_flush_commands() != 0 and self.ops.command.acquire();
    }
    pub fn fixed(self: *Execution, kind: u32, pipeline: ?*const Pipeline, args: Mv, part: u32, count: u32, group: c.ds4_gpu_moe_expert_group_args, split: c.ds4_gpu_stream_expert_split_args) bool {
        const program = pipeline orelse return false;
        if (args.ne00 <= 0 or args.ne01 <= 0 or args.nei0 <= 0 or args.nei1 <= 0) return false;
        const is_sum = kind == c.DS4_SINGLE_ARRAY_SUM or kind == c.DS4_SINGLE_GROUP_SUM or kind == c.DS4_SINGLE_GROUP_ARRAY_SUM or kind == c.DS4_SINGLE_GROUP24_SUM or kind == c.DS4_SINGLE_Q4_ADDRESS_SUM or kind == c.DS4_SINGLE_MASKED_SUM;
        var launch = std.mem.zeroes(c.ds4_metal_single_launch);
        launch.kind = kind;
        launch.pipeline = program;
        launch.mv = args;
        launch.activation = Batch.activation(self.q, false, false);
        launch.group = group;
        launch.split = split;
        launch.part = part;
        launch.array_count = count;
        launch.groups_x = (@as(u64, @intCast(args.ne01)) + @as(u64, @intCast(args.nr0)) * 2 - 1) / (@as(u64, @intCast(args.nr0)) * 2);
        launch.groups_y = if (is_sum) @intCast(args.nei1) else 1;
        launch.groups_z = if (is_sum) 1 else @as(u64, @intCast(args.nei0)) * @as(u64, @intCast(args.nei1));
        launch.threads_x = 32;
        launch.threads_y = 2;
        launch.shared_bytes = Batch.shared(if (is_sum or part == 2) self.q.down_type else self.q.gate_type);
        launch.queue_residency = self.policy.queue;
        launch.use_resources = Batch.present("DS4_METAL_Q4_ADDR_USE_RESOURCES");
        return c.ds4_metal_single_encode(self.resource, self.ops.command.token.?, &launch);
    }
    pub fn pair(self: *Execution, kind: u32, pipeline: ?*const Pipeline, count: u32) bool {
        return self.fixed(kind, pipeline, self.gate, 0, count, std.mem.zeroes(c.ds4_gpu_moe_expert_group_args), std.mem.zeroes(c.ds4_gpu_stream_expert_split_args));
    }
    pub fn sum(self: *Execution, kind: u32, pipeline: ?*const Pipeline, count: u32) bool {
        return self.fixed(kind, pipeline, self.down, 2, count, std.mem.zeroes(c.ds4_gpu_moe_expert_group_args), std.mem.zeroes(c.ds4_gpu_stream_expert_split_args));
    }
    pub fn masked(self: *Execution, down: bool, active: u32) bool {
        return self.fixed(if (down) c.DS4_SINGLE_MASKED_SUM else c.DS4_SINGLE_MASKED_PAIR, self.facts.cached[if (down) c.DS4_SINGLE_MASK_SUM else c.DS4_SINGLE_MASK_PAIR], if (down) self.down else self.gate, if (down) 2 else 0, 0, std.mem.zeroes(c.ds4_gpu_moe_expert_group_args), .{ .active_mask = active, .accumulate = 0 });
    }
    pub fn gather(self: *Execution, part: u32) bool {
        const program = self.facts.cached[c.DS4_SINGLE_Q4_GATHER] orelse return false;
        const size = if (part == 2) self.q.down_expert_bytes else self.q.gate_expert_bytes;
        if (size == 0 or size & 15 != 0 or self.q.used_experts == 0 or self.q.used_experts > 6) return false;
        const chunks = size >> 4;
        var launch = std.mem.zeroes(c.ds4_metal_single_launch);
        launch.kind = c.DS4_SINGLE_GATHER;
        launch.pipeline = program;
        launch.gather = .{ .expert_bytes = size, .group_size = 64, .n_slots = self.q.used_experts };
        launch.part = part;
        launch.groups_x = (chunks + 255) / 256;
        launch.groups_y = self.q.used_experts;
        launch.groups_z = 1;
        launch.threads_x = 256;
        launch.threads_y = 1;
        return c.ds4_metal_single_encode(self.resource, self.ops.command.token.?, &launch);
    }
    pub fn window(self: *Execution, down: bool, first: u32, count: u32, accumulate: bool) bool {
        const q = self.q;
        const size = if (down) q.down_expert_bytes else q.gate_expert_bytes;
        if (first > std.math.maxInt(u64) / size or count > std.math.maxInt(u64) / size) return false;
        const relative = @as(u64, first) * size;
        const length = @as(u64, count) * size;
        if (relative > std.math.maxInt(u64) - (if (down) q.down_offset else q.gate_offset) or (!down and relative > std.math.maxInt(u64) - q.up_offset)) return false;
        const mode: u32 = if (Batch.present("DS4_METAL_Q4_GROUPED_CACHE_VIEWS")) 1 else 2;
        if (!c.ds4_metal_single_wrap(self.resource, if (down) 4 else 3, q.gate_offset +% relative, q.up_offset +% relative, q.down_offset +% relative, length, length, mode, std.math.maxInt(u32))) return false;
        return self.fixed(if (down) c.DS4_SINGLE_GROUP_SUM else c.DS4_SINGLE_GROUP_PAIR, Batch.cached(&self.facts.batch, if (down) c.DS4_ROUTED_GROUP_Q4_SUM else c.DS4_ROUTED_GROUP_Q4_SWI), if (down) self.down else self.gate, if (down) 2 else 0, 0, .{ .expert_base = first, .expert_count = count, .accumulate = @intFromBool(accumulate), .pad0 = 0 }, std.mem.zeroes(c.ds4_gpu_stream_expert_split_args));
    }
};
