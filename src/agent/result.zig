//! One tool-result transaction: import the file edit, fit/compact its owned
//! observation, then publish transcript/images and mirror in the old order.
//! Tool execution has already happened. A rejected edit becomes a receipt;
//! interruption aborts before the result is appended. C retains outer parser/
//! observation cleanup, status publication and the private summary operation.
const std = @import("std");
const c = @import("agent_runtime_c");

pub const Outcome = enum(c_int) {
    ok = c.DS4_AGENT_RESULT_OK,
    interrupted = c.DS4_AGENT_RESULT_INTERRUPTED,
    failed = c.DS4_AGENT_RESULT_FAILED,
};
fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn report(errors: []u8, message: [*:0]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message);
}
fn fallback(errors: []u8, message: [*:0]const u8) void {
    if (errors.len != 0 and errors[0] == 0) report(errors, message);
}
fn isInterrupted(errors: [*c]const u8) bool {
    return errors != null and c.strcmp(errors, "interrupted") == 0;
}
pub const Transaction = struct {
    worker: *c.agent_worker,

    fn contextSize(self: Transaction) c_int {
        const worker = self.worker;
        const current = if (worker.session != null) c.ds4_session_ctx(worker.session) else 0;
        return if (current > 0) current else worker.cfg.*.gen.ctx_size;
    }
    fn reserve(self: Transaction) c_int {
        const context = self.contextSize();
        var result: c_int = 1024;
        if (context > 0) result = @min(result, @max(16, @divTrunc(context, 8)));
        return @max(result, c.ds4_agent_compaction_reserve(self.worker.engine, context) + 128);
    }
    /// An ordinary rejected import does not discard tool output. Report it in
    /// the tool role; only interruption leaves the result transaction early.
    pub fn afterTools(self: Transaction, observation: *c.agent_tool_observation) Outcome {
        const worker = self.worker;
        if (worker.cfg.*.gen.live_context_path == null) return .ok;
        var errors: [160]u8 = @splat(0);
        const before = worker.transcript.len;
        const imported = c.ds4_agent_runtime_import(worker, &errors, errors.len);
        if (imported < 0 and isInterrupted(&errors)) return .interrupted;
        if (imported > 0) {
            var receipt: [192]u8 = undefined;
            _ = c.snprintf(&receipt, receipt.len, "Live context edit accepted: %d -> %d tokens; suffix reuse=%s.\n", before, worker.transcript.len, @as([*:0]const u8, if (imported == 2) "approximate" else "none"));
            c.ds4_agent_observation_puts(observation, &receipt);
        } else if (imported < 0) {
            c.ds4_agent_observation_puts(observation, "Live context edit rejected: ");
            c.ds4_agent_observation_puts(observation, &errors);
            c.ds4_agent_observation_puts(observation, "\n");
        }
        return .ok;
    }
    fn fits(self: Transaction, observation: *const c.agent_tool_observation, reservation: c_int, projected: [*c]c_int, errors: []u8) c_int {
        return c.ds4_agent_observation_fits(self.worker.engine, &self.worker.transcript, self.contextSize(), observation, reservation, projected, errors.ptr, errors.len);
    }
    /// Build into a separate owner first. Only a successful multimodal render
    /// can replace the old transcript and transfer image span embeddings.
    pub fn commit(self: Transaction, observation: *const c.agent_tool_observation, errors: []u8) bool {
        const worker = self.worker;
        var next = zero(c.ds4_tokens);
        var spans: [*c]c.ds4_vision_span = null;
        if (!c.ds4_agent_observation_build(worker.engine, &worker.transcript, observation, &next, &spans, errors.ptr, errors.len)) return false;
        c.ds4_tokens_free(&worker.transcript);
        worker.transcript = next;
        c.ds4_agent_runtime_images_append(worker, spans, observation.image_count);
        c.free(spans);
        return true;
    }
    /// -1 from fits is a rendering/embedding error, never a compact request.
    /// A too-large result gets one compaction attempt and then a small error
    /// receipt. That receipt itself must fit before any transcript publication.
    pub fn append(self: Transaction, observation: *c.agent_tool_observation, errors: []u8) Outcome {
        var projected: c_int = 0;
        const reservation = self.reserve();
        var admission = self.fits(observation, reservation, &projected, errors);
        if (admission < 0) {
            fallback(errors, "unable to append tool observation");
            return .failed;
        }
        if (admission == 0) {
            var compact_error: [160]u8 = @splat(0);
            if (!c.ds4_agent_result_compact(self.worker, "tool result would exceed context", &compact_error, compact_error.len)) {
                if (isInterrupted(&compact_error)) return .interrupted;
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", @as([*c]const u8, if (compact_error[0] != 0) &compact_error else "context compaction failed"));
                return .failed;
            }
            admission = self.fits(observation, reservation, &projected, errors);
            if (admission < 0) {
                fallback(errors, "unable to append tool observation");
                return .failed;
            }
            if (admission == 0) {
                c.ds4_agent_observation_free(observation);
                c.ds4_agent_observation_init(observation);
                var receipt: [256]u8 = undefined;
                _ = c.snprintf(&receipt, receipt.len, "Tool error: tool result still does not fit after context compaction " ++
                    "(projected_prompt=%d tokens, ctx=%d, reserve=%d). " ++
                    "Retry with a smaller read/search/bash output.\n", projected, self.contextSize(), reservation);
                c.ds4_agent_observation_puts(observation, &receipt);
                admission = self.fits(observation, 16, null, errors);
                if (admission < 0) {
                    fallback(errors, "unable to append tool observation");
                    return .failed;
                }
                if (admission == 0) {
                    report(errors, "context full after compaction");
                    return .failed;
                }
            }
        }
        if (!self.commit(observation, errors) or !c.ds4_agent_runtime_export(self.worker, errors.ptr, errors.len)) {
            fallback(errors, "unable to append tool observation");
            return .failed;
        }
        return .ok;
    }
};
