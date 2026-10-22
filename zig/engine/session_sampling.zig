//! Public selection/logprob policy over the actual committed Session frontier.
//! Raw logits stay untouched; only sampling and greedy selection apply the
//! optional overthinking penalty. Numeric filtering/RNG remains Sampling's job.
const std = @import("std");
const c = @import("gguf_c");
const Frontier = @import("deepseek41/frontier.zig").Frontier;
const negative_infinity: f32 = -1.0e30; // Existing finite sentinel, not -inf.

fn tokens(state: *const c.ds4_session_frontier) []const c_int {
    return if (state.checkpoint.len <= 0) &.{} else state.checkpoint.v[0..@intCast(state.checkpoint.len)];
}
fn insideThinking(history: []const c_int, start: c_int, end: c_int) bool {
    var position = history.len;
    while (position != 0) {
        position -= 1;
        if (end >= 0 and history[position] == end) return false;
        if (start >= 0 and history[position] == start) return true;
    }
    return false;
}

// Input/cache history is excluded. Reopened thinking accumulates in the same
// assistant turn; a new assistant marker resets it. Controls do not consume
// the threshold. Successful sync/restore sets the generated-reasoning floor.
fn generatedThinking(history: []const c_int, prompt_length: c_int, vocab: *const c.ds4_vocab) c_int {
    if (history.len == 0 or vocab.think_start_id < 0 or vocab.think_end_id < 0) return 0;
    var turn_start: usize = 0;
    if (vocab.assistant_id >= 0) {
        var position = history.len;
        while (position != 0) {
            position -= 1;
            if (history[position] == vocab.assistant_id) {
                turn_start = position + 1;
                break;
            }
        }
    }
    const prompt: usize = @min(history.len, @as(usize, @intCast(@max(0, prompt_length))));
    var inside = false;
    var count: c_int = 0;
    for (history[turn_start..], turn_start..) |token, position| {
        if (token == vocab.think_start_id) {
            inside = true;
        } else if (token == vocab.think_end_id) {
            inside = false;
        } else if (inside and position >= prompt) count += 1;
    }
    return count;
}

pub const Selection = struct {
    state: *c.ds4_session_frontier,
    policy: ?*const c.ds4_overthink_policy,
    vocab: ?*const c.ds4_vocab,
    count: u32,

    pub fn init(view: *const c.ds4_session_sampling_view) Selection {
        return .{
            .state = @ptrCast(view.frontier),
            .policy = if (view.policy == null) null else @ptrCast(view.policy),
            .vocab = if (view.vocab == null) null else @ptrCast(view.vocab),
            .count = view.count,
        };
    }
    fn raw(self: Selection) []const f32 {
        return self.state.logits[0..self.count];
    }
    fn adjusted(self: Selection) []const f32 {
        @setFloatMode(.strict);
        const policy = self.policy orelse return self.raw();
        if (policy.penalty <= 0.0) return self.raw();
        const vocab = self.vocab orelse return self.raw();
        const history = tokens(self.state);
        if (policy.think_only and !insideThinking(history, vocab.think_start_id, vocab.think_end_id)) return self.raw();
        if (policy.after > 0 and generatedThinking(history, self.state.overthink_prompt_len, vocab) < policy.after) return self.raw();
        if (self.state.sampling_logits == null) {
            // Same allocation-admission/fatal-OOM path as the C engine, and
            // the outer Session continues to free this lazy libc allocation.
            const memory = c.ds4_sampling_alloc(@as(usize, self.count) * @sizeOf(f32)) orelse unreachable;
            self.state.sampling_logits = @ptrCast(@alignCast(memory));
        }
        const output = self.state.sampling_logits[0..self.count];
        @memcpy(output, self.raw());
        for (0..@as(usize, @intCast(@max(0, policy.count)))) |index| {
            const token = policy.ids[index];
            if (token >= 0 and @as(u32, @intCast(token)) < self.count)
                output[@intCast(token)] -= policy.penalty;
        }
        return output;
    }

    pub fn argmax(self: Selection) c_int {
        return c.ds4_sampling_argmax(self.adjusted().ptr, self.count);
    }
    pub fn argmaxExcluding(self: Selection, excluded: c_int) c_int {
        return c.ds4_sampling_argmax_excluding(self.adjusted().ptr, self.count, excluded);
    }
    pub fn stop(self: Selection, token: c_int, mode: c.ds4_think_mode) bool {
        const vocab = self.vocab orelse return false;
        if (token < 0) return false;
        if (token == vocab.eos_id or (vocab.endoftext_id >= 0 and token == vocab.endoftext_id)) return true;
        // No-thinking prompts already supplied the protocol close tag. A
        // repeated thinking control is not printable assistant content.
        return !c.ds4_think_mode_enabled(mode) and
            ((vocab.think_start_id >= 0 and token == vocab.think_start_id) or
                (vocab.think_end_id >= 0 and token == vocab.think_end_id));
    }
    pub fn argmaxIgnoringEos(self: Selection, mode: c.ds4_think_mode) c_int {
        @setFloatMode(.strict);
        var best: c_int = -1;
        var maximum = negative_infinity;
        for (self.adjusted(), 0..) |value, index| {
            const token: c_int = @intCast(index);
            if (self.stop(token, mode)) continue;
            if (best < 0 or value > maximum) {
                best = token;
                maximum = value;
            }
        }
        return best;
    }
    pub fn sample(self: Selection, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: *u64) c_int {
        return c.ds4_sampling_pick(self.adjusted().ptr, self.count, temperature, top_k, top_p, min_p, rng, self.state.sample_probs);
    }

    fn logNormalizer(self: Selection, maximum: f32) f64 {
        @setFloatMode(.strict);
        var sum: f64 = 0.0;
        for (self.raw()) |value| {
            if (std.math.isFinite(value)) sum += c.exp(@as(f64, value) - @as(f64, maximum));
        }
        return @as(f64, maximum) + c.log(sum);
    }
    pub fn topLogprobs(self: Selection, output: []c.ds4_token_score) c_int {
        @setFloatMode(.strict);
        for (output) |*score| score.* = .{ .id = -1, .logit = negative_infinity, .logprob = negative_infinity };
        var maximum = negative_infinity;
        for (self.raw(), 0..) |value, index| {
            if (!std.math.isFinite(value)) continue;
            if (value > maximum) maximum = value;
            for (output, 0..) |*score, rank| {
                if (score.id < 0 or value > score.logit) {
                    var position = output.len - 1;
                    while (position > rank) : (position -= 1) output[position] = output[position - 1];
                    score.id = @intCast(index);
                    score.logit = value;
                    break;
                }
            }
        }
        // Preserve the finite sentinel even if every logit is nonfinite. C
        // returns the requested k with initialized -1 entries in this case.
        if (!std.math.isFinite(maximum)) return 0;
        const normalizer = self.logNormalizer(maximum);
        for (output) |*score| {
            if (score.id < 0) break;
            score.logprob = if (std.math.isFinite(score.logit)) @floatCast(@as(f64, score.logit) - normalizer) else negative_infinity;
        }
        return @intCast(output.len);
    }
    pub fn tokenLogprob(self: Selection, token: c_int, output: *c.ds4_token_score) c_int {
        @setFloatMode(.strict);
        var maximum = negative_infinity;
        for (self.raw()) |value| {
            if (std.math.isFinite(value) and value > maximum) maximum = value;
        }
        if (!std.math.isFinite(maximum)) return 0;
        const normalizer = self.logNormalizer(maximum);
        output.id = token;
        output.logit = self.raw()[@intCast(token)];
        output.logprob = if (std.math.isFinite(output.logit)) @floatCast(@as(f64, output.logit) - normalizer) else negative_infinity;
        return 1;
    }
    pub fn copyRaw(self: Selection, output: [*c]f32) c_int {
        // Preserve the public memcpy boundary, including its caller ownership
        // contract, rather than introducing native overlap checks here.
        _ = c.memcpy(output, self.state.logits, @as(usize, self.count) * @sizeOf(f32));
        return @intCast(self.count);
    }
    pub fn copySampling(self: Selection, output: [*c]f32) c_int {
        _ = c.memcpy(output, self.adjusted().ptr, @as(usize, self.count) * @sizeOf(f32));
        return @intCast(self.count);
    }
    pub fn setRaw(self: Selection, input: [*c]const f32) c_int {
        _ = c.memcpy(self.state.logits, input, @as(usize, self.count) * @sizeOf(f32));
        return 0;
    }
};

pub fn commonPrefix(state: *c.ds4_session_frontier, prompt: *const c.ds4_tokens) c_int {
    // An invalid frontier returns before reading the prompt, just as C does.
    if (!state.checkpoint_valid) return 0;
    const input = if (prompt.len <= 0) &.{} else prompt.v[0..@intCast(prompt.len)];
    return @intCast((Frontier{ .state = state }).commonPrefix(input));
}
