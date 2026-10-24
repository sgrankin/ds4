//! Common loaded vocabulary and overthinking-policy configuration. The model
//! selects its tokenizer explicitly; no active-model globals or Engine layout
//! are consulted. Token strings continue to borrow the actual model mapping.
const std = @import("std");
const c = @import("gguf_c");

fn hasText(value: [*c]const u8) bool {
    return value != null and value[0] != 0;
}
fn fail(message: [:0]const u8) bool {
    _ = c.fputs(message.ptr, c.__stderrp);
    return false;
}

pub fn load(core: *c.ds4_engine_core, family: c.ds4_model_family) void {
    const model = &core.model;
    var tokens: c.ds4_array_ref = std.mem.zeroes(c.ds4_array_ref);
    var merges: c.ds4_array_ref = std.mem.zeroes(c.ds4_array_ref);
    if (!c.ds4_gguf_get_array(model, "tokenizer.ggml.tokens", &tokens) or
        tokens.type != c.GGUF_VALUE_STRING or tokens.len > std.math.maxInt(i32))
        c.ds4_engine_model_die("GGUF tokenizer token table is missing or invalid");
    if (!c.ds4_gguf_get_array(model, "tokenizer.ggml.merges", &merges) or merges.type != c.GGUF_VALUE_STRING)
        c.ds4_engine_model_die("GGUF tokenizer merge table is missing or invalid");
    var source = c.ds4_vocab_source{
        .tokens = .{ .base = model.map, .size = model.size, .pos = tokens.data_pos, .@"error" = @splat(0) },
        .merges = .{ .base = model.map, .size = model.size, .pos = merges.data_pos, .@"error" = @splat(0) },
        .token_count = tokens.len,
        .merge_count = merges.len,
        .family = switch (family) {
            c.DS4_MODEL_FAMILY_GLM_DSA => c.DS4_CHAT_GLM4,
            c.DS4_MODEL_FAMILY_QWEN4_EXP => c.DS4_CHAT_QWEN35,
            else => c.DS4_CHAT_DEEPSEEK41,
        },
        .glm_bos_id = -1,
        .glm_eos_id = -1,
    };
    if (family == c.DS4_MODEL_FAMILY_GLM_DSA) {
        _ = c.ds4_gguf_get_token_id(model, "tokenizer.ggml.bos_token_id", &source.glm_bos_id);
        _ = c.ds4_gguf_get_token_id(model, "tokenizer.ggml.eos_token_id", &source.glm_eos_id);
    }
    var errors: [256]u8 = undefined;
    if (!c.ds4_vocab_build(&core.vocab, &source, &errors)) c.ds4_engine_model_die(&errors);
}

fn mapMarkers(core: *c.ds4_engine_core, family: c.ds4_model_family, tokenizer: ?*const c.flint_tokenizer) c_int {
    const policy = &core.overthink;
    if (policy.count != 0 or core.vocab.n_vocab == 0) return policy.count;
    // Paper 2606.00206, table8: an exact leading-space vocabulary token only.
    // Penalizing the first piece of a split word would affect unrelated words.
    const words = [_][]const u8{
        "perhaps",       "maybe",         "wait",    "Wait",    "actually", "hold",      "Hmm",         "hmm",
        "Alternatively", "alternatively", "However", "however", "instead",  "Instead",   "But",         "but",
        "though",        "although",      "yet",     "rather",  "unless",   "otherwise", "nonetheless", "nevertheless",
        "regardless",    "still",         "anyway",  "Or",      "or",       "either",    "whether",     "uncertain",
        "unsure",        "possibly",      "might",   "could",   "another",  "different", "reconsider",  "rethink",
        "backtrack",     "retry",         "recheck", "revisit", "doubt",    "confused",  "wrong",       "mistake",
        "error",         "incorrect",
    };
    const pretokenizer: c.ds4_pretokenizer_family = switch (family) {
        c.DS4_MODEL_FAMILY_GLM_DSA => c.DS4_PRETOKENIZER_GLM4,
        c.DS4_MODEL_FAMILY_QWEN4_EXP => c.DS4_PRETOKENIZER_QWEN35,
        else => c.DS4_PRETOKENIZER_JOYAI,
    };
    for (words) |word| {
        var buffer: [32]u8 = undefined;
        const spelling = std.mem.printSentinel(&buffer, " {s}", .{word}, 0) catch unreachable;
        var tokens = std.mem.zeroes(c.ds4_tokens);
        defer c.ds4_tokens_free(&tokens);
        if (tokenizer != null) {
            if (!c.flint_tokenizer_encode(tokenizer, spelling.ptr, spelling.len, &tokens)) c.ds4_engine_model_die("out of memory");
        } else {
            c.ds4_vocab_admit_text(spelling.len);
            if (!c.ds4_pretokenize(pretokenizer, core.vocab.token_to_id, core.vocab.merge_rank, spelling.ptr, spelling.len, &tokens)) c.ds4_engine_model_die("out of memory");
        }
        if (tokens.len != 1) continue;
        var length: usize = 0;
        const decoded = if (tokenizer != null)
            c.flint_tokenizer_piece(tokenizer, tokens.v[0], &length)
        else
            c.ds4_vocab_decode_token(&core.vocab, tokens.v[0], &length);
        if (decoded == null) c.ds4_engine_model_die("out of memory");
        defer if (tokenizer == null) c.free(@constCast(decoded));
        if (length == spelling.len and std.mem.eql(u8, decoded[0..length], spelling)) {
            policy.ids[@intCast(policy.count)] = tokens.v[0];
            policy.count += 1;
        }
    }
    return policy.count;
}

pub fn set(core: *c.ds4_engine_core, family: c.ds4_model_family, tokenizer: ?*const c.flint_tokenizer, penalty: f32, think_only: bool) bool {
    const bits: u32 = @bitCast(penalty);
    if ((bits & 0x7f800000) == 0x7f800000 or penalty < 0) return false;
    if (penalty > 0 and mapMarkers(core, family, tokenizer) == 0) return false;
    core.overthink.penalty = penalty;
    core.overthink.think_only = think_only;
    return true;
}

pub fn configure(core: *c.ds4_engine_core, family: c.ds4_model_family, tokenizer: ?*const c.flint_tokenizer) bool {
    const policy = &core.overthink;
    const after = c.getenv("DS4_OVERTHINK_AFTER");
    if (hasText(after)) {
        var end: [*c]u8 = null;
        c.__error().* = 0;
        const value = c.strtoul(after, &end, 10);
        if (c.__error().* != 0 or end == after or end[0] != 0 or
            c.strspn(after, "0123456789") != c.strlen(after) or value > std.math.maxInt(c_int))
            return fail("ds4: DS4_OVERTHINK_AFTER must be a nonnegative integer\n");
        policy.after = @intCast(value);
    }
    const setting = c.getenv("DS4_OVERTHINK_PENALTY");
    if (!hasText(setting)) return true;
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const penalty = c.strtod(setting, &end);
    if (c.__error().* != 0 or end == setting or end[0] != 0 or
        c.strspn(setting, "0123456789.eE+-") != c.strlen(setting) or
        !(penalty >= 0 and penalty <= std.math.floatMax(f32)))
        return fail("ds4: DS4_OVERTHINK_PENALTY must be a finite nonnegative number\n");
    if (penalty == 0) return true;
    const scope = c.getenv("DS4_OVERTHINK_SCOPE");
    policy.think_only = scope == null or c.strcmp(scope, "think") == 0;
    if (!policy.think_only and c.strcmp(scope, "all") != 0) return fail("ds4: DS4_OVERTHINK_SCOPE must be think or all\n");
    if (core.vocab.n_vocab == 0) return true;
    if (!set(core, family, tokenizer, @floatCast(penalty), policy.think_only)) return fail("ds4: no exact single-token overthinking markers in this vocabulary\n");
    _ = c.fprintf(c.__stderrp, "ds4: overthinking logit penalty %.6g, scope=%s, after=%d generated thinking tokens, mapped=%d/50; speculation disabled\n", @as(f64, policy.penalty), @as([*c]const u8, if (policy.think_only) "think" else "all"), policy.after, policy.count);
    return true;
}
