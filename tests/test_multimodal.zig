//! CPU transaction checks. Tokenizer/renderer calls are existing boundaries;
//! byte tokens below make role wrapping, image order and ownership observable.
const std = @import("std");
const c = @import("gguf_c");
const Vision = @import("vision_runtime").Vision;
const QwenInput = @import("vision_runtime").Qwen;
var family: c.ds4_chat_family = 0;
var ordinary_calls: usize = 0;
var encoder_ok = true;
var prepared: usize = 0;
var retired: usize = 0;
var position_calls: usize = 0;

export fn ds4_tokens_push(tokens: *c.ds4_tokens, token: c_int) void {
    if (tokens.len == tokens.cap) {
        tokens.cap = @max(8, tokens.cap * 2);
        tokens.v = @ptrCast(@alignCast(c.realloc(tokens.v, @as(usize, @intCast(tokens.cap)) * @sizeOf(c_int)) orelse @panic("OOM")));
    }
    tokens.v[@intCast(tokens.len)] = token;
    tokens.len += 1;
}
fn emit(tokens: *c.ds4_tokens, value: []const u8) void {
    for (value) |byte| ds4_tokens_push(tokens, byte);
}
export fn ds4_vocab_admit_text(_: u64) void {}
export fn ds4_engine_model_die(_: [*c]const u8) noreturn {
    @panic("unexpected fatal allocation failure");
}
export fn ds4_pretokenize(lexer: c.ds4_pretokenizer_family, _: ?*const c.ds4_token_map, _: ?*const c.ds4_token_map, bytes: [*c]const u8, length: u64, tokens: *c.ds4_tokens) bool {
    const expected: c.ds4_pretokenizer_family = if (family == c.DS4_CHAT_QWEN35) c.DS4_PRETOKENIZER_QWEN35 else c.DS4_PRETOKENIZER_GLM4;
    std.debug.assert(lexer == expected);
    emit(tokens, bytes[0..@intCast(length)]);
    return true;
}
export fn ds4_vocab_tokenize_rendered(_: *const c.ds4_vocab, selected: c.ds4_chat_family, value: [*c]const u8, tokens: *c.ds4_tokens) bool {
    std.debug.assert(selected == family);
    ds4_tokens_push(tokens, if (std.mem.eql(u8, std.mem.span(value), "<tool_response>")) 503 else 504);
    return true;
}
export fn ds4_vocab_tokenize_payload(_: *const c.ds4_vocab, selected: c.ds4_chat_family, value: [*c]const u8, closing: [*c]const u8, tokens: *c.ds4_tokens) bool {
    std.debug.assert(selected == family and std.mem.eql(u8, std.mem.span(closing), "</tool_response>"));
    emit(tokens, std.mem.span(value));
    return true;
}
export fn ds4_vocab_chat_append(_: *const c.ds4_vocab, selected: c.ds4_chat_family, _: [*c]const u8, _: [*c]const u8, tokens: *c.ds4_tokens) bool {
    std.debug.assert(selected == family);
    ordinary_calls += 1;
    ds4_tokens_push(tokens, 600);
    return true;
}
export fn ds4_deepseek41_image_layout_build(_: *c.ds4_deepseek4_image_layout, _: u32, _: u32, _: [*c]u8, _: usize) c_int {
    @panic("generic image append entered DeepSeek layout");
}
export fn ds4_deepseek4_image_layout_free(_: *c.ds4_deepseek4_image_layout) void {}

export fn ds4_backend_name(_: c.ds4_backend) [*:0]const u8 {
    return "metal";
}
export fn ds4_gpu_init() c_int {
    return 1;
}
export fn ds4_gpu_set_model_map_range(_: ?*const anyopaque, _: u64, _: u64, _: u64, _: u64) c_int {
    return 1;
}
fn patches(out: *c.ds4_image_patches) void {
    prepared += 1;
    out.* = std.mem.zeroes(c.ds4_image_patches);
    out.content_width = 7;
    out.content_height = 11;
    out.grid_width = 2;
    out.grid_height = 2;
    out.patch_count = 4;
    out.image_token_count = 1;
    out.patches = @ptrCast(@alignCast(c.malloc(4).?));
}
export fn ds4_image_preprocess_qwen4(out: *c.ds4_image_patches, _: *const c.ds4_image, minimum: u32, maximum: u32, _: [*c]u8, _: usize) c_int {
    std.debug.assert(minimum == 64 and maximum == 1024);
    patches(out);
    return 1;
}
export fn ds4_image_preprocess_glm53(out: *c.ds4_image_patches, _: *const c.ds4_image, minimum: u32, maximum: u32, _: [*c]u8, _: usize) c_int {
    std.debug.assert(minimum == 16 and maximum == 8000);
    patches(out);
    return 1;
}
export fn ds4_image_patches_free(out: *c.ds4_image_patches) void {
    retired += 1;
    c.free(out.patches);
    out.* = std.mem.zeroes(c.ds4_image_patches);
}
export fn ds4_image_preprocess_deepseek41(out: *c.ds4_deepseek4_image_patches, _: *const c.ds4_image, _: [*c]u8, _: usize) c_int {
    prepared += 1;
    out.* = std.mem.zeroes(c.ds4_deepseek4_image_patches);
    out.content_width = 7;
    out.content_height = 11;
    out.grid_width = 2;
    out.grid_height = 2;
    out.llm_grid_width = 1;
    out.llm_grid_height = 1;
    out.patches = @ptrCast(@alignCast(c.malloc(4).?));
    return 1;
}
export fn ds4_deepseek4_image_patches_free(out: *c.ds4_deepseek4_image_patches) void {
    retired += 1;
    c.free(out.patches);
    out.* = std.mem.zeroes(c.ds4_deepseek4_image_patches);
}
export fn ds4_native_qwen4_vision_pos_embed(_: [*c]const f32, _: u32, width: u32, height: u32, columns: u32, out: [*c]f32) void {
    position_calls += 1;
    @memset(out[0 .. width * height * columns], 0);
}
export fn ds4_gpu_qwen4_vision_encode(out: [*c]f32, _: [*c]const f32, _: [*c]const f32, _: u32, _: u32, _: ?*const anyopaque, _: u64, _: *const c.ds4_qwen4_vision_weights) c_int {
    out[0] = 17;
    return @intFromBool(encoder_ok);
}
export fn ds4_gpu_glm53_vision_encode(out: [*c]f32, _: [*c]const f32, _: u32, _: u32, _: ?*const anyopaque, _: u64, _: *const c.ds4_glm53_vision_weights) c_int {
    out[0] = 17;
    return @intFromBool(encoder_ok);
}
export fn ds4_gpu_deepseek4_vision_encode(out: [*c]f32, _: [*c]const f32, _: u32, _: u32, _: ?*const anyopaque, _: u64, _: *const c.ds4_deepseek4_vision_weights) c_int {
    out[0] = 17;
    return @intFromBool(encoder_ok);
}

fn embedding(count: u32) c.ds4_vision_embedding {
    const data: [*c]f32 = @ptrCast(@alignCast(c.malloc(@sizeOf(f32)).?));
    data[0] = 17;
    return .{ .data = data, .token_count = count, .fingerprint = @splat(9) };
}
fn inputs(core: *c.ds4_engine_core, shape: *c.ds4_shape, qwen: bool) Vision {
    core.* = std.mem.zeroes(c.ds4_engine_core);
    shape.* = std.mem.zeroes(c.ds4_shape);
    shape.family = if (qwen) c.DS4_MODEL_FAMILY_QWEN4_EXP else c.DS4_MODEL_FAMILY_GLM_DSA;
    core.vision_ready = true;
    core.vision_kind = if (qwen) c.DS4_VISION_QWEN4 else c.DS4_VISION_GLM53;
    core.vocab.user_id = 501;
    core.vocab.observation_id = 502;
    core.vocab.im_start_id = 505;
    core.vocab.im_end_id = 506;
    core.vision_start_token = 510;
    core.vision_image_token = 511;
    core.vision_end_token = 512;
    family = if (qwen) c.DS4_CHAT_QWEN35 else c.DS4_CHAT_GLM4;
    return .{ .core = core, .shape = shape };
}
fn imageTokens(tokens: *c.ds4_tokens) void {
    for ([_]c_int{ 510, 511, 512 }) |token| ds4_tokens_push(tokens, token);
}

test "Qwen and GLM preserve mixed text/image order, wrappers and ownership" {
    for ([_]bool{ false, true }) |qwen| {
        var core: c.ds4_engine_core = undefined;
        var shape: c.ds4_shape = undefined;
        const vision = inputs(&core, &shape, qwen);
        for ([_][*:0]const u8{ "user", "tool", "function" }, 0..) |role, index| {
            const tool = index != 0;
            var actual = std.mem.zeroes(c.ds4_tokens);
            defer c.free(actual.v);
            var expected = std.mem.zeroes(c.ds4_tokens);
            defer c.free(expected.v);
            ds4_tokens_push(&actual, 77);
            ds4_tokens_push(&expected, 77);
            var images = [_]c.ds4_vision_embedding{ embedding(1), embedding(1) };
            const original = images;
            var spans = std.mem.zeroes([2]c.ds4_vision_span);
            var errors: [128]u8 = @splat(0);
            const parts = [_][*c]const u8{ "A", null, "B" };
            try std.testing.expect(vision.message(&actual, role, &parts, &images, images.len, &spans, &errors));
            defer for (&spans) |*span| c.free(span.embedding.data);
            if (qwen) {
                ds4_tokens_push(&expected, 505);
                emit(&expected, if (tool) "user\n<tool_response>\n" else "user\n");
            } else {
                ds4_tokens_push(&expected, if (tool) 502 else 501);
                if (tool) ds4_tokens_push(&expected, 503);
            }
            emit(&expected, "A");
            try std.testing.expectEqual(@as(u32, @intCast(expected.len + 1)), spans[0].token_start);
            imageTokens(&expected);
            try std.testing.expectEqual(@as(u32, @intCast(expected.len + 1)), spans[1].token_start);
            imageTokens(&expected);
            emit(&expected, "B");
            if (qwen) {
                if (tool) emit(&expected, "\n</tool_response>");
                ds4_tokens_push(&expected, 506);
                emit(&expected, "\n");
            } else if (tool) ds4_tokens_push(&expected, 504);
            try std.testing.expectEqualSlices(c_int, expected.v[0..@intCast(expected.len)], actual.v[0..@intCast(actual.len)]);
            for (images, spans, original) |image, span, previous| {
                try std.testing.expect(image.data == null and span.embedding.data == previous.data);
                try std.testing.expectEqualSlices(u8, &previous.fingerprint, &span.embedding.fingerprint);
            }
        }
    }
}

test "failed second image returns ownership and text-only keeps existing family policy" {
    for ([_]bool{ false, true }) |qwen| {
        var core: c.ds4_engine_core = undefined;
        var shape: c.ds4_shape = undefined;
        const vision = inputs(&core, &shape, qwen);
        var tokens = std.mem.zeroes(c.ds4_tokens);
        defer c.free(tokens.v);
        ds4_tokens_push(&tokens, 77);
        var images = [_]c.ds4_vision_embedding{ embedding(1), embedding(std.math.maxInt(c_int)) };
        const before = images;
        defer for (&images) |*image| c.free(image.data);
        var spans = std.mem.zeroes([2]c.ds4_vision_span);
        var errors: [128]u8 = @splat(0);
        const parts = [_][*c]const u8{ "A", "B", "C" };
        try std.testing.expect(!vision.message(&tokens, "tool", &parts, &images, 2, &spans, &errors));
        try std.testing.expectEqual(@as(c_int, 1), tokens.len);
        for (images, spans, before) |image, span, previous| try std.testing.expect(image.data == previous.data and span.embedding.data == null);
        try std.testing.expect(!vision.message(&tokens, "assistant", &parts, &images, 2, &spans, &errors));
        try std.testing.expectEqual(@as(c_int, 1), tokens.len);
        core.vision_ready = false;
        ordinary_calls = 0;
        try std.testing.expect(vision.message(&tokens, "tool", &parts, null, 0, null, &errors));
        try std.testing.expectEqual(@as(usize, if (qwen) 0 else 1), ordinary_calls);
    }
}

test "vision dispatch publishes metadata only after encoding and always retires patches" {
    _ = c.unsetenv("DS4_QWEN4_IMAGE_MAX_TOKENS");
    for ([_]c.ds4_vision_kind{ c.DS4_VISION_DEEPSEEK4, c.DS4_VISION_QWEN4, c.DS4_VISION_GLM53 }) |kind| {
        var core: c.ds4_engine_core = undefined;
        var shape: c.ds4_shape = undefined;
        const vision = inputs(&core, &shape, kind == c.DS4_VISION_QWEN4);
        core.vision_kind = kind;
        shape.n_embd = 2;
        var table: [4]f32 = @splat(0);
        core.vision_model.map = @ptrCast(&table);
        core.vision_model.size = @sizeOf(@TypeOf(table));
        core.qwen4_vision_weights.n_embd = 2;
        core.qwen4_vision_weights.n_out = 2;
        var image = std.mem.zeroes(c.ds4_image);
        image.width = 23;
        image.height = 29;
        image.fingerprint = @splat(9);
        for ([_]bool{ true, false }) |success| {
            encoder_ok = success;
            prepared = 0;
            retired = 0;
            position_calls = 0;
            var out = std.mem.zeroes(c.ds4_vision_embedding);
            defer c.free(out.data);
            var errors: [128]u8 = @splat(0);
            try std.testing.expectEqual(success, vision.encodeImage(&image, &out, &errors));
            try std.testing.expectEqual(@as(usize, 1), prepared);
            try std.testing.expectEqual(prepared, retired);
            try std.testing.expectEqual(@as(usize, if (kind == c.DS4_VISION_QWEN4) 1 else 0), position_calls);
            if (success) {
                try std.testing.expectEqual(@as(u32, 1), out.token_count);
                try std.testing.expectEqual(@as(f32, 17), out.data[0]);
                try std.testing.expectEqual(image.width, out.width);
                try std.testing.expectEqual(image.height, out.height);
                try std.testing.expectEqual(@as(u32, 7), out.content_width);
                try std.testing.expectEqual(@as(u32, 11), out.content_height);
                try std.testing.expectEqual(@as(u32, if (kind == c.DS4_VISION_GLM53) 0 else 1), out.grid_width);
                try std.testing.expectEqualSlices(u8, &image.fingerprint, &out.fingerprint);
            } else try std.testing.expect(out.data == null and out.token_count == 0 and errors[0] != 0);
        }
    }
    encoder_ok = true;
}

test "Qwen image budget preserves existing atol range semantics" {
    defer _ = c.unsetenv("DS4_QWEN4_IMAGE_MAX_TOKENS");
    _ = c.setenv("DS4_QWEN4_IMAGE_MAX_TOKENS", "64junk", 1);
    try std.testing.expectEqual(@as(u32, 64), QwenInput.maxTokens());
    _ = c.setenv("DS4_QWEN4_IMAGE_MAX_TOKENS", "16383", 1);
    try std.testing.expectEqual(@as(u32, 16383), QwenInput.maxTokens());
    _ = c.setenv("DS4_QWEN4_IMAGE_MAX_TOKENS", "16384", 1);
    try std.testing.expectEqual(@as(u32, 1024), QwenInput.maxTokens());
}
