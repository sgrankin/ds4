//! Common public vision transaction. Core/Shape are the actual shared
//! storage; Engine remains opaque at the C family/backend dispatch boundary.
//! Natural image rows are expanded into prompt rows only when ownership moves
//! into a span. Failed construction leaves the caller's embedding untouched.
const std = @import("std");
const c = @import("gguf_c");
pub const Qwen = @import("qwen4/vision_input.zig");
const Shape = c.ds4_shape;
const Core = c.ds4_engine_core;
const Embedding = c.ds4_vision_embedding;
const Span = c.ds4_vision_span;
const Tokens = c.ds4_tokens;
fn errorPointer(errors: []u8) [*c]u8 {
    return if (errors.len == 0) null else errors.ptr;
}
fn fail(errors: []u8, message: []const u8) bool {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return false;
}
fn die() noreturn {
    c.ds4_engine_model_die("out of memory");
    unreachable;
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
pub fn freeEmbedding(optional: ?*Embedding) void {
    const embedding = optional orelse return;
    c.free(embedding.data);
    embedding.* = std.mem.zeroes(Embedding);
}
pub const Vision = struct {
    core: *Core,
    shape: *const Shape,

    pub fn encodeFile(self: Vision, path: [*c]const u8, out: ?*Embedding, errors: []u8) bool {
        var image = std.mem.zeroes(c.ds4_image);
        if (c.ds4_image_decode_file(&image, path, errorPointer(errors), errors.len) == 0) return false;
        defer c.ds4_image_free(&image);
        return self.encodeImage(&image, out, errors);
    }
    pub fn encodeMemory(self: Vision, encoded: [*c]const u8, length: usize, out: ?*Embedding, errors: []u8) bool {
        var image = std.mem.zeroes(c.ds4_image);
        if (c.ds4_image_decode_memory(&image, encoded, length, errorPointer(errors), errors.len) == 0) return false;
        defer c.ds4_image_free(&image);
        return self.encodeImage(&image, out, errors);
    }
    pub fn encodeImage(self: Vision, image: *const c.ds4_image, optional_out: ?*Embedding, errors: []u8) bool {
        if (optional_out == null or !self.core.vision_ready) return fail(errors, "vision encoder is not loaded");
        const out = optional_out.?;
        out.* = std.mem.zeroes(Embedding);
        if (!self.core.metal_ready) self.core.metal_ready = c.ds4_gpu_init() != 0;
        const model = &self.core.vision_model;
        if (self.core.metal_ready and !self.core.vision_map_ready) {
            self.core.vision_map_ready = c.ds4_gpu_set_model_map_range(model.map, model.size, model.tensor_data_pos, model.size - model.tensor_data_pos, model.max_tensor_bytes) != 0;
        }
        if (!self.core.metal_ready or !self.core.vision_map_ready) {
            if (errors.len != 0) _ = c.snprintf(errorPointer(errors), errors.len, "unable to map vision weights on %s", c.ds4_backend_name(self.core.backend));
            return false;
        }
        var data: [*c]f32 = null;
        var count: u32 = 0;
        var content_width: u32 = 0;
        var content_height: u32 = 0;
        var grid_width: u32 = 0;
        var grid_height: u32 = 0;
        var layout: u32 = 0;
        const ok = encoded: {
            switch (self.core.vision_kind) {
                c.DS4_VISION_DEEPSEEK4 => {
                    var patches = std.mem.zeroes(c.ds4_deepseek4_image_patches);
                    if (c.ds4_image_preprocess_deepseek41(&patches, image, errorPointer(errors), errors.len) == 0) return false;
                    defer c.ds4_deepseek4_image_patches_free(&patches);
                    count = patches.llm_grid_height *% patches.llm_grid_width;
                    const allocation = c.malloc(@as(usize, count) *% self.shape.n_embd *% @sizeOf(f32));
                    data = if (allocation) |pointer| @ptrCast(@alignCast(pointer)) else null;
                    const success = data != null and c.ds4_gpu_deepseek4_vision_encode(data, patches.patches, patches.grid_height, patches.grid_width, model.map, model.size, &self.core.deepseek4_vision_weights) != 0;
                    content_width = patches.content_width;
                    content_height = patches.content_height;
                    grid_width = patches.llm_grid_width;
                    grid_height = patches.llm_grid_height;
                    layout = c.DS4_VISION_LAYOUT_DEEPSEEK4_NATURAL;
                    break :encoded success;
                },
                c.DS4_VISION_QWEN4 => {
                    var patches = std.mem.zeroes(c.ds4_image_patches);
                    defer c.ds4_image_patches_free(&patches);
                    if (!Qwen.encode(model, &self.core.qwen4_vision_weights, image, 64, Qwen.maxTokens(), &data, &patches, errors)) return false;
                    count = patches.image_token_count;
                    content_width = patches.content_width;
                    content_height = patches.content_height;
                    grid_width = patches.grid_width / 2;
                    grid_height = patches.grid_height / 2;
                    break :encoded true;
                },
                else => {
                    var patches = std.mem.zeroes(c.ds4_image_patches);
                    if (c.ds4_image_preprocess_glm53(&patches, image, 16, 8000, errorPointer(errors), errors.len) == 0) return false;
                    defer c.ds4_image_patches_free(&patches);
                    count = patches.image_token_count;
                    const allocation = c.malloc(@as(usize, count) * 4096 * @sizeOf(f32));
                    data = if (allocation) |pointer| @ptrCast(@alignCast(pointer)) else null;
                    const success = data != null and c.ds4_gpu_glm53_vision_encode(data, patches.patches, patches.grid_height, patches.grid_width, model.map, model.size, &self.core.vision_weights) != 0;
                    content_width = patches.content_width;
                    content_height = patches.content_height;
                    break :encoded success;
                },
            }
        };
        if (data == null) return fail(errors, "unable to allocate vision output");
        if (!ok) {
            c.free(data);
            return fail(errors, switch (self.core.vision_kind) {
                c.DS4_VISION_DEEPSEEK4 => "DeepSeek V4 vision inference failed",
                c.DS4_VISION_QWEN4 => "Qwen3.8 vision inference failed",
                else => "GLM-5.3 vision inference failed",
            });
        }
        out.data = data;
        out.token_count = count;
        out.layout = layout;
        out.grid_width = grid_width;
        out.grid_height = grid_height;
        out.width = image.width;
        out.height = image.height;
        out.content_width = content_width;
        out.content_height = content_height;
        out.fingerprint = image.fingerprint;
        return true;
    }
    fn sentinel(self: Vision, offset: u64) ?[*]align(1) const u16 {
        const model = &self.core.vision_model;
        const bytes = @as(u64, self.shape.n_embd) * @sizeOf(u16);
        if (model.map == null or offset > model.size or bytes > model.size - offset) return null;
        return @ptrCast(model.map + offset);
    }
    pub fn append(self: Vision, optional_tokens: ?*Tokens, optional_span: ?*Span, optional_embedding: ?*Embedding, errors: []u8) bool {
        if (optional_tokens == null or optional_span == null or optional_embedding == null or
            optional_embedding.?.data == null or optional_embedding.?.token_count == 0 or !self.core.vision_ready) return fail(errors, "invalid vision prompt input");
        const tokens = optional_tokens.?;
        const span = optional_span.?;
        const embedding = optional_embedding.?;
        if (self.core.vision_kind != c.DS4_VISION_DEEPSEEK4) {
            if (@as(u64, @intCast(tokens.len)) + embedding.token_count + 2 > std.math.maxInt(c_int)) return fail(errors, "vision prompt is too large");
            span.* = std.mem.zeroes(Span);
            c.ds4_tokens_push(tokens, self.core.vision_start_token);
            span.token_start = @intCast(tokens.len);
            for (0..embedding.token_count) |_| c.ds4_tokens_push(tokens, self.core.vision_image_token);
            c.ds4_tokens_push(tokens, self.core.vision_end_token);
            span.embedding = embedding.*;
            embedding.* = std.mem.zeroes(Embedding);
            return true;
        }
        if (embedding.layout != c.DS4_VISION_LAYOUT_DEEPSEEK4_NATURAL or embedding.grid_height == 0 or
            embedding.grid_width == 0 or embedding.token_count != embedding.grid_height *% embedding.grid_width) return fail(errors, "invalid DeepSeek vision embedding layout");
        var layout = std.mem.zeroes(c.ds4_deepseek4_image_layout);
        if (c.ds4_deepseek41_image_layout_build(&layout, embedding.grid_height, embedding.grid_width, errorPointer(errors), errors.len) == 0) return false;
        defer c.ds4_deepseek4_image_layout_free(&layout);
        if (layout.image_count != embedding.token_count or layout.token_count > @as(u32, @intCast(std.math.maxInt(c_int) - tokens.len))) return fail(errors, "DeepSeek vision prompt is too large");
        const count = @as(u64, layout.token_count) * self.shape.n_embd;
        if (count > std.math.maxInt(usize) / @sizeOf(f32)) return fail(errors, "DeepSeek vision embedding is too large");
        const allocation = c.malloc(@as(usize, @intCast(count)) * @sizeOf(f32)) orelse return fail(errors, "unable to allocate DeepSeek image block");
        var keep = false;
        defer if (!keep) c.free(allocation);
        const block: [*]f32 = @ptrCast(@alignCast(allocation));
        const weights = &self.core.deepseek4_vision_weights;
        const sentinels = [_]?[*]align(1) const u16{
            self.sentinel(weights.image_start),   null,                             null,
            self.sentinel(weights.image_newline), self.sentinel(weights.image_end),
        };
        if (sentinels[c.DS4_DEEPSEEK4_IMAGE_START] == null or sentinels[c.DS4_DEEPSEEK4_IMAGE_NEWLINE] == null or sentinels[c.DS4_DEEPSEEK4_IMAGE_END] == null) return fail(errors, "DeepSeek vision sentinels are not mapped");
        var image_position: u32 = 0;
        for (layout.types[0..layout.token_count], 0..) |kind, row| {
            const destination = block[row * self.shape.n_embd ..][0..self.shape.n_embd];
            if (kind == c.DS4_DEEPSEEK4_IMAGE) {
                if (image_position >= layout.image_count) return fail(errors, "invalid DeepSeek image embedding permutation");
                const source = layout.perm[image_position];
                image_position += 1;
                @memcpy(destination, embedding.data[@as(usize, source) * self.shape.n_embd ..][0..self.shape.n_embd]);
            } else {
                const source = sentinels[kind].?;
                for (destination, source[0..self.shape.n_embd]) |*value, half| value.* = @bitCast(@as(u32, half) << 16);
            }
        }
        if (image_position != layout.image_count) return fail(errors, "invalid DeepSeek image embedding count");
        const start: u32 = @intCast(tokens.len);
        for (0..layout.token_count) |_| c.ds4_tokens_push(tokens, self.core.vision_image_token);
        c.free(embedding.data);
        embedding.data = block;
        embedding.token_count = layout.token_count;
        embedding.layout = 0;
        embedding.grid_width = 0;
        embedding.grid_height = 0;
        span.* = std.mem.zeroes(Span);
        span.token_start = start;
        span.embedding = embedding.*;
        embedding.* = std.mem.zeroes(Embedding);
        keep = true;
        return true;
    }
    fn family(self: Vision) c.ds4_chat_family {
        return switch (self.shape.family) {
            c.DS4_MODEL_FAMILY_QWEN4_EXP => c.DS4_CHAT_QWEN35,
            c.DS4_MODEL_FAMILY_GLM_DSA => c.DS4_CHAT_GLM4,
            else => c.DS4_CHAT_DEEPSEEK41,
        };
    }
    fn emit(self: Vision, tokens: *Tokens, input: [*c]const u8) void {
        const bytes = text(input);
        if (bytes.len == 0) return;
        c.ds4_vocab_admit_text(bytes.len);
        const vocab = &self.core.vocab;
        const lexer: c.ds4_pretokenizer_family = switch (self.family()) {
            c.DS4_CHAT_GLM4 => c.DS4_PRETOKENIZER_GLM4,
            c.DS4_CHAT_QWEN35 => c.DS4_PRETOKENIZER_QWEN35,
            else => c.DS4_PRETOKENIZER_JOYAI,
        };
        if (!c.ds4_pretokenize(lexer, vocab.token_to_id, vocab.merge_rank, bytes.ptr, bytes.len, tokens)) die();
    }
    pub fn message(self: Vision, optional_tokens: ?*Tokens, role: [*c]const u8, parts: [*c]const [*c]const u8, embeddings: [*c]Embedding, count: usize, spans: [*c]Span, errors: []u8) bool {
        if (optional_tokens == null or role == null or parts == null or (count != 0 and (embeddings == null or spans == null))) return fail(errors, "invalid multimodal message");
        const tokens = optional_tokens.?;
        const name = text(role);
        const tool = std.mem.eql(u8, name, "tool") or std.mem.eql(u8, name, "function");
        const user = std.mem.eql(u8, name, "user");
        const qwen = self.shape.family == c.DS4_MODEL_FAMILY_QWEN4_EXP;
        const glm = self.shape.family == c.DS4_MODEL_FAMILY_GLM_DSA;
        if (!tool and !user) return fail(errors, if (qwen) "multimodal messages require a user or tool role" else "multimodal messages require a supported user or tool role");
        // Text-only tool results retain the ordinary wrapper and payload
        // escaping, and require no loaded vision encoder.
        if (!qwen and count == 0) {
            if (!c.ds4_vocab_chat_append(&self.core.vocab, self.family(), role, parts[0], tokens)) die();
            return true;
        }
        if (!qwen and !glm and self.core.vision_kind != c.DS4_VISION_DEEPSEEK4) return fail(errors, "model does not support image messages");
        if (count != 0) for (embeddings[0..count]) |embedding| if (embedding.data == null or embedding.token_count == 0) return fail(errors, "invalid image embedding");
        const old_length = tokens.len;
        const vocab = &self.core.vocab;
        const glm_tool = glm and tool;
        if (qwen) {
            // Image tool results use a user ChatML turn. Keep newlines outside
            // the escaped payload, as in the official model template.
            c.ds4_tokens_push(tokens, vocab.im_start_id);
            self.emit(tokens, "user");
            self.emit(tokens, "\n");
            if (tool) self.emit(tokens, "<tool_response>\n");
        } else if (glm_tool) {
            if (vocab.observation_id >= 0) c.ds4_tokens_push(tokens, vocab.observation_id);
            if (!c.ds4_vocab_tokenize_rendered(vocab, self.family(), "<tool_response>", tokens)) die();
        } else {
            c.ds4_tokens_push(tokens, vocab.user_id);
            if (tool) self.emit(tokens, "<tool_result>");
        }
        var moved: usize = 0;
        for (0..count + 1) |index| {
            const part: [*c]const u8 = if (parts[index] == null) "" else parts[index];
            if (tool) {
                const closing: [*:0]const u8 = if (qwen or glm_tool) "</tool_response>" else "</tool_result>";
                if (!c.ds4_vocab_tokenize_payload(vocab, self.family(), part, closing, tokens)) die();
            } else self.emit(tokens, part);
            if (index == count) break;
            if (!self.append(tokens, @ptrCast(&spans[index]), @ptrCast(&embeddings[index]), errors)) {
                for (0..moved) |previous| {
                    embeddings[previous] = spans[previous].embedding;
                    spans[previous] = std.mem.zeroes(Span);
                }
                tokens.len = old_length;
                return false;
            }
            moved += 1;
        }
        if (qwen) {
            if (tool) self.emit(tokens, "\n</tool_response>");
            c.ds4_tokens_push(tokens, vocab.im_end_id);
            self.emit(tokens, "\n");
        } else if (glm_tool) {
            if (!c.ds4_vocab_tokenize_rendered(vocab, self.family(), "</tool_response>", tokens)) die();
        } else if (tool) self.emit(tokens, "</tool_result>");
        return true;
    }
};
