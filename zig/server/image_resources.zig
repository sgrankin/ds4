//! Request-local image bytes, exact-byte embedding cache and multimodal prompt
//! assembly. The actual Server owns its cache and mutex; every cache hit gives
//! the caller a private copy because prompt layout can expand it in place.
const std = @import("std");
const c = @import("server_images_c");
const allocator = std.heap.c_allocator;
const Input = c.server_image_input;
const Inputs = c.server_image_inputs;
const Cache = c.server_image_cache;
const Entry = c.server_image_cache_entry;
const Embedding = c.ds4_vision_embedding;
const marker_bytes = c.SERVER_IMAGE_MARKER_BYTES;
fn fatal() noreturn {
    _ = c.fputs("ds4-server: out of memory\n", c.__stderrp);
    c.exit(1);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn values(comptime T: type, pointer: [*c]T, count: usize) []T {
    return if (pointer == null and count == 0) &.{} else @as([*]T, @ptrCast(pointer))[0..count];
}
fn fail(errors: []u8, message: []const u8) bool {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return false;
}
pub fn free(optional: ?*Inputs) void {
    const images = optional orelse return;
    for (values(Input, images.v, images.len)) |image| c.free(image.encoded);
    c.free(images.v);
    images.* = std.mem.zeroes(Inputs);
}

fn decode(source: []const u8) ?[]u8 {
    if (source.len == 0 or source.len % 4 != 0 or source.len > 64 * 1024 * 1024) return null;
    const data = allocator.alloc(u8, source.len / 4 * 3) catch fatal();
    var keep = false;
    defer if (!keep) allocator.free(data);
    // The standard decoder handles all complete, unpadded groups. Legacy HTTP
    // input accepts nonzero spare bits in the final padded sextet; its tiny
    // tail below deliberately retains that grammar instead of tightening it.
    const prefix = source.len - 4;
    std.base64.standard_no_pad.Decoder.decode(data[0 .. prefix / 4 * 3], source[0..prefix]) catch return null;
    const tail = source[prefix..][0..4];
    const pad2 = tail[2] == '=';
    const pad3 = tail[3] == '=';
    const table = &std.base64.standard.Decoder.char_to_index;
    const a = table[tail[0]];
    const b = table[tail[1]];
    const third = if (pad2) 0 else table[tail[2]];
    const fourth = if (pad3) 0 else table[tail[3]];
    if (a == 255 or b == 255 or third == 255 or fourth == 255 or (pad2 and !pad3)) return null;
    const bits = (@as(u32, a) << 18) | (@as(u32, b) << 12) | (@as(u32, third) << 6) | fourth;
    var used = prefix / 4 * 3;
    data[used] = @truncate(bits >> 16);
    used += 1;
    if (!pad2) {
        data[used] = @truncate(bits >> 8);
        used += 1;
    }
    if (!pad3) {
        data[used] = @truncate(bits);
        used += 1;
    }
    keep = true;
    return data[0..used];
}
fn random(destination: []u8) bool {
    const fd = c.open("/dev/urandom", c.O_RDONLY);
    if (fd < 0) return false;
    defer _ = c.close(fd);
    var position: usize = 0;
    while (position < destination.len) {
        const count = c.read(fd, destination.ptr + position, destination.len - position);
        if (count < 0 and c.__error().* == c.EINTR) continue;
        if (count <= 0) return false;
        position += @intCast(count);
    }
    return true;
}
pub fn base64(images: *Inputs, media_type: [*c]const u8, source: [*c]const u8, marker: *[marker_bytes]u8) bool {
    if (media_type == null or (c.strcasecmp(media_type, "image/png") != 0 and
        c.strcasecmp(media_type, "image/jpeg") != 0 and c.strcasecmp(media_type, "image/jpg") != 0)) return false;
    const data = decode(text(source)) orelse return false;
    var image = std.mem.zeroes(Input);
    image.encoded = data.ptr;
    image.encoded_len = data.len;
    var nonce: [12]u8 = undefined;
    if (!random(&nonce)) {
        const timestamp: u64 = @bitCast(@as(i64, c.time(null)));
        const fallback = timestamp ^ (@as(u64, @intCast(c.getpid())) << 32) ^ @intFromPtr(images);
        @memcpy(nonce[0..8], std.mem.asBytes(&fallback));
        @memset(nonce[8..], 0);
    }
    const prefix = "\x1eDS4_IMAGE_";
    @memcpy(image.marker[0..prefix.len], prefix);
    var position: usize = prefix.len;
    const hex = "0123456789abcdef";
    for (nonce) |byte| {
        image.marker[position] = hex[byte >> 4];
        image.marker[position + 1] = hex[byte & 15];
        position += 2;
    }
    image.marker[position] = '\x1f';
    image.marker[position + 1] = 0;
    if (images.len == images.cap) {
        const capacity = if (images.cap == 0) 2 else images.cap *% 2;
        images.v = @ptrCast(@alignCast(c.realloc(images.v, capacity *% @sizeOf(Input)) orelse fatal()));
        images.cap = capacity;
    }
    images.v[images.len] = image;
    images.len += 1;
    // C snprintf copied only the string and terminator, preserving bytes after
    // the sentinel in the caller's fixed marker storage.
    @memcpy(marker[0 .. position + 2], image.marker[0 .. position + 2]);
    return true;
}
pub fn dataUri(images: *Inputs, uri: [*c]const u8, marker: *[marker_bytes]u8) bool {
    const source = text(uri);
    inline for (.{ "image/png", "image/jpeg", "image/jpg" }) |media_type| {
        const prefix = "data:" ++ media_type ++ ";base64,";
        if (std.mem.startsWith(u8, source, prefix)) return base64(images, media_type, uri + prefix.len, marker);
    }
    return false;
}

fn remove(cache: *Cache, entry: *Entry) void {
    cache.bytes -%= entry.encoded_len +% entry.data_bytes;
    c.free(entry.encoded);
    c.ds4_vision_embedding_free(&entry.embedding);
    entry.* = std.mem.zeroes(Entry);
}
pub fn clear(cache: *Cache) void {
    for (&cache.entries) |*entry| remove(cache, entry);
    cache.clock = 0;
}
pub fn get(cache: *Cache, input: *const Input, out: *Embedding) bool {
    for (&cache.entries) |*entry| {
        if (entry.encoded == null or entry.encoded_len != input.encoded_len or
            !std.mem.eql(u8, values(u8, entry.encoded, entry.encoded_len), values(u8, input.encoded, input.encoded_len))) continue;
        const data = c.malloc(entry.data_bytes) orelse return false;
        @memcpy(@as([*]u8, @ptrCast(data))[0..entry.data_bytes], @as([*]const u8, @ptrCast(entry.embedding.data))[0..entry.data_bytes]);
        out.* = entry.embedding;
        out.data = @ptrCast(@alignCast(data));
        cache.clock +%= 1;
        entry.used = cache.clock;
        return true;
    }
    return false;
}
/// Cache only successful encodes. Exact byte keys do not trust client hashes;
/// callers receive a copy since DeepSeek expands natural embeddings in place.
pub fn put(cache: *Cache, input: *const Input, embedding: *const Embedding, dim: u32, budget: usize) void {
    if (embedding.data == null or dim == 0 or embedding.token_count == 0 or input.encoded_len == 0 or
        embedding.token_count > budget / @sizeOf(f32) / dim) return;
    const data_bytes = @as(usize, embedding.token_count) * dim * @sizeOf(f32);
    if (input.encoded_len > budget - data_bytes) return;
    const total = input.encoded_len + data_bytes;
    var destination: *Entry = undefined;
    while (true) {
        destination = &cache.entries[0];
        for (cache.entries[1..]) |*entry| if (entry.used < destination.used) {
            destination = entry;
        };
        if (destination.encoded == null and cache.bytes <= budget - total) break;
        if (destination.encoded == null) for (&cache.entries) |*entry| {
            if (entry.encoded != null and (destination.encoded == null or entry.used < destination.used)) destination = entry;
        };
        remove(cache, destination);
    }
    const encoded = c.malloc(input.encoded_len);
    const data = c.malloc(data_bytes);
    if (encoded == null or data == null) {
        c.free(encoded);
        c.free(data);
        return;
    }
    @memcpy(@as([*]u8, @ptrCast(encoded.?))[0..input.encoded_len], values(u8, input.encoded, input.encoded_len));
    @memcpy(@as([*]u8, @ptrCast(data.?))[0..data_bytes], @as([*]const u8, @ptrCast(embedding.data))[0..data_bytes]);
    cache.clock +%= 1;
    destination.* = .{ .encoded = @ptrCast(encoded.?), .encoded_len = input.encoded_len, .embedding = embedding.*, .data_bytes = data_bytes, .used = cache.clock };
    destination.embedding.data = @ptrCast(@alignCast(data.?));
    cache.bytes += total;
}

fn encode(runtime: *c.server, input: *const Input, out: *Embedding, errors: []u8) bool {
    // Caller holds inference_mu across the cache lookup and encoder operation.
    if (get(&runtime.image_cache, input, out)) {
        c.ds4_server_persist_log(c.DS4_LOG_KVCACHE, "ds4-server: vision embedding cache hit");
        return true;
    }
    if (c.ds4_engine_vision_encode_memory(runtime.engine, input.encoded, input.encoded_len, out, if (errors.len == 0) null else errors.ptr, errors.len) == 0) return false;
    // Projection dim equals n_embd at model load. Qwen3.8 uses2560, V4.1
    // uses5120: a legacy4096 width would over-read the encoder allocation.
    put(&runtime.image_cache, input, out, @intCast(c.ds4_engine_embd_dim(runtime.engine)), c.SERVER_IMAGE_CACHE_BYTES);
    var line: [128]u8 = undefined;
    _ = c.snprintf(&line, line.len, "ds4-server: vision embedding encoded; cache=%zu bytes", runtime.image_cache.bytes);
    c.ds4_server_persist_log(c.DS4_LOG_KVCACHE, &line);
    return true;
}
pub fn tokenize(engine: ?*c.ds4_engine, runtime: ?*c.server, request: *c.request, optional_messages: ?*const c.chat_msgs, errors: []u8) bool {
    const messages: []const c.chat_msg = if (optional_messages) |m| (if (m.len <= 0) &.{} else values(c.chat_msg, m.v, @intCast(m.len))) else &.{};
    var count: usize = 0;
    for (messages) |message| count +%= message.images.len;
    if (count == 0) {
        c.ds4_tokenize_rendered_chat(engine, request.prompt_text, &request.prompt);
        return true;
    }
    if (count > 16) return fail(errors, "too many images; at most 16 are allowed");
    if (engine == null or runtime == null or !c.ds4_engine_has_vision(engine)) return fail(errors, "image input requires starting ds4-server with --vision FILE");
    const srv = runtime.?;
    const ok = transaction: {
        const inputs = allocator.alloc(*const Input, count) catch fatal();
        defer allocator.free(inputs);
        const embeddings = allocator.alloc(Embedding, count) catch fatal();
        @memset(embeddings, std.mem.zeroes(Embedding));
        defer {
            for (embeddings) |*embedding| c.ds4_vision_embedding_free(embedding);
            allocator.free(embeddings);
        }
        var next: usize = 0;
        for (messages) |*message| for (values(Input, message.images.v, message.images.len)) |*input| {
            inputs[next] = input;
            next += 1;
        };
        var success = true;
        {
            _ = c.pthread_mutex_lock(&srv.inference_mu);
            defer _ = c.pthread_mutex_unlock(&srv.inference_mu);
            for (inputs, embeddings) |input, *embedding| if (!encode(srv, input, embedding, errors)) {
                success = false;
                break;
            };
        }
        if (success) {
            const spans = allocator.alloc(c.ds4_vision_span, count) catch fatal();
            @memset(spans, std.mem.zeroes(c.ds4_vision_span));
            request.images = spans.ptr;
            const markers = allocator.alloc([marker_bytes]u8, count) catch fatal();
            request.image_markers = markers.ptr;
            var cursor: [*c]const u8 = request.prompt_text;
            for (inputs, embeddings, 0..) |input, *embedding, i| {
                const found = c.strstr(cursor, &input.marker);
                if (found == null) {
                    success = fail(errors, "image marker was lost while rendering the request");
                    break;
                }
                const size = @intFromPtr(found) - @intFromPtr(cursor);
                const prefix = allocator.dupeSentinel(u8, cursor[0..size], 0) catch fatal();
                c.ds4_tokenize_rendered_chat(engine, prefix.ptr, &request.prompt);
                allocator.free(prefix);
                if (c.ds4_prompt_append_vision(engine, &request.prompt, &spans[i], embedding, if (errors.len == 0) null else errors.ptr, errors.len) == 0) {
                    success = false;
                    break;
                }
                request.image_count += 1;
                markers[i] = input.marker;
                cursor = found + c.strlen(&input.marker);
            }
            if (success) c.ds4_tokenize_rendered_chat(engine, cursor, &request.prompt);
        }
        break :transaction success;
    };
    if (!ok) {
        c.ds4_tokens_free(&request.prompt);
        for (values(c.ds4_vision_span, request.images, request.image_count)) |*span| c.ds4_vision_embedding_free(&span.embedding);
        c.free(request.images);
        c.free(request.image_markers);
        request.images = null;
        request.image_markers = null;
        request.image_count = 0;
    }
    return ok;
}
