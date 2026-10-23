//! Image decoding, EXIF orientation and decoded-pixel identity.
//! Iris PNG/JPEG codecs remain in C, including the existing limits and local
//! progressive-JPEG fixes. Resizing and patch layout live in image_preprocess.zig;
//! this module owns the complete decode-memory/file/free lifecycle.
const std = @import("std");
const c = @import("image_c");
const Image = c.ds4_image;
const allocator = std.heap.c_allocator;

fn fail(message: [*:0]const u8, error_text: [*c]u8, capacity: usize) c_int {
    if (error_text != null and capacity != 0) _ = c.snprintf(error_text, capacity, "%s", message);
    return 0;
}

fn fileError(path: [*c]const u8, error_text: [*c]u8, capacity: usize) c_int {
    if (error_text != null and capacity != 0) _ = c.snprintf(error_text, capacity, "%s: %s", path, c.strerror(c.__error().*));
    return 0;
}

fn read16(bytes: []const u8, endian: std.builtin.Endian) u16 {
    return std.mem.readInt(u16, bytes[0..2], endian);
}

fn read32(bytes: []const u8, endian: std.builtin.Endian) u32 {
    return std.mem.readInt(u32, bytes[0..4], endian);
}

// Read only the existing EXIF orientation subset. Malformed/missing EXIF keeps
// orientation 1; this is not a general TIFF parser or a metadata rewrite.
fn jpegOrientation(data: []const u8) u16 {
    var position: usize = 2;
    while (position + 4 <= data.len and data[position] == 0xff) {
        const marker = data[position + 1];
        position += 2;
        if (marker == 0xd8 or (marker >= 0xd0 and marker <= 0xd9)) continue;
        if (position + 2 > data.len) break;
        const segment: usize = read16(data[position..], .big);
        if (segment < 2 or segment > data.len - position) break;
        if (marker == 0xe1 and segment >= 14 and std.mem.eql(u8, data[position + 2 ..][0..6], "Exif\x00\x00")) {
            const tiff = data[position + 8 .. position + segment];
            if (tiff.len < 8) break;
            const endian: std.builtin.Endian = if (std.mem.eql(u8, tiff[0..2], "II")) .little else if (std.mem.eql(u8, tiff[0..2], "MM")) .big else break;
            if (read16(tiff[2..], endian) != 42) break;
            const directory: usize = read32(tiff[4..], endian);
            if (directory > tiff.len or tiff.len - directory < 2) break;
            const count = read16(tiff[directory..], endian);
            const entries_bytes = @as(usize, count) * 12;
            if (entries_bytes > tiff.len - directory - 2) break;
            for (0..count) |index| {
                const entry = tiff[directory + 2 + index * 12 ..][0..12];
                if (read16(entry, endian) == 0x0112 and read16(entry[2..], endian) == 3 and read32(entry[4..], endian) == 1) {
                    const orientation = read16(entry[8..], endian);
                    return if (orientation >= 1 and orientation <= 8) orientation else 1;
                }
            }
        }
        if (marker == 0xda) break;
        position += segment;
    }
    return 1;
}

fn orientedRgb(output: *Image, pixels: []const u8, width: u32, height: u32, channels: u32, orientation: u16) bool {
    const output_width = if (orientation >= 5) height else width;
    const output_height = if (orientation >= 5) width else height;
    const rgb = allocator.alloc(u8, @as(usize, output_width) * output_height * 3) catch return false;
    for (0..height) |y| {
        for (0..width) |x| {
            var ox = x;
            var oy = y;
            switch (orientation) {
                2 => ox = width - 1 - x,
                3 => {
                    ox = width - 1 - x;
                    oy = height - 1 - y;
                },
                4 => oy = height - 1 - y,
                5 => {
                    ox = y;
                    oy = x;
                },
                6 => {
                    ox = height - 1 - y;
                    oy = x;
                },
                7 => {
                    ox = height - 1 - y;
                    oy = width - 1 - x;
                },
                8 => {
                    ox = y;
                    oy = width - 1 - x;
                },
                else => {},
            }
            const src = pixels[(y * width + x) * channels ..];
            const dst = rgb[(oy * output_width + ox) * 3 ..][0..3];
            if (channels < 3) @memset(dst, src[0]) else @memcpy(dst, src[0..3]);
        }
    }
    output.width = output_width;
    output.height = output_height;
    output.rgb = rgb.ptr; // libc allocation transfers to the public C owner.

    // Preserve the fingerprint FORMAT: SHA256(native-u32 width, native-u32
    // height, oriented RGB bytes). This identifies decoded pixels, not encoded
    // file bytes; changing to a serialized endian or hashing EXIF would differ.
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.asBytes(&output_width));
    hash.update(std.mem.asBytes(&output_height));
    hash.update(rgb);
    hash.final(&output.fingerprint);
    return true;
}

export fn ds4_image_decode_memory(maybe_output: ?*Image, encoded: [*c]const u8, encoded_len: usize, error_text: [*c]u8, error_capacity: usize) c_int {
    const output = maybe_output orelse return 0;
    output.* = std.mem.zeroes(Image);
    if (encoded == null or encoded_len == 0 or encoded_len > c.DS4_IMAGE_MAX_ENCODED_BYTES) return fail("image is empty or exceeds the 64 MiB encoded limit", error_text, error_capacity);
    const bytes = encoded[0..encoded_len];
    if (std.mem.startsWith(u8, bytes, "\x89PNG\r\n\x1a\n")) {
        const decoded = c.png_load_mem(encoded, encoded_len);
        if (decoded == null) return fail("invalid or unsupported PNG image", error_text, error_capacity);
        defer c.png_free(decoded);
        const width: u32 = @intCast(decoded.*.width);
        const height: u32 = @intCast(decoded.*.height);
        const channels: u32 = @intCast(decoded.*.channels);
        const pixels = decoded.*.data[0 .. @as(usize, width) * height * channels];
        if (!orientedRgb(output, pixels, width, height, channels, 1)) return fail("unable to allocate decoded PNG pixels", error_text, error_capacity);
        return 1;
    }
    if (bytes.len >= 2 and bytes[0] == 0xff and bytes[1] == 0xd8) {
        const decoded = c.jpeg_load_mem(encoded, encoded_len);
        if (decoded == null) return fail("invalid or unsupported JPEG image", error_text, error_capacity);
        defer c.jpeg_free(decoded);
        const width: u32 = @intCast(decoded.*.width);
        const height: u32 = @intCast(decoded.*.height);
        const channels: u32 = @intCast(decoded.*.channels);
        const pixels = decoded.*.data[0 .. @as(usize, width) * height * channels];
        if (!orientedRgb(output, pixels, width, height, channels, jpegOrientation(bytes))) return fail("unable to allocate decoded JPEG pixels", error_text, error_capacity);
        return 1;
    }
    return fail("image must be JPEG or PNG", error_text, error_capacity);
}

export fn ds4_image_decode_file(output: ?*Image, path: [*c]const u8, error_text: [*c]u8, error_capacity: usize) c_int {
    // Keep the original partial-failure contract: path/I/O failures leave the
    // caller's record alone; decode_memory clears it when actual decoding starts.
    if (output == null or path == null or path[0] == 0) return fail("image path is empty", error_text, error_capacity);
    const file = c.fopen(path, "rb");
    if (file == null) return fileError(path, error_text, error_capacity);
    var closed = false;
    defer if (!closed) {
        _ = c.fclose(file);
    };
    if (c.fseek(file, 0, c.SEEK_END) != 0) return fileError(path, error_text, error_capacity);
    const length = c.ftell(file);
    if (length <= 0 or length > c.DS4_IMAGE_MAX_ENCODED_BYTES) return fail("image is empty or exceeds the 64 MiB encoded limit", error_text, error_capacity);
    if (c.fseek(file, 0, c.SEEK_SET) != 0) return fileError(path, error_text, error_capacity);
    const data = allocator.alloc(u8, @intCast(length)) catch return fail("unable to allocate encoded image buffer", error_text, error_capacity);
    defer allocator.free(data);
    if (c.fread(data.ptr, 1, data.len, file) != data.len) return fileError(path, error_text, error_capacity);
    _ = c.fclose(file);
    closed = true;
    return ds4_image_decode_memory(output, data.ptr, data.len, error_text, error_capacity);
}

export fn ds4_image_free(maybe_image: ?*Image) void {
    const image = maybe_image orelse return;
    c.free(image.rgb);
    image.* = std.mem.zeroes(Image);
}
