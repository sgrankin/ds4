//! Bounded GGUF byte reads and metadata indexing. Allocation, mapped-file
//! ownership remain with the C loader; typed descriptors borrow that mapping.
const std = @import("std");
const c = @import("gguf_c");
const Cursor = c.ds4_cursor;
const String = c.ds4_str;
const Metadata = c.ds4_gguf_kv;

fn recordError(cursor: *Cursor, comptime message: []const u8) void {
    if (cursor.@"error"[0] != 0) return;
    // Every fixed message plus the decimal u64 offset fits the 256-byte field.
    _ = std.mem.printSentinel(&cursor.@"error", "{s} at byte {d}", .{ message, cursor.pos }, 0) catch unreachable;
}

fn hasBytes(cursor: *Cursor, bytes: u64) bool {
    // Subtract only after checking bytes, avoiding overflow in pos + bytes.
    if (bytes > cursor.size or cursor.pos > cursor.size - bytes) {
        recordError(cursor, "truncated GGUF file");
        return false;
    }
    return true;
}

export fn ds4_gguf_read(cursor: *Cursor, dst: ?*anyopaque, bytes: u64) bool {
    if (!hasBytes(cursor, bytes)) return false;
    if (bytes != 0) {
        const count: usize = @intCast(bytes);
        const source = (cursor.base + @as(usize, @intCast(cursor.pos)))[0..count];
        const destination = @as([*]u8, @ptrCast(dst.?))[0..count];
        @memcpy(destination, source);
    }
    cursor.pos += bytes;
    return true;
}

export fn ds4_gguf_skip(cursor: *Cursor, bytes: u64) bool {
    if (!hasBytes(cursor, bytes)) return false;
    cursor.pos += bytes;
    return true;
}

export fn ds4_gguf_u32(cursor: *Cursor, value: *u32) bool {
    return ds4_gguf_read(cursor, value, @sizeOf(u32));
}

export fn ds4_gguf_u64(cursor: *Cursor, value: *u64) bool {
    return ds4_gguf_read(cursor, value, @sizeOf(u64));
}

export fn ds4_gguf_string(cursor: *Cursor, value: *String) bool {
    var length: u64 = undefined;
    if (!ds4_gguf_u64(cursor, &length)) return false;
    // Preserve the consumed length prefix if the string body is truncated.
    if (!hasBytes(cursor, length)) return false;
    value.* = .{
        .ptr = @ptrCast(cursor.base + @as(usize, @intCast(cursor.pos))),
        .len = length,
    };
    cursor.pos += length;
    return true;
}

fn scalarWidth(value_type: u32) u64 {
    return switch (value_type) {
        c.GGUF_VALUE_UINT8, c.GGUF_VALUE_INT8, c.GGUF_VALUE_BOOL => 1,
        c.GGUF_VALUE_UINT16, c.GGUF_VALUE_INT16 => 2,
        c.GGUF_VALUE_UINT32, c.GGUF_VALUE_INT32, c.GGUF_VALUE_FLOAT32 => 4,
        c.GGUF_VALUE_UINT64, c.GGUF_VALUE_INT64, c.GGUF_VALUE_FLOAT64 => 8,
        else => 0,
    };
}

export fn ds4_gguf_skip_value(cursor: *Cursor, value_type: u32, depth: c_int) bool {
    if (depth > 8) {
        recordError(cursor, "metadata array nesting is too deep");
        return false;
    }
    const scalar = scalarWidth(value_type);
    if (scalar != 0) return ds4_gguf_skip(cursor, scalar);

    if (value_type == c.GGUF_VALUE_STRING) {
        var ignored: String = undefined;
        return ds4_gguf_string(cursor, &ignored);
    }
    if (value_type == c.GGUF_VALUE_ARRAY) {
        var item_type: u32 = undefined;
        var count: u64 = undefined;
        if (!ds4_gguf_u32(cursor, &item_type)) return false;
        if (!ds4_gguf_u64(cursor, &count)) return false;

        const width = scalarWidth(item_type);
        if (width != 0) {
            if (count > std.math.maxInt(u64) / width) {
                recordError(cursor, "metadata array is too large");
                return false;
            }
            return ds4_gguf_skip(cursor, count * width);
        }
        // Empty arrays consume only their header, even for an unknown item tag.
        var i: u64 = 0;
        while (i < count) : (i += 1) {
            if (!ds4_gguf_skip_value(cursor, item_type, depth + 1)) return false;
        }
        return true;
    }
    recordError(cursor, "unknown GGUF metadata type");
    return false;
}

// Read only the metadata directory. Values stay in the mapping; the C model
// owner allocates these records and uses their absolute offsets for later reads.
export fn ds4_gguf_parse_metadata(cursor: *Cursor, entries: [*c]Metadata, count: u64, alignment: *u64) bool {
    alignment.* = 32;
    if (count == 0) return true;
    // The caller supplies count records. Narrow the nonempty C pointer once so
    // addresses of their fields have ordinary non-null Zig pointer types.
    const records: [*]Metadata = entries;
    var index: u64 = 0;
    while (index < count) : (index += 1) {
        const entry = &records[@as(usize, @intCast(index))];
        if (!ds4_gguf_string(cursor, &entry.key)) return false;
        if (!ds4_gguf_u32(cursor, &entry.type)) return false;
        entry.value_pos = cursor.pos;

        const alignment_key = "general.alignment";
        if (entry.type == c.GGUF_VALUE_UINT32 and entry.key.len == alignment_key.len and
            std.mem.eql(u8, entry.key.ptr[0..alignment_key.len], alignment_key))
        {
            // Peek with an independent error field, as the C loader did. The
            // main cursor still skips the complete value and reports truncation
            // at its own position. Zero is ignored; no power-of-two rule is added.
            var peek = cursor.*;
            peek.@"error" = @splat(0);
            var value: u32 = undefined;
            if (ds4_gguf_u32(&peek, &value) and value != 0) alignment.* = value;
        }
        if (!ds4_gguf_skip_value(cursor, entry.type, 0)) return false;
    }
    return true;
}

const Model = c.ds4_model;
const Array = c.ds4_array_ref;

fn cursorAt(model: *const Model, position: u64) Cursor {
    return .{ .base = model.map, .size = model.size, .pos = position, .@"error" = @splat(0) };
}

// Metadata keeps file order; the first matching key wins, even for duplicates.
export fn ds4_gguf_find_kv(model: *const Model, key: [*c]const u8) [*c]Metadata {
    const wanted = std.mem.span(key);
    if (model.n_kv == 0) return null;
    const entries: [*]Metadata = model.kv;
    for (entries[0..@intCast(model.n_kv)]) |*entry| {
        if (entry.key.len == wanted.len and (wanted.len == 0 or std.mem.eql(u8, entry.key.ptr[0..wanted.len], wanted))) return entry;
    }
    return null;
}

export fn ds4_gguf_get_string(model: *const Model, key: [*c]const u8, output: *String) bool {
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null or entry.*.type != c.GGUF_VALUE_STRING) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    return ds4_gguf_string(&cursor, output);
}
export fn ds4_gguf_get_u32(model: *const Model, key: [*c]const u8, output: *u32) bool {
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null or entry.*.type != c.GGUF_VALUE_UINT32) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    return ds4_gguf_u32(&cursor, output);
}

// Token IDs accept the original four integer tags, but never negative or a
// value outside C int. Rejected conversions leave the caller output unchanged.
export fn ds4_gguf_get_token_id(model: *const Model, key: [*c]const u8, output: *c_int) bool {
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    const maximum = std.math.maxInt(c_int);
    switch (entry.*.type) {
        c.GGUF_VALUE_UINT32 => {
            var value: u32 = undefined;
            if (!ds4_gguf_u32(&cursor, &value) or value > maximum) return false;
            output.* = @intCast(value);
        },
        c.GGUF_VALUE_INT32 => {
            var value: i32 = undefined;
            if (!ds4_gguf_read(&cursor, &value, @sizeOf(i32)) or value < 0) return false;
            output.* = @intCast(value);
        },
        c.GGUF_VALUE_UINT64 => {
            var value: u64 = undefined;
            if (!ds4_gguf_u64(&cursor, &value) or value > maximum) return false;
            output.* = @intCast(value);
        },
        c.GGUF_VALUE_INT64 => {
            var value: i64 = undefined;
            if (!ds4_gguf_read(&cursor, &value, @sizeOf(i64)) or value < 0 or value > maximum) return false;
            output.* = @intCast(value);
        },
        else => return false,
    }
    return true;
}
export fn ds4_gguf_get_u64_compat(model: *const Model, key: [*c]const u8, output: *u64) bool {
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    if (entry.*.type == c.GGUF_VALUE_UINT64) return ds4_gguf_u64(&cursor, output);
    if (entry.*.type == c.GGUF_VALUE_UINT32) {
        var value: u32 = undefined;
        if (!ds4_gguf_u32(&cursor, &value)) return false;
        output.* = value;
        return true;
    }
    return false;
}
export fn ds4_gguf_get_f32_compat(model: *const Model, key: [*c]const u8, output: *f32) bool {
    @setFloatMode(.strict);
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    switch (entry.*.type) {
        c.GGUF_VALUE_FLOAT32 => return ds4_gguf_read(&cursor, output, @sizeOf(f32)),
        c.GGUF_VALUE_FLOAT64 => {
            var value: f64 = undefined;
            if (!ds4_gguf_read(&cursor, &value, @sizeOf(f64))) return false;
            output.* = @floatCast(value);
        },
        c.GGUF_VALUE_UINT32 => {
            var value: u32 = undefined;
            if (!ds4_gguf_u32(&cursor, &value)) return false;
            output.* = @floatFromInt(value);
        },
        c.GGUF_VALUE_INT32 => {
            var value: i32 = undefined;
            if (!ds4_gguf_read(&cursor, &value, @sizeOf(i32))) return false;
            output.* = @floatFromInt(value);
        },
        else => return false,
    }
    return true;
}
export fn ds4_gguf_get_bool(model: *const Model, key: [*c]const u8, output: *bool) bool {
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null or entry.*.type != c.GGUF_VALUE_BOOL) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    var value: u8 = undefined;
    if (!ds4_gguf_read(&cursor, &value, @sizeOf(u8))) return false;
    output.* = value != 0;
    return true;
}
export fn ds4_gguf_get_array(model: *const Model, key: [*c]const u8, output: *Array) bool {
    const entry = ds4_gguf_find_kv(model, key);
    if (entry == null or entry.*.type != c.GGUF_VALUE_ARRAY) return false;
    var cursor = cursorAt(model, entry.*.value_pos);
    // Preserve partial fields: a truncated length may still update the tag.
    if (!ds4_gguf_u32(&cursor, &output.type)) return false;
    if (!ds4_gguf_u64(&cursor, &output.len)) return false;
    output.data_pos = cursor.pos;
    return true;
}

export fn ds4_gguf_tensor_nbytes(tag: u32, elements: u64, output: *u64) bool {
    // One authoritative C type table supplies the format/block constants.
    const info = c.ds4_gguf_tensor_type(tag);
    if (info == null or info.*.block_elems == 0) return false;
    const block_elements: u64 = info.*.block_elems;
    const block_bytes: u64 = info.*.block_bytes;
    // Retain the original unsigned rounding expression, including its wrapping
    // numerator. Do not silently change file-admission policy during this port.
    const blocks = (elements +% block_elements -% 1) / block_elements;
    if (blocks > std.math.maxInt(u64) / block_bytes) return false;
    output.* = blocks * block_bytes;
    return true;
}

// Read the tensor directory, then convert relative GGUF offsets to absolute
// mmap offsets. Strings and weight bytes stay borrowed; C allocates the array
// and retains count admission, fatal errors, mapping and disk-only-tail policy.
export fn ds4_gguf_parse_tensors(model: *Model, cursor: *Cursor) [*c]const u8 {
    const tensors: []c.ds4_tensor = if (model.n_tensors == 0) &.{} else @as([*]c.ds4_tensor, model.tensors)[0..@intCast(model.n_tensors)];
    for (tensors) |*tensor| {
        if (!ds4_gguf_string(cursor, &tensor.name)) return &cursor.@"error";
        if (!ds4_gguf_u32(cursor, &tensor.ndim)) return &cursor.@"error";
        if (tensor.ndim == 0 or tensor.ndim > c.DS4_MAX_DIMS) return "tensor has an unsupported number of dimensions";
        tensor.elements = 1;
        for (tensor.dim[0..tensor.ndim]) |*dimension| {
            if (!ds4_gguf_u64(cursor, dimension)) return &cursor.@"error";
            if (dimension.* != 0 and tensor.elements > std.math.maxInt(u64) / dimension.*) return "tensor element count overflow";
            tensor.elements *= dimension.*;
        }
        if (!ds4_gguf_u32(cursor, &tensor.type)) return &cursor.@"error";
        if (!ds4_gguf_u64(cursor, &tensor.rel_offset)) return &cursor.@"error";
        if (!ds4_gguf_tensor_nbytes(tensor.type, tensor.elements, &tensor.bytes)) c.ds4_gguf_warn_tensor_type(tensor);
    }
    const remainder = cursor.pos % model.alignment;
    // Keep align_up's original unsigned modular addition/subtraction.
    model.tensor_data_pos = if (remainder == 0) cursor.pos else cursor.pos +% model.alignment -% remainder;
    for (tensors) |*tensor| {
        if (tensor.rel_offset > std.math.maxInt(u64) - model.tensor_data_pos) return "tensor offset overflow";
        tensor.abs_offset = model.tensor_data_pos + tensor.rel_offset;
        if (tensor.bytes != 0 and (tensor.abs_offset > model.size or tensor.bytes > model.size - tensor.abs_offset)) return "tensor points outside GGUF file";
        if (tensor.bytes > model.max_tensor_bytes) model.max_tensor_bytes = tensor.bytes;
    }
    return null;
}
