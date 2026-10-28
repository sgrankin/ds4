//! DS41 keeps its 13-word header, token offsets and span order. Version 2 is
//! F32 main/index; 3 packs index; 4 packs main; 5 packs both. Main FP4 rows keep
//! 256 code bytes and 32 E4M3 scale bytes, index rows 64 + 4 E8M0 scale bytes.
pub const f32_version: u32 = 2;
pub const fp4_version: u32 = 3;
pub const main_fp4_version: u32 = 4;
pub const both_fp4_version: u32 = 5;
pub const f32_main_bytes: u64 = 512 * 4;
pub const fp4_main_bytes: u64 = 288;
pub const f32_index_bytes: u64 = 128 * 4;
pub const fp4_index_bytes: u64 = 68;

pub fn validVersion(wire_version: u32) bool {
    return wire_version >= f32_version and wire_version <= both_fp4_version;
}

pub fn version(main_fp4: bool, index_fp4: bool) u32 {
    return f32_version + 2 * @as(u32, @intFromBool(main_fp4)) + @as(u32, @intFromBool(index_fp4));
}

pub fn mainFp4(wire_version: u32) bool {
    return wire_version == main_fp4_version or wire_version == both_fp4_version;
}

pub fn indexFp4(wire_version: u32) bool {
    return wire_version == fp4_version or wire_version == both_fp4_version;
}

pub fn mainRowBytes(packed_main: bool) u64 {
    return if (packed_main) fp4_main_bytes else f32_main_bytes;
}

pub fn indexRowBytes(packed_index: bool) u64 {
    return if (packed_index) fp4_index_bytes else f32_index_bytes;
}

pub fn baseBytes(position: u32, packed_main: bool, packed_index: bool, vocab: u32) u64 {
    const raw: u64 = @min(position, 128);
    var bytes: u64 = 52 + (@as(u64, position) + vocab) * 4 + 40 * raw * 512 * 4;
    for (0..4) |owner| {
        const rows = position / @as(u32, if (owner < 3) 2 else 1);
        bytes += @as(u64, rows) * (mainRowBytes(packed_main) + indexRowBytes(packed_index));
        if (owner < 3 and position & 1 != 0) bytes += 2 * 512 * 4;
    }
    return bytes;
}
