//! DS41 keeps its 13-word header and token offsets in both physical layouts.
//! Version 2 stores F32 index rows; version 3 stores 64 FP4 bytes plus four
//! E8M0 scale bytes. Main-cache wire values remain F32 in either version.
pub const f32_version: u32 = 2;
pub const fp4_version: u32 = 3;
pub const f32_index_bytes: u64 = 128 * 4;
pub const fp4_index_bytes: u64 = 68;

pub fn validVersion(version: u32) bool {
    return version == f32_version or version == fp4_version;
}

pub fn indexRowBytes(packed_index: bool) u64 {
    return if (packed_index) fp4_index_bytes else f32_index_bytes;
}

pub fn baseBytes(position: u32, packed_index: bool, vocab: u32) u64 {
    const raw: u64 = @min(position, 128);
    var bytes: u64 = 52 + (@as(u64, position) + vocab) * 4 + 40 * raw * 512 * 4;
    for (0..4) |owner| {
        const rows = position / @as(u32, if (owner < 3) 2 else 1);
        bytes += @as(u64, rows) * (512 * 4 + indexRowBytes(packed_index));
        if (owner < 3 and position & 1 != 0) bytes += 2 * 512 * 4;
    }
    return bytes;
}
