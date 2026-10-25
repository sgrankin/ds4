//! CPU media owns these arrays; the vision/audio towers borrow them during one
//! inference transaction. Patch rows follow the published HF tensor layout.
const std = @import("std");

pub const Rgb = struct {
    width: u32,
    height: u32,
    pixels: []u8,

    pub fn deinit(self: *Rgb, allocator: std.mem.Allocator) void {
        allocator.free(self.pixels);
        self.* = undefined;
    }
};

pub const ImageParams = struct {
    min_pixels: usize = 65536,
    max_pixels: usize = 16777216,
    patch: u32 = 16,
    merge: u32 = 2,
    temporal: u32 = 2,

    pub fn omni() ImageParams {
        // The official Omni adapter supplies these overrides to its processor.
        return .{ .min_pixels = 65536, .max_pixels = 1048576 };
    }
};

pub const ImagePatches = struct {
    width: u32,
    height: u32,
    grid: [3]u32,
    patch_dim: u32,
    token_count: usize,
    patches: []f32,

    pub fn deinit(self: *ImagePatches, allocator: std.mem.Allocator) void {
        allocator.free(self.patches);
        self.* = undefined;
    }
};

pub const VideoPatches = struct {
    width: u32,
    height: u32,
    grid: [3]u32,
    patch_dim: u32,
    token_count: usize,
    patches: []f32,
    /// One timestamp per temporal pair, at the midpoint of its source frames.
    timestamps: []f64,

    pub fn deinit(self: *VideoPatches, allocator: std.mem.Allocator) void {
        allocator.free(self.patches);
        allocator.free(self.timestamps);
        self.* = undefined;
    }
};

pub const AudioFeatures = struct {
    /// Channel-major [128][frames], excluding right padding the tower won't read.
    mel: []f32,
    frames: usize,
    padded_frames: usize,
    token_count: usize,

    pub fn deinit(self: *AudioFeatures, allocator: std.mem.Allocator) void {
        allocator.free(self.mel);
        self.* = undefined;
    }
};
