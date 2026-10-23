//! Match Recursive's resident schedules. Explicit selection lets a larger chip
//! exercise a smaller profile; it cannot prove that chip's actual residency.
const std = @import("std");

pub const Profile = enum {
    m4_pro,
    m5_pro,
    m5_max,

    pub fn name(self: Profile) [:0]const u8 {
        return switch (self) {
            .m4_pro => "m4-pro",
            .m5_pro => "m5-pro",
            .m5_max => "m5-max",
        };
    }
    pub fn workerCount(self: Profile) u32 {
        return if (self == .m5_max) 76 else 36;
    }
    pub fn mlpWorkers(self: Profile) u32 {
        return if (self == .m5_max) 64 else 32;
    }
    pub fn prefillChunk(self: Profile) u32 {
        return if (self == .m4_pro) 2048 else 4096;
    }
    pub fn usesNax(self: Profile) bool {
        return self != .m4_pro;
    }
};

pub fn parse(name: []const u8) !Profile {
    for ([_]Profile{ .m4_pro, .m5_pro, .m5_max }) |profile| {
        if (std.mem.eql(u8, name, profile.name())) return profile;
    }
    return error.UnknownFlintProfile;
}

pub fn detect(device: []const u8) !Profile {
    if (std.mem.eql(u8, device, "Apple M4 Pro")) return .m4_pro;
    if (std.mem.eql(u8, device, "Apple M5 Pro")) return .m5_pro;
    if (std.mem.eql(u8, device, "Apple M5 Max")) return .m5_max;
    return error.UnsupportedResidentDevice;
}

pub fn select(device: []const u8, requested: ?[]const u8) !Profile {
    const physical = try detect(device);
    const name = requested orelse return physical;
    if (std.mem.eql(u8, name, "auto")) return physical;
    const selected = try parse(name);
    // Never launch a larger residency envelope or request the M5 tensor path
    // on M4. Smaller schedules can be exercised on the supported M5 Max.
    if (selected.workerCount() > physical.workerCount() or
        (selected.usesNax() and !physical.usesNax())) return error.UnsupportedFlintProfile;
    return selected;
}
