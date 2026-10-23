//! CPU-only checks for resident-kernel selection. A larger GPU may exercise a
//! smaller schedule; an override must never exceed the physical GPU's envelope.
const std = @import("std");
const profiles = @import("profile.zig");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

pub fn run() !void {
    const cases = .{
        .{ "Apple M4 Pro", profiles.Profile.m4_pro, @as(u32, 36), @as(u32, 32), @as(u32, 2048), false },
        .{ "Apple M5 Pro", profiles.Profile.m5_pro, @as(u32, 36), @as(u32, 32), @as(u32, 4096), true },
        .{ "Apple M5 Max", profiles.Profile.m5_max, @as(u32, 76), @as(u32, 64), @as(u32, 4096), true },
    };
    inline for (cases) |case| {
        const profile = case[1];
        try expectEqual(profile, try profiles.detect(case[0]));
        try expectEqual(profile, try profiles.parse(profile.name()));
        try expectEqual(profile, try profiles.select(case[0], null));
        try expectEqual(profile, try profiles.select(case[0], "auto"));
        try expectEqual(profile, try profiles.select(case[0], profile.name()));
        try expectEqual(case[2], profile.workerCount());
        try expectEqual(case[3], profile.mlpWorkers());
        try expectEqual(case[4], profile.prefillChunk());
        try expectEqual(case[5], profile.usesNax());
        try expect(profile.mlpWorkers() < profile.workerCount());
    }

    // Cross-validation on M5 Max uses the real smaller schedules, including
    // M4's non-NAX path. This does not establish residency on an actual Pro.
    try expectEqual(.m4_pro, try profiles.select("Apple M5 Max", "m4-pro"));
    try expectEqual(.m5_pro, try profiles.select("Apple M5 Max", "m5-pro"));
    try expectEqual(.m4_pro, try profiles.select("Apple M5 Pro", "m4-pro"));

    // A wrong override can otherwise deadlock a persistent dispatch at its
    // global barrier, or request an instruction family unavailable on M4.
    try expectError(error.UnsupportedFlintProfile, profiles.select("Apple M4 Pro", "m5-pro"));
    try expectError(error.UnsupportedFlintProfile, profiles.select("Apple M4 Pro", "m5-max"));
    try expectError(error.UnsupportedFlintProfile, profiles.select("Apple M5 Pro", "m5-max"));
    try expectError(error.UnsupportedResidentDevice, profiles.select("Apple M3 Max", "m4-pro"));
    try expectError(error.UnknownFlintProfile, profiles.select("Apple M5 Max", "unknown"));
    std.debug.print("flint: CPU hardware-profile selection passed\n", .{});
}

pub fn main() !void {
    try run();
}
