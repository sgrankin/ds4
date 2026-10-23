//! Common committed token/logit/image frontier, borrowed from the ONE actual
//! record embedded in the C session. No Engine or Session layout is imported.
const std = @import("std");
const c = @import("gguf_c");
const State = c.ds4_session_frontier;
const Images = []const c.ds4_vision_span;

pub const Frontier = struct {
    state: *State,

    pub fn clearApproximation(self: Frontier) void {
        self.state.approximate_suffix = false;
        self.state.approximate_decoder_replay = false;
        self.state.decoder_replay_rows = 0;
        self.state.decoder_replay_selected = false;
    }

    pub fn forgetImages(self: Frontier) void {
        c.free(self.state.checkpoint_images);
        self.state.checkpoint_images = null;
        self.state.checkpoint_image_count = 0;
    }

    pub fn invalidate(self: Frontier) void {
        self.clearApproximation();
        self.state.checkpoint_valid = false;
        self.state.checkpoint.len = 0;
        self.state.mtp_draft_valid = false;
        self.forgetImages();
    }

    pub fn commonPrefix(self: Frontier, prompt: []const c_int) usize {
        const state = self.state;
        if (!state.checkpoint_valid) return 0;
        const count = @min(@as(usize, @intCast(state.checkpoint.len)), prompt.len);
        var common: usize = 0;
        while (common < count and state.checkpoint.v[common] == prompt[common]) : (common += 1) {}
        return common;
    }

    pub fn startsWithCheckpoint(self: Frontier, prompt: []const c_int) bool {
        const count: usize = @intCast(self.state.checkpoint.len);
        if (prompt.len < count) return false;
        return count == 0 or std.mem.eql(c_int, self.state.checkpoint.v[0..count], prompt[0..count]);
    }

    pub fn visionPrefixMatches(self: Frontier, images: Images) bool {
        const state = self.state;
        if (!state.checkpoint_valid or state.checkpoint_image_count > images.len) return false;
        // NULL is legal when count is zero. Form no C-pointer slice then.
        for (0..state.checkpoint_image_count) |index| {
            const previous = state.checkpoint_images[index];
            const current = images[index];
            if (previous.token_start != current.token_start or
                previous.token_count != current.embedding.token_count or
                !std.mem.eql(u8, &previous.fingerprint, &current.embedding.fingerprint)) return false;
        }
        if (state.checkpoint_image_count < images.len and
            images[state.checkpoint_image_count].token_start < @as(u32, @intCast(state.checkpoint.len))) return false;
        return true;
    }

    pub fn visionStateMatches(self: Frontier, images: Images) bool {
        return self.state.checkpoint_image_count == images.len and self.visionPrefixMatches(images);
    }

    pub fn visionFingerprintPrefixMatches(self: Frontier, images: Images) bool {
        const state = self.state;
        if (!state.checkpoint_valid or state.checkpoint_image_count > images.len) return false;
        for (0..state.checkpoint_image_count) |index| {
            const previous = state.checkpoint_images[index];
            const current = images[index].embedding;
            if (previous.token_count != current.token_count or
                !std.mem.eql(u8, &previous.fingerprint, &current.fingerprint)) return false;
        }
        return true;
    }

    pub fn rebaseVision(self: Frontier, images: []c.ds4_vision_span) bool {
        const state = self.state;
        if (state.checkpoint_image_count != images.len or !self.visionFingerprintPrefixMatches(images)) return false;
        // Check the WHOLE set before changing any caller image offsets.
        for (images, 0..) |*image, index| image.token_start = state.checkpoint_images[index].token_start;
        return true;
    }

    pub fn visionRangeOverlaps(self: Frontier, start: u32, count: u32) bool {
        const end = @as(u64, start) + count;
        for (0..self.state.sync_image_count) |index| {
            const image = self.state.sync_images[index];
            const image_end = @as(u64, image.token_start) + image.embedding.token_count;
            if (start < image_end and image.token_start < end) return true;
        }
        return false;
    }

    pub fn rememberImages(self: Frontier) bool {
        const state = self.state;
        var copy: [*c]c.ds4_vision_identity = null;
        if (state.sync_image_count != 0) {
            if (state.sync_image_count > std.math.maxInt(usize) / @sizeOf(c.ds4_vision_identity)) return false;
            copy = @ptrCast(@alignCast(c.calloc(state.sync_image_count, @sizeOf(c.ds4_vision_identity))));
            if (copy == null) return false;
            for (0..state.sync_image_count) |index| {
                const image = state.sync_images[index];
                copy[index] = .{
                    .token_start = image.token_start,
                    .token_count = image.embedding.token_count,
                    .fingerprint = image.embedding.fingerprint,
                };
            }
        }
        self.forgetImages();
        state.checkpoint_images = copy;
        state.checkpoint_image_count = state.sync_image_count;
        return true;
    }
};
