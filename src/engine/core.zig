//! Loaded V4.1 Metal engine core. The C routing prologue opens/validates the
//! target once to select its family; this owner finishes the selected startup
//! transaction and retires the same actual common record after TP unbind.
//! Other-family workspaces and explicit multi-GPU placement remain outside.
const std = @import("std");
const Tokens = @import("engine_lifecycle_tokens.zig");
const c = @import("gguf_c");
const Config = @import("config.zig").ModelConfig;
const Mapping = @import("model.zig");
const Binding = @import("weights.zig");
const Memory = @import("deepseek41/memory.zig").Memory;
const Layout = @import("deepseek41/layout.zig").Layout;
const Residency = @import("deepseek41/target_residency.zig");
const Record = c.ds4_engine_core;
const Support = @import("support_weights.zig");
const VisionWeights = @import("vision_weights.zig");
const Scope = @import("metal_scope.zig").Scope(c);
const RouteServices = @import("deepseek41/route_services.zig");

fn hasText(value: [*c]const u8) bool {
    return value != null and value[0] != 0;
}
fn text(value: c.ds4_str) []const u8 {
    return if (value.len == 0) &.{} else value.ptr[0..@intCast(value.len)];
}
fn fail(message: [*:0]const u8) bool {
    _ = c.fputs(message, c.__stderrp);
    return false;
}
fn openModel(record: *c.ds4_model, path: [*c]const u8, metal: bool, prefetch: bool) void {
    var failure = Mapping.Failure{};
    var owner = Mapping.Model.init(path, metal, prefetch, &failure) catch failure.die();
    record.* = owner.take();
}

pub const Core = struct {
    state: *Record,
    config: Config,

    // Constructing this view does not create another owner. The Engine embeds
    // Record once; target/support/vision descriptor strings borrow its maps.
    // finishOpen failures leave those partial owners for the same deinit path.
    pub fn finishOpen(self: *Core, options: *const c.ds4_engine_options, shard_model_bytes: *u64) bool {
        @setFloatMode(.optimized);
        const state = self.state;
        if (state.host_context == null) {
            state.host_context = c.ds4_host_context_create(@intCast(@max(options.n_threads, 0)));
            if (state.host_context == null) return fail("ds4: cannot allocate host engine owner\n");
        }
        if (state.backend == c.DS4_BACKEND_METAL and state.metal_context == null) {
            state.metal_context = c.ds4_metal_context_create();
            if (state.metal_context == null) return fail("ds4: cannot allocate Metal engine owner\n");
        }
        const scope = Scope.enter(state);
        defer scope.leave();
        if (!options.inspect_only) {
            const supported = state.backend == c.DS4_BACKEND_METAL and
                options.distributed.role == c.DS4_DISTRIBUTED_NONE and
                !options.load_slice and !options.glm_mtp and
                !options.first_token_test and !options.metal_graph_test and
                state.power_percent == 100 and options.context_size <= 1048576;
            if (!supported) return fail("ds4: native V4.1 requires local Metal without legacy diagnostics (maximum context 1048576)\n");
            // The agent can chdir before its first session. Engram readers
            // reopen this absolute path and compare it against model.fd.
            state.model_path = c.realpath(options.model_path, null);
            if (state.model_path == null) {
                const code = c.__error().*;
                _ = c.fprintf(c.__stderrp, "ds4: cannot resolve V4.1 model path: %s\n", c.strerror(code));
                return false;
            }
        }

        if (hasText(options.vision_path)) {
            openModel(&state.vision_model, options.vision_path, true, false);
            // Binding and the 32-block encoder are separate native owners.
            VisionWeights.bind(&state.deepseek4_vision_weights, &state.vision_model, &self.config.shape);
            var language = std.mem.zeroes(c.ds4_str);
            var vision = std.mem.zeroes(c.ds4_str);
            if (!c.ds4_gguf_get_string(&state.model, "general.source.revision", &language) or
                !c.ds4_gguf_get_string(&state.vision_model, "general.source.revision", &vision) or
                !std.mem.eql(u8, text(language), text(vision)))
                c.ds4_engine_model_die("DeepSeek vision sidecar does not match the language checkpoint");
            state.vision_start_token = 129264;
            state.vision_image_token = 129264;
            state.vision_end_token = 129264;
            state.vision_kind = c.DS4_VISION_DEEPSEEK4;
            state.vision_ready = true;
        }

        var load = Binding.LoadOptions{
            .slice = options.load_slice,
            .start = options.load_layer_start,
            .end = options.load_layer_end,
            .require_output = options.load_output,
        };
        if (options.distributed.role != c.DS4_DISTRIBUTED_NONE and options.distributed.layers.set) {
            load.slice = true;
            load.start = options.distributed.layers.start;
            load.end = options.distributed.layers.end;
            load.require_output = options.distributed.layers.has_output;
            load.optional_output = options.distributed.role == c.DS4_DISTRIBUTED_COORDINATOR;
        }
        if (load.slice and load.end == std.math.maxInt(u32)) load.end = self.config.shape.n_layer - 1;
        if (state.distributed.role != c.DS4_DISTRIBUTED_NONE and state.distributed.layers.set)
            state.distributed.layers.end = load.end;

        var profile = options.expert_profile_path;
        if (!hasText(profile)) profile = c.getenv("DS4_EXPERT_PROFILE");
        const hotlist = c.getenv("DS4_EXPERT_HOTLIST");
        if (hasText(profile) or hasText(hotlist)) c.ds4_core_profile_init(profile, hotlist);

        var weights = Binding.Weights{ .config = self.config };
        var error_text: [256]u8 = @splat(0);
        weights.bind(&state.model, load, &error_text) catch c.ds4_engine_model_die(&error_text);
        state.weights = weights.table;
        if (state.vision_ready) {
            for (state.weights.layer[0..self.config.shape.n_layer]) |*layer| {
                if (layer.ffn_exp_probs_vl == null)
                    return fail("ds4: V4.1 vision requires visual router biases in the language GGUF\n");
            }
        }

        // A network rank owns replicated dense weights and one contiguous
        // expert-ID half. Publish the compatibility counter at the same point
        // as C, before either memory-admission pass or the inspect-only return.
        const shard = options.tp.role != c.DS4_TP_NONE and !state.ssd_streaming;
        const rank: c_int = if (options.tp.role == c.DS4_TP_WORKER) 1 else 0;
        shard_model_bytes.* = 0;
        var memory = self.memoryOwner(options, shard_model_bytes.*);
        const placement = c.ds4_model_residency_inputs{
            .load_slice = load.slice,
            .layer_start = load.start,
            .layer_end = load.end,
            .load_output = load.require_output or (load.optional_output and self.hasOutput()),
            .tp_shard = shard,
            .tp_rank = rank,
            .context = options.context_size,
            .chunk = state.prefill_chunk,
        };
        var residency = Residency.TargetResidency{ .memory = memory, .model = &state.model, .placement = placement };
        if (shard) shard_model_bytes.* = residency.shardBytes();
        memory = self.memoryOwner(options, shard_model_bytes.*);
        residency.memory = memory;
        if (!options.inspect_only) {
            if (!self.admit(&memory, options)) return false;
            // Streaming and TP have their own bounded preparation. Never
            // fault every unowned expert merely because the GGUF is mapped.
            if (options.warm_weights and !state.ssd_streaming and !options.cuda_tensor_parallel and
                options.tp.role == c.DS4_TP_NONE and options.distributed.role == c.DS4_DISTRIBUTED_NONE)
                c.ds4_core_warm_model(&state.model);
            self.loadVocab();
        }
        if (!self.configureOverthink()) return false;
        if (options.glm_mtp) return fail("ds4: --mtp requires a model with embedded MTP weights; use --mtp-model FILE for external support\n");
        memory.capRequestedCache(options.context_size, state.prefill_chunk);
        if (options.inspect_only) {
            if (hasText(options.mtp_path) and options.distributed.role == c.DS4_DISTRIBUTED_NONE)
                return self.loadSupport(options, true);
            return true;
        }

        if (hasText(options.mtp_path) and options.distributed.role == c.DS4_DISTRIBUTED_NONE) {
            if (!self.loadSupport(options, false)) return false;
        }
        // The support map is absent at the first admission pass in C. Refresh
        // immutable resource facts only after support loading, before pass two.
        memory = self.memoryOwner(options, shard_model_bytes.*);
        residency.memory = memory;
        if (shard and c.ds4_core_wired_limit_bytes() == 0) {
            _ = c.fputs("ds4: iogpu.wired_limit_mb is 0 -- TP expert shard will page lazily; raise it (e.g. sudo sysctl iogpu.wired_limit_mb=120000) for full residency\n", c.__stderrp);
            c.ds4_gpu_model_residency_skip(1);
        }
        state.metal_ready = c.ds4_gpu_init() != 0;
        if (!state.metal_ready) return fail("ds4: metal backend unavailable; aborting startup\n");
        c.ds4_gpu_set_quality(state.quality);
        c.ds4_gpu_set_glm_model(false);
        c.ds4_gpu_set_ssd_streaming(state.ssd_streaming);
        if (!memory.automaticCache(options.context_size, state.prefill_chunk)) return false;
        // These are process-wide compatibility resets, not Engine accessors.
        c.ds4_core_reset_streaming();
        state.ssd_streaming_static_decode_map = false;
        if (!memory.reservePrefill(state.ssd_streaming_full_layers_set) or !self.admit(&memory, options)) return false;

        var selection = std.mem.zeroes(c.ds4_model_map_selection);
        defer Residency.freeSelection(&selection);
        if (!residency.install(&selection)) return false;
        if (shard) c.ds4_core_warm_shard(&state.model, &state.weights, rank);

        const support_ready = state.mtp_ready or (state.support_kind == c.DS4_SUPPORT_DSPARK and state.dspark);
        if (support_ready and !mapModel(&state.mtp_model))
            return fail("ds4: metal failed to map support model views; aborting startup. This is commonly caused by insufficient memory or accelerator VM budget.\n");
        // PRO Q4 preload is ineligible for FLASH41. The optional accelerator
        // tensor-copy cache is CUDA-only; its complete Metal action is a no-op.
        _ = c.ds4_gpu_set_model_fd_for_map(state.model.fd, state.model.map);
        Residency.freeSelection(&selection);
        if (support_ready) {
            _ = c.ds4_gpu_set_model_fd_for_map(state.mtp_model.fd, state.mtp_model.map);
            _ = c.ds4_gpu_set_model_fd_for_map(state.model.fd, state.model.map);
        }
        if (state.vision_ready) {
            state.vision_map_ready = mapModel(&state.vision_model);
            if (!state.vision_map_ready) return fail("ds4: metal failed to map the GLM-5.3 vision encoder\n");
        }
        _ = c.fputs("ds4: metal backend initialized for graph diagnostics\n", c.__stderrp);
        c.ds4_core_memory_summary(state, options.context_size);
        return true;
    }

    fn hasOutput(self: *const Core) bool {
        return self.state.weights.output_norm != null and self.state.weights.output != null;
    }
    fn memoryOwner(self: *const Core, options: *const c.ds4_engine_options, shard_bytes: u64) Memory {
        const state = self.state;
        return .{
            .state = &state.memory,
            .config = &self.config,
            .weights = &state.weights,
            .inputs = .{
                .target_model_bytes = if (shard_bytes != 0) shard_bytes else state.model.size,
                .vision_model_bytes = state.vision_model.size,
                .mtp_model_bytes = state.mtp_model.size,
                .session_count = if (options.placement_session_count_hint > 0) @intCast(options.placement_session_count_hint) else 1,
                .ssd_streaming = state.ssd_streaming,
                .dspark = state.dspark,
                // bind() validated the complete selected layer. No runtime
                // slice is admitted on this path; inspection does not map it.
                .include_token = state.weights.layer[0].hc_attn_fn != null,
                .include_output = self.hasOutput(),
            },
        };
    }
    fn admit(self: *const Core, memory_: *const Memory, options: *const c.ds4_engine_options) bool {
        const context: u32 = if (options.context_size > 0) @intCast(options.context_size) else 4096;
        const layout = Layout.init(&self.config, context, self.state.prefill_chunk);
        return memory_.admit(layout.bytes() *| memory_.inputs.session_count, true);
    }
    fn loadVocab(self: *Core) void {
        Tokens.load(self.state, c.DS4_MODEL_FAMILY_DEEPSEEK41);
    }
    fn loadSupport(self: *Core, options: *const c.ds4_engine_options, inspect: bool) bool {
        const state = self.state;
        openModel(&state.mtp_model, options.mtp_path, !inspect, !inspect);
        var summary = std.mem.zeroes(c.ds4_dspark_summary);
        state.support_kind = Support.detect(&state.mtp_model, &state.support_stages, &summary);
        if (!Support.compatible(&state.mtp_model, self.config.shape.family)) return false;
        if (inspect) {
            if (state.support_kind == c.DS4_SUPPORT_DSPARK)
                Support.bind(&state.dspark_weights, &state.mtp_model, &summary, &self.config.shape);
            return true;
        }
        switch (state.support_kind) {
            c.DS4_SUPPORT_MTP_LEGACY => {
                if (options.tp.role != c.DS4_TP_NONE) {
                    _ = c.fputs("ds4: legacy MTP support is ignored under tensor parallelism; using the normal TP decode path\n", c.__stderrp);
                    Mapping.closeRecordPointer(&state.mtp_model);
                    state.support_kind = c.DS4_SUPPORT_NONE;
                    state.support_stages = 0;
                } else {
                    Support.bindLegacy(&state.mtp_weights, &state.mtp_model, &self.config.shape);
                    state.mtp_ready = true;
                    _ = c.fprintf(c.__stderrp, "ds4: MTP support model loaded: %s (draft=%d)\n", options.mtp_path, state.mtp_draft_tokens);
                }
            },
            c.DS4_SUPPORT_DSPARK => {
                Support.bind(&state.dspark_weights, &state.mtp_model, &summary, &self.config.shape);
                const weights = &state.dspark_weights;
                if (weights.v41 and (weights.missing_tensors != 0 or weights.invalid_tensors != 0 or weights.metadata_errors != 0))
                    return fail("ds4: invalid V4.1 DSpark weights\n");
                _ = c.fprintf(c.__stderrp, "ds4: DSpark support model detected: %s (stages=%u block=%u markov_rank=%u tensors=%u missing=%u invalid=%u metadata_errors=%u); use --dspark to enable experimental runtime decode\n", options.mtp_path, state.support_stages, summary.block_size, summary.markov_rank, weights.present_tensors, weights.missing_tensors, weights.invalid_tensors, weights.metadata_errors);
                if (state.dspark and !state.quality and !state.dspark_strict)
                    _ = c.fputs("ds4: DSpark direct verifier-state commits enabled; output may differ from one-token decode due to batched floating-point operation order\n", c.__stderrp);
            },
            else => {
                _ = c.fprintf(c.__stderrp, "ds4: unsupported --mtp-model support model %s (detected=none); expected legacy MTP or DSpark tensors\n", options.mtp_path);
                return false;
            },
        }
        return true;
    }

    fn configureOverthink(self: *Core) bool {
        return Tokens.configure(self.state, c.DS4_MODEL_FAMILY_DEEPSEEK41, null);
    }

    /// Retire the sole embedded record, not this borrowed view.
    pub fn deinit(self: *Core) void {
        freeRecord(self.state);
    }
};

fn mapModel(model: *const c.ds4_model) bool {
    return c.ds4_gpu_set_model_map_range(model.map, model.size, model.tensor_data_pos, model.size -% model.tensor_data_pos, model.max_tensor_bytes) != 0;
}

/// Outer C unbinds TP first. Vocabulary keys and tensor pointers retire before
/// their Model maps. Preserve cleanup even when init never reached GPU init:
/// inspect/manual-cap startup may already have initialized Metal.
pub fn freeRecord(state: *Record) void {
    const metal_context = state.metal_context;
    const host_context = state.host_context;
    const previous_host = c.ds4_host_context_select(host_context);
    const previous = c.ds4_metal_context_select(metal_context);
    defer {
        _ = c.ds4_host_context_select(if (previous_host == host_context) null else previous_host);
        _ = c.ds4_metal_context_select(if (previous == metal_context) null else previous);
        c.ds4_metal_context_destroy(metal_context);
        c.ds4_host_context_destroy(host_context);
    }
    c.ds4_core_profile_close();
    state.weights = std.mem.zeroes(c.ds4_weights);
    c.ds4_vocab_release(&state.vocab);
    c.ds4_core_threads_shutdown();
    RouteServices.cleanup();
    // Submitted commands and expert readers borrow the mapped weights/fds.
    // Drain and retire their views before releasing the underlying mappings.
    c.ds4_gpu_cleanup();
    if (state.mtp_model.map != null) Mapping.closeRecordPointer(&state.mtp_model);
    if (state.vision_model.map != null) Mapping.closeRecordPointer(&state.vision_model);
    Mapping.closeRecordPointer(&state.model);
    c.ds4_ssd_memory_lock_release(&state.simulated_memory);
    if (state.instance_lock_owned) c.ds4_core_release_instance_lock();
    if (state.native_admission_owned) c.ds4_core_native_admission_release();
    c.free(@ptrCast(state.directional_steering_dirs));
    c.free(state.directional_steering_file);
    c.free(state.model_path);
    state.* = std.mem.zeroes(Record);
    state.model.fd = -1;
    state.mtp_model.fd = -1;
    state.vision_model.fd = -1;
}
