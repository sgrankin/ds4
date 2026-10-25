//! The ordinary local Metal factory. It composes actual Core, configuration,
//! session-pool and model owners. C allocates its private outer shell and
//! publishes a compatibility metadata snapshot for legacy/reference callers;
//! neither Engine layout nor startup policy crosses this boundary.
const std = @import("std");
const c = @import("gguf_c");
const Mapping = @import("model.zig");
const Core41 = @import("core.zig");
const Tokens = @import("engine_lifecycle_tokens.zig");
const Scope = @import("metal_scope.zig").Scope(c);

fn hasText(value: [*c]const u8) bool {
    return value != null and value[0] != 0;
}
fn fail(message: [:0]const u8) bool {
    _ = c.fputs(message.ptr, c.__stderrp);
    return false;
}
fn equals(value: c.ds4_str, expected: []const u8) bool {
    return value.len == expected.len and std.mem.eql(u8, value.ptr[0..expected.len], expected);
}
fn duplicate(value: [*c]const u8) [*c]u8 {
    return c.strdup(value) orelse {
        c.ds4_engine_model_die("out of memory");
    };
}
fn mapModel(model: *const c.ds4_model) bool {
    return c.ds4_gpu_set_model_map_range(model.map, model.size, model.tensor_data_pos, model.size - model.tensor_data_pos, model.max_tensor_bytes) != 0;
}

pub const Lifecycle = struct {
    core: *c.ds4_engine_core,
    configuration: *c.ds4_engine_configuration,
    pool: *c.ds4_session_resource_pool,
    flint: *?*c.flint_engine,
    streaming: *c.ds4_streaming_policy,

    fn shape(self: *const Lifecycle) *const c.ds4_shape {
        return &self.configuration.shape;
    }

    fn options(self: *const Lifecycle, request: *const c.ds4_engine_options) bool {
        const core = self.core;
        core.backend = request.backend;
        core.quality = request.quality;
        core.glm_mtp = request.glm_mtp;
        core.glm_mtp_timing = request.glm_mtp_timing;
        core.dspark = request.dspark;
        core.dspark_strict = request.dspark_strict;
        core.dspark_exact_sampling = request.dspark_exact_sampling;
        core.cuda_tensor_parallel = request.cuda_tensor_parallel;
        core.glm_tp_token_prefill = request.tp.glm_token_prefill;
        core.ssd_streaming = request.ssd_streaming;
        core.ssd_streaming_cold = request.ssd_streaming_cold;
        core.ssd_streaming_full_layers_set = request.ssd_streaming_full_layers_set;
        core.distributed = request.distributed;
        core.power_percent = @min(if (request.power_percent > 0) request.power_percent else 100, 100);
        core.prefill_chunk = request.prefill_chunk;
        core.memory.ssd_streaming_cache_experts = request.ssd_streaming_cache_experts;
        core.memory.ssd_streaming_cache_bytes = request.ssd_streaming_cache_bytes;
        core.memory.ssd_streaming_full_layers = request.ssd_streaming_full_layers;
        core.memory.ssd_streaming_preload_experts = request.ssd_streaming_preload_experts;
        core.mtp_draft_tokens = @min(if (request.mtp_draft_tokens > 0) request.mtp_draft_tokens else 1, 16);
        core.mtp_margin = if (request.mtp_margin >= 0) request.mtp_margin else 3;
        core.dspark_confidence_threshold = if (request.dspark_confidence_threshold_set) request.dspark_confidence_threshold else 0.6;
        core.dspark_confidence_threshold_set = request.dspark_confidence_threshold_set;
        core.share_session_prefill_workspace = request.share_session_prefill_workspace;
        if (request.dspark and !hasText(request.mtp_path)) return fail("ds4: --dspark requires --mtp-model FILE\n");
        return self.steering(request);
    }

    fn steering(self: *const Lifecycle, request: *const c.ds4_engine_options) bool {
        const core = self.core;
        if (request.directional_steering_mode != c.DS4_STEERING_PROJECT and request.directional_steering_mode != c.DS4_STEERING_ADD)
            return fail("ds4: steering mode must be project or add\n");
        if (!std.math.isFinite(request.directional_steering_attn) or !std.math.isFinite(request.directional_steering_ffn))
            return fail("ds4: steering scales must be finite\n");
        if ((request.directional_steering_attn != 0 or request.directional_steering_ffn != 0) and !hasText(request.directional_steering_file))
            return fail("ds4: directional steering needs --dir-steering-file\n");
        if (hasText(request.directional_steering_file)) {
            core.directional_steering_file = duplicate(request.directional_steering_file);
            core.directional_steering_attn_scale = request.directional_steering_attn;
            core.directional_steering_ffn_scale = request.directional_steering_ffn;
            core.directional_steering_mode = request.directional_steering_mode;
        }
        return true;
    }

    fn openFlint(self: *const Lifecycle, request: *const c.ds4_engine_options) bool {
        var errors: [256]u8 = @splat(0);
        self.flint.* = c.flint_engine_open(request.model_path, request.prefill_chunk, request.flint_profile, &errors, errors.len);
        if (self.flint.* == null) {
            _ = c.fprintf(c.__stderrp, "ds4: Flint: %s\n", &errors);
            return false;
        }
        if (!self.steering(request)) return false;
        if (hasText(request.directional_steering_file) and !c.flint_engine_load_steering(self.flint.*, request.directional_steering_file, request.directional_steering_attn, request.directional_steering_ffn, @intCast(request.directional_steering_mode), &errors, errors.len)) {
            _ = c.fprintf(c.__stderrp, "ds4: Flint steering: %s\n", &errors);
            return false;
        }
        const tokenizer = c.flint_engine_tokenizer(self.flint.*);
        const core = self.core;
        core.backend = c.DS4_BACKEND_METAL;
        core.metal_ready = true;
        core.prefill_chunk = c.flint_engine_prefill_chunk(self.flint.*);
        core.power_percent = if (request.power_percent > 0) request.power_percent else 100;
        core.model_path = duplicate(request.model_path);
        core.vocab.n_vocab = c.flint_tokenizer_vocab_size(tokenizer);
        const fields = .{
            .{ "bos_id", "<|begin_of_text|>" },   .{ "eos_id", "<|end_of_text|>" },
            .{ "system_id", "<|system|>" },       .{ "user_id", "<|user|>" },
            .{ "assistant_id", "<|assistant|>" }, .{ "observation_id", "<|observation|>" },
            .{ "think_start_id", "<think>" },     .{ "think_end_id", "</think>" },
        };
        inline for (fields) |field| @field(core.vocab, field[0]) = c.flint_tokenizer_special_id(tokenizer, field[1]);
        core.vocab.endoftext_id = core.vocab.eos_id;
        return Tokens.configure(core, c.DS4_MODEL_FAMILY_DEEPSEEK41, tokenizer);
    }

    /// Returns LEGACY only for a different architecture. Invalid metadata or
    /// quantization in an accepted family is a real error, never a fallback.
    fn readConfiguration(self: *const Lifecycle) bool {
        var architecture: c.ds4_str = std.mem.zeroes(c.ds4_str);
        if (!c.ds4_gguf_get_string(&self.core.model, "general.architecture", &architecture)) return false;
        var errors: [512]u8 = @splat(0);
        const accepted = if (equals(architecture, "deepseek41"))
            c.ds4_model_config_deepseek41(&self.core.model, &self.configuration.ds41, &errors, errors.len)
        else if (equals(architecture, "qwen4exp"))
            c.ds4_native_qwen4_model_config(&self.core.model, &self.configuration.qwen, &errors, errors.len)
        else if (equals(architecture, "glm5-next"))
            c.ds4_native_glm53_model_config(&self.core.model, &self.configuration.glm, &errors, errors.len)
        else
            return false;
        if (!accepted) c.ds4_engine_model_die(&errors);
        c.ds4_core_publish_configuration(self.configuration);
        if (self.shape().family == c.DS4_MODEL_FAMILY_QWEN4_EXP)
            c.ds4_gpu_qwen4_set_rope(&self.configuration.qwen.rope_freq, self.shape().n_rot / 2, self.configuration.qwen.rope_mscale);
        return true;
    }

    fn supported(self: *const Lifecycle, request: *const c.ds4_engine_options) bool {
        if (request.inspect_only) return true;
        const core = self.core;
        if (self.shape().family == c.DS4_MODEL_FAMILY_QWEN4_EXP) {
            if (request.load_slice or core.ssd_streaming or request.dspark or core.power_percent != 100 or hasText(request.mtp_path))
                return fail("ds4: Qwen3.8 requires Metal or single-GPU CUDA (or --cpu --first-token-test); tensor parallelism, pipeline execution, SSD streaming, DSpark, external MTP models and power throttling are not supported\n");
            return true;
        }
        if (core.power_percent < 100) return fail("ds4: --power is not supported for GLM 5.3 Flash yet\n");
        if (request.prefill_chunk != 0) return fail("ds4: --prefill-chunk is not supported for GLM 5.3 Flash; GLM uses graph-selected prefill chunks\n");
        if (hasText(request.mtp_path)) return fail("ds4: --mtp-model is not supported for GLM yet\n");
        if (request.first_token_test) return fail("ds4: GLM first-token test is CPU-only; pass --cpu\n");
        return true;
    }

    fn bindVision(self: *const Lifecycle, request: *const c.ds4_engine_options) void {
        if (!hasText(request.vision_path)) return;
        const core = self.core;
        Mapping.openRecord(&core.vision_model, request.vision_path, true, false);
        var errors: [512]u8 = @splat(0);
        if (self.shape().family == c.DS4_MODEL_FAMILY_QWEN4_EXP) {
            if (!c.ds4_native_qwen4_vision_weights_bind(&core.qwen4_vision_weights, &core.vision_model, &errors, errors.len)) c.ds4_engine_model_die(&errors);
            if (core.qwen4_vision_weights.n_out != self.shape().n_embd) {
                _ = c.snprintf(&errors, errors.len, "vision projection_dim mismatch: got %u, expected %u", core.qwen4_vision_weights.n_out, self.shape().n_embd);
                c.ds4_engine_model_die(&errors);
            }
            core.vision_kind = c.DS4_VISION_QWEN4;
        } else {
            if (!c.ds4_native_glm53_vision_weights_bind(&core.vision_weights, &core.vision_model, &errors, errors.len)) c.ds4_engine_model_die(&errors);
            const keys = .{
                .{ "vision_image_token", "glm5-next-vision.image_token_id", 154854 },
                .{ "vision_start_token", "glm5-next-vision.image_start_token_id", 154830 },
                .{ "vision_end_token", "glm5-next-vision.image_end_token_id", 154831 },
            };
            inline for (keys) |item| {
                var value: u32 = 0;
                if (!c.ds4_gguf_get_u32(&core.vision_model, item[1], &value)) {
                    _ = c.snprintf(&errors, errors.len, "required metadata key is missing: %s", item[1]);
                    c.ds4_engine_model_die(&errors);
                }
                @field(core, item[0]) = @intCast(value);
                if (value != item[2]) c.ds4_engine_model_die("unexpected GLM-5.3 vision token IDs");
            }
            core.vision_kind = c.DS4_VISION_GLM53;
        }
        core.vision_ready = true;
    }

    fn bindWeights(self: *const Lifecycle, request: *const c.ds4_engine_options) void {
        var profile = request.expert_profile_path;
        if (!hasText(profile)) profile = c.getenv("DS4_EXPERT_PROFILE");
        const hotlist = c.getenv("DS4_EXPERT_HOTLIST");
        if (hasText(profile) or hasText(hotlist)) c.ds4_core_profile_init(profile, hotlist);
        var errors: [512]u8 = @splat(0);
        const core = self.core;
        const ok = if (self.shape().family == c.DS4_MODEL_FAMILY_QWEN4_EXP)
            c.ds4_native_qwen4_weights_bind(&core.weights, &core.model, &self.configuration.qwen, false, 0, self.shape().n_layer - 1, true, false, &errors, errors.len)
        else
            c.ds4_native_glm53_weights_bind(&core.weights, &core.model, &self.configuration.glm, false, 0, self.shape().n_layer - 1, true, false, &errors, errors.len);
        if (!ok) c.ds4_engine_model_die(&errors);
        if (core.model.ngram_tensor != null)
            _ = c.fprintf(c.__stderrp, "ds4: Qwen BF16 n-grams: %.2f GiB, disk reads only\n", @as(f64, @floatFromInt(core.model.ngram_tensor.*.bytes)) / (1024 * 1024 * 1024));
    }

    fn placement(self: *const Lifecycle, request: *const c.ds4_engine_options) c.ds4_model_residency_inputs {
        return .{ .load_slice = false, .layer_start = 0, .layer_end = self.shape().n_layer - 1, .load_output = true, .tp_shard = false, .tp_rank = 0, .context = request.context_size, .chunk = self.core.prefill_chunk };
    }

    fn initialAdmission(self: *const Lifecycle, request: *const c.ds4_engine_options) bool {
        if (self.shape().family != c.DS4_MODEL_FAMILY_GLM_DSA or request.context_size <= 0) return true;
        var context: u32 = 0;
        if (!c.ds4_native_glm_context_request(self.shape(), request.context_size, &context)) return false;
        const residency = self.placement(request);
        // GLM53's compact cache keeps the complete logical context; there is
        // no alternate expanded-cache geometry.
        return c.ds4_native_engine_admit_glm(self.core, self.shape(), &residency, context, 0, "before GLM graph allocation");
    }

    fn lookup(self: *const Lifecycle, spelling: [:0]const u8) c_int {
        var token: c_int = -1;
        if (!c.ds4_token_map_get(self.core.vocab.token_to_id, spelling.ptr, spelling.len, &token)) {
            _ = c.fprintf(c.__stderrp, "ds4: required tokenizer token is missing: %s\n", spelling.ptr);
            c.exit(1);
        }
        return token;
    }

    fn lockStatic(self: *const Lifecycle, request: *const c.ds4_engine_options) void {
        const core = self.core;
        if (!core.ssd_streaming or c.getenv("DS4_METAL_DISABLE_STREAMING_STATIC_LOCK") != null) return;
        var static_bytes: u64 = 0;
        const budget = c.ds4_native_engine_manual_safe_bytes(core, self.shape(), request.context_size, core.prefill_chunk);
        const experts = c.ds4_native_cache_dynamic_bytes(&core.memory, true, self.shape(), &core.weights) +| core.memory.ssd_streaming_prefill_headroom_bytes +| core.memory.ssd_streaming_full_layer_bytes;
        if (!c.ds4_native_model_spans_static_bytes(&core.weights, self.shape(), true, true, &static_bytes) or static_bytes > budget or experts > budget - static_bytes) {
            _ = c.fputs("ds4: Metal SSD static weights remain pageable to preserve runtime headroom\n", c.__stderrp);
            return;
        }
        var spans: c.ds4_model_map_span_vec = undefined;
        if (!c.ds4_native_model_spans_static(&core.weights, self.shape(), 0, std.math.maxInt(u32), true, true, &spans)) return;
        defer c.free(spans.v);
        var locked: u64 = 0;
        var failed: u64 = 0;
        var can_lock = true;
        const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
        for (spans.v[0..spans.len]) |span| {
            var offset = span.off / page * page;
            if (span.end > core.model.size or span.end <= offset) continue;
            while (offset < span.end) {
                const bytes = @min(span.end - offset, 256 * 1024 * 1024);
                if (can_lock and c.mlock(core.model.map + offset, bytes) == 0) {
                    locked += bytes;
                } else {
                    can_lock = false;
                    failed += bytes;
                }
                offset += bytes;
            }
        }
        _ = c.fprintf(c.__stderrp, "ds4: Metal SSD static weights locked %.2f GiB; pageable %.2f GiB\n", @as(f64, @floatFromInt(locked)) / (1024 * 1024 * 1024), @as(f64, @floatFromInt(failed)) / (1024 * 1024 * 1024));
    }

    fn configureSlabs(self: *const Lifecycle) void {
        const core = self.core;
        if (!core.ssd_streaming) return;
        var per_expert: u64 = 0;
        if (!c.ds4_native_cache_per_expert(self.shape(), &core.weights, &per_expert)) return;
        c.ds4_gpu_set_streaming_expert_cache_expert_bytes(per_expert);
        var routed: u32 = 0;
        var cacheable: u64 = 0;
        var uniform: u32 = 0;
        for (core.weights.layer[0..self.shape().n_layer]) |layer| {
            if (layer.ffn_gate_exps != null and layer.ffn_up_exps != null and layer.ffn_down_exps != null) routed += 1;
        }
        _ = c.ds4_native_cacheable_count(self.shape(), &core.weights, &cacheable, &uniform);
        const boosted = routed - uniform;
        if (boosted > 0) _ = c.fprintf(c.__stderrp, "ds4: SSD streaming mixed-precision model: %u/%u routed layers off the slab size class will bypass the expert cache and read experts via mapped model views\n", boosted, routed);
        if (boosted * 2 > routed) _ = c.fprintf(c.__stderrp, "ds4: WARNING: the majority of routed layers (%u/%u) are off the slab size class (is the FIRST routed layer itself boosted?); expert-cache hit rate will be catastrophic\n", boosted, routed);
        const minimum = @as(u64, uniform) * self.shape().n_expert_used;
        if (minimum != 0 and core.memory.ssd_streaming_cache_experts != 0 and core.memory.ssd_streaming_cache_experts < 2 * minimum)
            _ = c.fprintf(c.__stderrp, "ds4: WARNING: SSD streaming expert cache (%u experts) is under twice the per-token routed working set (%u layers x %u experts = %llu); expect heavy thrashing below %.2f GiB\n", core.memory.ssd_streaming_cache_experts, uniform, self.shape().n_expert_used, @as(c_ulonglong, minimum), @as(f64, @floatFromInt(2 * minimum * per_expert)) / (1024 * 1024 * 1024));
    }

    fn mapTarget(self: *const Lifecycle) bool {
        const core = self.core;
        if (!core.ssd_streaming) {
            core.memory.startup_model_span_bytes = core.model.size - core.model.tensor_data_pos;
            return mapModel(&core.model);
        }
        var spans: c.ds4_model_map_span_vec = undefined;
        if (!c.ds4_native_model_spans_token(&core.weights, &spans)) return fail("ds4: invalid SSD streaming initial token embedding map\n");
        defer c.free(spans.v);
        const allocator = std.heap.c_allocator;
        const offsets = allocator.alloc(u64, spans.len) catch c.ds4_engine_model_die("out of memory");
        defer allocator.free(offsets);
        const sizes = allocator.alloc(u64, spans.len) catch c.ds4_engine_model_die("out of memory");
        defer allocator.free(sizes);
        var total: u64 = 0;
        for (spans.v[0..spans.len], offsets, sizes) |span, *offset, *bytes| {
            offset.* = span.off;
            bytes.* = span.end - span.off;
            total += bytes.*;
        }
        core.memory.startup_model_span_bytes = total;
        _ = c.fprintf(c.__stderrp, "ds4: SSD streaming initial metal model map (%u spans, %.2f GiB tensor span)\n", spans.len, @as(f64, @floatFromInt(total)) / (1024 * 1024 * 1024));
        return c.ds4_gpu_set_model_map_spans(core.model.map, core.model.size, offsets.ptr, sizes.ptr, spans.len, spans.max_tensor_bytes) != 0;
    }

    fn finishMetal(self: *const Lifecycle, request: *const c.ds4_engine_options) bool {
        const core = self.core;
        core.metal_ready = c.ds4_gpu_init() != 0;
        if (!core.metal_ready) return fail("ds4: metal backend unavailable; aborting startup\n");
        c.ds4_gpu_set_quality(core.quality);
        c.ds4_gpu_set_glm_model(self.shape().family == c.DS4_MODEL_FAMILY_GLM_DSA);
        c.ds4_gpu_set_ssd_streaming(core.ssd_streaming);
        if (!c.ds4_native_engine_auto_cache(core, self.shape(), request.context_size, core.prefill_chunk) or
            !c.ds4_native_engine_reserve_cache(core, self.shape(), self.streaming)) return false;
        const residency = self.placement(request);
        if (self.shape().family == c.DS4_MODEL_FAMILY_GLM_DSA) {
            c.ds4_native_engine_fit_glm_cache(core, self.pool, self.shape(), self.streaming, &residency, request.placement_session_count_hint);
            // The second guard includes the transient reservation published by
            // auto/reserve/fit; it is not another graph-only admission pass.
            const extra = if (core.ssd_streaming)
                c.ds4_native_cache_dynamic_bytes(&core.memory, true, self.shape(), &core.weights) +|
                    core.memory.ssd_streaming_prefill_headroom_bytes +| core.memory.ssd_streaming_decode_map_extra_bytes
            else
                0;
            if (request.context_size > 0 and extra != 0) {
                var context: u32 = 0;
                if (!c.ds4_native_glm_context_request(self.shape(), request.context_size, &context)) return false;
                if (!c.ds4_native_engine_admit_glm(core, self.shape(), &residency, context, extra, "after GLM streaming cache budget")) return false;
            }
        }
        c.ds4_gpu_set_streaming_expert_cache_budget(core.memory.ssd_streaming_cache_experts);
        self.lockStatic(request);
        self.configureSlabs();
        _ = c.ds4_gpu_set_model_fd(core.model.fd);
        if (!self.mapTarget()) return fail("ds4: metal failed to map model views; aborting startup. This is commonly caused by insufficient memory or accelerator VM budget.\n");
        // The optional tensor-copy cache and PRO Q4 table preload are ineligible
        // for these Metal model families. Publish the same target fd after map.
        _ = c.ds4_gpu_set_model_fd_for_map(core.model.fd, core.model.map);
        if (core.vision_ready) {
            core.vision_map_ready = mapModel(&core.vision_model);
            if (!core.vision_map_ready) return fail("ds4: metal failed to map the GLM-5.3 vision encoder\n");
        }
        _ = c.fputs("ds4: metal backend initialized for graph diagnostics\n", c.__stderrp);
        c.ds4_core_memory_summary(core, request.context_size);
        return true;
    }

    pub fn open(self: *Lifecycle, request: *const c.ds4_engine_options, owned: *bool) c_int {
        owned.* = false;
        // A failed partial startup uses the same close transaction. Initialize
        // descriptors even on the Flint branch, before a model has been opened.
        self.core.model.fd = -1;
        self.core.mtp_model.fd = -1;
        self.core.vision_model.fd = -1;
        if (request.backend != c.DS4_BACKEND_METAL or request.load_slice or request.cuda_tensor_parallel or
            request.tp.role != c.DS4_TP_NONE or request.distributed.role != c.DS4_DISTRIBUTED_NONE)
            return c.DS4_ENGINE_OPEN_LEGACY;
        if (c.flint_engine_detect(request.model_path)) {
            const ok = self.openFlint(request);
            // A failed Flint factory retires its own partial allocation. Once
            // it returned a handle, even tokenizer-policy failure must close it.
            owned.* = self.flint.* != null;
            return if (ok) c.DS4_ENGINE_OPEN_OK else c.DS4_ENGINE_OPEN_ERROR;
        }
        if (hasText(request.flint_profile)) {
            _ = fail("ds4: --flint-profile requires a Flint checkpoint\n");
            return c.DS4_ENGINE_OPEN_ERROR;
        }
        if (!self.options(request)) return c.DS4_ENGINE_OPEN_ERROR;
        // Prior validation errors own only the caller's shell. They must not
        // shut down process-global GPU/thread/profile state or release its lock.
        owned.* = true;
        self.core.host_context = c.ds4_host_context_create(@intCast(@max(request.n_threads, 0)));
        self.core.metal_context = c.ds4_metal_context_create();
        if (self.core.host_context == null or self.core.metal_context == null) {
            _ = fail("ds4: cannot allocate engine context\n");
            return c.DS4_ENGINE_OPEN_ERROR;
        }
        const scope = Scope.enter(self.core);
        defer scope.leave();
        self.streaming = c.ds4_host_context_streaming(self.core.host_context);
        if (request.simulate_used_memory_bytes != 0 and !c.ds4_ssd_memory_lock_acquire(&self.core.simulated_memory, request.simulate_used_memory_bytes)) return c.DS4_ENGINE_OPEN_ERROR;
        Mapping.openRecord(&self.core.model, request.model_path, true, !request.inspect_only);
        if (!self.readConfiguration()) {
            self.close();
            owned.* = false;
            return c.DS4_ENGINE_OPEN_LEGACY;
        }
        if (self.shape().family == c.DS4_MODEL_FAMILY_DEEPSEEK41) {
            if (request.tp.role != c.DS4_TP_NONE) {
                c.ds4_core_acquire_instance_lock(request.n_threads);
                self.core.instance_lock_owned = true;
            } else {
                c.ds4_core_native_admission_acquire();
                self.core.native_admission_owned = true;
            }
            const config = &self.configuration.ds41;
            var owner = Core41.Core{ .state = self.core, .config = .{ .shape = config.shape, .compress_ratios = config.compress_ratios } };
            var shard_bytes: u64 = 0;
            return if (owner.finishOpen(request, &shard_bytes)) c.DS4_ENGINE_OPEN_OK else c.DS4_ENGINE_OPEN_ERROR;
        }
        // The selected DS4.1 owner path no longer shares model state. Retained
        // families keep the existing admission guard until separately tested.
        c.ds4_core_acquire_instance_lock(request.n_threads);
        self.core.instance_lock_owned = true;
        if (!self.supported(request)) return c.DS4_ENGINE_OPEN_ERROR;
        if (request.warm_weights and !request.inspect_only and !request.ssd_streaming) c.ds4_core_warm_model(&self.core.model);
        self.bindVision(request);
        self.bindWeights(request);
        // Qwen/GLM inspection exits here in the original factory: no tokenizer
        // or Metal setup is required to inspect the bound model tables.
        if (request.inspect_only) return c.DS4_ENGINE_OPEN_OK;
        if (self.shape().family == c.DS4_MODEL_FAMILY_QWEN4_EXP and request.first_token_test) {
            if (c.ds4_gpu_init() == 0 or c.ds4_gpu_set_model_map(self.core.model.map, self.core.model.size) == 0) {
                _ = fail("ds4: Qwen3.8 test model map setup failed\n");
                return c.DS4_ENGINE_OPEN_ERROR;
            }
            if (c.getenv("DS4_QWEN4_GPU") != null) _ = c.ds4_gpu_set_model_fd_for_map(self.core.model.fd, self.core.model.map);
            Tokens.load(self.core, self.shape().family);
            return c.DS4_ENGINE_OPEN_OK;
        }
        if (!self.initialAdmission(request)) return c.DS4_ENGINE_OPEN_ERROR;
        Tokens.load(self.core, self.shape().family);
        if (!Tokens.configure(self.core, self.shape().family, null)) return c.DS4_ENGINE_OPEN_ERROR;
        if (self.core.vision_kind == c.DS4_VISION_QWEN4) {
            self.core.vision_start_token = self.lookup("<|vision_start|>");
            self.core.vision_image_token = self.lookup("<|image_pad|>");
            self.core.vision_end_token = self.lookup("<|vision_end|>");
        }
        if (request.glm_mtp and self.shape().n_nextn_predict == 0) {
            _ = fail("ds4: --mtp requires a model with embedded MTP weights; use --mtp-model FILE for external support\n");
            return c.DS4_ENGINE_OPEN_ERROR;
        }
        if (self.core.ssd_streaming and self.core.memory.ssd_streaming_cache_bytes != 0) {
            const requested = self.core.memory.ssd_streaming_cache_bytes;
            const safe = c.ds4_native_engine_manual_safe_bytes(self.core, self.shape(), request.context_size, self.core.prefill_chunk);
            if (safe != 0 and requested > safe) {
                self.core.memory.ssd_streaming_cache_bytes = safe;
                _ = c.fprintf(c.__stderrp, "ds4: metal SSD streaming cache budget %.2f GiB capped to %.2f GiB to stay below the graph working-set pressure budget\n", @as(f64, @floatFromInt(requested)) / (1024 * 1024 * 1024), @as(f64, @floatFromInt(safe)) / (1024 * 1024 * 1024));
            }
        }
        return if (self.finishMetal(request)) c.DS4_ENGINE_OPEN_OK else c.DS4_ENGINE_OPEN_ERROR;
    }

    pub fn close(self: *const Lifecycle) void {
        closeOwners(self.core, self.pool, self.flint);
    }
};

/// Retire GPU pool views before common maps/device state. The private shell
/// must outlive sessions; it performs its one free only after this returns.
pub fn closeOwners(core: *c.ds4_engine_core, pool: *c.ds4_session_resource_pool, flint: *?*c.flint_engine) void {
    if (flint.* != null) {
        c.flint_engine_close(flint.*);
        flint.* = null;
        c.free(core.model_path);
        c.free(core.directional_steering_file);
        core.* = std.mem.zeroes(c.ds4_engine_core);
        core.model.fd = -1;
        core.mtp_model.fd = -1;
        core.vision_model.fd = -1;
        return;
    }
    c.ds4_native_qwen4_pool_free(&pool.qwen4_pool);
    if (pool.qwen4_shared_workspace != null) {
        c.ds4_native_qwen4_graph_free(pool.qwen4_shared_workspace);
        c.free(pool.qwen4_shared_workspace);
        pool.qwen4_shared_workspace = null;
    }
    pool.* = std.mem.zeroes(c.ds4_session_resource_pool);
    Core41.freeRecord(core);
}
