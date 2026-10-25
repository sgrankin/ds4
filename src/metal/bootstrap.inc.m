/* Fixed ARC factories: native Bootstrap owns feature/env/error and phase
 * sequencing. Literal PSO and strong-reference catalogues stay here rather
 * than becoming mechanically copied pointer assignments in Zig. */
#include "bootstrap.h"


ds4_metal_bootstrap_context ds4_metal_bootstrap_driver(void) {
    return (ds4_metal_bootstrap_context){ &g_bootstrap, &g_schedule,
        &g_expert_cache, &g_expert_load, g_ssd_streaming_mode != 0 };
}
void ds4_metal_bootstrap_reset_fast_cache(void) { ds4_gpu_decode_pipeline_fast_cache_reset(); }
bool ds4_metal_bootstrap_device(void) {
    if (g_runtime_owned) return g_device && g_queue;
    void *device = NULL, *queue = NULL;
    if (!ds4_runtime_metal_acquire(&device, &queue)) {
        fprintf(stderr, "ds4: Metal device/queue not available\n");
        return false;
    }
    g_device = (__bridge id<MTLDevice>)device;
    g_queue = (__bridge id<MTLCommandQueue>)queue;
    g_runtime_owned = true;
    return true;
}
void ds4_metal_bootstrap_diagnostics(void) {
    ds4_gpu_timeline_probe(g_device);
    ds4_gpu_print_device_summary();
}
const char *ds4_metal_bootstrap_device_name(void) { return [[g_device name] UTF8String]; }
ds4_metal_bootstrap_features ds4_metal_bootstrap_features_query(bool disabled) {
    ds4_metal_bootstrap_features result = {0};
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        result.runtime = true;
        result.family = !disabled && [g_device supportsFamily:MTLGPUFamilyMetal4];
        result.queue = [g_device respondsToSelector:@selector(newMTL4CommandQueue)];
    }
#else
    (void)disabled;
#endif
    return result;
}
int ds4_metal_bootstrap_tensor_probe(void) { return ds4_gpu_compile_tensor_probe(); }
bool ds4_metal_bootstrap_queue(void) {
    return g_queue != nil;
}

bool ds4_metal_bootstrap_bookkeeping(void) {
    ds4_metal_exact_cache_init();
    g_q4_expert_table_cache = [NSMutableDictionary dictionary];
    g_q4_expert_layer_residency_cache = [NSMutableDictionary dictionary];
    g_pipeline_cache = [NSMutableDictionary dictionary];
    g_dsv4_completion_cache = [NSCache new];
    g_dsv4_completion_cache.countLimit = 256u;
    g_transient_buffers = [NSMutableArray array];
    g_pending_cbs = [NSMutableArray array];
    if (
        !g_q4_expert_table_cache ||
        !g_q4_expert_layer_residency_cache ||
        !g_pipeline_cache || !g_dsv4_completion_cache ||
        !g_transient_buffers || !g_pending_cbs) {
        fprintf(stderr, "ds4: Metal bookkeeping allocation failed\n");
        g_pending_cbs = nil;
        g_transient_buffers = nil;
        g_dsv4_completion_cache = nil;
        g_pipeline_cache = nil;
        g_q4_expert_layer_residency_cache = nil;
        g_q4_expert_table_cache = nil;
        ds4_metal_exact_cache_cleanup();
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    return true;
}

bool ds4_metal_bootstrap_source(ds4_metal_shader_resources *out) {
    NSString *source = ds4_gpu_full_source();
    if (!source) {
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    NSMutableDictionary *macros = [NSMutableDictionary new];
    if (g_metal4_tensor_api_enabled) {
        macros[@"DS4_METAL_HAS_TENSOR"] = @"1";
        fprintf(stderr, "ds4: Metal 4 tensor API enabled for Tensor kernels\n");
    }

    out->source = source ? (__bridge_retained void *)source : NULL;
    out->options = options ? (__bridge_retained void *)options : NULL;
    out->macros = macros ? (__bridge_retained void *)macros : NULL;
    return true;
}
void ds4_metal_bootstrap_source_release(ds4_metal_shader_resources *source) {
    if (source->source) { id value = (__bridge_transfer id)source->source; (void)value; }
    if (source->options) { id value = (__bridge_transfer id)source->options; (void)value; }
    if (source->macros) { id value = (__bridge_transfer id)source->macros; (void)value; }
    *source = (ds4_metal_shader_resources){0};
}
bool ds4_metal_bootstrap_library(const ds4_metal_shader_resources *resources,
                                  const ds4_metal_shader_flags *flags) {
    NSString *source = (__bridge NSString *)resources->source;
    MTLCompileOptions *options = (__bridge MTLCompileOptions *)resources->options;
    NSMutableDictionary *macros = (__bridge NSMutableDictionary *)resources->macros;
    NSError *error = nil;
    if (flags->math_safe) {
        // MTLCompileOptions.fastMathEnabled defaults to YES and Apple's
        // headers explicitly say this "may violate the IEEE 754 standard".
        // Different fast-math optimizations get applied across the
        // matmul2d cooperative-tensor path and the legacy
        // simdgroup_multiply_accumulate path on M5, amplifying the
        // mismatch. MTLMathModeSafe pins the entire library to strict
        // IEEE-754 semantics. Diagnostic-only: useful to localize drift
        // sources but not to ship as a default.
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            fprintf(stderr, "ds4: Metal shader library math mode = safe (strict IEEE-754) by DS4_METAL_MATH_SAFE\n");
        } else {
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
    #pragma clang diagnostic pop
            fprintf(stderr, "ds4: Metal shader library fast-math disabled by DS4_METAL_MATH_SAFE (pre-macOS 15)\n");
        }
    }

    if (flags->hc_stable)      macros[@"DS4_METAL_HC_STABLE"]          = @"1";
    if (flags->norm_unify)     macros[@"DS4_METAL_NORM_RSQRT_DISABLE"] = @"1";
    if (flags->raw_f32)     macros[@"DS4_METAL_KV_RAW_F32"]         = @"1";
    if (flags->rope_exp2) macros[@"DS4_METAL_ROPE_EXP2_LOG2"]     = @"1";
    fprintf(stderr,
            "ds4: drift-patch flags hc_stable=%s norm_unify=%s kv_raw_f32=%s rope_exp2_log2=%s math_safe=%s tensor_matmul=%s\n",
            flags->hc_stable      ? "on"  : "off",
            flags->norm_unify     ? "on"  : "off",
            flags->raw_f32     ? "on"  : "off",
            flags->rope_exp2 ? "on"  : "off",
            flags->math_safe      ? "on"  : "off",
            g_metal4_tensor_api_enabled ? "on" : "off");
    options.preprocessorMacros = macros;
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        fprintf(stderr, "ds4: Metal shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_library = library;

    return true;
}

bool ds4_metal_bootstrap_base_pipelines(void) {
    id<MTLLibrary> library = g_library;
    NSError *error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@"kernel_get_rows_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_get_rows_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_get_rows_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_get_rows_f16"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_f16 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_get_rows_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_get_rows_f16_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_f16 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_get_rows_i32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_i32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_get_rows_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_get_rows_i32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_i32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_get_rows_q8_0_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_q8_0_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_get_rows_q8_0_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_get_rows_q8_0_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_q8_0_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_get_rows_q4_0_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_q4_0_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_get_rows_q4_0_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_get_rows_q4_0_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_q4_0_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_get_rows_q4_K_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_q4_K_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_get_rows_q4_K_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_get_rows_q4_K_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_get_rows_q4_K_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_repeat_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_repeat_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_repeat_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_repeat_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_repeat_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_set_rows_f32_i32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_set_rows_f32_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_set_rows_f32_i32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_concat"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_concat function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_concat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_concat_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_concat pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_f32_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_f32_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_f32_f16"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_f32_f16_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_contig_f32_f16_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_contig_f32_f16_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_contig_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_contig_f32_f16_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_contig_f32_f16_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_f16_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_f16_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_f16_f16"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f16_f16 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_f16_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_f16_f16_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_f16_f16 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_contig_f16_f32_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f32_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_contig_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_contig_f16_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f32_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_cpy_contig_f16_f16_bits_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f16_bits_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_cpy_contig_f16_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_cpy_contig_f16_f16_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f16_bits_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_dsv4_flash_kv_stage_f16"];
    if (fn) {
        g_flash_kv_stage_f16_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_flash_kv_stage_f16_pipeline) {
            fprintf(stderr,
                    "ds4: optional Metal gathered KV staging pipeline unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
    } else {
        fprintf(stderr,
                "ds4: optional Metal gathered KV staging kernel unavailable\n");
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_fp8_kv_quantize_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_fp8_kv_quantize_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_fp8_kv_quantize_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_indexer_hadamard_fp4_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_indexer_qat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_indexer_qat_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_kv_fp8_store_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_kv_fp8_store_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_kv_fp8_store_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_kv_rope_fp8_store_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_kv_rope_fp8_store_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    error = nil;
    g_dsv4_kv_rope_fp8_store_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_kv_rope_fp8_store_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_kv_rope_fp8_store_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_ratio4_shift_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_ratio4_shift_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_ratio4_shift_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_swiglu_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_swiglu_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_swiglu_flat_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_swiglu_flat_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    g_swiglu_flat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_swiglu_flat_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_swiglu_flat_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_moe_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum8_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum8_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_moe_sum8_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_sum8_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum8_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *bin_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t bin_op = 0;
    int16_t bin_f = 1;
    bool bin_rb = false;
    bool bin_cb = false;
    [bin_constants setConstantValue:&bin_op type:MTLDataTypeShort atIndex:1300];
    [bin_constants setConstantValue:&bin_f  type:MTLDataTypeShort atIndex:1301];
    [bin_constants setConstantValue:&bin_rb type:MTLDataTypeBool  atIndex:1302];
    [bin_constants setConstantValue:&bin_cb type:MTLDataTypeBool  atIndex:1303];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                       constantValues:bin_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_add_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_add_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_add2_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_add2_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    g_add2_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_add2_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_add2_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_add3_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_add3_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    g_add3_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_add3_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_add3_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *bin_mul_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t bin_mul_plain_op = 2;
    int16_t bin_mul_plain_f = 1;
    bool bin_mul_plain_rb = false;
    bool bin_mul_plain_cb = false;
    [bin_mul_constants setConstantValue:&bin_mul_plain_op type:MTLDataTypeShort atIndex:1300];
    [bin_mul_constants setConstantValue:&bin_mul_plain_f  type:MTLDataTypeShort atIndex:1301];
    [bin_mul_constants setConstantValue:&bin_mul_plain_rb type:MTLDataTypeBool  atIndex:1302];
    [bin_mul_constants setConstantValue:&bin_mul_plain_cb type:MTLDataTypeBool  atIndex:1303];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                       constantValues:bin_mul_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_mul_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *bin_mul_scalar_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t bin_mul_op = 2;
    int16_t bin_mul_f = 1;
    bool bin_mul_rb = false;
    bool bin_mul_cb = true;
    [bin_mul_scalar_constants setConstantValue:&bin_mul_op type:MTLDataTypeShort atIndex:1300];
    [bin_mul_scalar_constants setConstantValue:&bin_mul_f  type:MTLDataTypeShort atIndex:1301];
    [bin_mul_scalar_constants setConstantValue:&bin_mul_rb type:MTLDataTypeBool  atIndex:1302];
    [bin_mul_scalar_constants setConstantValue:&bin_mul_cb type:MTLDataTypeBool  atIndex:1303];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                       constantValues:bin_mul_scalar_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_bin_mul_scalar_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_bin_mul_scalar_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *bin_div_row_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t bin_div_op = 3;
    int16_t bin_div_f = 1;
    bool bin_div_rb = false;
    bool bin_div_cb = true;
    [bin_div_row_constants setConstantValue:&bin_div_op type:MTLDataTypeShort atIndex:1300];
    [bin_div_row_constants setConstantValue:&bin_div_f  type:MTLDataTypeShort atIndex:1301];
    [bin_div_row_constants setConstantValue:&bin_div_rb type:MTLDataTypeBool  atIndex:1302];
    [bin_div_row_constants setConstantValue:&bin_div_cb type:MTLDataTypeBool  atIndex:1303];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                       constantValues:bin_div_row_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_bin_div_row_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_bin_div_row_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_rms_norm_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_rms_norm_f32_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_rms_norm_plain_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_rms_norm_plain_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_add_rms_norm_mul_f32_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_add_rms_norm_mul_f32_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_add_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_add_rms_norm_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_add_rms_norm_mul_f32_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_qkv_rms_norm_f32_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_qkv_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_qkv_rms_norm_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_head_rms_norm_rope_tail_f32"];
    if (!fn) {
        fprintf(stderr,
                "ds4: Metal kernel_dsv4_head_rms_norm_rope_tail_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    error = nil;
    g_dsv4_head_rms_norm_rope_tail_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_head_rms_norm_rope_tail_pipeline) {
        fprintf(stderr,
                "ds4: Metal kernel_dsv4_head_rms_norm_rope_tail_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    return true;
}

bool ds4_metal_bootstrap_remaining_pipelines(void) {
    id<MTLLibrary> library = g_library;
    id<MTLFunction> fn = nil;
    NSError *error = nil;

    MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t moe_mv_id_nsg = 2;
    [moe_mv_id_constants setConstantValue:&moe_mv_id_nsg type:MTLDataTypeShort atIndex:600];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_iq2_xxs_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_iq2_xxs_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_pack2_overlap_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (fn) {
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
    }
    if (!g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline) {
        fprintf(stderr,
                "ds4: optional Metal IQ2 pair pack2 pipeline unavailable: %s\n",
                error ? [[error localizedDescription] UTF8String] :
                        "function not found");
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_q2_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_q2_k_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_q2_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_iq2_xxs_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_iq2_xxs_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_q2_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q2_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_slots6_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_slots6_q2_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q2_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_iq2_xxs_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_addr_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_addr_iq2_xxs_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q2_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_addr_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_addr_q2_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q2_K_sum6_masked_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_masked_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_masked_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_stream_expert_cache_validate"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_stream_expert_cache_validate function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_stream_expert_cache_validate_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_stream_expert_cache_validate_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_stream_expert_cache_validate pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_q4_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_q4_k_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_q4_k_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_q4_k_pair_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_id_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_id_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group6_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group6_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group6_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group6_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group8_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group8_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group8_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group8_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group24_q4_K_id_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_id_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group24_q4_k_id_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group24_q4_k_id_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_id_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_group24_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_group24_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_group24_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_slots6_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_slots6_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_moe_mul_mv_id_mxfp4_pipeline =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_mxfp4_f32", 2);
    g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_mxfp4_pair_swiglu_f32", 2);
    g_moe_mul_mv_id_mxfp4_sum6_pipeline =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_mxfp4_sum6_f32", 2);
    /* Single-simdgroup variants of the MXFP4 decode MoE kernels.  The
     * per-row per-lane arithmetic is identical; only the row-to-simdgroup
     * mapping changes, so outputs remain bit-exact.  Smaller threadgroups
     * measurably improve GPU scheduling granularity on these
     * latency-bound small-grid matvecs.  Pipelines are built
     * unconditionally; the dispatch-time selection gate reads the
     * environment so A/B harnesses can toggle it after engine creation. */
    g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1 =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_mxfp4_pair_swiglu_f32", 1);
    g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1 =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_mxfp4_sum6_f32", 1);
    g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_pair_swiglu_f32", 1);
    g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_sum6_f32", 1);
    g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_f32", 1);
    g_moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_sum6_fixed_route_f32", 1);
    g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_f32", 1);
    g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_f32", 1);
    g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_f32", 1);
    g_moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_pair_swiglu_tp_static_f32", 1);
    g_moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1 =
        ds4_gpu_new_mul_mv_tg_multiple_pipeline(
            "kernel_mul_mv_id_mxfp4_sum6_tp_full_rows_static_f32", 1);
    if (!g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple ||
        !g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple ||
        !g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1 ||
        !g_moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1) {
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_slots6_mxfp4_pair_swiglu_f32", 2);
    g_moe_mul_mv_slots6_mxfp4_sum6_pipeline =
        ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_slots6_mxfp4_sum6_f32", 2);
    if (!g_moe_mul_mv_id_mxfp4_pipeline ||
        !g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline ||
        !g_moe_mul_mv_id_mxfp4_sum6_pipeline ||
        !g_moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline ||
        !g_moe_mul_mv_slots6_mxfp4_sum6_pipeline) {
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_q4_gather_slots6"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_q4_gather_slots6 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_q4_gather_slots6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_q4_gather_slots6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_q4_gather_slots6 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_table_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_pair_swiglu_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_table_q4_pair_gate_encoder = [fn newArgumentEncoderWithBufferIndex:2];
    g_moe_table_q4_pair_up_encoder = [fn newArgumentEncoderWithBufferIndex:3];
    if (!g_moe_table_q4_pair_gate_encoder || !g_moe_table_q4_pair_up_encoder) {
        fprintf(stderr, "ds4: Metal Q4 expert-table pair argument encoder creation failed\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_table_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_sum6_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_table_q4_sum_down_encoder = [fn newArgumentEncoderWithBufferIndex:1];
    if (!g_moe_table_q4_sum_down_encoder) {
        fprintf(stderr, "ds4: Metal Q4 expert-table down argument encoder creation failed\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_moe_mul_mv_table_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_mul_mv_table_q4_k_sum6_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_sum6_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q4_K_pair_swiglu_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (fn) {
        g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_pair_swiglu_f32 pipeline unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
    } else {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_pair_swiglu_f32 function unavailable: %s\n",
                [[error localizedDescription] UTF8String]);
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q4_K_sum6_f32"
                       constantValues:moe_mv_id_constants
                                error:&error];
    if (fn) {
        g_moe_mul_mv_addr_q4_k_sum6_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_sum6_f32 pipeline unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
    } else {
        fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_sum6_f32 function unavailable: %s\n",
                [[error localizedDescription] UTF8String]);
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_rope_tail_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_rope_tail_batch_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32_inplace_pair"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32_inplace_pair function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_rope_tail_inplace_pair_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_rope_tail_inplace_pair_pipeline) {
        fprintf(stderr,
                "ds4: Metal kernel_dsv4_rope_tail_f32_inplace_pair pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32_inplace_pair_shared4"];
    if (!fn) {
        fprintf(stderr,
                "ds4: optional Metal shared-head RoPE kernel unavailable; using per-head path\n");
    } else {
        g_rope_tail_inplace_pair_shared4_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_inplace_pair_shared4_pipeline) {
            fprintf(stderr,
                    "ds4: optional Metal shared-head RoPE pipeline unavailable; using per-head path: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32_inplace_pair_affine"];
    if (fn) {
        g_rope_tail_inplace_pair_affine_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_inplace_pair_affine_pipeline) {
            fprintf(stderr,
                    "ds4: optional Metal affine-position RoPE pair pipeline unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
    } else {
        fprintf(stderr,
                "ds4: optional Metal affine-position RoPE pair kernel unavailable\n");
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_dsv4_softmax_pool"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_softmax_pool_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_softmax_pool_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_soft_max_f32"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_soft_max_f32 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_soft_max_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_soft_max_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_soft_max_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_soft_max_f32_4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_soft_max_f32_4_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_soft_max_f32_4_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_argsort_f32_i32_desc"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_argsort_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_argsort_f32_i32_desc_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_argsort_merge_f32_i32_desc"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_argsort_merge_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_argsort_merge_f32_i32_desc_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *sum_rows_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t sum_rows_op = 10;
    [sum_rows_constants setConstantValue:&sum_rows_op type:MTLDataTypeShort atIndex:1400];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                       constantValues:sum_rows_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_sum_rows_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_sum_rows_f32_f32_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_topk_mask_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_topk_mask_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask_scatter"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_topk_mask_scatter_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_topk_mask_scatter_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_indexer_weighted_sum"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_dsv4_indexer_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_dsv4_indexer_weighted_sum_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_sinkhorn"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_hc_split_sinkhorn_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_hc_split_sinkhorn_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_hc_split_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_hc_split_weighted_sum_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum_norm4"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_hc_split_weighted_sum_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_hc_split_weighted_sum_norm_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_hc_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_hc_weighted_sum_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_dsv4_output_hc_weights4"];
    if (fn) {
        g_output_hc_weights4_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_output_hc_weights4_pipeline) {
            fprintf(stderr,
                    "ds4: optional Metal output HC weights4 pipeline unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
    } else {
        fprintf(stderr,
                "ds4: optional Metal output HC weights4 kernel unavailable\n");
    }

    MTLFunctionConstantValues *unary_sigmoid_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_sigmoid_op = 102;
    bool unary_cnt = false;
    [unary_sigmoid_constants setConstantValue:&unary_sigmoid_op type:MTLDataTypeShort atIndex:1200];
    [unary_sigmoid_constants setConstantValue:&unary_cnt        type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                       constantValues:unary_sigmoid_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_sigmoid_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_sigmoid_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *unary_silu_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_silu_op = 106;
    [unary_silu_constants setConstantValue:&unary_silu_op type:MTLDataTypeShort atIndex:1200];
    [unary_silu_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                       constantValues:unary_silu_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_silu_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *unary_softplus_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_softplus_op = 115;
    [unary_softplus_constants setConstantValue:&unary_softplus_op type:MTLDataTypeShort atIndex:1200];
    [unary_softplus_constants setConstantValue:&unary_cnt         type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                       constantValues:unary_softplus_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_softplus_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_softplus_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *unary_sqrt_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_sqrt_op = 14;
    [unary_sqrt_constants setConstantValue:&unary_sqrt_op type:MTLDataTypeShort atIndex:1200];
    [unary_sqrt_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                       constantValues:unary_sqrt_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_sqrt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_sqrt_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *unary_clamp_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_clamp_op = 12;
    [unary_clamp_constants setConstantValue:&unary_clamp_op type:MTLDataTypeShort atIndex:1200];
    [unary_clamp_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32"
                       constantValues:unary_clamp_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_clamp_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_clamp_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *unary_scale_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_scale_op = 10;
    [unary_scale_constants setConstantValue:&unary_scale_op type:MTLDataTypeShort atIndex:1200];
    [unary_scale_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                       constantValues:unary_scale_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_scale_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    MTLFunctionConstantValues *unary_fill_constants = [[MTLFunctionConstantValues alloc] init];
    int16_t unary_fill_op = 11;
    [unary_fill_constants setConstantValue:&unary_fill_op type:MTLDataTypeShort atIndex:1200];
    [unary_fill_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                       constantValues:unary_fill_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_fill_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_fill_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    error = nil;
    fn = [library newFunctionWithName:@"kernel_unary_f16_f16"
                       constantValues:unary_fill_constants
                                error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_unary_fill_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_unary_fill_f16_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    fn = [library newFunctionWithName:@"kernel_dsv4_hc_expand"];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand function not found\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_hc_expand_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_hc_expand_pipeline) {
        fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    g_dsv4_indexer_score_one_direct_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_indexer_score_one_direct");
    g_dsv4_compressor_store_one_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_compressor_store_one");
    g_dsv4_compressor_pack_ratio4_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_compressor_pack_ratio4");
    g_dsv4_compressor_pack_ratio4_decode_ggml_pipeline =
        ds4_gpu_get_pipeline(
            "kernel_dsv4_compressor_pack_ratio4_decode_ggml");
    g_dsv4_compressor_exact_softmax_product_pipeline =
        ds4_gpu_get_pipeline(
            "kernel_dsv4_compressor_exact_softmax_product_ratio4");
    g_dsv4_compressor_exact_pool_ratio4_pipeline =
        ds4_gpu_get_pipeline(
            "kernel_dsv4_compressor_exact_pool_ratio4_decode_ggml");
    g_dsv4_softmax_pool_ratio4_direct_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_softmax_pool_ratio4_direct");
    g_rms_norm_scale_pipeline =
        ds4_gpu_get_pipeline("kernel_rms_norm_scale_f32_4");
    g_dsv4_sort_i32_rows_asc_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_sort_i32_rows_asc");
    g_dsv4_indexed_attention_heads8_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8");
    g_dsv4_indexed_attention_heads8_rb16_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb16");
    g_dsv4_indexed_attention_heads16_dual_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads16_dual");
    g_dsv4_indexed_attention_heads8_split_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_split");
    g_dsv4_indexed_attention_heads8_split_reduce_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_split_reduce");
    g_dsv4_softplus_sqrt_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_softplus_sqrt_f32_4");
    g_dsv4_router_finalize_one_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one");
    g_dsv4_router_finalize_one_simd_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one_simd");
    g_dsv4_router_finalize_weights_one_simd_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_weights_one_simd");
    g_dsv4_router_transform_finalize_weights_one_simd_pipeline =
        ds4_gpu_get_pipeline(
            "kernel_dsv4_router_transform_finalize_weights_one_simd");
    g_dsv4_router_select_visual_batch_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_router_select_visual_batch");
    g_dsv4_router_weights_one_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_router_weights_one");
    g_glm_router_select_one_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_router_select_one");
    g_glm_kv_lora_rms_norm_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_kv_lora_rms_norm");
    g_glm_k_b_project_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_k_b_project_q8_0");
    g_glm_store_compact_kv_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_store_compact_kv");
    g_glm_qkv_norm_store_compact_kv_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_qkv_norm_store_compact_kv");
    g_glm_store_indexer_k_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_store_indexer_k");
    g_glm53_indexer_pool_update_pipeline =
        ds4_gpu_get_pipeline("kernel_glm53_indexer_pool_update");
    g_glm_build_kv_cache_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_build_kv_cache");
    g_glm_build_kv_cache_decode_group4_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_build_kv_cache_decode_group4");
    g_glm_build_kv_cache_flash_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_build_kv_cache_flash");
    g_glm_attention_full_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_full");
    g_glm_fill_selected_range_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_fill_selected_range");
    g_glm_fill_selected_range_batch_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_fill_selected_range_batch");
    g_glm53_expand_pool_selection_pipeline =
        ds4_gpu_get_pipeline("kernel_glm53_expand_pool_selection");
    g_glm_indexer_rope_tail_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_indexer_rope_tail_f32");
    g_glm_indexer_score_one_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_indexer_score_one");
    g_glm_indexer_score_one_direct_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_indexer_score_one_direct");
    g_glm_indexer_scores_batch_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_indexer_scores_batch");
    g_glm_indexer_scores_tiled_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_indexer_scores_tiled");
    g_glm_indexer_scores_tiled_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_indexer_scores_tiled_f32");
    g_glm_qk_lowrank_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0");
    g_glm_qk_lowrank_glm52_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_glm52");
    g_glm_qk_lowrank_glm52_sg_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_glm52_sg");
    g_glm_qk_lowrank_batch_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_batch");
    g_glm_qk_lowrank_batch_glm52_t4_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_batch_glm52_t4");
    g_glm_value_project_q8_0_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_value_project_q8_0");
    g_glm_value_project_q8_0_batch_heads_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_value_project_q8_0_batch_heads");
    g_glm_value_project_q8_0_batch_heads_mma_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_value_project_q8_0_batch_heads_mma");
    g_glm_attention_indexed_decode_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode");
    g_glm_attention_indexed_decode_split_group8_partial_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_partial");
    g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads");
    g_glm_attention_indexed_decode_split_group8_reduce_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_reduce");
    g_glm_attention_indexed_decode_split_group8_reduce16_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_reduce16");
    g_glm_attention_indexed_batch_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch");
    g_glm_attention_indexed_batch_group2_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_group2");
    g_glm_attention_indexed_batch_q2_group4_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_q2_group4");
    g_glm_attention_indexed_batch_group8_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_group8");
    g_glm_attention_indexed_batch_lora_group8_vec_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec");
    g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_valid");
    g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads");
    g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_causal");
    g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads");
    g_glm_q4_k_pair_swiglu_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu_f32");
    g_glm_q4_k_pair_swiglu2_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_f32");
    g_glm_q4_k_pair_swiglu4_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu4_f32");
    g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_mapped_f32");
    g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_mapped_row_f32");
    g_glm_q2_k_pair_swiglu_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q2_K_pair_swiglu_f32");
    g_glm_q2_k_addr_pair_swiglu2_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q2_K_addr_pair_swiglu2_f32");
    g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q2_K_addr_pair_swiglu2_f32_masked");
    g_glm_q4_k_addr_pair_swiglu_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_addr_pair_swiglu_f32");
    g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_addr_pair_swiglu_f32_masked");
    g_glm_q2_k_down_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q2_K_down_f32");
    g_glm_q4_k_down_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_down_simd_f32");
    g_glm_q2_k_addr_down_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q2_K_addr_down_f32");
    g_glm_q4_k_addr_down_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q4_K_addr_down_simd_f32");
    g_glm_q5_k_pair_swiglu_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_f32");
    g_glm_q5_k_pair_swiglu_mapped_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_mapped_f32");
    g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_mapped_row_f32");
    g_glm_q5_k_down_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q5_K_down_f32");
    g_glm_q6_k_down_f32_pipeline =
        ds4_gpu_get_pipeline("kernel_glm_q6_K_down_f32");
    g_dsv4_router_weights_batch_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_router_weights_batch");
    g_dsv4_hc_expand4_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_hc_expand4");
    if (!g_dsv4_indexer_score_one_direct_pipeline ||
        !g_dsv4_compressor_store_one_pipeline ||
        !g_dsv4_sort_i32_rows_asc_pipeline ||
        !g_dsv4_indexed_attention_heads8_pipeline ||
        !g_dsv4_indexed_attention_heads8_rb16_pipeline ||
        !g_dsv4_indexed_attention_heads16_dual_pipeline ||
        !g_dsv4_indexed_attention_heads8_split_pipeline ||
        !g_dsv4_indexed_attention_heads8_split_reduce_pipeline ||
        !g_dsv4_softplus_sqrt_pipeline ||
        !g_dsv4_router_finalize_one_pipeline ||
        !g_dsv4_router_select_visual_batch_pipeline ||
        !g_dsv4_router_weights_one_pipeline ||
        !g_glm_router_select_one_pipeline ||
        !g_glm_kv_lora_rms_norm_pipeline ||
        !g_glm_k_b_project_pipeline ||
        !g_glm_store_compact_kv_pipeline ||
        !g_glm_qkv_norm_store_compact_kv_pipeline ||
        !g_glm_store_indexer_k_pipeline ||
        !g_glm53_indexer_pool_update_pipeline ||
        !g_glm_build_kv_cache_pipeline ||
        !g_glm_build_kv_cache_decode_group4_pipeline ||
        !g_glm_build_kv_cache_flash_pipeline ||
        !g_glm_attention_full_pipeline ||
        !g_glm_fill_selected_range_pipeline ||
        !g_glm_fill_selected_range_batch_pipeline ||
        !g_glm53_expand_pool_selection_pipeline ||
        !g_glm_indexer_rope_tail_pipeline ||
        !g_glm_indexer_score_one_pipeline ||
        !g_glm_indexer_score_one_direct_pipeline ||
        !g_glm_indexer_scores_batch_pipeline ||
        !g_glm_indexer_scores_tiled_pipeline ||
        !g_glm_indexer_scores_tiled_f32_pipeline ||
        !g_glm_qk_lowrank_pipeline ||
        !g_glm_qk_lowrank_glm52_pipeline ||
        !g_glm_qk_lowrank_glm52_sg_pipeline ||
        !g_glm_qk_lowrank_batch_pipeline ||
        !g_glm_qk_lowrank_batch_glm52_t4_pipeline ||
        !g_glm_value_project_q8_0_pipeline ||
        !g_glm_value_project_q8_0_batch_heads_pipeline ||
        !g_glm_value_project_q8_0_batch_heads_mma_pipeline ||
        !g_glm_attention_indexed_decode_pipeline ||
        !g_glm_attention_indexed_decode_split_group8_partial_pipeline ||
        !g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline ||
        !g_glm_attention_indexed_decode_split_group8_reduce_pipeline ||
        !g_glm_attention_indexed_decode_split_group8_reduce16_pipeline ||
        !g_glm_attention_indexed_batch_pipeline ||
        !g_glm_attention_indexed_batch_group2_pipeline ||
        !g_glm_attention_indexed_batch_q2_group4_pipeline ||
        !g_glm_attention_indexed_batch_group8_pipeline ||
        !g_glm_attention_indexed_batch_lora_group8_vec_pipeline ||
        !g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline ||
        !g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline ||
        !g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline ||
        !g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline ||
        !g_glm_q4_k_pair_swiglu_f32_pipeline ||
        !g_glm_q4_k_pair_swiglu2_f32_pipeline ||
        !g_glm_q4_k_pair_swiglu4_f32_pipeline ||
        !g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline ||
        !g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline ||
        !g_glm_q2_k_pair_swiglu_f32_pipeline ||
        !g_glm_q2_k_addr_pair_swiglu2_f32_pipeline ||
        !g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline ||
        !g_glm_q4_k_addr_pair_swiglu_f32_pipeline ||
        !g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline ||
        !g_glm_q2_k_down_f32_pipeline ||
        !g_glm_q4_k_down_f32_pipeline ||
        !g_glm_q2_k_addr_down_f32_pipeline ||
        !g_glm_q4_k_addr_down_f32_pipeline ||
        !g_glm_q5_k_pair_swiglu_f32_pipeline ||
        !g_glm_q5_k_pair_swiglu_mapped_f32_pipeline ||
        !g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline ||
        !g_glm_q5_k_down_f32_pipeline ||
        !g_glm_q6_k_down_f32_pipeline ||
        !g_dsv4_hc_expand4_pipeline) {
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    return true;
}

void ds4_metal_bootstrap_stop_keepalive(void) { ds4_gpu_queue_keepalive_stop_thread(); }
void ds4_metal_bootstrap_commit_wait(void *command) {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    ds4_runtime_metal_commit((__bridge void *)cb, &g_bootstrap, true);
}
void ds4_metal_bootstrap_memory_report(void) { ds4_gpu_print_memory_report("at cleanup"); }
void ds4_metal_bootstrap_retire_readback(void) {
    g_selected_readback_event = nil;
    g_selected_readback_event_value = 0;
    [g_transient_buffers removeAllObjects];
}

void ds4_metal_bootstrap_retire_addresses(void) {
    for (uint32_t layer = 0; layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; layer++) {
        g_stream_expert_cache_gate_addr_buffers[layer] = nil;
        g_stream_expert_cache_up_addr_buffers[layer] = nil;
        g_stream_expert_cache_down_addr_buffers[layer] = nil;
        g_stream_compact_gate_addr_buffers[layer] = nil;
        g_stream_compact_up_addr_buffers[layer] = nil;
        g_stream_compact_down_addr_buffers[layer] = nil;
        g_stream_compact_selected_buffers[layer] = nil;
        g_stream_selected_id_buffers[layer] = nil;
    }
}

void ds4_metal_bootstrap_retire_pipelines(void) {
    g_set_rows_f32_i32_pipeline = nil;
    g_get_rows_f32_pipeline = nil;
    g_get_rows_f16_pipeline = nil;
    g_get_rows_i32_pipeline = nil;
    g_get_rows_q8_0_pipeline = nil;
    g_get_rows_q4_0_pipeline = nil;
    g_get_rows_q4_K_pipeline = nil;
    g_repeat_f32_pipeline = nil;
    g_concat_pipeline = nil;
    g_cpy_f32_f32_pipeline = nil;
    g_cpy_f32_f16_pipeline = nil;
    g_cpy_contig_f32_f16_pipeline = nil;
    g_cpy_f16_f32_pipeline = nil;
    g_cpy_f16_f16_pipeline = nil;
    g_cpy_contig_f16_f32_pipeline = nil;
    g_cpy_contig_f16_f16_pipeline = nil;
    g_flash_kv_stage_f16_pipeline = nil;
    g_swiglu_pipeline = nil;
    g_swiglu_flat_pipeline = nil;
    g_add_pipeline = nil;
    g_add2_pipeline = nil;
    g_add3_pipeline = nil;
    g_moe_sum6_pipeline = nil;
    g_moe_sum8_pipeline = nil;
    g_mul_pipeline = nil;
    g_bin_mul_scalar_pipeline = nil;
    g_bin_div_row_pipeline = nil;
    g_unary_sigmoid_pipeline = nil;
    g_unary_silu_pipeline = nil;
    g_unary_softplus_pipeline = nil;
    g_unary_sqrt_pipeline = nil;
    g_unary_clamp_pipeline = nil;
    g_unary_scale_pipeline = nil;
    g_unary_fill_pipeline = nil;
    g_unary_fill_f16_pipeline = nil;
    g_rms_norm_pipeline = nil;
    g_rms_norm_plain_pipeline = nil;
    g_add_rms_norm_pipeline = nil;
    g_rms_norm_scale_pipeline = nil;
    g_dsv4_qkv_rms_norm_pipeline = nil;
    g_dsv4_head_rms_norm_rope_tail_pipeline = nil;
    g_use_dsv4_head_rms_norm_rope_tail_pipeline = false;
    g_hc_split_sinkhorn_pipeline = nil;
    g_hc_split_weighted_sum_pipeline = nil;
    g_hc_split_weighted_sum_norm_pipeline = nil;
    g_dsv4_hc_producer_pre_norm_pipeline = nil;
    g_hc_weighted_sum_pipeline = nil;
    g_output_hc_weights4_pipeline = nil;
    g_hc_expand_pipeline = nil;
    g_moe_mul_mv_id_iq2_xxs_pipeline = nil;
    g_moe_mul_mv_id_iq2_xxs_pair_pipeline = nil;
    g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline = nil;
    g_moe_mul_mv_id_q2_k_pipeline = nil;
    g_moe_mul_mv_id_q2_k_sum6_pipeline = nil;
    g_moe_mul_mv_id_iq2_xxs_sum6_pipeline = nil;
    g_moe_mul_mv_id_q4_k_pipeline = nil;
    g_moe_mul_mv_id_q4_k_pair_pipeline = nil;
    g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_id_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_group_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_group6_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_group8_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_group24_q4_k_id_pipeline = nil;
    g_moe_mul_mv_group24_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_slots6_q2_k_sum6_pipeline = nil;
    g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_slots6_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_id_mxfp4_pipeline = nil;
    g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_id_mxfp4_sum6_pipeline = nil;
    g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple = nil;
    g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple = nil;
    g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1 = nil;
    g_moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1 = nil;
    g_moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_slots6_mxfp4_sum6_pipeline = nil;
    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_addr_iq2_xxs_pipeline = nil;
    g_moe_mul_mv_addr_q2_k_sum6_pipeline = nil;
    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline = nil;
    g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline = nil;
    g_moe_stream_expert_cache_validate_pipeline = nil;
    g_moe_q4_gather_slots6_pipeline = nil;
    g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_table_q4_k_sum6_pipeline = nil;
    g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline = nil;
    g_moe_mul_mv_addr_q4_k_sum6_pipeline = nil;
    g_moe_table_q4_pair_gate_encoder = nil;
    g_moe_table_q4_pair_up_encoder = nil;
    g_moe_table_q4_sum_down_encoder = nil;
    g_rope_tail_batch_pipeline = nil;
    g_rope_tail_inplace_pair_pipeline = nil;
    g_rope_tail_inplace_pair_shared4_pipeline = nil;
    g_rope_tail_inplace_pair_affine_pipeline = nil;
    g_dsv4_fp8_kv_quantize_pipeline = nil;
    g_dsv4_indexer_qat_pipeline = nil;
    g_dsv4_kv_fp8_store_pipeline = nil;
    g_dsv4_kv_rope_fp8_store_pipeline = nil;
    g_dsv4_ratio4_shift_pipeline = nil;
    g_dsv4_compressor_pack_ratio4_pipeline = nil;
    g_dsv4_compressor_pack_ratio4_decode_ggml_pipeline = nil;
    g_dsv4_compressor_exact_softmax_product_pipeline = nil;
    g_dsv4_compressor_exact_pool_ratio4_pipeline = nil;
    g_dsv4_softmax_pool_ratio4_direct_pipeline = nil;
    g_dsv4_softmax_pool_pipeline = nil;
    g_soft_max_f32_pipeline = nil;
    g_soft_max_f32_4_pipeline = nil;
    g_argsort_f32_i32_desc_pipeline = nil;
    g_argsort_merge_f32_i32_desc_pipeline = nil;
    g_sum_rows_f32_f32_pipeline = nil;
    g_dsv4_topk_mask_pipeline = nil;
    g_dsv4_topk_mask_scatter_pipeline = nil;
    g_dsv4_indexer_weighted_sum_pipeline = nil;
    g_dsv4_indexer_score_one_direct_pipeline = nil;
    g_dsv4_compressor_store_one_pipeline = nil;
    g_dsv4_sort_i32_rows_asc_pipeline = nil;
    g_dsv4_indexed_attention_heads8_pipeline = nil;
    g_dsv4_indexed_attention_heads8_rb16_pipeline = nil;
    g_dsv4_indexed_attention_heads16_dual_pipeline = nil;
    g_dsv4_indexed_attention_heads8_split_pipeline = nil;
    g_dsv4_indexed_attention_heads8_split_reduce_pipeline = nil;
    g_dsv4_softplus_sqrt_pipeline = nil;
    g_dsv4_router_finalize_one_pipeline = nil;
    g_dsv4_router_finalize_one_simd_pipeline = nil;
    g_dsv4_router_finalize_weights_one_simd_pipeline = nil;
    g_dsv4_router_transform_finalize_weights_one_simd_pipeline = nil;
    g_dsv4_router_select_visual_batch_pipeline = nil;
    g_dsv4_hc_producer_last_completion = nil;
    g_dsv4_hc_producer_last_mix_buffer = nil;
    g_dsv4_hc_producer_last_mix_offset = 0;
    [g_dsv4_completion_cache removeAllObjects];
    g_dsv4_completion_cache = nil;
    g_dsv4_router_weights_one_pipeline = nil;
    g_glm_router_select_one_pipeline = nil;
    g_glm_kv_lora_rms_norm_pipeline = nil;
    g_glm_k_b_project_pipeline = nil;
    g_glm_store_compact_kv_pipeline = nil;
    g_glm_qkv_norm_store_compact_kv_pipeline = nil;
    g_glm_store_indexer_k_pipeline = nil;
    g_glm53_indexer_pool_update_pipeline = nil;
    g_glm_build_kv_cache_pipeline = nil;
    g_glm_build_kv_cache_decode_group4_pipeline = nil;
    g_glm_build_kv_cache_flash_pipeline = nil;
    g_glm_attention_full_pipeline = nil;
    g_glm_fill_selected_range_pipeline = nil;
    g_glm_fill_selected_range_batch_pipeline = nil;
    g_glm53_expand_pool_selection_pipeline = nil;
    g_glm_indexer_rope_tail_pipeline = nil;
    g_glm_indexer_score_one_pipeline = nil;
    g_glm_indexer_score_one_direct_pipeline = nil;
    g_glm_indexer_scores_batch_pipeline = nil;
    g_glm_indexer_scores_tiled_pipeline = nil;
    g_glm_indexer_scores_tiled_f32_pipeline = nil;
    g_glm_qk_lowrank_pipeline = nil;
    g_glm_qk_lowrank_glm52_pipeline = nil;
    g_glm_qk_lowrank_glm52_sg_pipeline = nil;
    g_glm_qk_lowrank_batch_pipeline = nil;
    g_glm_qk_lowrank_batch_glm52_t4_pipeline = nil;
    g_glm_value_project_q8_0_pipeline = nil;
    g_glm_value_project_q8_0_batch_heads_pipeline = nil;
    g_glm_value_project_q8_0_batch_heads_mma_pipeline = nil;
    g_glm_attention_indexed_decode_pipeline = nil;
    g_glm_attention_indexed_decode_split_group8_partial_pipeline = nil;
    g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline = nil;
    g_glm_attention_indexed_decode_split_group8_reduce_pipeline = nil;
    g_glm_attention_indexed_decode_split_group8_reduce16_pipeline = nil;
    g_glm_attention_indexed_batch_pipeline = nil;
    g_glm_attention_indexed_batch_group2_pipeline = nil;
    g_glm_attention_indexed_batch_q2_group4_pipeline = nil;
    g_glm_attention_indexed_batch_group8_pipeline = nil;
    g_glm_attention_indexed_batch_lora_group8_vec_pipeline = nil;
    g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline = nil;
    g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline = nil;
    g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline = nil;
    g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline = nil;
    g_glm_q4_k_pair_swiglu_f32_pipeline = nil;
    g_glm_q4_k_pair_swiglu2_f32_pipeline = nil;
    g_glm_q4_k_pair_swiglu4_f32_pipeline = nil;
    g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline = nil;
    g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline = nil;
    g_glm_q2_k_pair_swiglu_f32_pipeline = nil;
    g_glm_q2_k_addr_pair_swiglu2_f32_pipeline = nil;
    g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline = nil;
    g_glm_q4_k_addr_pair_swiglu_f32_pipeline = nil;
    g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline = nil;
    g_glm_q2_k_down_f32_pipeline = nil;
    g_glm_q4_k_down_f32_pipeline = nil;
    g_glm_q2_k_addr_down_f32_pipeline = nil;
    g_glm_q4_k_addr_down_f32_pipeline = nil;
    g_glm_q5_k_pair_swiglu_f32_pipeline = nil;
    g_glm_q5_k_pair_swiglu_mapped_f32_pipeline = nil;
    g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline = nil;
    g_glm_q5_k_down_f32_pipeline = nil;
    g_glm_q6_k_down_f32_pipeline = nil;
    g_dsv4_router_weights_batch_pipeline = nil;
    g_dsv4_hc_expand4_pipeline = nil;
}

void ds4_metal_bootstrap_retire_scratch(void) {
    g_flash_attn_mask_buffer = nil;
    g_flash_attn_zero_mask_buffer = nil;
    g_flash_attn_pad_buffer = nil;
    g_flash_attn_tmp_buffer = nil;
    g_flash_attn_blk_buffer = nil;
    ds4_gpu_clear_zero_prefix_prefill_mask_cache();
    g_flash_attn_ring_buffer = nil;
    g_flash_attn_kv_buffer = nil;
    g_glm_flash_attn_mask_buffer = nil;
    g_compressor_pool_kv_buffer = nil;
    g_compressor_pool_score_buffer = nil;
    g_compressor_pool_score_cont_buffer = nil;
    g_compressor_pool_softmax_buffer = nil;
    g_compressor_pool_product_buffer = nil;
    g_compressor_store_ape_buffer = nil;
    g_compressor_store_score_buffer = nil;
    g_embed_rows_buffer = nil;
    g_router_selection_buffer = nil;
    g_router_weight_sum_buffer = nil;
    g_indexer_head_scores_buffer = nil;
    g_indexer_topk_buffer = nil;
    g_indexed_topk_buffer = nil;
    g_stream_expert_validate_status_buffer = nil;
    g_f16_round_scratch_buffer = nil;
    g_q8_prefill_scratch_buffer = nil;
    g_qwen4_conv_halo_buffer = nil;
    g_qwen4_conv_halo_bytes = 0;
    g_raw_store_round_buffer = nil;
    g_moe_gate_scratch_buffer = nil;
    g_moe_down_scratch_buffer = nil;
    g_moe_id_map_buffer = nil;
    g_moe_packed_rhs_buffer = nil;
    g_moe_packed_rhs_bytes = 0;
    g_moe_q4_gate_slots_buffer = nil;
    g_moe_q4_up_slots_buffer = nil;
    g_moe_q4_down_slots_buffer = nil;
    g_attn_out_group_ids_buffer = nil;
}

void ds4_metal_bootstrap_retire_maps(void) {
    g_model_fd = -1;
    g_model_views.published_map = NULL;
    g_model_views.published_size = 0;
    g_model_views.published_offset = 0;
    g_model_views.published_bytes = 0;
    g_model_views.published_max_tensor_bytes = 0;
    qwen4_nax_release_scratch();
    qwen4_batch_release_scratch();
    ds4_gpu_tensor_tracking_reset();
    g_flash_attn_mask_bytes = 0;
    g_flash_attn_zero_mask_bytes = 0;
    g_flash_attn_pad_bytes = 0;
    g_flash_attn_tmp_bytes = 0;
    g_flash_attn_blk_bytes = 0;
    g_flash_attn_ring_bytes = 0;
    g_flash_attn_kv_bytes = 0;
    g_glm_flash_attn_mask_bytes = 0;
    g_glm_flash_attn_mask_valid = 0;
    g_glm_flash_attn_mask_pos0 = 0;
    g_glm_flash_attn_mask_tokens = 0;
    g_glm_flash_attn_mask_cache_len = 0;
    g_compressor_pool_kv_bytes = 0;
    g_compressor_pool_score_bytes = 0;
    g_compressor_pool_score_cont_bytes = 0;
    g_compressor_pool_softmax_bytes = 0;
    g_compressor_pool_product_bytes = 0;
    g_compressor_store_ape_bytes = 0;
    g_compressor_store_score_bytes = 0;
    g_embed_rows_bytes = 0;
    g_router_selection_bytes = 0;
    g_router_weight_sum_bytes = 0;
    g_indexer_head_scores_bytes = 0;
    g_indexer_topk_bytes = 0;
    g_indexed_topk_bytes = 0;
    g_f16_round_scratch_bytes = 0;
    g_q8_prefill_scratch_bytes = 0;
    g_raw_store_round_bytes = 0;
    g_moe_gate_scratch_bytes = 0;
    g_moe_down_scratch_bytes = 0;
    g_moe_id_map_bytes = 0;
    g_moe_q4_gate_slots_bytes = 0;
    g_moe_q4_up_slots_bytes = 0;
    g_moe_q4_down_slots_bytes = 0;
    g_attn_out_group_ids_bytes = 0;
    g_model_views.wrap_count = 0;
    g_model_views.wrap_bytes = 0;
    g_model_views.wrap_max_bytes = 0;
}

void ds4_metal_bootstrap_retire_bookkeeping(void) {
    ds4_native_metal_selected_diagnostics_cleanup();
    ds4_gpu_model_residency_clear();
    ds4_gpu_model_views_clear();
    [g_pipeline_cache removeAllObjects];
    g_pipeline_cache = nil;
    [g_q4_expert_layer_residency_cache removeAllObjects];
    g_q4_expert_layer_residency_cache = nil;
    [g_q4_expert_table_cache removeAllObjects];
    g_q4_expert_table_cache = nil;
    ds4_metal_exact_cache_cleanup();
    g_transient_buffers = nil;
    g_pending_cbs = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    if (g_runtime_owned) {
        g_runtime_owned = false;
        ds4_runtime_metal_release();
    }
}

