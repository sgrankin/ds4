/* Include after the existing vision leaf encoders. The native controller
 * selects operations; these wrappers expose only the existing Metal leaves. */
#include "../engine/vision_api.h"

int ds4_native_vision_rms(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const void *map, uint64_t size, uint64_t offset, uint32_t width,
        uint32_t rows, float epsilon, const char *label) {
    @autoreleasepool {
        return glm53_vision_dispatch_rows("kernel_glm53_vision_rms_bf16",
            out, input, map, size, offset, width, rows, epsilon, label);
    }
}
int ds4_native_vision_bias(const char *kernel, ds4_gpu_tensor *tensor,
        const ds4_gpu_tensor *residual, const void *map, uint64_t size,
        uint64_t offset, uint32_t width, uint32_t rows, const char *label) {
    @autoreleasepool {
        return glm53_vision_dispatch_bias(kernel, tensor, residual, map, size,
            offset, width, rows, label);
    }
}
int ds4_native_vision_qkv(ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
        const ds4_gpu_tensor *qkv, const void *map, uint64_t size,
        uint64_t bias, uint32_t rows, uint32_t grid_width) {
    @autoreleasepool {
        return deepseek4_vision_dispatch_qkv(q, k, v, qkv, map, size, bias,
            rows, grid_width);
    }
}
int ds4_native_vision_attention(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, uint32_t rows) {
    @autoreleasepool {
        return glm53_vision_dispatch_attention(out, q, k, v, rows);
    }
}
int ds4_native_vision_round(ds4_gpu_tensor *tensor, uint32_t width, uint32_t rows) {
    @autoreleasepool {
        return deepseek4_vision_dispatch_round(tensor, width, rows);
    }
}
int ds4_native_vision_two_rows(const char *kernel, ds4_gpu_tensor *first,
        const ds4_gpu_tensor *second, uint32_t width, uint32_t rows,
        const char *label) {
    @autoreleasepool {
        return deepseek4_vision_dispatch_two_rows(kernel, first, second,
            width, rows, label);
    }
}
int ds4_native_vision_reorder(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        uint32_t height, uint32_t width, uint32_t rows) {
    @autoreleasepool {
        return deepseek4_vision_dispatch_aligner_reorder(out, input,
            height, width, rows);
    }
}
int ds4_native_vision_gelu_bias(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t rows, uint32_t width) {
    @autoreleasepool {
        return deepseek4_vision_dispatch_gelu_bias(out, input, map, size,
            offset, rows, width);
    }
}
