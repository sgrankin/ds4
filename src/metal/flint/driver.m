#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "driver.h"

/* Borrow the runtime's device/queue, keeping ARC at this existing boundary. */
extern bool ds4_gpu_metal_shared_context(void **device, void **queue);

@interface FlintMetalContext : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> queue;
@property(strong) NSMutableArray<id<MTLLibrary>> *libraries;
@property(strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(strong) NSString *error;
@end
@implementation FlintMetalContext
@end

static FlintMetalContext *context(flint_gpu *gpu) {
    return (__bridge FlintMetalContext *)gpu;
}

flint_gpu *flint_gpu_create(void) {
    @autoreleasepool {
        void *device = NULL, *queue = NULL;
        if (!ds4_gpu_metal_shared_context(&device, &queue)) return NULL;
        FlintMetalContext *ctx = [FlintMetalContext new];
        ctx.device = (__bridge id<MTLDevice>)device;
        ctx.queue = (__bridge id<MTLCommandQueue>)queue;
        ctx.libraries = [NSMutableArray array];
        ctx.pipelines = [NSMutableDictionary dictionary];
        return (flint_gpu *)CFBridgingRetain(ctx);
    }
}

void flint_gpu_free(flint_gpu *gpu) {
    if (gpu) { id released = CFBridgingRelease(gpu); (void)released; }
}
const char *flint_gpu_device_name(flint_gpu *gpu) {
    return [context(gpu).device.name UTF8String];
}
const char *flint_gpu_error(flint_gpu *gpu) {
    return [context(gpu).error UTF8String];
}

bool flint_gpu_compile(flint_gpu *gpu, const char *source, size_t length,
                       char *error, size_t error_size) {
    @autoreleasepool {
        FlintMetalContext *ctx = context(gpu);
        NSString *text = [[NSString alloc] initWithBytes:source length:length
                                              encoding:NSUTF8StringEncoding];
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        NSError *failure = nil;
        id<MTLLibrary> library = [ctx.device newLibraryWithSource:text options:options error:&failure];
        if (!library) {
            ctx.error = failure.localizedDescription ?: @"Metal library compilation failed";
            if (error && error_size) snprintf(error, error_size, "%s", ctx.error.UTF8String);
            return false;
        }
        [ctx.libraries addObject:library];
        return true;
    }
}

void *flint_gpu_buffer(flint_gpu *gpu, size_t bytes) {
    @autoreleasepool {
        if (!bytes) bytes = 4;
        id<MTLBuffer> buffer = [context(gpu).device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
        return buffer ? (void *)CFBridgingRetain(buffer) : NULL;
    }
}
void *flint_gpu_contents(void *buffer) {
    return [(__bridge id<MTLBuffer>)buffer contents];
}
size_t flint_gpu_buffer_size(void *buffer) {
    return [(__bridge id<MTLBuffer>)buffer length];
}
void flint_gpu_buffer_free(void *buffer) {
    if (buffer) { id released = CFBridgingRelease(buffer); (void)released; }
}

void *flint_gpu_command_begin(flint_gpu *gpu) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [context(gpu).queue commandBuffer];
        return command ? (void *)CFBridgingRetain(command) : NULL;
    }
}
bool flint_gpu_command_end(flint_gpu *gpu, void *opaque, bool wait) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = CFBridgingRelease(opaque);
        [command commit];
        if (wait) [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
            context(gpu).error = command.error.localizedDescription;
            return false;
        }
        return true;
    }
}

static inline id<MTLComputePipelineState> flint_pipeline(FlintMetalContext *ctx, NSString *name) {
    id<MTLComputePipelineState> pipeline = ctx.pipelines[name];
    if (!pipeline) {
        id<MTLFunction> function = nil;
        for (id<MTLLibrary> library in [ctx.libraries reverseObjectEnumerator]) {
            function = [library newFunctionWithName:name];
            if (function) break;
        }
        if (!function) {
            ctx.error = [@"Missing Metal kernel: " stringByAppendingString:name];
            fprintf(stderr, "flint: %s\n", ctx.error.UTF8String);
            for (id<MTLLibrary> library in ctx.libraries)
                fprintf(stderr, "flint: library functions: %s\n", [[library.functionNames componentsJoinedByString:@", "] UTF8String]);
            return nil;
        }
        NSError *error = nil;
        pipeline = [ctx.device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            ctx.error = error.localizedDescription;
            fprintf(stderr, "flint: pipeline %s: %s\n", name.UTF8String, ctx.error.UTF8String);
            return nil;
        }
        ctx.pipelines[name] = pipeline;
    }
    return pipeline;
}

bool flint_gpu_prepare(flint_gpu *gpu, const char *kernel) {
    @autoreleasepool {
        return flint_pipeline(context(gpu), [NSString stringWithUTF8String:kernel]) != nil;
    }
}

bool flint_gpu_dispatch(flint_gpu *gpu, void *opaque, const char *kernel,
                        const void *params, size_t params_bytes,
                        const flint_gpu_binding *bindings, size_t count,
                        const uint32_t groups[3], const uint32_t threads[3]) {
    @autoreleasepool {
        FlintMetalContext *ctx = context(gpu);
        NSString *name = [NSString stringWithUTF8String:kernel];
        id<MTLComputePipelineState> pipeline = flint_pipeline(ctx, name);
        if (!pipeline) return false;
        if ((uint64_t)threads[0]*threads[1]*threads[2] > pipeline.maxTotalThreadsPerThreadgroup) {
            ctx.error = [@"Threadgroup too large: " stringByAppendingString:name];
            fprintf(stderr, "flint: %s (requested %u, maximum %lu)\n", ctx.error.UTF8String,
                    threads[0]*threads[1]*threads[2], (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
            return false;
        }
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)opaque;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            ctx.error = @"Failed to create Metal compute encoder";
            fprintf(stderr, "flint: %s\n", ctx.error.UTF8String);
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        if (params_bytes) [encoder setBytes:params length:params_bytes atIndex:0];
        for (size_t i = 0; i < count; ++i) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)bindings[i].handle
                        offset:bindings[i].offset atIndex:bindings[i].index];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(groups[0],groups[1],groups[2])
                threadsPerThreadgroup:MTLSizeMake(threads[0],threads[1],threads[2])];
        [encoder endEncoding];
        return true;
    }
}
