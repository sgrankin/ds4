/* The process runtime owns the actual Metal device/queue. Model drivers keep
 * independent pipeline/buffer owners and never re-create another model's queue. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "api.h"
#include <assert.h>

static pthread_mutex_t owners_mutex = PTHREAD_MUTEX_INITIALIZER;
static id<MTLDevice> runtime_device;
static id<MTLCommandQueue> runtime_queue;
static size_t owners;

bool ds4_runtime_metal_acquire(void **device, void **queue) {
    @autoreleasepool {
        pthread_mutex_lock(&owners_mutex);
        if (!runtime_device) {
            id<MTLDevice> created_device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> created_queue = [created_device newCommandQueueWithMaxCommandBufferCount:256];
            if (!created_device || !created_queue) {
                pthread_mutex_unlock(&owners_mutex);
                return false;
            }
            runtime_device = created_device;
            runtime_queue = created_queue;
        }
        owners++;
        *device = (__bridge void *)runtime_device;
        *queue = (__bridge void *)runtime_queue;
        pthread_mutex_unlock(&owners_mutex);
        return true;
    }
}
void ds4_runtime_metal_release(void) {
    pthread_mutex_lock(&owners_mutex);
    assert(owners > 0);
    if (--owners == 0) {
        runtime_queue = nil;
        runtime_device = nil;
    }
    pthread_mutex_unlock(&owners_mutex);
}
size_t ds4_runtime_metal_users(void) {
    pthread_mutex_lock(&owners_mutex);
    size_t count = owners;
    pthread_mutex_unlock(&owners_mutex);
    return count;
}
void ds4_runtime_metal_commit(void *opaque, const void *owner, bool wait) {
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)opaque;
    assert(command && owner);
    ds4_runtime_gpu_begin(owner);
    if (!wait) {
        [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            (void)completed;
            ds4_runtime_gpu_end(owner);
        }];
    }
    [command commit];
    if (wait) {
        [command waitUntilCompleted];
        ds4_runtime_gpu_end(owner);
    }
}
