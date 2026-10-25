#ifndef DS4_RUNTIME_API_H
#define DS4_RUNTIME_API_H
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>
#include <stdlib.h>

/* Model-independent Metal objects. Every successful acquire owns one reference.
 * A model retires its buffers/pipelines separately before releasing its borrow. */
bool ds4_runtime_metal_acquire(void **device, void **queue);
void ds4_runtime_metal_release(void);
size_t ds4_runtime_metal_users(void);

/* One engine's submitted GPU group at a time. The same owner may have several
 * command buffers in flight, preserving resident/streamed overlap and dependent
 * submissions. Completion may release on another thread. CPU work holds no lease. */
void ds4_runtime_gpu_begin(const void *owner);
void ds4_runtime_gpu_end(const void *owner);
void ds4_runtime_metal_commit(void *command, const void *owner, bool wait);

static const pthread_mutex_t ds4_runtime_mutex_init = PTHREAD_MUTEX_INITIALIZER;
static const pthread_cond_t ds4_runtime_cond_init = PTHREAD_COND_INITIALIZER;
#endif
