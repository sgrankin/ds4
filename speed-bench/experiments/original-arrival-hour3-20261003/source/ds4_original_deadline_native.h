#ifndef DS4_ORIGINAL_DEADLINE_NATIVE_H
#define DS4_ORIGINAL_DEADLINE_NATIVE_H
/* Private Metal SOURCE prototype; no native compile/test/model evidence.
 * Include after ds4_gpu_stream_expert_table is declared. Int1 success/0 fatal,
 * except arm returns DS4_OD_*; callbacks own cache metadata, reader owns FD IO.
 */
#include "ds4_original_deadline.h"
#ifdef __APPLE__
int ds4_gpu_original_deadline_token_begin(const void *owner,const void *map,
        uint64_t size,int bound_fd);
int ds4_gpu_original_deadline_close(const void *owner);
/* Matching model close joins first; unrelated CPU/model owners are untouched. */
int ds4_gpu_original_deadline_close_model(const void *map,uint64_t size,int bound_fd);
int ds4_gpu_original_deadline_future_selected(const void *owner,
        const ds4_gpu_stream_expert_table *,const int32_t native[6]);
int ds4_gpu_original_deadline_arm_selected(const void *owner,
        const ds4_gpu_stream_expert_table *,const int32_t native[6],
        const int32_t candidate[6]);
int ds4_gpu_original_deadline_finish_selected(const void *owner,
        int32_t final_ids[6],int *promoted);
int ds4_gpu_original_deadline_bind_ready(const void *owner);
int ds4_gpu_original_deadline_stats(const void *owner,ds4_od_stats *);
#endif
#endif
