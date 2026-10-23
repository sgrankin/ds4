#ifndef DS4_AGENT_COMPACTION_H
#define DS4_AGENT_COMPACTION_H

#include "ds4.h"

/* Existing policy, shared once by the retained C implementation and native
 * planner. Small contexts cap the free-token threshold proportionally. */
enum {
    DS4_AGENT_COMPACT_SOFT_PERCENT = 85,
    DS4_AGENT_COMPACT_MIN_FREE_TOKENS = 8192,
    DS4_AGENT_COMPACT_TAIL_DIVISOR = 10,
    DS4_AGENT_COMPACT_TAIL_CAP_TOKENS = 50000,
    DS4_AGENT_COMPACT_SUMMARY_MAX_TOKENS = 4096,
};

/* One two-phase text plan, not a Worker or Session snapshot. prepare owns the
 * private request; rebuild owns the replacement transcript. C may take either
 * actual vector by copying it and zeroing its field before plan_free. Images
 * and their embeddings are always borrowed; runtime publication/rollback stays
 * with C. Plan construction never evaluates a token or mutates live state. */
typedef struct {
    ds4_tokens prompt;
    ds4_tokens compacted;
    int summary_bottom, summary_room, summary_max;
    size_t summary_images;
    int tail_start, tail_dst_start, resumed_start;
} ds4_agent_compaction_plan;

bool ds4_agent_compaction_should_compact(ds4_engine *engine, int ctx, int used);
int ds4_agent_compaction_tail_boundary(const ds4_tokens *tokens, int bottom,
    int system_len, int tail_budget, int user_id);
int ds4_agent_compaction_summary_budget(int ctx);
int ds4_agent_compaction_reserve(ds4_engine *engine, int ctx);
int ds4_agent_compaction_image_boundary(const ds4_vision_span *images,
    size_t count, bool glm, int position);
bool ds4_agent_compaction_prepare(ds4_agent_compaction_plan *plan,
    ds4_engine *engine, const ds4_tokens *transcript, const ds4_tokens *system,
    const ds4_vision_span *images, size_t image_count, int ctx, const char *reason,
    char *err, size_t errlen);
bool ds4_agent_compaction_rebuild(ds4_agent_compaction_plan *plan,
    ds4_engine *engine, const ds4_tokens *transcript, const ds4_tokens *system,
    const ds4_vision_span *images, size_t image_count, int ctx, bool prerope,
    const char *summary, const int *open_assistant, char *err, size_t errlen);
void ds4_agent_compaction_free(ds4_agent_compaction_plan *plan);

#endif
