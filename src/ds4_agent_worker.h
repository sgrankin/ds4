#ifndef DS4_AGENT_WORKER_H
#define DS4_AGENT_WORKER_H
/* Actual shared Agent state, moved once from ds4_agent.c. UI and worker
 * threads retain their existing ownership/locking rules. Engine/Session are
 * opaque public handles; these are not replicas of engine-private layouts. */
#include "ds4.h"
#include "ds4_agent_context.h"
#include "ds4_agent_executor.h"
#include "ds4_agent_sessions.h"
#include "ds4_prompt_prefix.h"
#include "ds4_state_cache.h"
#include <pthread.h>
#include <stdio.h>

typedef struct {
    const char *prompt;
    char *prompt_owned;
    const char *system;
    ds4_prompt_prefix prefix;
    const char *trace_path;
    const char *live_context_path;
    bool raw_prompt;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    ds4_think_mode think_mode;
} agent_generation_options;

typedef struct {
    ds4_engine_options engine;
    agent_generation_options gen;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *chdir_path;
    bool non_interactive;
    bool edit_upto;
} agent_config;

typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_DRAINING,
    AGENT_WORKER_SAVING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    unsigned prefill_label;
    double prefill_tps;
    int generated;
    double gen_tps;
    bool greedy_sampling;
    int ctx_used;
    int ctx_size;
    int power_percent;
    char error[256];
} agent_status;


/* Runtime preference only; never part of saved session metadata. */
typedef struct {
    bool enabled;
    bool instructed;
    enum {
        AGENT_HINTS_UNSET,
        AGENT_HINTS_OFF,
        AGENT_HINTS_ON,
        AGENT_HINTS_UNKNOWN,
    } applied;
} agent_hints;

typedef struct {
    ds4_suffix_cache cache;
    ds4_tokens prefix;
    ds4_engine *engine;
    int ctx;
    char disk_key[41];
    uint64_t disk_state_bytes, disk_prerope_bytes;
    bool disk_approximate;
} agent_context_frontier;


typedef struct {
    ds4_engine *engine;
    agent_config *cfg;
    ds4_session *session;
    ds4_state_cache *state_cache;
    char state_cache_profile[41];
    uint64_t state_cache_staged_bytes;
    ds4_tokens transcript;
    ds4_agent_context live_context;
    ds4_vision_span *images;
    size_t image_count;
    size_t image_cap;
    char *cache_dir;
    char *sysprompt_path;
    agent_context_frontier context_frontiers[4];
    int context_frontier_count;
    uint64_t context_frontier_bytes;
    ds4_agent_session_identity identity;
    bool user_activity;
    bool session_dirty;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int wake_fd[2];
    FILE *trace;
    bool wake_pending;
    bool stop;
    bool interrupt;
    bool initialized;
    bool save_requested;
    bool compact_requested;
    bool power_requested;
    int requested_power;
    bool think_requested;
    ds4_think_mode requested_think;
    int progress_base;
    bool progress_direct;
    double progress_started_at;
    char *cmd_text;
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    bool web_approval_pending;
    bool web_approval_answered;
    bool web_approval_result;
    char web_approval_message[256];
    char web_approval_error[160];
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;
    char *queued_user_drain_text;
    bool datetime_context_injected;
    agent_hints hints;
    int last_system_prompt_reminder_at;
    ds4_agent_tool_runtime tools;
    bool raw_mode_needs_restore;
} agent_worker;

/* Actual suffix-plan packet: sizeof is part of staging-budget accounting. */
typedef struct {
    int source_frontier;
    ds4_state_cache_ref *disk_ref;
    uint64_t disk_state_bytes, disk_prerope_bytes;
    uint32_t source_start, destination_start, source_end, destination_end;
    uint32_t reused_tokens, fresh_tokens;
} agent_context_suffix_plan;

#endif
