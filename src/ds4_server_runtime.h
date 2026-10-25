#ifndef DS4_SERVER_RUNTIME_H
#define DS4_SERVER_RUNTIME_H

#include "ds4_server_input.h"
#include "ds4_kvstore.h"
#include "ds4_state_cache.h"
#include "ds4_tp.h"
#include "clef.h"
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>

/* Actual persistent server/slot/job records, shared once during the staged
 * runtime migration. Engine and Session stay opaque public API handles. These
 * are not parallel snapshots: the C cache/routing/generation paths and native
 * coordinator operate on these same mutexes and fields. */
typedef struct server server;
typedef struct server_slot server_slot;
typedef struct job job;
struct rax;
typedef struct tool_memory_entry tool_memory_entry;
typedef ds4_kvstore_entry kv_entry;
typedef ds4_kvstore_options kv_cache_options;
typedef ds4_kvstore kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    struct rax *by_id;
    struct rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

/* Image markers have request-local nonces. Normalize only actual marker spans
 * for visible-history matching and preserve byte offsets; literal client text
 * cannot impersonate an image. Full fingerprints still gate live KV reuse. */
typedef struct {
    size_t count;
    size_t offsets[16];
} visible_image_key;

typedef struct {
    bool valid;
    /* Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /* Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;
    visible_image_key images;
    /* Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    /* Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;
    size_t visible_len;
    visible_image_key images;
    /* Keep exact disk replay for clients that preserve reasoning. Switch to
     * the visible key once a request actually continues without reasoning. */
    bool token_text_disk_key;
} visible_live_state;

struct server_slot {
    server *srv;
    int id;
    ds4_session *session;
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state thinking_live;
    int continued_last_store_tokens;
    /* Wall time of the last completed job on this slot, stamped by the slot
     * worker; drives the staleness tiers in job_slot_score(). */
    time_t last_used;
    /* Cached rendered text of the checkpoint, refreshed by the slot worker
     * after each job; lets the reuse probe (slot_probe_reuse_locked) answer
     * the memory-text question with a memcmp instead of a detokenization. */
    char *live_text;
    size_t live_text_len;
    int live_text_pos;              /* checkpoint.len at render time; 0 = stale */
    char context_cache_profile[41]; /* Immutable suffix sources for this hint. */
    char last_response_id[40]; /* Approximate live state belongs to this parent. */

    job *assigned;
    job *running;
    bool busy;
    bool prefill_waiting;

    bool decode_pending;
    bool decode_in_flight;
    bool decode_done;
    int decode_token;
    bool decode_speculative;
    int decode_accepted[2];
    int decode_accepted_count;
    int decode_rc;
    char decode_err[160];
};

#define SERVER_IMAGE_CACHE_ENTRIES 32
#define SERVER_IMAGE_CACHE_BYTES (128u * 1024u * 1024u)

typedef struct {
    uint8_t *encoded;
    size_t encoded_len;
    ds4_vision_embedding embedding;
    size_t data_bytes;
    uint64_t used;
} server_image_cache_entry;

typedef struct {
    server_image_cache_entry entries[SERVER_IMAGE_CACHE_ENTRIES];
    size_t bytes;
    uint64_t clock;
} server_image_cache;

struct server {
    ds4_engine *engine;
    ds4_decision_registry *decision_models;
    ds4_tp *tp_leader;
    server_slot *slots;
    int slot_count;
    int ctx_size;
    bool batched_mode;
    pthread_t *slot_threads;
    pthread_t decode_thread;
    int default_tokens;
    kv_disk_cache kv;
    ds4_state_cache *state_cache;
    char state_cache_profile[41];
    bool context_suffix_reuse;
    bool context_suffix_canonical;
    bool context_suffix_pad;
    bool context_suffix_right;
    int context_suffix_pad_token;
    int context_suffix_fresh;
    const char *responses_store_dir;
    tool_memory tool_mem;
    server_image_cache image_cache; /* Protected by inference_mu. */
    bool disable_exact_dsml_tool_replay;
    bool enable_cors;
    pthread_mutex_t tool_mu;
    pthread_mutex_t kv_mu;
    pthread_mutex_t inference_mu;
    pthread_mutex_t model_mu;
    pthread_cond_t model_cv;
    bool model_busy;
    bool qwen4_batch_mtp;
    bool model_stopping;
    int decode_pending;
    int active_generations;
    int mixed_prefill_quantum;
    int last_prefill_slot;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    int clients;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
};

/* Jobs are stack-owned by client threads. Completion must follow the last
 * backend use of their cancellation callback, request data and socket. */

struct job {
    int fd;
    request req;
    bool done;
    bool cancelled;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
};

/* Complete model-admission/batching actions; callers retain record allocation,
 * mutex initialization, accept/slot-worker thread ownership and job routing. */
int ds4_server_coordinator_next_prefill(const server *state);
bool ds4_server_coordinator_prefill_enter(server *state, server_slot *slot,
    const volatile sig_atomic_t *stop);
void ds4_server_coordinator_prefill_leave(server *state);
int ds4_server_coordinator_quantum_for(const server *state, bool generation_active);
int ds4_server_coordinator_quantum(server *state);
void ds4_server_coordinator_generation_enter(server *state);
void ds4_server_coordinator_generation_leave(server *state);
bool ds4_server_coordinator_cancel_pending(server *state, server_slot *slot);
int ds4_server_coordinator_eval(server *state, server_slot *slot, int token,
    bool speculative, int accepted[2], int *accepted_count, char *err,
    size_t errlen, const volatile sig_atomic_t *stop);
long ds4_server_coordinator_coalesce_us(void);
void ds4_server_coordinator_run(server *state);
void ds4_server_coordinator_stop(server *state);
/* Existing synchronized server logger, retained as one normal diagnostic action
 * rather than a callback/accessor into model or scheduler state. */
void ds4_server_coordinator_log_batch(int count, double milliseconds, int status);

#endif
