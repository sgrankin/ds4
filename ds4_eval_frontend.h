#ifndef DS4_EVAL_FRONTEND_H
#define DS4_EVAL_FRONTEND_H
#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_tp.h"
#include "ds4_eval_cases.h"
#include <stdbool.h>
#include <stdio.h>
#define EVAL_ANSWER_MAX 256
#define EVAL_MAX_CONTEXT 1000000
/* Actual frontend records/constants moved once. Model/Session stay opaque. */
typedef enum {
    EVAL_PENDING,
    EVAL_PREFILL,
    EVAL_THINKING,
    EVAL_SKIPPED,
    EVAL_STOPPED,
    EVAL_INCOMPLETE,
    EVAL_PASSED,
    EVAL_FAILED,
} eval_status;

typedef enum {
    EVAL_RUN_OK,
    EVAL_RUN_ERROR,
    EVAL_RUN_SWITCH,
    EVAL_RUN_QUIT,
} eval_run_result;

typedef enum {
    EVAL_THINK_CLOSE_NONE,
    EVAL_THINK_CLOSE_NATURAL,
    EVAL_THINK_CLOSE_SOFT,
    EVAL_THINK_CLOSE_HARD,
} eval_think_close_kind;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *trace_path;
    const char *regrade_trace_path;
    const char *case_sequence;
    const char *source_filter;
    const char *domain_filter;
    const char *case_id_filter;
    uint32_t suite_mask;
    ds4_backend backend;
    int threads;
    int ctx_size;
    int max_tokens;
    int question_limit;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    int pause_ms;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    int soft_limit_reply_budget;
    int hard_limit_reply_budget;
    int soft_limit_think_close_rank;
    ds4_think_mode think_mode;
    ds4_dist_options dist;
    ds4_tp_options tp;
    bool plain;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool self_test_extractors;
    bool validate_cases;
    bool list_cases;
    bool retry_incomplete;
    bool max_tokens_set;
} eval_config;

typedef struct {
    eval_think_close_kind kind;
    int token_index;
    int remaining_budget;
    int rank;
} eval_think_close_info;
typedef enum {
    REGRADE_NOT_GRADED,
    REGRADE_PASSED,
    REGRADE_FAILED,
} regrade_outcome;

extern const eval_case eval_core_cases[];
extern const size_t ds4_eval_core_case_count;
/* Read-only aliases let generated bindings borrow counted C data arrays. */
extern const eval_case *const ds4_eval_core_data;
extern const eval_case *const ds4_eval_hard_data;
extern const eval_source *const ds4_eval_source_data;

/* These whole operations let the original C extractor fixture call the
 * selected production implementation; no grading algorithm is copied. */
void ds4_eval_find_case_answer(const eval_case *, const char *, char *, size_t);
bool ds4_eval_find_case_answer_strict(const eval_case *, const char *, char *, size_t);
bool ds4_eval_answer_matches(const eval_case *, const char *);
const char *ds4_eval_trace_find_next_case(const char *, const char *);
char *ds4_eval_trace_copy_model_output(const char *, const char *);
regrade_outcome ds4_eval_regrade_case_outcome(const eval_case *, const char *, const char *, char *, size_t);
int ds4_eval_validate_cases(bool);
int ds4_eval_run_extractor_self_tests(void);
void ds4_eval_install_signals(void (*handler)(int));
void ds4_eval_reraise(int signal_number);
#endif
