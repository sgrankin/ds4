#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_eval_frontend.h"
#include "ds4_help.h"
#include "ds4_tp.h"

/* ds4-eval: small built-in benchmark integration test.
 *
 * This program is deliberately not a unit test.  It loads the real model,
 * renders chat prompts, prefills them through ds4_session_sync(), samples the
 * continuation token by token, and grades the final answer.  The terminal UI is
 * also intentionally simple: no ncurses, just ANSI cursor movement, colors, and
 * a fixed two-pane layout.
 *
 * The original core suite contains fixed subsets of GPQA Diamond, SuperGPQA,
 * AIME 2025, and COMPSEC. The separate hard suite adds MMLU-Pro,
 * OlympiadBench, LiveBench, and NIST Juliet-derived cases. See EVAL_DATA.md for
 * source revisions, licenses, and transformations.
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define ANSI_RESET "\x1b[0m"
#define ANSI_DIM "\x1b[90m"
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BLUE "\x1b[34m"
#define ANSI_CYAN "\x1b[36m"
#define ANSI_BOLD "\x1b[1m"







typedef struct {
    char *v;
    size_t len;
    size_t cap;
} byte_buf;

typedef struct {
    unsigned char *v;
    size_t len;
    size_t cap;
} style_buf;



typedef struct {
    int cols;
    int rows;
    int left_w;
    int right_x;
    int right_w;
    int body_y;
    int body_h;
    bool active;
    bool enabled;
    const eval_case *cases;
    int ncases;
    eval_status *status;
    char (*guess)[EVAL_ANSWER_MAX];
    int *prompt_tokens;
    int *generated_tokens;
    int active_case;
    int generated;
    int max_tokens;
    int think_max_tokens;
    int prefill_current;
    int prefill_total;
    double phase_start_sec;
    double speed_tps;
    double run_elapsed_sec;
    double run_last_sec;
    bool run_clock_active;
    bool selection_active;
    int selected_case;
    int requested_case;
    bool paused;
    bool quit_requested;
    byte_buf stream;
    style_buf styles;
    bool in_think;
    char pending_tag[16];
    size_t pending_tag_len;
} eval_ui;

static eval_ui *global_ui;

static double run_clock_sec(void);
static int validate_eval_cases(bool verbose);
static void list_eval_cases(const eval_case *cases, int ncases);

static void tui_run_clock_tick(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    double now = run_clock_sec();
    double delta = now - ui->run_last_sec;
    if (delta > 0.0) ui->run_elapsed_sec += delta;
    ui->run_last_sec = now;
}

static void tui_run_clock_start(eval_ui *ui) {
    if (!ui || ui->run_clock_active) return;
    ui->run_last_sec = run_clock_sec();
    ui->run_clock_active = true;
}

static void tui_run_clock_stop(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    tui_run_clock_tick(ui);
    ui->run_clock_active = false;
}

static double tui_run_clock_visible_sec(const eval_ui *ui) {
    if (!ui) return 0.0;
    double elapsed = ui->run_elapsed_sec;
    if (ui->run_clock_active) {
        double delta = run_clock_sec() - ui->run_last_sec;
        if (delta > 0.0) elapsed += delta;
    }
    return elapsed;
}

static void format_run_elapsed(char *dst, size_t dstlen, double sec) {
    if (sec < 0.0) sec = 0.0;
    unsigned long long minutes = (unsigned long long)(sec / 60.0);
    unsigned long long hours = minutes / 60ull;
    minutes %= 60ull;
    snprintf(dst, dstlen, "%02lluh:%02llum", hours, minutes);
}

typedef struct {
    bool enabled;
    bool raw_mode;
    bool thread_started;
    volatile sig_atomic_t running;
    pthread_t thread;
    pthread_mutex_t mu;
    struct termios orig_termios;
    int move_delta;
    bool enter_pressed;
    bool pause_pressed;
    bool quit_pressed;
} eval_input;

static eval_input global_input = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
};

static void buf_append(byte_buf *b, const char *p, size_t n) {
    if (n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n + 1) cap *= 2;
        char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memcpy(b->v + b->len, p, n);
    b->len += n;
    b->v[b->len] = '\0';
}

static void buf_appendf(byte_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (n < 0) {
        va_end(ap);
        return;
    }
    char *tmp = malloc((size_t)n + 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    vsnprintf(tmp, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_append(b, tmp, (size_t)n);
    free(tmp);
}

static int eval_case_nchoices(const eval_case *tc) {
    int n = 0;
    while (n < EVAL_MAX_CHOICES && tc->choice[n]) n++;
    return n;
}

static bool eval_case_is_multiple_choice(const eval_case *tc) {
    return eval_case_nchoices(tc) > 0;
}

static bool eval_case_is_compsec(const eval_case *tc) {
    return tc->source && !strcmp(tc->source, "COMPSEC");
}

static size_t eval_core_case_count(void) {
    return ds4_eval_core_case_count;
}

static eval_answer_kind eval_case_answer_kind(const eval_case *tc) {
    if (tc->answer_kind != EVAL_ANSWER_AUTO) return tc->answer_kind;
    if (eval_case_is_multiple_choice(tc)) return EVAL_ANSWER_CHOICE;
    if (eval_case_is_compsec(tc)) return EVAL_ANSWER_LINE_SET;
    return EVAL_ANSWER_INTEGER;
}

static const char *eval_answer_kind_name(eval_answer_kind kind) {
    switch (kind) {
    case EVAL_ANSWER_CHOICE: return "choice";
    case EVAL_ANSWER_INTEGER: return "integer";
    case EVAL_ANSWER_RATIONAL: return "reduced_rational";
    case EVAL_ANSWER_EXACT_TEXT: return "exact_text";
    case EVAL_ANSWER_ORDERED_SEQUENCE: return "ordered_sequence";
    case EVAL_ANSWER_LINE_SET: return "security_line_set";
    case EVAL_ANSWER_AUTO:
    default: return "auto";
    }
}

static uint32_t parse_suite(const char *s, const char *opt) {
    if (!strcmp(s, "core")) return EVAL_SUITE_CORE;
    if (!strcmp(s, "hard")) return EVAL_SUITE_HARD;
    if (!strcmp(s, "all")) return EVAL_SUITE_CORE | EVAL_SUITE_HARD;
    if (!strcmp(s, "hard-smoke")) return EVAL_SUITE_HARD_SMOKE;
    fprintf(stderr,
            "ds4-eval: invalid value for %s: %s (expected core, hard, all, or hard-smoke)\n",
            opt, s);
    exit(2);
}

static const char *eval_suite_name(uint32_t mask) {
    if (mask == EVAL_SUITE_CORE) return "core";
    if (mask == EVAL_SUITE_HARD) return "hard";
    if (mask == (EVAL_SUITE_CORE | EVAL_SUITE_HARD)) return "all";
    if (mask == EVAL_SUITE_HARD_SMOKE) return "hard-smoke";
    return "custom";
}

static bool eval_case_matches_filters(const eval_case *tc,
                                      const eval_config *cfg) {
    if (cfg->source_filter && strcasecmp(tc->source, cfg->source_filter)) return false;
    if (cfg->domain_filter && strcasecmp(tc->domain, cfg->domain_filter)) return false;
    if (cfg->case_id_filter && strcmp(tc->id, cfg->case_id_filter)) return false;
    return true;
}

static eval_case *select_eval_cases(const eval_config *cfg, int *count_out) {
    const size_t core_count = eval_core_case_count();
    const size_t max_count = core_count + eval_hard_case_count;
    eval_case *selected = calloc(max_count ? max_count : 1, sizeof(*selected));
    if (!selected) {
        fprintf(stderr, "ds4-eval: out of memory while selecting cases\n");
        return NULL;
    }

    int count = 0;
    if (cfg->suite_mask & EVAL_SUITE_CORE) {
        for (size_t i = 0; i < core_count; i++) {
            if (eval_case_matches_filters(&eval_core_cases[i], cfg))
                selected[count++] = eval_core_cases[i];
        }
    }
    if (cfg->suite_mask & (EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE)) {
        for (size_t i = 0; i < eval_hard_case_count; i++) {
            const eval_case *tc = &eval_hard_cases[i];
            if ((cfg->suite_mask & EVAL_SUITE_HARD_SMOKE) &&
                !(cfg->suite_mask & EVAL_SUITE_HARD) &&
                !(tc->suites & EVAL_SUITE_HARD_SMOKE))
                continue;
            if (eval_case_matches_filters(tc, cfg)) selected[count++] = *tc;
        }
    }

    if (cfg->question_limit > count) {
        fprintf(stderr, "ds4-eval: only %d questions match the selected suite and filters\n",
                count);
        free(selected);
        return NULL;
    }
    if (cfg->question_limit > 0 && cfg->question_limit < count)
        count = cfg->question_limit;
    if (count == 0) {
        fprintf(stderr, "ds4-eval: no questions match the selected suite and filters\n");
        free(selected);
        return NULL;
    }
    *count_out = count;
    return selected;
}

static void style_append(style_buf *b, unsigned char style, size_t n) {
    if (n == 0) return;
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n) cap *= 2;
        unsigned char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memset(b->v + b->len, style, n);
    b->len += n;
}

static void buf_free(byte_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

static void style_free(style_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double run_clock_sec(void) {
    struct timespec ts;
#ifdef CLOCK_UPTIME_RAW
    /* On Darwin this clock excludes system sleep, which is exactly what the TUI
     * elapsed counter wants: benchmark runtime, not lid-closed wall time. */
    clock_gettime(CLOCK_UPTIME_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64_arg(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_arg(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-eval: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-eval: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-eval: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_EVAL, topic);
}

static eval_config parse_options(int argc, char **argv) {
    eval_config c = {
        .model_path = "ds4flash.gguf",
        .backend = default_backend(),
        .max_tokens = 16000,
        .suite_mask = EVAL_SUITE_CORE,
        .top_p = DS4_DEFAULT_TOP_P,
        .min_p = DS4_DEFAULT_MIN_P,
        .pause_ms = 350,
        .soft_limit_reply_budget = 1024,
        .hard_limit_reply_budget = 512,
        .soft_limit_think_close_rank = 3,
        .think_mode = DS4_THINK_HIGH,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-eval: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        char tp_parse_err[256] = {0};
        ds4_tp_cli_parse_result tp_parse =
            ds4_tp_parse_cli_arg(arg,
                                 &i,
                                 argc,
                                 argv,
                                 &c.tp,
                                 tp_parse_err,
                                 sizeof(tp_parse_err));
        if (tp_parse == DS4_TP_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-eval: %s\n",
                    tp_parse_err[0] ? tp_parse_err : "invalid tensor-parallel option");
            exit(2);
        }
        if (tp_parse == DS4_TP_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.max_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            c.max_tokens_set = true;
        } else if (!strcmp(arg, "--questions")) {
            c.question_limit = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--case-sequence")) {
            c.case_sequence = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--suite")) {
            c.suite_mask = parse_suite(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--source")) {
            c.source_filter = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--domain")) {
            c.domain_filter = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--case-id")) {
            c.case_id_filter = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--list-cases")) {
            c.list_cases = true;
        } else if (!strcmp(arg, "--validate-cases")) {
            c.validate_cases = true;
        } else if (!strcmp(arg, "--retry-incomplete")) {
            c.retry_incomplete = true;
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--regrade-trace")) {
            c.regrade_trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--soft-limit-reply-budget")) {
            c.soft_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--hard-limit-reply-budget")) {
            c.hard_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--soft-limit-think-close-rank")) {
            c.soft_limit_think_close_rank = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--pause-ms")) {
            c.pause_ms = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-eval: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int_arg(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-eval: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-eval: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-eval: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-eval: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--think")) {
            c.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--plain")) {
            c.plain = true;
        } else if (!strcmp(arg, "--self-test-extractors")) {
            c.self_test_extractors = true;
        } else {
            fprintf(stderr, "ds4-eval: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }
    if (c.self_test_extractors || c.validate_cases || c.list_cases ||
        c.regrade_trace_path)
        return c;

    char tp_err[256];
    if (!ds4_tp_adopt_distributed_options(&c.tp, &c.dist,
                                          tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-eval: %s\n", tp_err);
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-eval: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-eval: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    if (c.tp.role == DS4_TP_WORKER) {
        fprintf(stderr, "ds4-eval: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }

    if (c.max_tokens > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --tokens (%d) exceeds the %d token context cap\n",
                c.max_tokens, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (c.ctx_size > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --ctx (%d) exceeds the %d token context cap\n",
                c.ctx_size, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (ds4_think_mode_enabled(c.think_mode) &&
        c.hard_limit_reply_budget >= c.max_tokens) {
        fprintf(stderr,
                "ds4-eval: --hard-limit-reply-budget (%d) must be smaller than --tokens (%d)\n",
                c.hard_limit_reply_budget, c.max_tokens);
        exit(2);
    }
    if (ds4_think_mode_enabled(c.think_mode) &&
        c.soft_limit_reply_budget < c.hard_limit_reply_budget) {
        fprintf(stderr,
                "ds4-eval: --soft-limit-reply-budget (%d) must be >= --hard-limit-reply-budget (%d)\n",
                c.soft_limit_reply_budget, c.hard_limit_reply_budget);
        exit(2);
    }
    return c;
}

static int terminal_size(int *cols, int *rows) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0 || ws.ws_row == 0) {
        *cols = 80;
        *rows = 24;
        return -1;
    }
    *cols = ws.ws_col;
    *rows = ws.ws_row;
    return 0;
}

static void term_move(int row, int col) {
    printf("\x1b[%d;%dH", row, col);
}

static void term_clear_to_eol(void) {
    fputs("\x1b[K", stdout);
}

static int tui_right_text_w(const eval_ui *ui) {
    /* Avoid writing the physical last column.  Many terminals defer wrapping
     * until the next byte after the last column, which would spill right-pane
     * status text into the next row's left pane. */
    return ui->right_w > 1 ? ui->right_w - 1 : ui->right_w;
}

static void tui_clear_left_line(eval_ui *ui, int row) {
    term_move(row, 1);
    for (int i = 0; i < ui->left_w; i++) fputc(' ', stdout);
    term_move(row, ui->left_w + 1);
    fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    term_move(row, 1);
}

static void term_set_color_for_status(eval_status st) {
    switch (st) {
    case EVAL_PENDING:  fputs(ANSI_DIM, stdout); break;
    case EVAL_PREFILL:  fputs(ANSI_CYAN, stdout); break;
    case EVAL_THINKING: fputs(ANSI_YELLOW, stdout); break;
    case EVAL_SKIPPED:  fputs(ANSI_DIM, stdout); break;
    case EVAL_STOPPED:  fputs(ANSI_YELLOW ANSI_BOLD, stdout); break;
    case EVAL_INCOMPLETE:fputs(ANSI_YELLOW ANSI_BOLD, stdout); break;
    case EVAL_PASSED:   fputs(ANSI_GREEN ANSI_BOLD, stdout); break;
    case EVAL_FAILED:   fputs(ANSI_RED ANSI_BOLD, stdout); break;
    }
}

static const char *status_name(eval_status st) {
    switch (st) {
    case EVAL_PENDING: return "PEND";
    case EVAL_PREFILL: return "FILL";
    case EVAL_THINKING: return "RUN";
    case EVAL_SKIPPED: return "SKIP";
    case EVAL_STOPPED: return "STOP";
    case EVAL_INCOMPLETE: return "INC";
    case EVAL_PASSED: return "PASS";
    case EVAL_FAILED: return "FAIL";
    }
    return "?";
}

static bool eval_status_running(eval_status st) {
    return st == EVAL_PREFILL || st == EVAL_THINKING;
}

static void print_trimmed(const char *s, int width) {
    if (width <= 0) return;
    int len = (int)strlen(s);
    if (len <= width) {
        fputs(s, stdout);
        return;
    }
    if (width <= 3) {
        for (int i = 0; i < width; i++) fputc('.', stdout);
        return;
    }
    fwrite(s, 1, (size_t)width - 3, stdout);
    fputs("...", stdout);
}

static void tui_draw_question_preview(eval_ui *ui, const eval_case *tc) {
    const char *q = tc->question;
    size_t pos = 0;
    const int width = tui_right_text_w(ui);

    for (int row = 0; row < 3; row++) {
        term_move(2 + row, ui->right_x);
        term_clear_to_eol();
        fputs(ANSI_BLUE, stdout);

        int printed = 0;
        bool last_space = true;
        while (q[pos] && printed < width) {
            unsigned char c = (unsigned char)q[pos++];
            if (isspace(c)) {
                if (last_space) continue;
                c = ' ';
                last_space = true;
            } else {
                last_space = false;
            }
            fputc(c, stdout);
            printed++;
        }

        if (row == 2) {
            while (q[pos] && isspace((unsigned char)q[pos])) pos++;
            if (q[pos] && width >= 3) {
                int ell_col = ui->right_x + (printed >= 3 ? printed - 3 : 0);
                term_move(2 + row, ell_col);
                fputs("...", stdout);
            }
        }
        fputs(ANSI_RESET, stdout);
    }

    term_move(5, ui->right_x);
    term_clear_to_eol();
}

static void input_queue_key(int key) {
    pthread_mutex_lock(&global_input.mu);
    if (key == 'A') {
        global_input.move_delta--;
    } else if (key == 'B') {
        global_input.move_delta++;
    } else if (key == '\r' || key == '\n') {
        global_input.enter_pressed = true;
    } else if (key == 'p' || key == 'P') {
        global_input.pause_pressed = true;
    } else if (key == 'q' || key == 'Q') {
        global_input.quit_pressed = true;
    }
    pthread_mutex_unlock(&global_input.mu);
}

static void *input_thread_main(void *arg) {
    (void)arg;
    int esc_state = 0;
    char buf[64];

    while (global_input.running) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) {
            usleep(5000);
            continue;
        }
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)buf[i];
            if (esc_state == 0) {
                if (c == 27) esc_state = 1;
                else if (c == 'p' || c == 'P' || c == 'q' || c == 'Q') input_queue_key(c);
                else if (c == '\r' || c == '\n') input_queue_key(c);
            } else if (esc_state == 1) {
                esc_state = (c == '[' || c == 'O') ? 2 : 0;
            } else {
                if (c == 'A' || c == 'B') input_queue_key(c);
                esc_state = 0;
            }
        }
    }
    return NULL;
}

static void tui_start_input(void) {
    if (!isatty(STDIN_FILENO)) return;
    if (tcgetattr(STDIN_FILENO, &global_input.orig_termios) != 0) return;

    /* The input thread is intentionally boring: it never writes to the terminal,
     * it only queues navigation/control state.  Rendering remains owned by the
     * main thread and follows the same full-frame redraw path as the
     * noninteractive UI. Keep ISIG set so Ctrl-C still restores the alternate
     * screen. */
    struct termios raw = global_input.orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return;

    pthread_mutex_lock(&global_input.mu);
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    global_input.raw_mode = true;
    global_input.enabled = true;
    global_input.running = 1;
    if (pthread_create(&global_input.thread, NULL, input_thread_main, NULL) == 0) {
        global_input.thread_started = true;
    } else {
        global_input.running = 0;
        global_input.enabled = false;
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
}

static void tui_stop_input(void) {
    if (global_input.thread_started) {
        global_input.running = 0;
        pthread_join(global_input.thread, NULL);
        global_input.thread_started = false;
    }
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    global_input.enabled = false;
}

static void tui_consume_input(eval_ui *ui) {
    if (!ui->enabled || !global_input.enabled) return;

    pthread_mutex_lock(&global_input.mu);
    int move = global_input.move_delta;
    bool enter = global_input.enter_pressed;
    bool pause = global_input.pause_pressed;
    bool quit = global_input.quit_pressed;
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    if (quit) {
        ui->quit_requested = true;
        ui->paused = false;
        return;
    }

    if (pause) ui->paused = !ui->paused;

    if (move) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        int selected = ui->selected_case + move;
        if (selected < 0) selected = 0;
        if (selected >= ui->ncases) selected = ui->ncases - 1;
        ui->selected_case = selected;
    }

    if (enter) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        if (ui->selected_case != ui->active_case ||
            !eval_status_running(ui->status[ui->active_case]))
        {
            ui->requested_case = ui->selected_case;
        }
    }
}

static void tui_restore(void) {
    eval_ui *ui = global_ui;
    tui_stop_input();
    if (!ui || !ui->active) return;
    fputs(ANSI_RESET "\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
    ui->active = false;
}

static void tui_signal_restore(int sig) {
    /* Signal handlers cannot safely run the full tui_restore() path: that path
     * joins the input thread and uses stdio.  For Ctrl-C / termination, do the
     * minimal terminal repair directly, then re-raise the signal with its
     * default action so the process status remains correct. */
    eval_ui *ui = global_ui;
    global_input.running = 0;
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    if (ui && ui->active) {
        const char restore[] = ANSI_RESET "\x1b[?25h\x1b[?1049l";
        if (write(STDOUT_FILENO, restore, sizeof(restore) - 1) == -1) {}
        ui->active = false;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

static void tui_draw_title(eval_ui *ui) {
    term_move(1, 1);
    tui_clear_left_line(ui, 1);
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));
    fputs("ds4-eval (" ANSI_BOLD "p" ANSI_RESET ")ause (" ANSI_BOLD "q" ANSI_RESET ")uit", stdout);
    printf(" %s", elapsed);
    if (ui->paused) {
        fputs(" " ANSI_RED ANSI_BOLD "PAUSED" ANSI_RESET, stdout);
    }
}

static void tui_draw_frame(eval_ui *ui) {
    fputs("\x1b[2J", stdout);
    tui_draw_title(ui);
    term_move(1, ui->right_x);
    fputs(ANSI_BOLD "live sampled tokens" ANSI_RESET, stdout);

    for (int row = 1; row <= ui->rows; row++) {
        term_move(row, ui->left_w + 1);
        fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    }
}

static void tui_draw_left(eval_ui *ui) {
    int passed = 0;
    int failed = 0;
    int incomplete = 0;
    for (int i = 0; i < ui->ncases; i++) {
        if (ui->status[i] == EVAL_PASSED) passed++;
        else if (ui->status[i] == EVAL_FAILED) failed++;
        else if (ui->status[i] == EVAL_INCOMPLETE) incomplete++;
    }

    tui_draw_title(ui);

    term_move(2, 1);
    tui_clear_left_line(ui, 2);
    printf("score %s%d%s/%d  failed %s%d%s  incomplete %s%d%s",
           ANSI_GREEN, passed, ANSI_RESET,
           ui->ncases,
           failed ? ANSI_RED : ANSI_DIM, failed, ANSI_RESET,
           incomplete ? ANSI_YELLOW : ANSI_DIM, incomplete, ANSI_RESET);

    const int first_row = 4;
    const int visible_rows = ui->rows >= first_row ? ui->rows - first_row + 1 : 0;
    const int shown = ui->ncases < visible_rows ? ui->ncases : visible_rows;
    int first = 0;
    if (shown > 0 && ui->ncases > shown) {
        /* The list follows the selection cursor, not the running test.  This
         * makes arrow navigation visible immediately.  When normal execution
         * advances, main() moves the selection to the new active case so the
         * viewport naturally follows the work again. */
        int anchor = ui->selected_case;
        if (anchor < 0) anchor = ui->active_case;
        if (anchor >= ui->ncases) anchor = ui->ncases - 1;
        first = anchor - shown / 2;
        if (first < 0) first = 0;
        if (first > ui->ncases - shown) first = ui->ncases - shown;
    }

    for (int row = 0; row < visible_rows; row++) {
        const int screen_row = first_row + row;
        const int i = first + row;
        term_move(screen_row, 1);
        tui_clear_left_line(ui, screen_row);
        if (i >= ui->ncases) continue;

        if (i == ui->active_case) fputs(ANSI_BOLD, stdout);
        term_set_color_for_status(ui->status[i]);
        printf("%c%2d ", ui->selection_active && i == ui->selected_case ? '>' : ' ', i + 1);
        printf("%-4s", status_name(ui->status[i]));
        fputs(ANSI_RESET, stdout);

        const int answer_w = 18;
        const int answer_col = ui->left_w - answer_w + 1;
        int title_w = answer_col - 11;
        if (title_w > 0) {
            fputc(' ', stdout);
            char title[512];
            snprintf(title, sizeof(title), "%s: %s",
                     ui->cases[i].source, ui->cases[i].title);
            print_trimmed(title, title_w);
        }
        if (ui->status[i] == EVAL_FAILED || ui->status[i] == EVAL_PASSED ||
            ui->status[i] == EVAL_INCOMPLETE) {
            char answers[64];
            snprintf(answers, sizeof(answers), "%s/%s",
                     ui->guess[i][0] ? ui->guess[i] : "?",
                     ui->cases[i].answer);
            term_move(screen_row, answer_col);
            print_trimmed(answers, answer_w);
        }
        fputs(ANSI_RESET, stdout);
    }
}

static void tui_reset_stream(eval_ui *ui, const eval_case *tc, bool in_think) {
    ui->stream.len = 0;
    if (ui->stream.v) ui->stream.v[0] = '\0';
    ui->styles.len = 0;
    ui->in_think = in_think;
    ui->pending_tag_len = 0;
    ui->generated = 0;
    ui->prefill_current = 0;
    ui->prefill_total = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;

    for (int row = 2; row <= ui->rows; row++) {
        term_move(row, ui->right_x);
        term_clear_to_eol();
    }
    tui_draw_question_preview(ui, tc);
}

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void stream_append_visible(eval_ui *ui, const char *p, size_t n) {
    buf_append(&ui->stream, p, n);
    style_append(&ui->styles, ui->in_think ? 1 : 0, n);
}

static void stream_append_token_text(eval_ui *ui, const char *text, size_t len, bool finish) {
    const char *open = "<think>";
    const char *close = "</think>";
    size_t total = ui->pending_tag_len + len;
    char *tmp = malloc(total ? total : 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (ui->pending_tag_len) memcpy(tmp, ui->pending_tag, ui->pending_tag_len);
    if (len) memcpy(tmp + ui->pending_tag_len, text, len);
    ui->pending_tag_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = tmp + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, open)) {
            ui->in_think = true;
            i += strlen(open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, close)) {
            ui->in_think = false;
            stream_append_visible(ui, "\n", 1);
            i += strlen(close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, open) ||
             bytes_is_partial_prefix(cur, rem, close)))
        {
            if (rem < sizeof(ui->pending_tag)) {
                memcpy(ui->pending_tag, cur, rem);
                ui->pending_tag_len = rem;
            }
            break;
        }
        stream_append_visible(ui, cur, 1);
        i++;
    }
    free(tmp);
}

typedef struct {
    size_t start;
    size_t end;
} line_span;

static void tui_draw_stream(eval_ui *ui) {
    if (!ui->enabled) return;
    const int width = tui_right_text_w(ui);

    line_span *lines = NULL;
    int line_len = 0;
    int line_cap = 0;
    size_t line_start = 0;
    int col = 0;

    for (size_t i = 0; i < ui->stream.len; i++) {
        char c = ui->stream.v[i];
        bool end_line = false;
        size_t end = i;
        if (c == '\n') {
            end_line = true;
            end = i;
        } else {
            col++;
            if (col >= width) {
                end_line = true;
                end = i + 1;
            }
        }
        if (end_line) {
            if (line_len == line_cap) {
                line_cap = line_cap ? line_cap * 2 : 64;
                line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
                if (!v) {
                    fprintf(stderr, "ds4-eval: out of memory\n");
                    exit(1);
                }
                lines = v;
            }
            lines[line_len++] = (line_span){line_start, end};
            line_start = i + 1;
            col = 0;
        }
    }
    if (line_start <= ui->stream.len) {
        if (line_len == line_cap) {
            line_cap = line_cap ? line_cap * 2 : 64;
            line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
            if (!v) {
                fprintf(stderr, "ds4-eval: out of memory\n");
                exit(1);
            }
            lines = v;
        }
        lines[line_len++] = (line_span){line_start, ui->stream.len};
    }

    int start = line_len > ui->body_h ? line_len - ui->body_h : 0;
    for (int row = 0; row < ui->body_h; row++) {
        term_move(ui->body_y + row, ui->right_x);
        term_clear_to_eol();
        int li = start + row;
        if (li >= line_len) continue;
        unsigned char cur_style = 255;
        int printed = 0;
        for (size_t i = lines[li].start; i < lines[li].end && printed < width; i++) {
            unsigned char st = ui->styles.v[i];
            if (st != cur_style) {
                fputs(st ? ANSI_DIM : ANSI_RESET, stdout);
                cur_style = st;
            }
            char c = ui->stream.v[i];
            if (c == '\r' || c == '\n') continue;
            if (c == '\t') c = ' ';
            fputc((unsigned char)c, stdout);
            printed++;
        }
        fputs(ANSI_RESET, stdout);
    }
    free(lines);
}

static void format_short_count(char *dst, size_t dstlen, int n) {
    if (n >= 1000000) {
        snprintf(dst, dstlen, "%dm", (n + 500000) / 1000000);
    } else if (n >= 10000) {
        snprintf(dst, dstlen, "%dk", (n + 500) / 1000);
    } else {
        snprintf(dst, dstlen, "%d", n);
    }
}

static const char *short_phase_name(const char *phase) {
    if (!strcmp(phase, "prefill")) return "FILL";
    if (!strcmp(phase, "thinking")) return "THINK";
    if (!strcmp(phase, "answer")) return "ANS";
    if (!strcmp(phase, "passed")) return "PASS";
    if (!strcmp(phase, "failed")) return "FAIL";
    return "RUN";
}

static void tui_draw_right_status(eval_ui *ui, const char *phase) {
    term_move(1, ui->right_x);
    term_clear_to_eol();
    const eval_case *tc = &ui->cases[ui->active_case];
    char id[13];
    char gen[16], max[16], cur[16], total[16];
    char line[512];
    snprintf(id, sizeof(id), "%.12s", tc->id);
    if (ui->prefill_total > 0 && ui->status[ui->active_case] == EVAL_PREFILL) {
        double pct = 100.0 * (double)ui->prefill_current / (double)ui->prefill_total;
        format_short_count(cur, sizeof(cur), ui->prefill_current);
        format_short_count(total, sizeof(total), ui->prefill_total);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s %.0f%%  %s/%s",
                 ui->speed_tps, short_phase_name(phase), cur, total, pct,
                 tc->source, id);
    } else {
        int phase_max = !strcmp(phase, "thinking") ? ui->think_max_tokens : ui->max_tokens;
        int phase_gen = ui->generated;
        if (phase_max < 0) phase_max = 0;
        if (!strcmp(phase, "thinking") && phase_gen > phase_max) phase_gen = phase_max;
        format_short_count(gen, sizeof(gen), phase_gen);
        format_short_count(max, sizeof(max), phase_max);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s  %s/%s",
                 ui->speed_tps, short_phase_name(phase), gen, max,
                 tc->source, id);
    }
    print_trimmed(line, tui_right_text_w(ui));
}

static void tui_refresh(eval_ui *ui, const char *phase) {
    if (!ui->enabled) return;
    tui_draw_left(ui);
    tui_draw_right_status(ui, phase);
    tui_draw_stream(ui);
    fflush(stdout);
}

static void tui_start(eval_ui *ui, const eval_case *cases, int ncases, int max_tokens,
                      bool enabled) {
    memset(ui, 0, sizeof(*ui));
    ui->enabled = enabled;
    ui->cases = cases;
    ui->ncases = ncases;
    ui->max_tokens = max_tokens;
    ui->selected_case = 0;
    ui->requested_case = -1;
    ui->quit_requested = false;
    ui->status = calloc((size_t)ncases, sizeof(*ui->status));
    ui->guess = calloc((size_t)ncases, sizeof(*ui->guess));
    ui->prompt_tokens = calloc((size_t)ncases, sizeof(*ui->prompt_tokens));
    ui->generated_tokens = calloc((size_t)ncases, sizeof(*ui->generated_tokens));
    if (!ui->status || !ui->guess || !ui->prompt_tokens || !ui->generated_tokens) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (!enabled) return;

    terminal_size(&ui->cols, &ui->rows);
    if (ui->cols < 90 || ui->rows < 18) {
        ui->enabled = false;
        return;
    }
    ui->left_w = ui->cols / 2;
    if (ui->left_w < 42) ui->left_w = 42;
    if (ui->left_w > 72) ui->left_w = 72;
    ui->right_x = ui->left_w + 3;
    ui->right_w = ui->cols - ui->right_x + 1;
    ui->body_y = 6;
    ui->body_h = ui->rows - ui->body_y + 1;

    global_ui = ui;
    atexit(tui_restore);
    signal(SIGINT, tui_signal_restore);
    signal(SIGTERM, tui_signal_restore);
#ifdef SIGHUP
    signal(SIGHUP, tui_signal_restore);
#endif
#ifdef SIGQUIT
    signal(SIGQUIT, tui_signal_restore);
#endif

    fputs("\x1b[?1049h\x1b[?25l", stdout);
    ui->active = true;
    tui_start_input();
    tui_draw_frame(ui);
    tui_refresh(ui, "idle");
}

static void tui_free(eval_ui *ui) {
    if (ui->active) tui_restore();
    free(ui->status);
    free(ui->guess);
    free(ui->prompt_tokens);
    free(ui->generated_tokens);
    buf_free(&ui->stream);
    style_free(&ui->styles);
    memset(ui, 0, sizeof(*ui));
}

static void plain_set_thinking_color(bool use_color) {
    if (use_color) fputs(ANSI_DIM, stdout);
}

static void plain_reset_color(bool use_color) {
    if (use_color) fputs(ANSI_RESET, stdout);
}

static const char *eval_system_prompt(void) {
    return "You are solving a hard benchmark question. Reason carefully. "
           "The final answer must follow the requested format exactly.";
}

static char *build_question_prompt(const eval_case *tc) {
    byte_buf b = {0};
    int nchoices = eval_case_nchoices(tc);
    eval_answer_kind kind = eval_case_answer_kind(tc);
    buf_appendf(&b, "%s\n", tc->question);
    if (kind == EVAL_ANSWER_CHOICE) {
        buf_append(&b, "\nChoices:\n", strlen("\nChoices:\n"));
        for (int i = 0; i < nchoices; i++) {
            buf_appendf(&b, "%c. %s\n", 'A' + i, tc->choice[i]);
        }
        buf_append(&b,
            "\nSolve the question. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <letter>",
            strlen("\nSolve the question. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <letter>"));
    } else if (kind == EVAL_ANSWER_LINE_SET) {
        buf_append(&b,
            "\nAt the end, write exactly one final line in this format and do not "
            "write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>",
            strlen("\nAt the end, write exactly one final line in this format and do not "
                   "write anything after it:\n"
                   "Answer: <line number or comma-separated line numbers>"));
    } else if (kind == EVAL_ANSWER_INTEGER) {
        buf_append(&b,
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <integer>",
            strlen("\nSolve the problem. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <integer>"));
    } else if (kind == EVAL_ANSWER_RATIONAL) {
        buf_append(&b,
            "\nSolve the problem. Reduce the result. At the end, write exactly one final "
            "line in this format and do not write anything after it:\n"
            "Answer: <integer or reduced fraction>",
            strlen("\nSolve the problem. Reduce the result. At the end, write exactly one final "
                   "line in this format and do not write anything after it:\n"
                   "Answer: <integer or reduced fraction>"));
    } else if (kind == EVAL_ANSWER_ORDERED_SEQUENCE) {
        buf_append(&b,
            "\nSolve the problem. At the end, write exactly one final line containing the "
            "answers in the requested order, separated by commas, and do not write anything "
            "after it:\nAnswer: <ordered answers>",
            strlen("\nSolve the problem. At the end, write exactly one final line containing the "
                   "answers in the requested order, separated by commas, and do not write anything "
                   "after it:\nAnswer: <ordered answers>"));
    } else {
        buf_append(&b,
            "\nSolve the problem. At the end, write exactly one final line in this format "
            "and do not write anything after it:\nAnswer: <exact answer>",
            strlen("\nSolve the problem. At the end, write exactly one final line in this format "
                   "and do not write anything after it:\nAnswer: <exact answer>"));
    }
    if (b.v) return b.v;
    char *empty = malloc(1);
    if (empty) empty[0] = '\0';
    return empty;
}

static int eval_case_generation_budget(const eval_config *cfg,
                                       const eval_case *tc,
                                       bool retry) {
    uint64_t budget = (!cfg->max_tokens_set && tc->max_tokens) ?
                      tc->max_tokens : (uint32_t)cfg->max_tokens;
    if (retry) budget *= 2;
    if (budget > EVAL_MAX_CONTEXT) budget = EVAL_MAX_CONTEXT;
    return (int)budget;
}

static int eval_max_generation_budget(const eval_config *cfg,
                                      const eval_case *cases,
                                      int ncases) {
    int max_budget = 0;
    for (int i = 0; i < ncases; i++) {
        int budget = eval_case_generation_budget(cfg, &cases[i],
                                                 cfg->retry_incomplete);
        if (budget > max_budget) max_budget = budget;
    }
    return max_budget;
}

static int eval_max_prompt_tokens(ds4_engine *engine,
                                  const eval_config *cfg,
                                  const eval_case *cases,
                                  int ncases,
                                  int ctx_for_think_mode,
                                  int *max_case_out)
{
    int max_prompt = 0;
    int max_case = -1;
    const ds4_think_mode think_mode =
        ds4_think_mode_for_context(cfg->think_mode, ctx_for_think_mode);

    for (int i = 0; i < ncases; i++) {
        char *question = build_question_prompt(&cases[i]);
        if (!question) {
            fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
            exit(1);
        }
        ds4_tokens prompt = {0};
        ds4_encode_chat_prompt(engine, eval_system_prompt(), question, think_mode, &prompt);
        if (prompt.len > max_prompt) {
            max_prompt = prompt.len;
            max_case = i;
        }
        ds4_tokens_free(&prompt);
        free(question);
    }
    if (max_case_out) *max_case_out = max_case;
    return max_prompt;
}

static int eval_auto_context_size(ds4_engine *engine,
                                  eval_config *cfg,
                                  const eval_case *cases,
                                  int ncases,
                                  int max_generation_tokens,
                                  int *max_prompt_out,
                                  int *max_case_out)
{
    int ctx = EVAL_MAX_CONTEXT;
    int max_prompt = 0;
    int max_case = -1;
    const int min_ctx = cfg->think_mode == DS4_THINK_MAX ?
                        (int)ds4_think_max_min_context() : 1;

    /* Think Max downgrades to normal thinking under its minimum context.  Size
     * the prompts iteratively so the prompt tokenizer sees the same effective
     * thinking mode that the actual run will use. */
    for (int iter = 0; iter < 3; iter++) {
        max_prompt = eval_max_prompt_tokens(engine, cfg, cases, ncases, ctx, &max_case);
        long long required = (long long)max_prompt + (long long)max_generation_tokens;
        if (required < min_ctx) required = min_ctx;
        if (required > EVAL_MAX_CONTEXT) {
            fprintf(stderr,
                    "ds4-eval: largest prompt (%d tokens, case %d) + generation budget (%d) exceeds the %d token context cap\n",
                    max_prompt, max_case + 1, max_generation_tokens, EVAL_MAX_CONTEXT);
            exit(2);
        }
        if ((int)required == ctx) break;
        ctx = (int)required;
    }

    if (max_prompt_out) *max_prompt_out = max_prompt;
    if (max_case_out) *max_case_out = max_case;
    return ctx;
}

static void eval_warn_think_max_downgraded(const eval_config *cfg) {
    if (cfg->think_mode != DS4_THINK_MAX ||
        ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size) == DS4_THINK_MAX) {
        return;
    }
    fprintf(stderr,
            "ds4-eval: warning: --think-max needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
            ds4_think_max_min_context(),
            cfg->ctx_size);
}

static void eval_warn_context_budget(const eval_config *cfg, int max_prompt_tokens,
                                     int max_prompt_case, int max_generation_tokens) {
    if (max_prompt_tokens >= cfg->ctx_size) {
        fprintf(stderr,
                "ds4-eval: warning: largest prompt (%d tokens, case=%d) does not fit ctx=%d\n",
                max_prompt_tokens,
                max_prompt_case + 1,
                cfg->ctx_size);
        return;
    }

    const int room = cfg->ctx_size - max_prompt_tokens;
    if (room < max_generation_tokens) {
        fprintf(stderr,
                "ds4-eval: warning: largest prompt (%d tokens, case=%d) leaves %d generation tokens in ctx=%d; requested %d\n",
                max_prompt_tokens,
                max_prompt_case + 1,
                room,
                cfg->ctx_size,
                max_generation_tokens);
    }
}

static void trace_write_block(FILE *trace, const char *label, const char *text) {
    if (!trace) return;
    size_t len = text ? strlen(text) : 0;
    /* Counted blocks make regrading robust against model output that happens
     * to contain trace-looking delimiters or embedded CASE headers. */
    fprintf(trace, "%s_BEGIN bytes=%zu\n", label, len);
    if (len) {
        fwrite(text, 1, len, trace);
        if (text[len - 1] != '\n') fputc('\n', trace);
    }
    fprintf(trace, "%s_END\n", label);
}

static const char *think_close_kind_name(eval_think_close_kind kind) {
    switch (kind) {
    case EVAL_THINK_CLOSE_NATURAL: return "natural";
    case EVAL_THINK_CLOSE_SOFT: return "soft_forced";
    case EVAL_THINK_CLOSE_HARD: return "hard_forced";
    case EVAL_THINK_CLOSE_NONE:
    default: return "none";
    }
}

static int token_rank_in_top(ds4_session *session, int token, int max_rank) {
    if (token < 0 || max_rank <= 0) return 0;
    ds4_token_score *top = malloc((size_t)max_rank * sizeof(*top));
    if (!top) return 0;
    int n = ds4_session_top_logprobs(session, top, max_rank);
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (top[i].id == token) {
            rank = i + 1;
            break;
        }
    }
    free(top);
    return rank;
}

static void trace_write_header(FILE *trace, const eval_config *cfg,
                               const char *model_name,
                               int ncases,
                               int max_prompt_tokens,
                               int max_generation_tokens) {
    if (!trace) return;
    fprintf(trace,
            "# ds4-eval trace\n"
            "started_unix: %lld\n"
            "model: %s\n"
            "model_shape: %s\n"
            "backend: %s\n"
            "suite: %s\n"
            "source_filter: %s\n"
            "domain_filter: %s\n"
            "case_id_filter: %s\n"
            "ctx: %d\n"
            "max_tokens: %d\n"
            "max_generation_tokens: %d\n"
            "max_prompt_tokens: %d\n"
            "questions: %d\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "seed: %llu\n"
            "think_mode_requested: %s\n"
            "soft_limit_reply_budget: %d\n"
            "hard_limit_reply_budget: %d\n"
            "soft_limit_think_close_rank: %d\n"
            "\n",
            (long long)time(NULL),
            cfg->model_path,
            model_name ? model_name : "unknown",
            ds4_backend_name(cfg->backend),
            eval_suite_name(cfg->suite_mask),
            cfg->source_filter ? cfg->source_filter : "*",
            cfg->domain_filter ? cfg->domain_filter : "*",
            cfg->case_id_filter ? cfg->case_id_filter : "*",
            cfg->ctx_size,
            cfg->max_tokens,
            max_generation_tokens,
            max_prompt_tokens,
            ncases,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            (unsigned long long)cfg->seed,
            ds4_think_mode_name(cfg->think_mode),
            cfg->soft_limit_reply_budget,
            cfg->hard_limit_reply_budget,
            cfg->soft_limit_think_close_rank);
    fflush(trace);
}

static void trace_write_case(FILE *trace,
                             const eval_config *cfg,
                             const eval_case *tc,
                             int idx,
                             int ncases,
                             const char *status,
                             const char *error,
                             const char *system_prompt,
                             const char *question_prompt,
                             const char *model_output,
                             ds4_think_mode effective_think_mode,
                             int prompt_tokens,
                             int generated_tokens,
                             int generation_budget,
                             double elapsed_sec,
                             const char *picked,
                             const eval_think_close_info *think_close) {
    if (!trace) return;
    int nchoices = eval_case_nchoices(tc);
    fprintf(trace,
            "===== CASE %d/%d %s/%s =====\n"
            "timestamp_unix: %lld\n"
            "source: %s\n"
            "id: %s\n"
            "domain: %s\n"
            "title: %s\n"
            "status: %s\n"
            "picked: %s\n"
            "expected: %s\n"
            "prompt_tokens: %d\n"
            "generated_tokens: %d\n"
            "generation_budget: %d\n"
            "elapsed_sec: %.3f\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "think_mode_effective: %s\n",
            idx + 1, ncases, tc->source, tc->id,
            (long long)time(NULL),
            tc->source,
            tc->id,
            tc->domain,
            tc->title,
            status,
            picked && *picked ? picked : "?",
            tc->answer,
            prompt_tokens,
            generated_tokens,
            generation_budget,
            elapsed_sec,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            ds4_think_mode_name(effective_think_mode));
    if (think_close && think_close->kind != EVAL_THINK_CLOSE_NONE) {
        fprintf(trace,
                "think_close: %s\n"
                "think_close_token_index: %d\n"
                "think_close_remaining_budget: %d\n"
                "think_close_rank: %d\n",
                think_close_kind_name(think_close->kind),
                think_close->token_index,
                think_close->remaining_budget,
                think_close->rank);
    } else {
        fprintf(trace, "think_close: none\n");
    }
    if (error && *error) fprintf(trace, "error: %s\n", error);

    if (nchoices > 0) {
        fprintf(trace, "choices:\n");
        for (int i = 0; i < nchoices; i++) {
            fprintf(trace, "  %c. %s\n", 'A' + i, tc->choice[i]);
        }
    }
    fprintf(trace, "answer_kind: %s\n",
            eval_answer_kind_name(eval_case_answer_kind(tc)));
    trace_write_block(trace, "SYSTEM_PROMPT", system_prompt);
    trace_write_block(trace, "QUESTION_PROMPT", question_prompt);
    trace_write_block(trace, "MODEL_OUTPUT", model_output);
    fputc('\n', trace);
    fflush(trace);
}

/* Old traces predate strict final-line grading, so the low-level extractors keep
 * their loose fallback. Live runs call find_case_answer_strict() instead. */
static char *strcasestr_local(const char *hay, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return (char *)hay;
    for (; *hay; hay++) {
        if (tolower((unsigned char)*hay) == tolower((unsigned char)needle[0]) &&
            strncasecmp(hay, needle, nlen) == 0)
            return (char *)hay;
    }
    return NULL;
}

static bool is_letter_boundary(char before, char after) {
    return !isalpha((unsigned char)before) && !isalpha((unsigned char)after);
}

static char *find_last_answer_marker(const char *visible) {
    char *last = NULL;
    const size_t nlen = strlen("answer");

    for (const char *p = visible; *p; p++) {
        if (tolower((unsigned char)*p) != 'a' || strncasecmp(p, "answer", nlen) != 0)
            continue;
        char before = p == visible ? ' ' : p[-1];
        char after = p[nlen];
        if (!is_letter_boundary(before, after)) continue;

        const char *q = p + nlen;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q == ':') last = (char *)p;
    }
    return last ? last : strcasestr_local(visible, "answer");
}

/* True when the in-range capital at `letter` is the object of an explicit
 * rejection earlier on the same answer line -- "not B", "isn't B",
 * "rules out C", "eliminate E" -- so it is a distractor the model is
 * discarding, not its pick.  This rewrites only the lexical "first valid
 * letter wins" default for clear elimination phrasing; the cue set is kept
 * small and high-precision on purpose, and there is no general
 * selection-vs-rejection sentence parsing (issue #321).  It never looks before
 * `start` or across a newline, so "D, not B" still grades D: the pick is
 * reached and accepted before the rejected distractor is ever inspected. */
static bool mc_letter_is_negated(const char *start, const char *letter) {
    const char *p = letter;
    /* Step back over the gap to the previous word: spaces and light separating
     * punctuation only, and never across a line break. */
    while (p > start) {
        char c = p[-1];
        if (c == '\n') return false;
        if (c == ' ' || c == '\t' || c == ',' || c == ';') p--;
        else break;
    }
    /* Read the immediately preceding word (letters/apostrophe), lowercased. */
    const char *wend = p;
    while (p > start && (isalpha((unsigned char)p[-1]) || p[-1] == '\'')) p--;
    size_t wlen = (size_t)(wend - p);
    if (wlen == 0 || wlen >= 16) return false;
    char w[16];
    for (size_t i = 0; i < wlen; i++) w[i] = (char)tolower((unsigned char)p[i]);
    w[wlen] = '\0';

    /* Contraction form: isn't / aren't / wasn't / doesn't / won't / can't. */
    if (wlen >= 3 && strcmp(w + wlen - 3, "n't") == 0) return true;

    static const char *cues[] = {
        "not", "except", "excluding", "exclude", "excludes",
        "eliminate", "eliminates", "eliminated",
        "reject", "rejects", "rejected", "rejecting", NULL
    };
    for (int i = 0; cues[i]; i++) if (strcmp(w, cues[i]) == 0) return true;

    /* Two-word cue: "rule out" / "rules out" / "ruled out". */
    if (strcmp(w, "out") == 0) {
        const char *q = p;
        while (q > start && (q[-1] == ' ' || q[-1] == '\t')) q--;
        const char *rend = q;
        while (q > start && isalpha((unsigned char)q[-1])) q--;
        size_t rl = (size_t)(rend - q);
        if (rl && rl < 8) {
            char r[8];
            for (size_t i = 0; i < rl; i++) r[i] = (char)tolower((unsigned char)q[i]);
            r[rl] = '\0';
            if (!strcmp(r, "rule") || !strcmp(r, "rules") || !strcmp(r, "ruled"))
                return true;
        }
    }
    return false;
}

static char find_answer_letter(const char *generated, int nchoices) {
    if (nchoices <= 0) return '?';
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;
    char max_answer = (char)('A' + nchoices - 1);

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 96) end = answer + 96;
        for (const char *p = answer; p < end && *p; p++) {
            char c = (char)toupper((unsigned char)*p);
            if (c >= 'A' && c <= max_answer) {
                char before = p == visible ? ' ' : p[-1];
                char after = p[1];
                if (!is_letter_boundary(before, after)) continue;
                /* A standalone capital that begins a same-line English word
                 * ("A careful ...") or a contraction ("I'll ...") is prose,
                 * not the model's pick: the real letter comes later on the
                 * line. Skip it; the reverse scan below still recovers it if
                 * it was the only candidate (e.g. "Answer: A is correct"). */
                if (after == '\'') continue;
                if (after == ' ' || after == '\t') {
                    const char *w = p + 1;
                    while (*w == ' ' || *w == '\t') w++;
                    if (islower((unsigned char)*w)) continue;
                }
                /* A distractor explicitly rejected before the pick on the same
                 * line ("not B, ... D") must not win over the real choice. */
                if (mc_letter_is_negated(answer, p)) continue;
                return c;
            }
        }
    }

    for (const char *p = visible + strlen(visible); p > visible; p--) {
        char c = (char)toupper((unsigned char)p[-1]);
        if (c >= 'A' && c <= max_answer) {
            char before = p - 1 == visible ? ' ' : p[-2];
            char after = p[0];
            if (is_letter_boundary(before, after)) return c;
        }
    }
    return '?';
}

static void normalize_integer_answer(const char *p, size_t len,
                                     char *dst, size_t dstlen) {
    while (len > 1 && *p == '0') {
        p++;
        len--;
    }
    if (dstlen == 0) return;
    size_t n = len < dstlen - 1 ? len : dstlen - 1;
    memcpy(dst, p, n);
    dst[n] = '\0';
}

static bool scan_first_integer(const char *start, const char *end,
                               char *dst, size_t dstlen) {
    const char *p = start;
    while (p < end && *p) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (q < end && isdigit((unsigned char)*q)) q++;
            normalize_integer_answer(p, (size_t)(q - p), dst, dstlen);
            return true;
        }
        p++;
    }
    return false;
}

static void find_integer_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        /* Restrict to the final answer line: digits after it (continued
         * reasoning, footnotes, years) must not override the answer. */
        const char *nl = memchr(answer, '\n', (size_t)(end - answer));
        if (nl) end = nl;
        /* When the line shows arithmetic ("m+n = 256+37 = 293") the stated
         * result is the right-hand side of the LAST '='. Otherwise the first
         * integer on the line is the answer (keeps "Final answer: 082"). */
        const char *eq = NULL;
        for (const char *r = answer; r < end; r++) if (*r == '=') eq = r;
        if (scan_first_integer(eq ? eq + 1 : answer, end, dst, dstlen)) return;
        if (eq && scan_first_integer(answer, end, dst, dstlen)) return;
    }

    const char *last_start = NULL;
    const char *last_end = NULL;
    for (const char *p = visible; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (isdigit((unsigned char)*q)) q++;
            last_start = p;
            last_end = q;
            p = q - 1;
        }
    }
    if (last_start && last_end) {
        normalize_integer_answer(last_start, (size_t)(last_end - last_start),
                                 dst, dstlen);
    }
}

static void normalize_compsec_line_spec(const char *p, const char *end,
                                        char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    size_t n = 0;

    /* The model is instructed to finish with:
     *
     *     Answer: <line number or comma-separated line numbers>
     *
     * We still accept harmless surface variants in that final line: "line 9",
     * "9 and 15", "9, 15", or "20-22".  The hidden key decides what set of
     * exact line numbers is acceptable; this routine only normalizes the model
     * text into the same compact line-spec syntax.
     */
    for (; p < end && *p; p++) {
        if (!isdigit((unsigned char)*p)) continue;

        if (n > 0 && n + 1 < dstlen) dst[n++] = ',';
        while (p < end && isdigit((unsigned char)*p)) {
            if (n + 1 < dstlen) dst[n++] = *p;
            p++;
        }
        while (p < end && isspace((unsigned char)*p)) p++;

        if (p < end && *p == '-') {
            if (n + 1 < dstlen) dst[n++] = '-';
            p++;
            while (p < end && isspace((unsigned char)*p)) p++;
            while (p < end && isdigit((unsigned char)*p)) {
                if (n + 1 < dstlen) dst[n++] = *p;
                p++;
            }
        }
        if (p >= end || !*p) break;
    }
    while (n > 0 && (dst[n - 1] == ',' || dst[n - 1] == '-')) n--;
    dst[n] = '\0';
    if (n == 0) snprintf(dst, dstlen, "?");
}

static void find_compsec_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        const char *newline = memchr(answer, '\n', (size_t)(end - answer));
        if (newline) end = newline;
        normalize_compsec_line_spec(answer, end, dst, dstlen);
        if (strcmp(dst, "?") != 0) return;
    }
    find_integer_answer(generated, dst, dstlen);
}

static bool parse_line_spec(const char *spec, bool *set, size_t setlen) {
    bool any = false;
    const char *p = spec;
    while (p && *p) {
        while (*p && !isdigit((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            p = end;
        }
        if (a > b) {
            long tmp = a;
            a = b;
            b = tmp;
        }
        if (a < 0) a = 0;
        if (b >= (long)setlen) b = (long)setlen - 1;
        for (long i = a; i <= b; i++) {
            set[i] = true;
            any = true;
        }
    }
    return any;
}

static bool compsec_answer_matches(const char *expected_spec, const char *got_spec) {
    bool expected[256] = {0};
    bool got[256] = {0};
    if (!parse_line_spec(expected_spec, expected, sizeof(expected))) return false;
    if (!parse_line_spec(got_spec, got, sizeof(got))) return false;

    bool hit = false;
    for (size_t i = 0; i < sizeof(got); i++) {
        if (!got[i]) continue;
        if (!expected[i]) return false;
        hit = true;
    }
    /* The prompt asks the model for the single best line, or the smallest exact
     * set when the bug cannot be localized to one line.  The hidden expected
     * answer may be a small audited range when adjacent lines are equivalent
     * locations for the same bug.  Any model-supplied line must be inside that
     * accepted set, and at least one accepted line must be present. */
    return hit;
}

static const char *answer_payload(const char *line) {
    char *marker = find_last_answer_marker(line);
    if (!marker) return NULL;
    const char *p = marker + strlen("answer");
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') return NULL;
    p++;
    while (*p && (isspace((unsigned char)*p) || *p == '*')) p++;
    return p;
}

static void normalize_exact_answer(const char *src, char *dst, size_t dstlen) {
    size_t n = 0;
    if (dstlen == 0) return;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        if (isspace(*p) || *p == '$' || *p == '*' || *p == '{' || *p == '}')
            continue;
        if (*p == '\\') {
            if (!strncmp((const char *)p, "\\boxed", 6)) p += 5;
            else if (!strncmp((const char *)p, "\\left", 5)) p += 4;
            else if (!strncmp((const char *)p, "\\right", 6)) p += 5;
            continue;
        }
        if (n + 1 < dstlen) dst[n++] = (char)tolower(*p);
    }
    while (n > 0 && (dst[n - 1] == '.' || dst[n - 1] == ';')) n--;
    dst[n] = '\0';
}

static void normalize_ordered_answer(const char *src, char *dst, size_t dstlen) {
    size_t n = 0;
    bool in_tag = false;
    if (dstlen == 0) return;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        if (*p == '<') {
            in_tag = true;
            continue;
        }
        if (in_tag) {
            if (*p == '>') in_tag = false;
            continue;
        }
        if (isalnum(*p) || *p == ',' || *p == '-' || *p == '+' ||
            *p == '/' || *p == '.') {
            if (n + 1 < dstlen) dst[n++] = (char)tolower(*p);
        }
    }
    while (n > 0 && (dst[n - 1] == '.' || dst[n - 1] == ',')) n--;
    dst[n] = '\0';
}

static void normalize_rational_answer(const char *src, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    const char *frac = strstr(src, "\\frac");
    if (frac) {
        const char *a = strchr(frac, '{');
        const char *a_end = a ? strchr(a + 1, '}') : NULL;
        const char *b = a_end ? strchr(a_end + 1, '{') : NULL;
        const char *b_end = b ? strchr(b + 1, '}') : NULL;
        if (a && a_end && b && b_end) {
            size_t n = 0;
            for (const char *p = a + 1; p < a_end && n + 1 < dstlen; p++)
                if (!isspace((unsigned char)*p)) dst[n++] = *p;
            if (n + 1 < dstlen) dst[n++] = '/';
            for (const char *p = b + 1; p < b_end && n + 1 < dstlen; p++)
                if (!isspace((unsigned char)*p)) dst[n++] = *p;
            dst[n] = '\0';
            return;
        }
    }

    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        if (isdigit(*p) || *p == '-' || *p == '+' || *p == '/' || *p == '.') {
            if (n + 1 < dstlen) dst[n++] = (char)*p;
        }
    }
    while (n > 0 && dst[n - 1] == '.') n--;
    dst[n] = '\0';
}

static void find_payload_answer(const eval_case *tc, const char *generated,
                                char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;
    const char *payload = answer_payload(visible);
    if (!payload) return;

    switch (eval_case_answer_kind(tc)) {
    case EVAL_ANSWER_RATIONAL:
        normalize_rational_answer(payload, dst, dstlen);
        break;
    case EVAL_ANSWER_ORDERED_SEQUENCE:
        normalize_ordered_answer(payload, dst, dstlen);
        break;
    case EVAL_ANSWER_EXACT_TEXT:
        normalize_exact_answer(payload, dst, dstlen);
        break;
    default:
        break;
    }
    if (dst[0] == '\0') snprintf(dst, dstlen, "?");
}

static void find_case_answer(const eval_case *tc, const char *generated,
                             char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    switch (eval_case_answer_kind(tc)) {
    case EVAL_ANSWER_CHOICE:
        dst[0] = find_answer_letter(generated, eval_case_nchoices(tc));
        if (dstlen > 1) dst[1] = '\0';
        break;
    case EVAL_ANSWER_LINE_SET:
        find_compsec_answer(generated, dst, dstlen);
        break;
    case EVAL_ANSWER_INTEGER:
        find_integer_answer(generated, dst, dstlen);
        break;
    case EVAL_ANSWER_RATIONAL:
    case EVAL_ANSWER_EXACT_TEXT:
    case EVAL_ANSWER_ORDERED_SEQUENCE:
        find_payload_answer(tc, generated, dst, dstlen);
        break;
    case EVAL_ANSWER_AUTO:
    default:
        snprintf(dst, dstlen, "?");
        break;
    }
}

static bool find_case_answer_strict(const eval_case *tc, const char *generated,
                                    char *dst, size_t dstlen) {
    if (dstlen == 0) return false;
    snprintf(dst, dstlen, "?");

    const char *end = generated + strlen(generated);
    while (end > generated && isspace((unsigned char)end[-1])) end--;
    if (end == generated) return false;
    const char *start = end;
    while (start > generated && start[-1] != '\n' && start[-1] != '\r') start--;

    size_t len = (size_t)(end - start);
    char *line = malloc(len + 1);
    if (!line) {
        fprintf(stderr, "ds4-eval: out of memory while grading answer\n");
        return false;
    }
    memcpy(line, start, len);
    line[len] = '\0';
    if (!answer_payload(line)) {
        free(line);
        return false;
    }
    find_case_answer(tc, line, dst, dstlen);
    free(line);
    return true;
}

static void normalize_expected_answer(eval_answer_kind kind, const char *src,
                                      char *dst, size_t dstlen) {
    switch (kind) {
    case EVAL_ANSWER_RATIONAL:
        normalize_rational_answer(src, dst, dstlen);
        break;
    case EVAL_ANSWER_EXACT_TEXT:
        normalize_exact_answer(src, dst, dstlen);
        break;
    case EVAL_ANSWER_ORDERED_SEQUENCE:
        normalize_ordered_answer(src, dst, dstlen);
        break;
    case EVAL_ANSWER_INTEGER:
        normalize_integer_answer(src, strlen(src), dst, dstlen);
        break;
    default:
        snprintf(dst, dstlen, "%s", src);
        break;
    }
}

static bool answer_matches(const eval_case *tc, const char *got) {
    eval_answer_kind kind = eval_case_answer_kind(tc);
    if (kind == EVAL_ANSWER_CHOICE) {
        return got && got[0] && tc->answer && got[0] == tc->answer[0];
    }
    if (kind == EVAL_ANSWER_LINE_SET) {
        return got && tc->answer && compsec_answer_matches(tc->answer, got);
    }
    if (!got || !tc->answer) return false;
    char expected[EVAL_ANSWER_MAX];
    normalize_expected_answer(kind, tc->answer, expected, sizeof(expected));
    if (strcmp(got, expected) == 0) return true;
    for (int i = 0; i < EVAL_MAX_ALIASES && tc->alias[i]; i++) {
        normalize_expected_answer(kind, tc->alias[i], expected, sizeof(expected));
        if (strcmp(got, expected) == 0) return true;
    }
    return false;
}

static char *read_text_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-eval: cannot seek trace '%s': %s\n",
                path, strerror(errno));
        fclose(fp);
        return NULL;
    }
    long len_long = ftell(fp);
    if (len_long < 0) {
        fprintf(stderr, "ds4-eval: cannot tell trace size '%s': %s\n",
                path, strerror(errno));
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    size_t len = (size_t)len_long;
    char *buf = malloc(len + 1);
    if (!buf) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        fclose(fp);
        return NULL;
    }
    if (len && fread(buf, 1, len, fp) != len) {
        fprintf(stderr, "ds4-eval: cannot read trace '%s': %s\n",
                path, ferror(fp) ? strerror(errno) : "short read");
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    if (len_out) *len_out = len;
    return buf;
}

static const char *bounded_strstr(const char *start, const char *end,
                                  const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return start;
    if ((size_t)(end - start) < nlen) return NULL;
    for (const char *p = start; p + nlen <= end; p++) {
        if (!memcmp(p, needle, nlen)) return p;
    }
    return NULL;
}

static void copy_span(char *dst, size_t dstlen, const char *start, const char *end) {
    if (dstlen == 0) return;
    while (end > start && (end[-1] == '\r' || end[-1] == '\n')) end--;
    size_t n = (size_t)(end - start);
    if (n >= dstlen) n = dstlen - 1;
    memcpy(dst, start, n);
    dst[n] = '\0';
}

static const char *trace_skip_counted_block(const char *line,
                                            const char *line_end,
                                            const char *end) {
    /*
     * Trace regrading must scan metadata lines without being confused by model
     * output.  Prefer the byte count written by trace_write_block(); the older
     * delimiter-only parser is still accepted in trace_copy_model_output for
     * compatibility with existing trace files.
     */
    const char *begin = bounded_strstr(line, line_end, "_BEGIN bytes=");
    if (!begin || begin == line) return NULL;

    size_t label_len = (size_t)(begin - line);
    if (label_len > 96) return NULL;

    char end_marker[128];
    int marker_len = snprintf(end_marker, sizeof(end_marker), "%.*s_END",
                              (int)label_len, line);
    if (marker_len <= 0 || (size_t)marker_len >= sizeof(end_marker)) return NULL;

    const char *bytes = begin + strlen("_BEGIN bytes=");
    char *endptr = NULL;
    unsigned long long declared = strtoull(bytes, &endptr, 10);
    if (endptr == bytes || endptr > line_end) return NULL;

    const char *content = line_end < end ? line_end + 1 : end;
    if (declared > (unsigned long long)(end - content)) return NULL;

    const char *marker = content + (size_t)declared;
    if (marker < end && *marker == '\n') marker++;
    if ((size_t)(end - marker) < (size_t)marker_len ||
        memcmp(marker, end_marker, (size_t)marker_len) != 0)
        return NULL;

    const char *after_marker = marker + marker_len;
    const char *after_line = memchr(after_marker, '\n', (size_t)(end - after_marker));
    return after_line ? after_line + 1 : end;
}

static const char *trace_find_next_case(const char *start, const char *end) {
    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;

        const char *skip = trace_skip_counted_block(p, line_end, end);
        if (skip && skip > p) {
            p = skip;
            continue;
        }

        const char *case_marker = "===== CASE ";
        size_t marker_len = strlen(case_marker);
        if ((size_t)(line_end - p) >= marker_len &&
            !memcmp(p, case_marker, marker_len))
            return p;

        p = line_end < end ? line_end + 1 : end;
    }
    return NULL;
}

static const char *trace_find_block_begin(const char *start, const char *end,
                                          const char *label) {
    size_t label_len = strlen(label);
    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;

        if ((size_t)(line_end - p) >= label_len &&
            !memcmp(p, label, label_len))
            return p;

        const char *skip = trace_skip_counted_block(p, line_end, end);
        if (skip && skip > p) {
            p = skip;
            continue;
        }

        p = line_end < end ? line_end + 1 : end;
    }
    return NULL;
}

static bool trace_get_line_field(const char *start, const char *end,
                                 const char *key, char *dst, size_t dstlen) {
    size_t keylen = strlen(key);
    if (dstlen > 0) dst[0] = '\0';

    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;
        if ((size_t)(line_end - p) >= keylen && !memcmp(p, key, keylen)) {
            copy_span(dst, dstlen, p + keylen, line_end);
            return true;
        }
        p = line_end < end ? line_end + 1 : end;
    }
    return false;
}

static const eval_case *find_eval_case_by_source_id(const char *source,
                                                    const char *id) {
    for (size_t i = 0; i < eval_core_case_count(); i++) {
        if (eval_core_cases[i].source && eval_core_cases[i].id &&
            !strcmp(eval_core_cases[i].source, source) &&
            !strcmp(eval_core_cases[i].id, id))
            return &eval_core_cases[i];
    }
    for (size_t i = 0; i < eval_hard_case_count; i++) {
        if (eval_hard_cases[i].source && eval_hard_cases[i].id &&
            !strcmp(eval_hard_cases[i].source, source) &&
            !strcmp(eval_hard_cases[i].id, id))
            return &eval_hard_cases[i];
    }
    return NULL;
}

static char *trace_copy_model_output(const char *case_start, const char *case_end) {
    const char *begin = trace_find_block_begin(case_start, case_end, "MODEL_OUTPUT_BEGIN");
    if (!begin) return NULL;
    const char *line_end = memchr(begin, '\n', (size_t)(case_end - begin));
    if (!line_end) return NULL;
    const char *content = line_end + 1;

    size_t len = 0;
    const char *bytes = bounded_strstr(begin, line_end, "bytes=");
    if (bytes) {
        char *endptr = NULL;
        unsigned long long declared = strtoull(bytes + 6, &endptr, 10);
        if (endptr == bytes + 6 || endptr > line_end ||
            declared > (unsigned long long)(case_end - content))
            return NULL;
        len = (size_t)declared;
        const char *marker = content + len;
        if (marker < case_end && *marker == '\n') marker++;
        if ((size_t)(case_end - marker) < strlen("MODEL_OUTPUT_END") ||
            memcmp(marker, "MODEL_OUTPUT_END", strlen("MODEL_OUTPUT_END")) != 0)
            return NULL;
    } else {
        const char *finish = bounded_strstr(content, case_end, "\nMODEL_OUTPUT_END");
        if (!finish) return NULL;
        len = (size_t)(finish - content);
    }

    char *out = malloc(len + 1);
    if (!out) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        return NULL;
    }
    memcpy(out, content, len);
    out[len] = '\0';
    return out;
}


/* Decide how one trace case regrades. Only PASSED/FAILED traces were graded by
 * a completed run; STOPPED/SKIPPED/SWITCHED/ERROR cases carry partial or no
 * model output and must be reported as not-graded rather than counted, since
 * grading them inflates the totals and raises spurious "changed" drift. An
 * empty status keeps legacy traces (written before the status line) working. */
static regrade_outcome regrade_case_outcome(const eval_case *tc,
                                            const char *traced_status,
                                            const char *model_output,
                                            char *got, size_t gotlen) {
    bool gradeable = traced_status[0] == '\0' ||
                     !strcmp(traced_status, "PASSED") ||
                     !strcmp(traced_status, "FAILED");
    if (!gradeable) {
        if (gotlen) got[0] = '\0';
        return REGRADE_NOT_GRADED;
    }
    find_case_answer(tc, model_output, got, gotlen);
    return answer_matches(tc, got) ? REGRADE_PASSED : REGRADE_FAILED;
}

static int regrade_trace_file(const char *path) {
    size_t len = 0;
    char *text = read_text_file(path, &len);
    if (!text) return 2;

    const char *start = text;
    const char *end = text + len;
    int total = 0;
    int passed = 0;
    int failed = 0;
    int changed = 0;
    int unknown = 0;
    int parse_errors = 0;
    int not_graded = 0;

    while (true) {
        const char *case_start = trace_find_next_case(start, end);
        if (!case_start) break;
        const char *after_header = memchr(case_start, '\n', (size_t)(end - case_start));
        after_header = after_header ? after_header + 1 : end;
        const char *case_end = trace_find_next_case(after_header, end);
        if (!case_end) case_end = end;
        start = case_end;
        total++;

        char source[64];
        char id[128];
        char traced_status[32];
        char traced_pick[EVAL_ANSWER_MAX];
        if (!trace_get_line_field(case_start, case_end, "source: ", source, sizeof(source)) ||
            !trace_get_line_field(case_start, case_end, "id: ", id, sizeof(id))) {
            fprintf(stderr, "ds4-eval: trace case %d is missing source/id\n", total);
            parse_errors++;
            continue;
        }
        trace_get_line_field(case_start, case_end, "status: ", traced_status, sizeof(traced_status));
        trace_get_line_field(case_start, case_end, "picked: ", traced_pick, sizeof(traced_pick));

        const eval_case *tc = find_eval_case_by_source_id(source, id);
        if (!tc) {
            fprintf(stderr, "ds4-eval: trace case %d not found in embedded cases: %s/%s\n",
                    total, source, id);
            unknown++;
            continue;
        }

        char *model_output = trace_copy_model_output(case_start, case_end);
        if (!model_output) {
            fprintf(stderr, "ds4-eval: trace case %d is missing MODEL_OUTPUT block: %s/%s\n",
                    total, source, id);
            parse_errors++;
            continue;
        }

        char got[EVAL_ANSWER_MAX];
        regrade_outcome outcome =
            regrade_case_outcome(tc, traced_status, model_output, got, sizeof(got));
        if (outcome == REGRADE_NOT_GRADED) {
            not_graded++;
            free(model_output);
            continue;
        }
        bool ok = (outcome == REGRADE_PASSED);
        if (ok) passed++;
        else failed++;

        bool traced_ok = !strcmp(traced_status, "PASSED");
        if ((traced_status[0] && ok != traced_ok) ||
            (traced_pick[0] && strcmp(got, traced_pick) != 0)) {
            changed++;
            printf("case %d %s/%s: trace %s picked=%s -> regrade %s picked=%s expected=%s\n",
                   total, source, id,
                   traced_status[0] ? traced_status : "?",
                   traced_pick[0] ? traced_pick : "?",
                   ok ? "PASSED" : "FAILED", got, tc->answer ? tc->answer : "?");
        }
        free(model_output);
    }

    printf("ds4-eval: regraded %d cases from %s: passed=%d failed=%d changed=%d not_graded=%d unknown=%d parse_errors=%d\n",
           total, path, passed, failed, changed, not_graded, unknown, parse_errors);
    free(text);
    return (unknown || parse_errors || total == 0) ? 1 : 0;
}

#include "ds4_eval_selftests.inc"

static bool tui_has_switch_request(eval_ui *ui, int running_idx) {
    return ui->enabled &&
           ui->requested_case >= 0 &&
           ui->requested_case < ui->ncases &&
           ui->requested_case != running_idx;
}

static bool tui_has_quit_request(eval_ui *ui) {
    return ui->enabled && ui->quit_requested;
}

static void mark_case_pending(eval_ui *ui, int idx) {
    ui->status[idx] = EVAL_PENDING;
    ui->guess[idx][0] = '\0';
}

static double tui_wait_if_paused(eval_ui *ui, const char *phase) {
    if (!ui->enabled || !ui->paused) return 0.0;
    bool was_running = ui->run_clock_active;
    if (was_running) tui_run_clock_stop(ui);
    double start = now_sec();
    tui_refresh(ui, phase);
    while (ui->paused) {
        usleep(50000);
        tui_consume_input(ui);
        tui_refresh(ui, phase);
    }
    if (was_running) tui_run_clock_start(ui);
    tui_refresh(ui, phase);
    return now_sec() - start;
}

static void eval_prefill_progress(void *ud, const char *event, int current, int total) {
    eval_ui *ui = ud;
    if (!ui || !event) return;
    if (strcmp(event, "prefill_chunk") && strcmp(event, "prefill_display"))
        return;
    tui_consume_input(ui);
    tui_run_clock_tick(ui);
    ui->prefill_current = current;
    ui->prefill_total = total;
    double elapsed = now_sec() - ui->phase_start_sec;
    ui->speed_tps = elapsed > 0.001 ? (double)current / elapsed : 0.0;
    tui_refresh(ui, "prefill");
    double paused_sec = tui_wait_if_paused(ui, "prefill");
    if (paused_sec > 0.0) ui->phase_start_sec += paused_sec;
}

static eval_run_result run_one_case(ds4_engine *engine, ds4_session *session,
                                    const eval_config *cfg, eval_ui *ui,
                                    FILE *trace, int idx, int generation_budget,
                                    uint64_t *rng) {
    const eval_case *tc = &ui->cases[idx];
    const bool tty = ui->enabled;
    const bool use_plain_color = !tty && isatty(STDOUT_FILENO);
    const ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size);
    const char *system = eval_system_prompt();

    char *question = build_question_prompt(tc);
    if (!question) {
        fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
        return EVAL_RUN_ERROR;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, system, question, think_mode, &prompt);
    ui->prompt_tokens[idx] = prompt.len;
    ui->generated_tokens[idx] = 0;

    if (prompt.len >= cfg->ctx_size) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt does not fit context", system, question, "",
                         think_mode, prompt.len, 0, generation_budget, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    int generation_limit = generation_budget;
    int ctx_generation_limit = cfg->ctx_size - prompt.len;
    if (ctx_generation_limit < generation_limit) generation_limit = ctx_generation_limit;
    if (generation_limit < 1) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt leaves no generation room", system, question, "",
                         think_mode, prompt.len, 0, generation_limit, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    ui->active_case = idx;
    if (!ui->selection_active) ui->selected_case = idx;
    ui->requested_case = -1;
    ui->guess[idx][0] = '\0';
    ui->status[idx] = EVAL_PREFILL;
    ui->max_tokens = generation_limit;
    ui->think_max_tokens = generation_limit - cfg->hard_limit_reply_budget;
    if (ui->think_max_tokens < 0) ui->think_max_tokens = 0;
    tui_run_clock_start(ui);
    if (tty) {
        tui_reset_stream(ui, tc, ds4_think_mode_enabled(think_mode));
        tui_refresh(ui, "prefill");
    } else {
        printf("\n%s[%d/%d] %s%s%s/%s%s%s %s%s\n",
               use_plain_color ? ANSI_BOLD : "",
               idx + 1, ui->ncases,
               use_plain_color ? ANSI_CYAN : "", tc->source,
               use_plain_color ? ANSI_RESET : "",
               use_plain_color ? ANSI_CYAN : "", tc->domain,
               use_plain_color ? ANSI_RESET : "", tc->title,
               use_plain_color ? ANSI_RESET : "");
        fflush(stdout);
    }

    char err[256];
    ds4_session_set_progress(session, eval_prefill_progress, ui);
    ds4_session_set_display_progress(session, eval_prefill_progress, ui);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        tui_run_clock_stop(ui);
        fprintf(stderr, "ds4-eval: prefill failed for %s: %s\n", tc->id, err);
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                         system, question, "", think_mode, prompt.len, 0,
                         generation_limit, 0.0, "?", NULL);
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_ERROR;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    int prompt_tokens = prompt.len;
    ui->prompt_tokens[idx] = prompt_tokens;
    ds4_tokens_free(&prompt);

    tui_consume_input(ui);
    tui_wait_if_paused(ui, "prefill");
    if (tui_has_quit_request(ui)) {
        ui->status[idx] = EVAL_STOPPED;
        ui->generated_tokens[idx] = 0;
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0,
                         generation_limit, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_QUIT;
    }
    if (tui_has_switch_request(ui, idx)) {
        mark_case_pending(ui, idx);
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0,
                         generation_limit, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_SWITCH;
    }

    ui->status[idx] = EVAL_THINKING;
    ui->generated = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;
    byte_buf raw = {0};
    bool plain_in_think = ds4_think_mode_enabled(think_mode);
    bool generation_in_think = ds4_think_mode_enabled(think_mode);
    eval_think_close_info think_close = {0};
    ds4_tokens think_close_tokens = {0};
    if (generation_in_think) ds4_tokenize_text(engine, "</think>", &think_close_tokens);
    if (!tty && plain_in_think) plain_set_thinking_color(use_plain_color);
    tui_refresh(ui, "thinking");

    double t0 = ui->phase_start_sec;
    int forced_close_pos = -1;
    for (int i = 0; i < generation_limit; i++) {
        if (tty) {
            tui_consume_input(ui);
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, generation_limit,
                                 now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, generation_limit,
                                 now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
            double paused_sec = tui_wait_if_paused(ui, ui->in_think ? "thinking" : "answer");
            if (paused_sec > 0.0) {
                ui->phase_start_sec += paused_sec;
                t0 += paused_sec;
            }
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, generation_limit,
                                 now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, generation_limit,
                                 now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
        }

        int remaining_budget = generation_limit - ui->generated;
        int close_rank = 0;
        int token = -1;
        eval_think_close_kind close_kind = EVAL_THINK_CLOSE_NONE;

        /* Benchmarks usually cap generation length, but DeepSeek can spend the
         * entire budget in <think>.  This controller only acts while the model is
         * still in thinking mode.  The soft limit is conservative: it accepts the
         * model's own desire to end thinking when </think> is already near the
         * top of the distribution.  The hard limit is a uniform benchmark rule:
         * leave a fixed answer reserve instead of failing a case because all
         * tokens were spent before the visible answer could start. */
        if (generation_in_think && think_close_tokens.len > 0) {
            if (forced_close_pos >= 0) {
                token = think_close_tokens.v[forced_close_pos++];
                if (forced_close_pos >= think_close_tokens.len) forced_close_pos = -1;
            } else if (remaining_budget <= cfg->hard_limit_reply_budget) {
                close_kind = EVAL_THINK_CLOSE_HARD;
                token = think_close_tokens.v[0];
                forced_close_pos = think_close_tokens.len > 1 ? 1 : -1;
            } else if (remaining_budget <= cfg->soft_limit_reply_budget &&
                       think_close_tokens.len == 1) {
                close_rank = token_rank_in_top(session, think_close_tokens.v[0],
                                               cfg->soft_limit_think_close_rank);
                if (close_rank > 0) {
                    close_kind = EVAL_THINK_CLOSE_SOFT;
                    token = think_close_tokens.v[0];
                }
            }
        }
        if (token < 0)
            token = ds4_session_sample(session, cfg->temperature, 0,
                                       cfg->top_p, cfg->min_p, rng);
        if (ds4_token_is_stop(engine, token)) break;
        if (close_kind != EVAL_THINK_CLOSE_NONE &&
            think_close.kind == EVAL_THINK_CLOSE_NONE) {
            think_close.kind = close_kind;
            think_close.token_index = ui->generated + 1;
            think_close.remaining_budget = remaining_budget;
            think_close.rank = close_rank;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            plain_reset_color(use_plain_color);
            ui->generated_tokens[idx] = ui->generated;
            tui_run_clock_stop(ui);
            fprintf(stderr, "ds4-eval: decode failed for %s: %s\n", tc->id, err);
            trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                             system, question, raw.v ? raw.v : "", think_mode,
                             prompt_tokens, ui->generated, generation_limit,
                             now_sec() - t0, "?",
                             &think_close);
            free(question);
            ds4_tokens_free(&think_close_tokens);
            buf_free(&raw);
            return EVAL_RUN_ERROR;
        }

        size_t len = 0;
        char *text = ds4_token_text(engine, token, &len);
        buf_append(&raw, text, len);
        ui->generated++;
        ui->generated_tokens[idx] = ui->generated;
        tui_run_clock_tick(ui);
        if (generation_in_think && raw.v && strstr(raw.v, "</think>")) {
            generation_in_think = false;
            if (think_close.kind == EVAL_THINK_CLOSE_NONE) {
                think_close.kind = EVAL_THINK_CLOSE_NATURAL;
                think_close.token_index = ui->generated;
                think_close.remaining_budget = remaining_budget;
                think_close.rank = 0;
            }
        }
        double elapsed = now_sec() - ui->phase_start_sec;
        ui->speed_tps = elapsed > 0.001 ? (double)ui->generated / elapsed : 0.0;

        if (tty) {
            stream_append_token_text(ui, text, len, false);
            tui_refresh(ui, ui->in_think ? "thinking" : "answer");
        } else {
            if (plain_in_think && strstr(raw.v ? raw.v : "", "</think>")) {
                plain_in_think = false;
                plain_reset_color(use_plain_color);
            }
            fwrite(text, 1, len, stdout);
            fflush(stdout);
        }
        free(text);
    }
    if (tty) {
        stream_append_token_text(ui, NULL, 0, true);
        tui_draw_stream(ui);
    } else {
        plain_reset_color(use_plain_color);
        if (!raw.v || raw.len == 0 || raw.v[raw.len - 1] != '\n') fputc('\n', stdout);
    }

    char got[EVAL_ANSWER_MAX];
    bool complete = find_case_answer_strict(tc, raw.v ? raw.v : "",
                                            got, sizeof(got));
    snprintf(ui->guess[idx], EVAL_ANSWER_MAX, "%s", got);
    bool pass = complete && answer_matches(tc, got);
    ui->status[idx] = !complete ? EVAL_INCOMPLETE :
                      (pass ? EVAL_PASSED : EVAL_FAILED);
    ui->generated_tokens[idx] = ui->generated;
    tui_run_clock_stop(ui);
    double sec = now_sec() - t0;
    const char *result_name = !complete ? "INCOMPLETE" :
                              (pass ? "PASSED" : "FAILED");
    tui_refresh(ui, !complete ? "incomplete" : (pass ? "passed" : "failed"));
    trace_write_case(trace, cfg, tc, idx, ui->ncases, result_name, NULL,
                     system, question, raw.v ? raw.v : "", think_mode, prompt_tokens,
                     ui->generated, generation_limit, sec, got, &think_close);

    if (!tty) {
        printf("%s%s%s got %s expected %s (%.1fs, %d tokens)\n",
               use_plain_color ? (!complete ? ANSI_YELLOW :
                                  (pass ? ANSI_GREEN : ANSI_RED)) : "",
               !complete ? "INCOMPLETE" : (pass ? "PASSED" : "FAIL"),
               use_plain_color ? ANSI_RESET : "",
               got, tc->answer, sec, ui->generated);
    }

    if (tty && cfg->pause_ms > 0) usleep((useconds_t)cfg->pause_ms * 1000);
    free(question);
    ds4_tokens_free(&think_close_tokens);
    buf_free(&raw);
    return EVAL_RUN_OK;
}

static int next_pending_case(eval_ui *ui, int start) {
    if (ui->ncases <= 0) return -1;
    for (int off = 0; off < ui->ncases; off++) {
        int i = (start + off) % ui->ncases;
        if (ui->status[i] == EVAL_PENDING) return i;
    }
    return -1;
}

static int parse_case_sequence(const char *arg, int ncases, int **seq_out, int *len_out) {
    int cap = 8;
    int len = 0;
    int *seq = malloc((size_t)cap * sizeof(*seq));
    if (!seq) {
        fprintf(stderr, "ds4-eval: out of memory while parsing --case-sequence\n");
        return -1;
    }

    const char *p = arg;
    while (p && *p) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        errno = 0;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || errno != 0 || v < 1 || v > ncases) {
            fprintf(stderr,
                    "ds4-eval: invalid --case-sequence entry near '%s' "
                    "(valid range: 1..%d)\n",
                    p, ncases);
            free(seq);
            return -1;
        }
        if (len == cap) {
            cap *= 2;
            int *new_seq = realloc(seq, (size_t)cap * sizeof(*seq));
            if (!new_seq) {
                fprintf(stderr, "ds4-eval: out of memory while parsing --case-sequence\n");
                free(seq);
                return -1;
            }
            seq = new_seq;
        }
        seq[len++] = (int)v - 1;
        p = end;
        while (isspace((unsigned char)*p)) p++;
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p) {
            fprintf(stderr, "ds4-eval: expected comma in --case-sequence near '%s'\n", p);
            free(seq);
            return -1;
        }
    }
    if (len == 0) {
        fprintf(stderr, "ds4-eval: --case-sequence cannot be empty\n");
        free(seq);
        return -1;
    }
    *seq_out = seq;
    *len_out = len;
    return 0;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-eval: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-eval: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-eval: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-eval: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static const char *report_status_name(eval_status st) {
    switch (st) {
    case EVAL_PASSED: return "PASSED";
    case EVAL_FAILED: return "FAILED";
    case EVAL_INCOMPLETE: return "INCOMPLETE";
    case EVAL_SKIPPED: return "SKIPPED";
    case EVAL_STOPPED: return "STOPPED";
    case EVAL_PREFILL: return "PREFILL";
    case EVAL_THINKING: return "RUNNING";
    case EVAL_PENDING:
    default: return "PENDING";
    }
}

static void print_eval_report(const eval_ui *ui, int ncases, int passed, int failed,
                              int incomplete) {
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));

    printf("ds4-eval: %d/%d passed", passed, ncases);
    if (failed) printf(", %d failed", failed);
    if (incomplete) printf(", %d incomplete", incomplete);
    printf(", runtime %s\n", elapsed);
    printf("%-3s %-10s %8s %8s %8s %-20s %-20s %s\n",
           "#", "state", "prompt", "gen", "total", "given", "correct", "test");
    for (int i = 0; i < ncases; i++) {
        int prompt_tokens = ui->prompt_tokens ? ui->prompt_tokens[i] : 0;
        int generated_tokens = ui->generated_tokens ? ui->generated_tokens[i] : 0;
        int total_tokens = prompt_tokens + generated_tokens;
        const char *given = ui->guess && ui->guess[i][0] ? ui->guess[i] : "-";
        printf("%3d %-10s %8d %8d %8d %-20.20s %-20.20s %s/%s\n",
               i + 1,
               report_status_name(ui->status[i]),
               prompt_tokens,
               generated_tokens,
               total_tokens,
               given,
               ui->cases[i].answer,
               ui->cases[i].source,
               ui->cases[i].id);
    }
}

static const eval_case *eval_case_at(size_t index, bool *hard_out) {
    size_t core_count = eval_core_case_count();
    if (index < core_count) {
        if (hard_out) *hard_out = false;
        return &eval_core_cases[index];
    }
    if (hard_out) *hard_out = true;
    return &eval_hard_cases[index - core_count];
}

static bool eval_source_is_declared(const char *source) {
    for (size_t i = 0; i < eval_hard_source_count; i++)
        if (!strcmp(source, eval_hard_sources[i].name)) return true;
    return false;
}

static int validate_eval_cases(bool verbose) {
    const size_t core_count = eval_core_case_count();
    const size_t total = core_count + eval_hard_case_count;
    int errors = 0;
    size_t smoke_count = 0;

    for (size_t i = 0; i < total; i++) {
        bool hard = false;
        const eval_case *tc = eval_case_at(i, &hard);
        const char *where = hard ? "hard" : "core";
#define REQUIRE_CASE_FIELD(field) do { \
        if (!tc->field || !tc->field[0]) { \
            fprintf(stderr, "ds4-eval: %s case %zu has no %s\n", \
                    where, i + 1, #field); \
            errors++; \
        } \
    } while (0)
        REQUIRE_CASE_FIELD(source);
        REQUIRE_CASE_FIELD(id);
        REQUIRE_CASE_FIELD(domain);
        REQUIRE_CASE_FIELD(title);
        REQUIRE_CASE_FIELD(question);
        REQUIRE_CASE_FIELD(answer);
#undef REQUIRE_CASE_FIELD

        if (hard && (tc->suites & EVAL_SUITE_HARD_SMOKE)) smoke_count++;
        if (!tc->source || !tc->id || !tc->answer) continue;
        if (tc->answer_kind < EVAL_ANSWER_AUTO ||
            tc->answer_kind > EVAL_ANSWER_LINE_SET) {
            fprintf(stderr, "ds4-eval: %s/%s has an invalid answer kind\n",
                    tc->source, tc->id);
            errors++;
            continue;
        }

        int nchoices = eval_case_nchoices(tc);
        bool saw_gap = false;
        for (int c = 0; c < EVAL_MAX_CHOICES; c++) {
            if (!tc->choice[c]) saw_gap = true;
            else if (saw_gap) {
                fprintf(stderr, "ds4-eval: %s/%s has a gap in its choices\n",
                        tc->source, tc->id);
                errors++;
                break;
            }
        }

        eval_answer_kind kind = eval_case_answer_kind(tc);
        if (kind == EVAL_ANSWER_CHOICE) {
            if (nchoices < 2 || strlen(tc->answer) != 1 ||
                tc->answer[0] < 'A' || tc->answer[0] >= 'A' + nchoices) {
                fprintf(stderr, "ds4-eval: %s/%s has an invalid choice key\n",
                        tc->source, tc->id);
                errors++;
            }
        } else if (nchoices != 0) {
            fprintf(stderr, "ds4-eval: %s/%s has choices but is not choice-graded\n",
                    tc->source, tc->id);
            errors++;
        } else if (kind == EVAL_ANSWER_LINE_SET) {
            bool lines[256] = {0};
            if (!parse_line_spec(tc->answer, lines, sizeof(lines))) {
                fprintf(stderr, "ds4-eval: %s/%s has an invalid line-set key\n",
                        tc->source, tc->id);
                errors++;
            }
        } else {
            char normalized[EVAL_ANSWER_MAX];
            normalize_expected_answer(kind, tc->answer,
                                      normalized, sizeof(normalized));
            if (!normalized[0] || !strcmp(normalized, "?")) {
                fprintf(stderr, "ds4-eval: %s/%s has an invalid answer key\n",
                        tc->source, tc->id);
                errors++;
            }
        }

        if (hard) {
            if (tc->answer_kind == EVAL_ANSWER_AUTO ||
                !(tc->suites & EVAL_SUITE_HARD) || tc->max_tokens == 0 ||
                !eval_source_is_declared(tc->source)) {
                fprintf(stderr,
                        "ds4-eval: hard case %s/%s lacks type, suite, budget, or source metadata\n",
                        tc->source, tc->id);
                errors++;
            }
        }

        for (size_t j = i + 1; j < total; j++) {
            const eval_case *other = eval_case_at(j, NULL);
            if (other->source && other->id &&
                !strcmp(tc->source, other->source) && !strcmp(tc->id, other->id)) {
                fprintf(stderr, "ds4-eval: duplicate case ID %s/%s\n",
                        tc->source, tc->id);
                errors++;
            }
        }
    }

    if (eval_hard_case_count != 50 || smoke_count != 12) {
        fprintf(stderr,
                "ds4-eval: hard suite shape changed: cases=%zu smoke=%zu (expected 50/12)\n",
                eval_hard_case_count, smoke_count);
        errors++;
    }
    if (verbose || errors)
        printf("ds4-eval: validated %zu core and %zu hard cases (%zu hard-smoke): %s\n",
               core_count, eval_hard_case_count, smoke_count,
               errors ? "FAILED" : "ok");
    return errors ? 1 : 0;
}

static void list_eval_cases(const eval_case *cases, int ncases) {
    printf("#\tsource\tdomain\tid\tanswer-kind\ttitle\n");
    for (int i = 0; i < ncases; i++) {
        printf("%d\t%s\t%s\t%s\t%s\t%s\n",
               i + 1, cases[i].source, cases[i].domain, cases[i].id,
               eval_answer_kind_name(eval_case_answer_kind(&cases[i])),
               cases[i].title);
    }
}

int main(int argc, char **argv) {
    eval_config cfg = parse_options(argc, argv);
    if (cfg.self_test_extractors) return run_extractor_self_tests();
    if (cfg.validate_cases) return validate_eval_cases(true);
    if (cfg.regrade_trace_path) return regrade_trace_file(cfg.regrade_trace_path);

    int ncases = 0;
    eval_case *cases = select_eval_cases(&cfg, &ncases);
    if (!cases) return 2;
    if (cfg.list_cases) {
        list_eval_cases(cases, ncases);
        free(cases);
        return 0;
    }
    int *case_sequence = NULL;
    int case_sequence_len = 0;
    if (cfg.case_sequence &&
        parse_case_sequence(cfg.case_sequence, ncases, &case_sequence, &case_sequence_len) != 0) {
        free(cases);
        return 2;
    }
    if (!cfg.seed) {
        cfg.seed = (uint64_t)time(NULL) ^
                   ((uint64_t)getpid() << 32) ^
                   (uint64_t)clock();
    }

    FILE *trace = NULL;
    if (cfg.trace_path) {
        trace = fopen(cfg.trace_path, "w");
        if (!trace) {
            fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                    cfg.trace_path, strerror(errno));
            free(case_sequence);
            free(cases);
            return 2;
        }
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_size > 0 ? cfg.ctx_size : 0,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .power_percent = cfg.power_percent,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .distributed = cfg.dist,
        .tp = cfg.tp,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-eval: %s\n", dist_err);
        if (trace) fclose(trace);
        free(case_sequence);
        free(cases);
        return 2;
    }
    char tp_err[256];
    if (!ds4_tp_validate_engine_options(&opt, tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-eval: %s\n", tp_err);
        if (trace) fclose(trace);
        free(case_sequence);
        free(cases);
        return 2;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        if (trace) fclose(trace);
        free(case_sequence);
        free(cases);
        return 1;
    }

    int max_prompt_tokens = 0;
    int max_prompt_case = -1;
    int max_generation_tokens = eval_max_generation_budget(&cfg, cases, ncases);
    const bool auto_ctx = cfg.ctx_size <= 0;
    if (auto_ctx) {
        cfg.ctx_size = eval_auto_context_size(engine, &cfg, cases, ncases,
                                              max_generation_tokens,
                                              &max_prompt_tokens, &max_prompt_case);
        fprintf(stderr,
                "ds4-eval: context auto-sized to %d tokens "
                "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
                cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1,
                max_generation_tokens);
    } else {
        max_prompt_tokens = eval_max_prompt_tokens(engine, &cfg, cases, ncases,
                                                   cfg.ctx_size, &max_prompt_case);
        fprintf(stderr,
                "ds4-eval: context set to %d tokens "
                "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
                cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1,
                max_generation_tokens);
        eval_warn_context_budget(&cfg, max_prompt_tokens, max_prompt_case,
                                 max_generation_tokens);
    }
    fprintf(stderr, "ds4-eval: model shape %s\n", ds4_engine_model_name(engine));
    eval_warn_think_max_downgraded(&cfg);
    trace_write_header(trace, &cfg, ds4_engine_model_name(engine), ncases,
                       max_prompt_tokens, max_generation_tokens);
    log_context_memory(cfg.backend,
                       cfg.ctx_size,
                       cfg.prefill_chunk,
                       cfg.ssd_streaming);

    ds4_tp *tp_leader = NULL;
    if (cfg.tp.role == DS4_TP_LEADER) {
        ds4_tp_identity tp_id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)cfg.ctx_size,
        };
        ds4_engine_tp_gate_schedule(engine,
                                    &tp_id.gate_slot_start,
                                    &tp_id.gate_slot_step,
                                    &tp_id.gates_per_token,
                                    tp_id.gate_slot_mask);
        if (!ds4_tp_create(&tp_leader, &cfg.tp, &tp_id,
                           tp_err, sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp_leader, tp_err, sizeof(tp_err))) {
            fprintf(stderr, "ds4-eval: %s\n", tp_err);
            ds4_tp_free(tp_leader);
            if (trace) fclose(trace);
            ds4_engine_close(engine);
            free(case_sequence);
            free(cases);
            return 1;
        }
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        fprintf(stderr, "ds4-eval: failed to create session\n");
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        ds4_tp_free(tp_leader);
        free(case_sequence);
        free(cases);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        ds4_tp_free(tp_leader);
        free(case_sequence);
        free(cases);
        return 1;
    }

    eval_ui ui;
    bool split_ui = !cfg.plain && isatty(STDOUT_FILENO);
    tui_start(&ui, cases, ncases, max_generation_tokens, split_ui);

    uint64_t rng = cfg.seed;
    int rc = 0;
    int next = 0;
    int sequence_pos = 0;
    while (next >= 0) {
        if (case_sequence_len > 0) {
            if (sequence_pos >= case_sequence_len) break;
            next = case_sequence[sequence_pos++];
            ui.selected_case = next;
            ui.selection_active = false;
        }
        tui_consume_input(&ui);
        tui_wait_if_paused(&ui, "idle");
        if (ui.quit_requested) break;
        if (case_sequence_len == 0 && ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
        }

        int generation_budget = eval_case_generation_budget(&cfg, &cases[next], false);
        eval_run_result result = run_one_case(engine, session, &cfg, &ui, trace,
                                              next, generation_budget, &rng);
        if (result == EVAL_RUN_OK && ui.status[next] == EVAL_INCOMPLETE &&
            cfg.retry_incomplete) {
            int retry_budget = eval_case_generation_budget(&cfg, &cases[next], true);
            fprintf(stderr,
                    "ds4-eval: retrying incomplete case %s/%s with %d tokens\n",
                    cases[next].source, cases[next].id, retry_budget);
            result = run_one_case(engine, session, &cfg, &ui, trace, next,
                                  retry_budget, &rng);
        }
        if (result == EVAL_RUN_ERROR) {
            rc = 1;
            break;
        }
        if (result == EVAL_RUN_QUIT) break;
        /* A successful case should advance to the next pending benchmark.  If a
         * stale quit flag ever survives here, clear it; real q handling either
         * returns EVAL_RUN_QUIT from run_one_case() or is consumed at the top of
         * the next idle iteration. */
        ui.quit_requested = false;
        if (case_sequence_len > 0) continue;
        if (ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
            continue;
        }
        next = next_pending_case(&ui, next + 1);
        if (next >= 0) {
            ui.selected_case = next;
            ui.selection_active = false;
        }
    }

    int passed = 0;
    int failed = 0;
    int incomplete = 0;
    for (int i = 0; i < ncases; i++) {
        if (ui.status[i] == EVAL_PASSED) passed++;
        else if (ui.status[i] == EVAL_FAILED) failed++;
        else if (ui.status[i] == EVAL_INCOMPLETE) incomplete++;
    }

    if (ui.active) tui_restore();
    print_eval_report(&ui, ncases, passed, failed, incomplete);
    if (trace) {
        fprintf(trace,
                "===== SUMMARY =====\n"
                "passed: %d\n"
                "failed: %d\n"
                "incomplete: %d\n"
                "total: %d\n",
                passed, failed, incomplete, ncases);
        fflush(trace);
    }

    tui_free(&ui);
    ds4_session_free(session);
    if (tp_leader) ds4_tp_send_stop(tp_leader);
    ds4_engine_close(engine);
    ds4_tp_free(tp_leader);
    if (trace) fclose(trace);
    free(case_sequence);
    free(cases);
    return rc || failed || incomplete ? 1 : 0;
}
