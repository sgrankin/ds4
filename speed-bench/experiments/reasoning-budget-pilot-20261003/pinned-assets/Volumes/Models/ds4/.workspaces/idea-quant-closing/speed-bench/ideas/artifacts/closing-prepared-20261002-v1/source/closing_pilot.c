#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "overthink_diagnostics.h"
#include "closing_policy.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define PILOT_TEMPERATURE 0.6f

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', fp); fputc(c, fp); }
        else if (c < 32) fprintf(fp, "\\u%04x", c);
        else fputc(c, fp);
    }
    fputc('"', fp);
}

static int exact_answer(const char *answer, const char *expected) {
    return !strcmp(answer, expected);
}

typedef enum { ANSWER_INTEGER, ANSWER_FRACTION, ANSWER_LIST, ANSWER_INT_PAIR, ANSWER_TEXT } answer_kind;

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    return s;
}

/* Recognize a complete final answer, never a matching substring. A narrow
 * enclosing wrapper is allowed; extra prose/multiple proposed answers are
 * INVALID and remain available in the raw trace for manual review. */
static char *answer_body(char *s) {
    s = trim(s);
    size_t n = strlen(s);
    if (n >= 6 && !strncmp(s, "```", 3) && !strcmp(s + n - 3, "```")) {
        s[n - 3] = '\0';
        s += 3;
        if (!strncmp(s, "python\n", 7)) s += 7;
        else if (!strncmp(s, "text\n", 5)) s += 5;
        s = trim(s); n = strlen(s);
    }
    if (n >= 4 && !strncmp(s, "**", 2) && !strcmp(s + n - 2, "**")) {
        s[n - 2] = '\0'; s = trim(s + 2); n = strlen(s);
    }
    if (n >= 2 && s[0] == '`' && s[n - 1] == '`') {
        s[n - 1] = '\0'; s = trim(s + 1); n = strlen(s);
    }
    if (n >= 9 && !strncmp(s, "\\boxed{", 7) && s[n - 1] == '}') {
        s[n - 1] = '\0'; s = trim(s + 7);
    }
    return s;
}

static int integer_part(const char **p, long long *value) {
    while (isspace((unsigned char)**p)) (*p)++;
    if (!(isdigit((unsigned char)**p) || **p == '+' || **p == '-')) return 0;
    char *end = NULL;
    errno = 0;
    *value = strtoll(*p, &end, 10);
    if (errno || end == *p) return 0;
    *p = end;
    return 1;
}

static int parse_answer(char *raw, answer_kind kind, char *canonical, size_t cap) {
    const char *s = answer_body(raw);
    if (!*s) return 0;
    long long a = 0, b = 0;
    if (kind == ANSWER_INTEGER || kind == ANSWER_FRACTION || kind == ANSWER_INT_PAIR) {
        if (!integer_part(&s, &a)) return 0;
        if (kind == ANSWER_FRACTION) {
            while (isspace((unsigned char)*s)) s++;
            if (*s++ != '/' || !integer_part(&s, &b) || b == 0) return 0;
        } else if (kind == ANSWER_INT_PAIR) {
            if (!isspace((unsigned char)*s) || !integer_part(&s, &b)) return 0;
        }
        while (isspace((unsigned char)*s)) s++;
        if (*s) return 0;
        if (kind == ANSWER_INTEGER) snprintf(canonical, cap, "%lld", a);
        else if (kind == ANSWER_FRACTION) snprintf(canonical, cap, "%lld/%lld", a, b);
        else snprintf(canonical, cap, "%lld %lld", a, b);
        return 1;
    }
    if (kind == ANSWER_LIST) {
        if (*s++ != '[') return 0;
        size_t used = 1; canonical[0] = '[';
        while (isspace((unsigned char)*s)) s++;
        if (*s != ']') {
            for (;;) {
                if (!integer_part(&s, &a)) return 0;
                char number[32]; snprintf(number, sizeof(number), "%lld", a);
                const size_t n = strlen(number);
                if (used + n + 2 >= cap) return 0;
                memcpy(canonical + used, number, n); used += n;
                while (isspace((unsigned char)*s)) s++;
                if (*s != ',') break;
                s++; canonical[used++] = ',';
            }
        }
        if (*s++ != ']') return 0;
        while (isspace((unsigned char)*s)) s++;
        if (*s || used + 2 > cap) return 0;
        canonical[used++] = ']'; canonical[used] = '\0';
        return 1;
    }
    /* The text task requests one exact printed line. Quotes around that line
     * are accepted, but a multiline explanation is not a parsed answer. */
    size_t n = strlen(s);
    if (n >= 2 && ((s[0] == '"' && s[n - 1] == '"') || (s[0] == '\'' && s[n - 1] == '\''))) { s++; n -= 2; }
    if (n >= cap || memchr(s, '\n', n) || memchr(s, '\r', n)) return 0;
    memcpy(canonical, s, n); canonical[n] = '\0';
    return n != 0;
}

static int score_self_test(void) {
    const struct { const char *raw, *expected; answer_kind kind; int valid; } checks[] = {
        {" 255 \n", "255", ANSWER_INTEGER, 1},
        {"**-4**", "-4", ANSWER_INTEGER, 1},
        {"\\boxed{2/11}", "2/11", ANSWER_FRACTION, 1},
        {"```python\n[-1, -1, 0, 4]\n```", "[-1,-1,0,4]", ANSWER_LIST, 1},
        {"-3 2", "-3 2", ANSWER_INT_PAIR, 1},
        {"`could but error`", "could but error", ANSWER_TEXT, 1},
        {"1255", "1255", ANSWER_INTEGER, 1},
        {"255 or 256", "", ANSWER_INTEGER, 0},
        {"answer: 255", "", ANSWER_INTEGER, 0},
        {"255 trailing", "", ANSWER_INTEGER, 0},
        {"[1,9,25] or [1,9]", "", ANSWER_LIST, 0},
        {"-32", "", ANSWER_INT_PAIR, 0},
        {"2/0", "", ANSWER_FRACTION, 0},
        {"", "", ANSWER_INTEGER, 0}
    };
    for (size_t i = 0; i < sizeof(checks) / sizeof(checks[0]); i++) {
        char raw[256], parsed[256];
        snprintf(raw, sizeof(raw), "%s", checks[i].raw);
        const int valid = parse_answer(raw, checks[i].kind, parsed, sizeof(parsed));
        if (valid != checks[i].valid || (valid && !exact_answer(parsed, checks[i].expected))) {
            fprintf(stderr, "score self-test failed at %zu\n", i); return 1;
        }
    }
    if (exact_answer("couldbuterror", "could but error") ||
        exact_answer("could  but error", "could but error") ||
        exact_answer("-32", "-3 2")) {
        fputs("score self-test failed: exact canonical comparison\n", stderr);
        return 1;
    }
    puts("strict final-answer scorer: 17 checks passed");
    return 0;
}

static int cache_gib_value(const char *text, double *gib) {
    char *end = NULL;
    if (!text || !*text || strspn(text, "0123456789.eE+-") != strlen(text)) return 0;
    errno = 0;
    const double value = strtod(text, &end);
    if (errno || end == text || *end || !(value > 0.0 && value <= 1048576.0)) return 0;
    if (value * 1073741824.0 < 1.0) return 0;
    *gib = value;
    return 1;
}

static int bounded_int(const char *text, int min, int max, int *out) {
    char *end = NULL;
    if (!text || !*text || strspn(text, "0123456789") != strlen(text)) return 0;
    errno = 0;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno || end == text || *end || value < (unsigned long)min || value > (unsigned long)max) return 0;
    *out = (int)value;
    return 1;
}

static void lambda_label(float penalty, char label[32]);

static int helper_self_test(void) {
    if (score_self_test()) return 1;
    int value;
    if (!bounded_int("0", 0, 5, &value) || value != 0 ||
        bounded_int("-1", 0, 5, &value) || bounded_int("6", 0, 5, &value) ||
        bounded_int("", 0, 5, &value) || bounded_int("1.0", 0, 5, &value)) return 1;
    const float logits[] = {0.0f, 1.0f, -2.0f};
    const int ids[] = {0, 2};
    for (int pi = 0; pi < 4; pi++) {
        const double penalty = (double[]){0, .25, .5, 1}[pi];
        const double t = (double)PILOT_TEMPERATURE;
        const double m = overthink_marker_mass(logits, 3, ids, 2, t);
        const double a = penalty / t;
        double p[3], q[3], sp = 0, sq = 0, reference_kl = 0;
        for (int i = 0; i < 3; i++) {
            p[i] = exp(logits[i] / t);
            q[i] = exp(logits[i] / t - a * overthink_is_marker(i, ids, 2));
            sp += p[i]; sq += q[i];
        }
        for (int i = 0; i < 3; i++) { p[i] /= sp; q[i] /= sq; reference_kl += p[i] * log(p[i] / q[i]); }
        for (int i = 0; i < 3; i++) {
            double kl, dnll;
            overthink_policy_metrics(m, a, overthink_is_marker(i, ids, 2), &kl, &dnll);
            if (fabs(kl - reference_kl) > 1e-12 || fabs(dnll - log(p[i] / q[i])) > 1e-12) return 1;
        }
    }
    uint32_t bad_bits[] = {UINT32_C(0x7f800000), UINT32_C(0xff800000), UINT32_C(0x7fc00000)};
    for (int i = 0; i < 3; i++) { float bad; memcpy(&bad, bad_bits + i, sizeof(bad)); if (overthink_finite(bad)) return 1; }
    char label[32];
    lambda_label(.25f, label); if (strcmp(label, "0.25")) return 1;
    lambda_label(.5f, label); if (strcmp(label, "0.5")) return 1;
    lambda_label(0, label); if (strcmp(label, "0.0")) return 1;
    lambda_label(1, label); if (strcmp(label, "1.0")) return 1;
    puts("case selection and pre-filtering KL/delta-NLL diagnostic checks passed");
    return 0;
}

static void lambda_label(float penalty, char label[32]) {
    snprintf(label, 32, "%.9g", penalty);
    if (!strchr(label, '.') && !strchr(label, 'e')) strcat(label, ".0");
}

static void diagnostic_argmax(const float *logits, int vocab, const int *ids,
                              int count, float effective_penalty,
                              int *raw_best, int *policy_best, double *marker_margin) {
    float raw_value = -INFINITY, policy_value = -INFINITY;
    float best_marker = -INFINITY, best_other = -INFINITY;
    *raw_best = *policy_best = -1;
    for (int i = 0; i < vocab; i++) {
        if (!overthink_finite(logits[i])) continue;
        const int marked = overthink_is_marker(i, ids, count);
        const float value = logits[i] - effective_penalty * marked;
        if (*raw_best < 0 || logits[i] > raw_value) { raw_value = logits[i]; *raw_best = i; }
        if (*policy_best < 0 || value > policy_value) { policy_value = value; *policy_best = i; }
        if (marked && logits[i] > best_marker) best_marker = logits[i];
        if (!marked && logits[i] > best_other) best_other = logits[i];
    }
    *marker_margin = (double)best_marker - best_other;
}

static int native_copy(void *session,float *out,int n) { return ds4_session_copy_logits(session,out,n); }
static int native_set(void *session,const float *in,int n) { return ds4_session_set_logits(session,in,n); }
static int native_sample(void *session,float t,int k,float p,float m,uint64_t *rng) {
    return ds4_session_sample(session,t,k,p,m,rng);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--score-self-test")) return score_self_test();
    if (argc == 2 && !strcmp(argv[1], "--helper-self-test")) return helper_self_test();
    double cache_gib = 64.0;
    const char *cache_text = getenv("DS4_PILOT_CACHE_GIB");
    const char *case_text = getenv("DS4_PILOT_CASE_LIMIT");
    const char *offset_text = getenv("DS4_PILOT_CASE_OFFSET");
    const char *diagnostic_text = getenv("DS4_PILOT_DIAGNOSTIC_STRIDE");
    const char *boost_text = NULL;
    for (int i = 1; i < argc; i++) {
        const int cache_option = !strcmp(argv[i], "--cache-gb");
        const int case_option = !strcmp(argv[i], "--cases");
        const int offset_option = !strcmp(argv[i], "--case-offset");
        const int diagnostic_option = !strcmp(argv[i], "--diagnostic-stride");
        const int boost_option = !strcmp(argv[i], "--closing-boost");
        if (!cache_option && !case_option && !offset_option && !diagnostic_option && !boost_option) continue;
        if (i + 1 >= argc) {
            fputs("pilot named options require a value\n", stderr); return 2;
        }
        if (cache_option) cache_text = argv[i + 1];
        else if (case_option) case_text = argv[i + 1];
        else if (offset_option) offset_text = argv[i + 1];
        else if (diagnostic_option) diagnostic_text = argv[i + 1];
        else { if (boost_text) return 2; boost_text = argv[i + 1]; }
        for (int j = i; j + 2 < argc; j++) argv[j] = argv[j + 2];
        argc -= 2; i--;
    }
    if (cache_text && !cache_gib_value(cache_text, &cache_gib)) {
        fputs("cache size must be a finite positive GiB size\n", stderr); return 2;
    }
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL.gguf RUN_NAME [MAX_TOKENS=512] [THINK_LEVEL=75] [calibration|heldout|hard|counting] [--closing-boost 0|2] [--cache-gb GIB] [--cases N] [--case-offset N] [--diagnostic-stride N]\n", argv[0]);
        return 2;
    }
    const int cap = argc > 3 ? atoi(argv[3]) : 512;
    const int level = argc > 4 ? atoi(argv[4]) : 75;
    const char *suite = argc > 5 ? argv[5] : "calibration";
    const int heldout = !strcmp(suite, "heldout");
    const int hard = !strcmp(suite, "hard");
    const int counting = !strcmp(suite, "counting");
    if (!heldout && !hard && !counting && strcmp(suite, "calibration")) return 2;
    if (cap < 1 || cap > 4096 || level < 1 || level > 100) return 2;
    const int suite_cases = counting ? 4 : hard ? 6 : heldout ? 8 : 4;
    int case_offset = 0, diagnostic_stride = 0;
    if (offset_text && !bounded_int(offset_text, 0, suite_cases - 1, &case_offset)) {
        fputs("case offset must be a zero-based integer within the selected suite\n", stderr); return 2;
    }
    if (diagnostic_text && !bounded_int(diagnostic_text, 0, 4096, &diagnostic_stride)) {
        fputs("diagnostic stride must be an integer from 0 (disabled) through 4096\n", stderr); return 2;
    }
    int case_limit = suite_cases - case_offset;
    if (case_text) {
        if (!bounded_int(case_text, 1, suite_cases - case_offset, &case_limit)) {
            fputs("case limit plus offset must fit within the selected suite\n", stderr); return 2;
        }
    }
    const float boost = !boost_text || !strcmp(boost_text,"0") ? 0.0f : !strcmp(boost_text,"2") ? 2.0f : -1.0f;
    if (boost < 0 || cap != 2048 || level != 25 || !counting || case_limit != 4 || case_offset != 0 || diagnostic_stride != 32 || cache_gib != 64.0) return 2;
    const int arms = 1;
    if (!argv[2][0] || strspn(argv[2], "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(argv[2])) {
        fputs("RUN_NAME must contain only letters, digits, underscores, and hyphens\n", stderr); return 2;
    }
    if (mkdir("speed-bench/ideas", 0755) != 0 && errno != EEXIST) { perror("create ideas directory"); return 2; }
    if (mkdir("speed-bench/ideas/artifacts", 0755) != 0 && errno != EEXIST) { perror("create artifacts directory"); return 2; }
    char output_dir[4096];
    if (snprintf(output_dir, sizeof(output_dir), "speed-bench/ideas/artifacts/%s", argv[2]) >= (int)sizeof(output_dir) - 128) return 2;
    if (mkdir(output_dir, 0755) != 0) { perror("create fresh persistent output directory"); return 2; }
    char path[4096];
    snprintf(path, sizeof(path), "%s/results.jsonl", output_dir);
    FILE *results = fopen(path, "wx");
    if (!results) { perror(path); return 2; }

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_METAL;
    opt.n_threads = 2;
    opt.context_size = 8192;
    opt.prefill_chunk = 512;
    opt.power_percent = 100;
    opt.ssd_streaming = true;
    opt.ssd_streaming_cache_bytes = (uint64_t)(cache_gib * 1073741824.0 + 0.5);
    opt.mtp_draft_tokens = 1;
    fprintf(stderr, "pilot persistent artifacts=%s explicit expert cache=%.6g GiB cases=%d/%d offset=%d diagnostic_stride=%d\n", output_dir, cache_gib, case_limit, suite_cases, case_offset, diagnostic_stride);
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, opt.context_size) != 0) return 1;
    ds4_session_gpu_warmup(session);
    int marker_ids[50] = {0}, marker_count = 0;
    const int vocab = ds4_engine_vocab_size(engine);
    float *raw_logits = NULL;
    float *policy_logits = malloc((size_t)vocab * sizeof(*policy_logits));
    if (!policy_logits) return 1;
    const closing_api api = {session,native_copy,native_set,native_sample};
    if (diagnostic_stride) {
        if (!ds4_engine_set_overthinking_penalty(engine, 0.0f, true)) return 1;
        marker_count = 1; marker_ids[0] = CLOSING_TOKEN_ID;
        size_t close_n=0; char *close_piece=ds4_token_text(engine,CLOSING_TOKEN_ID,&close_n);
        const int close_valid=closing_piece_valid(CLOSING_TOKEN_ID,close_piece,close_n);
        free(close_piece);
        if (!close_valid) { fputs("closing-token exact-piece control failed\n",stderr); return 1; }
        raw_logits = malloc((size_t)vocab * sizeof(*raw_logits));
        if (!raw_logits) return 1;
        snprintf(path, sizeof(path), "%s/marker-map.json", output_dir);
        FILE *map = fopen(path, "wx");
        if (!map) return 1;
        fputs("[", map);
        for (int i = 0; i < marker_count; i++) {
            if (i) fputs(",", map);
            fprintf(map, "{\"id\":%d,\"text\":", marker_ids[i]);
            char *piece = ds4_token_text(engine, marker_ids[i], NULL);
            json_string(map, piece); free(piece); fputs("}", map);
        }
        fputs("]\n", map); fclose(map);
    }
    const ds4_think_mode mode = (ds4_think_mode)(DS4_THINK_LEVEL_BASE + level);
    const int after = CLOSING_AFTER;
    const struct { const char *id, *prompt, *answer; answer_kind kind; } cases[] = {
        {"binary", "What is the largest eight-digit base-2 integer, in decimal? Put only the decimal integer in your final answer.", "255", ANSWER_INTEGER},
        {"percentage", "A price rises by 20 percent and then falls by 20 percent. Starting from 100, what is the final price? Put only the number in your final answer.", "96", ANSWER_INTEGER},
        {"dice", "Two fair six-sided dice are rolled. Given that at least one die is 6, what is the probability that their sum is 8? Put only the reduced fraction in your final answer.", "2/11", ANSWER_FRACTION},
        {"python", "Evaluate this Python expression: [x*x for x in range(6) if x%2]. Put only the resulting Python list in your final answer.", "[1,9,25]", ANSWER_LIST},
        {"correct_product", "A student claims 7 times 8 is 54. Correct the student's error, then subtract 60 from the correct product. Put only the signed integer in your final answer.", "-4", ANSWER_INTEGER},
        {"extraneous_root", "A colleague claims both -2 and 3 solve sqrt(x+6)=x over the real numbers. Check the original equation and discard any incorrect candidate. Put only the valid root in your final answer.", "3", ANSWER_INTEGER},
        {"correct_factors", "Positive integers a and b satisfy a+b=10 and a*b=21. Someone proposes a=2,b=8, but that could be an error. Correct the proposal. What is (a-b)^2? Put only the integer in your final answer.", "16", ANSWER_INTEGER},
        {"floor_and_remainder", "What does this Python code print? print(-7//3, -7%3). Put only the exact printed line in your final answer.", "-3 2", ANSWER_INT_PAIR},
        {"exception_text", "What does this Python code print?\ntry:\n    raise ValueError('could but error')\nexcept ValueError as error:\n    print(str(error))\nPut only the exact printed line in your final answer.", "could but error", ANSWER_TEXT},
        {"named_functions", "What does this Python code print?\ndef could(x):\n    return x - 1\ndef error(x):\n    return could(x) * 2\nprint(error(0))\nPut only the signed integer in your final answer.", "-2", ANSWER_INTEGER},
        {"exception_branch", "What does this Python code print?\ndef could(x):\n    if x < 0:\n        raise ValueError('error')\n    return x*x\nresult = []\nfor x in [-2, -1, 0, 2]:\n    try:\n        result.append(could(x))\n    except ValueError:\n        result.append(-1)\nprint(result)\nPut only the resulting Python list in your final answer.", "[-1,-1,0,4]", ANSWER_LIST},
        {"signed_subtraction", "A student wrote that -3 minus -8 is -11. Recheck the signs and correct this mistake, then subtract 12 from the corrected result. Put only the signed integer in your final answer.", "-7", ANSWER_INTEGER},
        {"circular_binary", "How many binary strings of length 16 have exactly six 1s and no two adjacent 1s, including adjacency between positions 16 and 1? Positions are labeled; rotations are counted as different strings. Put only the integer count in your final answer.", "336", ANSWER_INTEGER},
        {"restricted_lattice", "Count paths from (0,0) to (7,7) that use only unit right and unit up steps, never have y>x, and do not pass through either (2,2) or (4,4). Put only the integer count in your final answer.", "295", ANSWER_INTEGER},
        {"primitive_triples", "How many triples of positive integers (a,b,c) satisfy a<=b<=c<=30, a+b+c=60, gcd(a,b,c)=1, and exactly one of a,b,c is even? Each nondecreasing triple is counted once. Put only the integer count in your final answer.", "52", ANSWER_INTEGER},
        {"alternating_digits", "How many six-digit decimal integers use six distinct digits from {0,1,2,3,4,5,6,7}, have alternating even and odd digits in adjacent positions, and are divisible by 7? The first digit cannot be zero, and zero is even. Put only the integer count in your final answer.", "156", ANSWER_INTEGER},
        {"primitive_triangles", "How many nondegenerate triangles have integer side lengths a<=b<=c=17, odd perimeter, and gcd(a,b,c)=1? Count side-length triples, not labeled arrangements. Put only the integer count in your final answer.", "44", ANSWER_INTEGER},
        {"even_permutations", "How many even permutations p of {1,2,3,4,5,6,7,8} have exactly two fixed points, with those two fixed positions not consecutive? A fixed point is a position i with p(i)=i, and an even permutation has an even number of inversions. Put only the integer count in your final answer.", "2730", ANSWER_INTEGER},
#include "overthink_counting_cases.h"
    };
    char err[256] = {0};
    int invalid = 0;
    const int first_case = (counting ? 18 : hard ? 12 : heldout ? 4 : 0) + case_offset;
    const int last_case = first_case + case_limit;
    for (int ci = first_case; ci < last_case; ci++) {
        /* One explicit local policy per fresh process; no speed promotion. */
        for (int ai = 0; ai < arms; ai++) {
            if (!ds4_engine_set_overthinking_penalty(engine, 0.0f, true)) return 1;
            ds4_tokens prompt = {0};
            ds4_chat_begin(engine, &prompt);
            ds4_chat_append_think_prefix(engine, &prompt, mode);
            ds4_chat_append_message(engine, &prompt, "user", cases[ci].prompt);
            ds4_chat_append_assistant_prefix(engine, &prompt, mode);
            const double prefill_t0 = seconds();
            if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
                fprintf(stderr, "sync: %s\n", err); return 1;
            }
            const double prefill_seconds = seconds() - prefill_t0;
            ds4_tokens_free(&prompt);
            ds4_overthinking_info case_info;
            if (!ds4_session_overthinking_info(session, &case_info)) return 1;
            size_t used = 0, final_offset = 0;
            char *text = calloc((size_t)cap * 512 + 1, 1);
            if (!text) return 1;
            uint64_t rng = UINT64_C(20261002);
            int generated = 0, reasoning_tokens = 0, closed = 0, stopped = 0;
            int sampling_steps = 0, active_steps = 0, thinking_marker_selections = 0;
            int active_marker_selections = 0, softmax_samples = 0, argmax_changes = 0;
            double marker_mass_sum = 0, marker_mass_max = 0, kl_sum = 0, selected_dnll_sum = 0;
            double diagnostic_seconds = 0;
            char label[32], raw_name[128], token_name[128];
            lambda_label(boost, label);
            snprintf(raw_name, sizeof(raw_name), "%s-boost%s.txt", cases[ci].id, label);
            snprintf(token_name, sizeof(token_name), "%s-boost%s.tokens.jsonl", cases[ci].id, label);
            FILE *tokens = NULL;
            if (diagnostic_stride) {
                snprintf(path, sizeof(path), "%s/%s", output_dir, token_name);
                tokens = fopen(path, "wx");
                if (!tokens) return 1;
            }
            const double decode_t0 = seconds();
            while (generated < cap) {
                ds4_overthinking_info info = {0};
                double mass = -1.0;
                int raw_best = -1, policy_best = -1;
                double marker_margin = 0;
                double rounded_delta = 0;
                int active = 0;
                if (diagnostic_stride) {
                    const double diag_t0 = seconds();
                    if (!ds4_session_overthinking_info(session, &info) || info.penalty != 0.0f || info.active || !info.think_only || !info.hesitation_only || info.after != 0) return 1;
                    active = closing_active(boost,info.inside_thinking,info.generated_thinking_tokens);
                    sampling_steps++;
                    active_steps += active;
                    if (generated % diagnostic_stride == 0 && info.inside_thinking) {
                        if (ds4_session_copy_logits(session, raw_logits, vocab) != vocab) return 1;
                        mass = overthink_marker_mass(raw_logits, vocab, marker_ids, marker_count,
                            (double)PILOT_TEMPERATURE);
                        if (mass < 0) return 1;
                        diagnostic_argmax(raw_logits, vocab, marker_ids, marker_count,
                            active ? -boost : 0, &raw_best, &policy_best, &marker_margin);
                        softmax_samples++;
                        marker_mass_sum += mass;
                        if (mass > marker_mass_max) marker_mass_max = mass;
                        argmax_changes += raw_best != policy_best;
                    }
                    diagnostic_seconds += seconds() - diag_t0;
                }
                const uint64_t rng_before = rng;
                const int token = closing_sample(api,vocab,CLOSING_TOKEN_ID,boost,
                    info.inside_thinking,info.generated_thinking_tokens,raw_logits,policy_logits,
                    PILOT_TEMPERATURE,0,0.95f,0.0f,&rng,&rounded_delta);
                if (token < 0) return 1;
                const int is_stop = ds4_token_is_stop_for_think_mode(engine, token, mode);
                size_t n = 0;
                char *piece = NULL;
                if (!is_stop || tokens) piece = ds4_token_text(engine, token, &n);
                if (tokens) {
                    const double diag_t0 = seconds();
                    const int selected_marker = overthink_is_marker(token, marker_ids, marker_count);
                    thinking_marker_selections += selected_marker && info.inside_thinking;
                    active_marker_selections += selected_marker && active;
                    fprintf(tokens, "{\"step\":%d,\"token_id\":%d,\"stop\":%s,\"inside_thinking\":%s,\"generated_thinking_tokens\":%d,\"policy_active\":%s,\"selected_marker\":%s",
                        generated, token, is_stop ? "true" : "false", info.inside_thinking ? "true" : "false",
                        info.generated_thinking_tokens, active ? "true" : "false", selected_marker ? "true" : "false");
                    fprintf(tokens, ",\"applied_logit_delta\":%.17g",rounded_delta);
                    fprintf(tokens, ",\"rng_before\":\"%016llx\",\"rng_after\":\"%016llx\",\"piece_hex\":\"",
                        (unsigned long long)rng_before, (unsigned long long)rng);
                    /* Individual tokens may split UTF-8. Hex retains their
                     * exact bytes while keeping each JSON row valid Unicode. */
                    for (size_t bi = 0; bi < n; bi++) fprintf(tokens, "%02x", (unsigned char)piece[bi]);
                    fputc('"', tokens);
                    if (mass >= 0) {
                        double kl, dnll;
                        closing_metrics(mass,rounded_delta,(double)PILOT_TEMPERATURE,
                            selected_marker,&kl,&dnll);
                        kl_sum += kl; selected_dnll_sum += dnll;
                        fprintf(tokens, ",\"raw_marker_mass_pre_filter\":%.12g,\"kl_raw_to_policy_pre_filter\":%.12g,\"selected_delta_nll_pre_filter\":%.12g,\"raw_argmax\":%d,\"policy_argmax\":%d,\"marker_best_minus_other_logit\":%.9g",
                            mass, kl, dnll, raw_best, policy_best, marker_margin);
                    }
                    fputs("}\n", tokens);
                    diagnostic_seconds += seconds() - diag_t0;
                }
                if (is_stop) { free(piece); stopped = 1; break; }
                if (used + n >= (size_t)cap * 512) return 1;
                memcpy(text + used, piece, n); used += n;
                if (!closed) reasoning_tokens++;
                if (n == 8 && !memcmp(piece, "</think>", 8)) { closed = 1; final_offset = used; }
                free(piece);
                generated++;
                if (generated < cap && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "eval: %s\n", err); return 1;
                }
            }
            const double decode_seconds = seconds() - decode_t0;
            if (tokens) fclose(tokens);
            snprintf(path, sizeof(path), "%s/%s", output_dir, raw_name);
            FILE *raw = fopen(path, "wx");
            if (!raw) { perror(path); return 1; }
            fwrite(text, 1, used, raw); fclose(raw);
            char *final_copy = strdup(closed ? text + final_offset : "");
            if (!final_copy) return 1;
            char parsed[1024] = {0};
            const int parse_ok = closed && parse_answer(final_copy, cases[ci].kind, parsed, sizeof(parsed));
            const int correct = parse_ok && (cases[ci].kind == ANSWER_TEXT ?
                !strcmp(parsed, cases[ci].answer) : exact_answer(parsed, cases[ci].answer));
            const char *status = !parse_ok ? "INVALID" : correct ? "CORRECT" : "INCORRECT";
            const char *parse_reason = parse_ok ? "none" : !closed ? "unclosed_reasoning" : !*trim(final_copy) ? "empty_final" : "unsupported_final_format";
            if (!parse_ok) invalid++;
            fprintf(results, "{\"case\":"); json_string(results, cases[ci].id);
            fprintf(results, ",\"suite\":\"%s\",\"status\":\"%s\",\"parse_ok\":%s,\"parse_reason\":\"%s\",\"scorer\":\"strict-final-v2\",\"parsed_answer\":",
                    suite, status, parse_ok ? "true" : "false", parse_reason);
            json_string(results, parsed);
            fprintf(results, ",\"marker_profile\":\"%s\",\"mapped_markers\":%d",
                "closing", 1);
            fprintf(results, ",\"closing_boost\":%.9g,\"policy_direction\":\"ADD\",\"closing_token_id\":128822,\"scope\":\"think\",\"engine_penalty\":0,\"engine_marker_profile\":\"hesitation\",\"engine_overthink_after\":0,\"think_level\":%d,\"seed\":20261002,\"temperature\":0.6,\"top_p\":0.95,\"min_p\":0,\"max_tokens\":%d,\"cache_gib_requested\":%.9g,\"closing_after\":%d,\"case_offset\":%d,\"case_limit\":%d,\"generated_tokens\":%d,\"reasoning_tokens\":%d,\"think_closed\":%s,\"stopped\":%s,\"cap_hit\":%s,\"correct_exact\":%s,\"prefill_seconds\":%.6f,\"decode_seconds\":%.6f,\"final\":",
                    boost, level, cap, cache_gib, after, case_offset, case_limit, generated, reasoning_tokens, closed ? "true" : "false", stopped ? "true" : "false", generated >= cap ? "true" : "false", correct ? "true" : "false", prefill_seconds, decode_seconds);
            json_string(results, closed ? text + final_offset : "");
            fprintf(results, ",\"expected\":"); json_string(results, cases[ci].answer);
            fprintf(results, ",\"prompt\":"); json_string(results, cases[ci].prompt);
            fprintf(results, ",\"raw_trace\":"); json_string(results, raw_name);
            fprintf(results, ",\"diagnostics\":{\"temperature_float_value\":%.17g,\"stride\":%d,\"sampling_steps\":%d,\"active_steps\":%d,\"thinking_marker_selections\":%d,\"active_marker_selections\":%d,\"pre_filter_samples\":%d,\"raw_marker_mass_mean\":%.12g,\"raw_marker_mass_max\":%.12g,\"policy_kl_mean_pre_filter\":%.12g,\"selected_delta_nll_sum_pre_filter\":%.12g,\"sampled_argmax_changes\":%d,\"cpu_seconds\":%.6f,\"token_trace\":",
                (double)PILOT_TEMPERATURE,
                diagnostic_stride, sampling_steps, active_steps, thinking_marker_selections,
                active_marker_selections, softmax_samples, softmax_samples ? marker_mass_sum / softmax_samples : 0,
                marker_mass_max, softmax_samples ? kl_sum / softmax_samples : 0,
                selected_dnll_sum, argmax_changes, diagnostic_seconds);
            json_string(results, diagnostic_stride ? token_name : ""); fputs("}", results);
            fputs("}\n", results); fflush(results);
            fprintf(stderr, "pilot suite=%s case=%s boost=%s tokens=%d closed=%d cap=%d status=%s decode=%.3fs active=%d marker_selected=%d sampled_argmax_changed=%d/%d\n", suite, cases[ci].id, label, generated, closed, generated >= cap, status, decode_seconds, active_steps, active_marker_selections, argmax_changes, softmax_samples);
            free(final_copy);
            free(text);
        }
    }
    fclose(results);
    free(raw_logits);
    free(policy_logits);
    ds4_session_free(session);
    ds4_engine_close(engine);
    if (invalid) fprintf(stderr, "pilot contains %d INVALID final-answer records; inspect cap hits and raw traces\n", invalid);
    return invalid ? 3 : 0;
}
