/* Paired, teacher-forced approximation screen. Reference logits are recorded
 * once and reused by subsequent builds; model inference is never duplicated
 * merely to compute KL. Native-endian records are local benchmark artifacts. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Optional policy hook from the overthinking experiment. */
#ifdef DS4_IDEAS_POLICY_HOOK
extern int ds4_session_copy_sampling_logits(ds4_session *, float *, int);
#endif

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static void fail(const char *message) {
    fprintf(stderr, "quality-probe: %s\n", message); exit(1);
}
static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fseek(f, 0, SEEK_END)) fail("seek failed");
    long n = ftell(f); if (n < 0 || n > 16 * 1024 * 1024) fail("bad text size");
    rewind(f); char *text = malloc((size_t)n + 1);
    if (!text || fread(text, 1, (size_t)n, f) != (size_t)n) fail("read failed");
    text[n] = 0; fclose(f); return text;
}
static uint64_t token_hash(const ds4_tokens *a, const ds4_tokens *b) {
    uint64_t h = UINT64_C(14695981039346656037);
    const ds4_tokens *seq[2] = {a, b};
    for (int s = 0; s < 2; s++) {
        h ^= (uint32_t)seq[s]->len; h *= UINT64_C(1099511628211);
        for (int i = 0; i < seq[s]->len; i++) {
            h ^= (uint32_t)seq[s]->v[i]; h *= UINT64_C(1099511628211);
        }
    }
    return h;
}
static double log_z(const float *z, int n) {
    double mx = -INFINITY, sum = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(z[i])) fail("nonfinite model logits");
        if (z[i] > mx) mx = z[i];
    }
    for (int i = 0; i < n; i++) sum += exp((double)z[i] - mx);
    return mx + log(sum);
}
static int argmax(const float *z, int n) {
    int best = 0; for (int i = 1; i < n; i++) if (z[i] > z[best]) best = i;
    return best;
}
static void distance(const float *p, double lp, const float *q, double lq,
                     int n, double *kl, double *tv) {
    *kl = *tv = 0;
    for (int i = 0; i < n; i++) {
        double a = exp((double)p[i] - lp), b = exp((double)q[i] - lq);
        *kl += a * (((double)p[i] - lp) - ((double)q[i] - lq));
        *tv += fabs(a - b) * 0.5;
    }
    if (*kl < 0 && *kl > -1e-10) *kl = 0;
}
static long positive(const char *s) {
    char *end; long n = strtol(s, &end, 10);
    if (end == s || *end || n < 1) fail("expected positive integer"); return n;
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        const float p[3] = {logf(0.6f), logf(0.3f), logf(0.1f)};
        const float q[3] = {logf(0.3f), logf(0.6f), logf(0.1f)};
        const float shifted[3] = {p[0]+4, p[1]+4, p[2]+4};
        double kl, tv, pz = log_z(p, 3), qz = log_z(q, 3);
        distance(p, pz, q, qz, 3, &kl, &tv);
        if (fabs(kl - 0.3*log(2.0)) > 1e-7 || fabs(tv-0.3) > 1e-7 ||
            fabs((qz-q[0])-(pz-p[0])-log(2.0)) > 1e-7)
            fail("distribution metric arithmetic failed");
        distance(p, pz, shifted, log_z(shifted, 3), 3, &kl, &tv);
        if (fabs(kl) > 1e-10 || tv > 1e-7) fail("constant-logit shift invariance failed");
        puts("quality-probe metrics: PASS"); return 0;
    }
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL MANIFEST SCORES.csv REFERENCE.bin record|compare [--cases N] [--tokens N] [--ctx N] [--cache-gb N] [--chunk N] [--think-level 1..100]\n", argv[0]);
        return 2;
    }
    const int recording = !strcmp(argv[5], "record");
    if (!recording && strcmp(argv[5], "compare")) fail("record or compare required");
    int cases = 8, limit = 24, ctx = 4096, cache_gb = 64, chunk = 512;
    ds4_think_mode think = DS4_THINK_NONE;
    for (int i = 6; i < argc; i += 2) {
        if (i + 1 >= argc) fail("missing option argument");
        long value = positive(argv[i + 1]);
        if (!strcmp(argv[i], "--cases")) cases = (int)value;
        else if (!strcmp(argv[i], "--tokens")) limit = (int)value;
        else if (!strcmp(argv[i], "--ctx")) ctx = (int)value;
        else if (!strcmp(argv[i], "--cache-gb")) cache_gb = (int)value;
        else if (!strcmp(argv[i], "--chunk")) chunk = (int)value;
        else if (!strcmp(argv[i], "--think-level") && value <= 100)
            think = (ds4_think_mode)(DS4_THINK_LEVEL_BASE + value);
        else fail("unknown option");
    }
    if (ctx > 131072 || cache_gb > 96 || chunk > 8192) fail("option out of range");
    FILE *manifest = fopen(argv[2], "r"), *scores = fopen(argv[3], "wx");
    FILE *reference = fopen(argv[4], recording ? "wbx" : "rb");
    if (!manifest || !scores || !reference) { perror("open benchmark file"); return 1; }
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = (uint32_t)ctx, .ssd_streaming = true, .power_percent = 100,
        .ssd_streaming_cache_bytes = (uint64_t)cache_gb << 30,
        .prefill_chunk = (uint32_t)chunk};
    ds4_engine *engine = NULL; ds4_session *session = NULL;
    double start = now_ms();
    if (ds4_engine_open(&engine, &opt) || ds4_session_create(&session, engine, ctx))
        fail("model/session initialization failed");
    const double startup_ms = now_ms() - start;
    const int vocab = ds4_engine_vocab_size(engine);
    uint32_t header[4] = {UINT32_C(0x44535131), (uint32_t)vocab, (uint32_t)cases, (uint32_t)limit};
    if (recording) {
        if (fwrite(header, sizeof(header), 1, reference) != 1) fail("reference header write failed");
    } else {
        uint32_t got[4];
        if (fread(got, sizeof(got), 1, reference) != 1 || memcmp(got, header, sizeof(got)))
            fail("reference dimensions/configuration differ");
    }
    float *raw = malloc((size_t)vocab * sizeof(float));
    float *policy = malloc((size_t)vocab * sizeof(float));
    float *base = malloc((size_t)vocab * sizeof(float));
    if (!raw || !policy || !base) fail("allocation failed");
    fprintf(scores, "case,step,target,prompt_tokens,base_nll,raw_nll,policy_nll,delta_raw_nll,delta_policy_nll,raw_kl,policy_kl,raw_tv,policy_tv,argmax_flip,decode_ms\n");
    char line[8192], err[256] = {0}; int case_n = 0, count = 0, flips = 0, bit_equal = 0;
    double total_nll = 0, total_delta = 0, total_policy_delta = 0, total_kl = 0,
        total_policy_kl = 0, total_tv = 0, prefill_ms = 0, decode_ms = 0, metrics_ms = 0;
    while (case_n < cases && fgets(line, sizeof(line), manifest)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *id = strtok(line, "\t\r\n"), *pp = strtok(NULL, "\t\r\n"),
             *cp = strtok(NULL, "\t\r\n");
        if (!id || !pp || !cp) fail("bad manifest row");
        char *prompt_text = read_text(pp), *cont_text = read_text(cp);
        ds4_tokens prompt = {0}, target = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt_text, think, &prompt);
        if (think == DS4_THINK_NONE) ds4_tokenize_text(engine, cont_text, &target);
        else ds4_tokenize_rendered_chat(engine, cont_text, &target);
        if (target.len > limit) target.len = limit;
        if (!target.len || prompt.len + target.len + 1 >= ctx) fail("empty target or insufficient context");
        uint64_t fingerprint = token_hash(&prompt, &target);
        ds4_session_invalidate(session);
        start = now_ms();
        if (ds4_session_sync(session, &prompt, err, sizeof(err))) fail(err);
        const double fill = now_ms() - start; prefill_ms += fill;
        double case_nll = 0, case_kl = 0, case_decode = 0;
        for (int i = 0; i < target.len; i++) {
            start = now_ms();
            if (ds4_session_copy_logits(session, raw, vocab) != vocab) fail("logit copy failed");
            memcpy(policy, raw, (size_t)vocab * sizeof(float));
#ifdef DS4_IDEAS_POLICY_HOOK
            if (
                ds4_session_copy_sampling_logits(session, policy, vocab) != vocab)
                fail("sampling-policy logit copy failed");
#endif
            uint32_t key[4] = {(uint32_t)case_n, (uint32_t)i, (uint32_t)target.v[i], (uint32_t)prompt.len};
            if (recording) {
                if (fwrite(key, sizeof(key), 1, reference) != 1 ||
                    fwrite(&fingerprint, sizeof(fingerprint), 1, reference) != 1 ||
                    fwrite(raw, sizeof(float), (size_t)vocab, reference) != (size_t)vocab)
                    fail("reference write failed");
                memcpy(base, raw, (size_t)vocab * sizeof(float));
            } else {
                uint32_t got[4]; uint64_t hash;
                if (fread(got, sizeof(got), 1, reference) != 1 || memcmp(key, got, sizeof(key)) ||
                    fread(&hash, sizeof(hash), 1, reference) != 1 || hash != fingerprint ||
                    fread(base, sizeof(float), (size_t)vocab, reference) != (size_t)vocab)
                    fail("reference token/prefix alignment mismatch or truncation");
            }
            double bz = log_z(base, vocab), rz = log_z(raw, vocab), pz = log_z(policy, vocab);
            double bnll = bz - base[target.v[i]], rnll = rz - raw[target.v[i]],
                   pnll = pz - policy[target.v[i]], kl, tv, pkl, ptv;
            distance(base, bz, raw, rz, vocab, &kl, &tv);
            distance(base, bz, policy, pz, vocab, &pkl, &ptv);
            int flip = argmax(base, vocab) != argmax(raw, vocab);
            bit_equal += memcmp(base, raw, (size_t)vocab * sizeof(float)) == 0;
            metrics_ms += now_ms() - start;
            start = now_ms();
            if (ds4_session_eval(session, target.v[i], err, sizeof(err))) fail(err);
            const double elapsed = now_ms() - start;
            fprintf(scores, "%s,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%.6f\n",
                id, i, target.v[i], prompt.len, bnll, rnll, pnll, rnll-bnll, pnll-bnll,
                kl, pkl, tv, ptv, flip, elapsed);
            total_nll += rnll; total_delta += rnll-bnll; total_policy_delta += pnll-bnll;
            total_kl += kl; total_policy_kl += pkl; total_tv += tv; flips += flip; count++;
            case_nll += rnll; case_kl += kl; case_decode += elapsed; decode_ms += elapsed;
        }
        fprintf(stderr, "PROBE case=%s tokens=%d nll=%.6f kl=%.6g prefill_ms=%.3f decode_ms=%.3f\n",
            id, target.len, case_nll/target.len, case_kl/target.len, fill, case_decode);
        fflush(scores); fflush(reference);
        ds4_tokens_free(&prompt); ds4_tokens_free(&target); free(prompt_text); free(cont_text); case_n++;
    }
    if (case_n != cases || !count) fail("manifest contains fewer cases than requested");
    if (!recording && fgetc(reference) != EOF) fail("unconsumed reference records");
    fprintf(stdout, "{\"cases\":%d,\"tokens\":%d,\"avg_nll\":%.12g,\"delta_nll\":%.12g,\"policy_delta_nll\":%.12g,\"kl_base_to_raw\":%.12g,\"kl_base_to_policy\":%.12g,\"tv\":%.12g,\"argmax_flips\":%d,\"bit_equal_rows\":%d,\"startup_ms\":%.3f,\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"metrics_ms\":%.3f}\n",
        case_n,count,total_nll/count,total_delta/count,total_policy_delta/count,
        total_kl/count,total_policy_kl/count,total_tv/count,flips,bit_equal,startup_ms,prefill_ms,decode_ms,metrics_ms);
    ds4_gpu_print_memory_report("quality probe final");
    free(raw); free(policy); free(base);
    ds4_session_free(session); ds4_engine_close(engine);
    if (fclose(reference) || fclose(scores) || fclose(manifest)) fail("file close failed");
    return 0;
}
