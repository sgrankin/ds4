#ifndef DS4_STATE_CACHE_RUNTIME_H
#define DS4_STATE_CACHE_RUNTIME_H

#include "ds4_state_cache.h"
#include "ds4_kvstore.h"
#include "ds4_tp.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Two frontends use the same opt-in cache settings and weight/policy identity.
 * A directory has one process owner; server clients share that owner. */
extern char **environ;
static int ds4_state_cache_env_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}
static bool ds4_state_cache_env_mb(const char *name, uint64_t fallback,
                                  uint64_t *bytes) {
    const char *s = getenv(name);
    if (!s) { *bytes = fallback << 20; return true; }
    char *end = NULL;
    errno = 0;
    unsigned long long mb = strtoull(s, &end, 10);
    if (errno || !*s || *s == '-' || *end || mb > (UINT64_MAX >> 20)) return false;
    *bytes = (uint64_t)mb << 20;
    return true;
}
static ds4_state_cache *ds4_state_cache_runtime_open(ds4_engine *engine,
        const ds4_engine_options *options, int ctx, char profile[41],
        char *err, size_t errlen) {
    const char *dir = getenv("DS4_STATE_CACHE_DIR");
    if (!dir || !*dir) return NULL;
    if (!ds4_engine_is_deepseek41(engine) || ds4_tp_enabled(&options->tp) ||
        options->directional_steering_file || options->backend != DS4_BACKEND_METAL) {
        snprintf(err, errlen, "state cache requires native text DS4.1 on Metal");
        return NULL;
    }
    uint64_t hot, ssd;
    struct stat st;
    const char *model_path = ds4_engine_model_path(engine);
    if (!ds4_state_cache_env_mb("DS4_STATE_CACHE_HOT_MB", 512, &hot) ||
        !ds4_state_cache_env_mb("DS4_STATE_CACHE_SSD_MB", 8192, &ssd) || !ssd ||
        !model_path || stat(model_path, &st)) {
        snprintf(err, errlen, "invalid state cache limits or model identity");
        return NULL;
    }
    size_t count = 0;
    for (char **p = environ; *p; p++) if (!strncmp(*p, "DS4_METAL_", 10)) count++;
    char **policy = malloc((count ? count : 1) * sizeof(*policy));
    char *identity = NULL; size_t len = 0;
    FILE *fp = open_memstream(&identity, &len);
    if (!policy || !fp) {
        free(policy); if (fp) fclose(fp); free(identity);
        snprintf(err, errlen, "cannot allocate state cache identity"); return NULL;
    }
    size_t at = 0;
    for (char **p = environ; *p; p++) if (!strncmp(*p, "DS4_METAL_", 10)) policy[at++] = *p;
    qsort(policy, count, sizeof(*policy), ds4_state_cache_env_cmp);
    fprintf(fp, "DSV4-v2-DPR1-20261003\n%s\n%llu:%llu:%llu:%lld:%ld:%d:%d\n",
            model_path, (unsigned long long)st.st_dev,
            (unsigned long long)st.st_ino, (unsigned long long)st.st_size,
            (long long)st.st_mtime,
#ifdef __APPLE__
            st.st_mtimespec.tv_nsec,
#else
            st.st_mtim.tv_nsec,
#endif
            ctx, ds4_engine_routed_quant_bits(engine));
    for (size_t i = 0; i < count; i++) fprintf(fp, "%s\n", policy[i]);
    fprintf(fp, "prefill_chunk=%u quality=%d\n",
            ds4_engine_prefill_chunk(engine), options->quality);
    /* Keep native identities stable; replay checkpoints belong to their
     * configured forward policy, including selected-expert execution. */
    const char *replay = getenv("DS4_V41_DECODER_REPLAY_ROWS");
    if (replay && strcmp(replay, "0")) {
        const char *selected = getenv("DS4_V41_DECODER_REPLAY_SELECTED");
        fprintf(fp, "decoder_replay=DRP1 rows=%s selected=%d\n", replay,
                selected && !strcmp(selected, "1"));
    }
    free(policy);
    if (fclose(fp)) { free(identity); snprintf(err, errlen, "cannot finish cache identity"); return NULL; }
    ds4_kvstore_sha1_bytes_hex(identity, len, profile);
    free(identity);
    return ds4_state_cache_open(dir, hot, ssd, err, errlen);
}

#endif
