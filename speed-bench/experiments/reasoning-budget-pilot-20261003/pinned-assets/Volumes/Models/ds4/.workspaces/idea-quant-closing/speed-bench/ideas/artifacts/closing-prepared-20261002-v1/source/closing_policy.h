#ifndef DS4_CLOSING_POLICY_H
#define DS4_CLOSING_POLICY_H
#include "overthink_diagnostics.h"
#include <stddef.h>

#define CLOSING_TOKEN_ID 128822
#define CLOSING_AFTER 1024

static int closing_piece_valid(int id, const char *piece, size_t n) {
    return id == CLOSING_TOKEN_ID && piece && n == 8 && !memcmp(piece,"</think>",8);
}

static int closing_active(float boost, int inside, int prior_thinking) {
    return boost > 0.0f && inside && prior_thinking >= CLOSING_AFTER;
}

/* The measured delta is the actual rounded FP32 operation, not nominal boost. */
static int closing_adjust(const float *raw, float *policy, int vocab, int id,
                          float boost, double *delta) {
    if (!raw || !policy || !delta || id < 0 || id >= vocab ||
        !overthink_finite(boost) || boost < 0 || !overthink_finite(raw[id])) return 0;
    memcpy(policy,raw,(size_t)vocab*sizeof(*raw));
    policy[id] = raw[id] + boost;
    if (!overthink_finite(policy[id])) return 0;
    *delta = (double)policy[id] - (double)raw[id];
    return 1;
}

typedef struct {
    void *session;
    int (*copy)(void *,float *,int);
    int (*set)(void *,const float *,int);
    int (*sample)(void *,float,int,float,float,uint64_t *);
} closing_api;

/* Inactive/default zero calls the native sampler directly. Active changes one
 * cell, uses the same one sampler/RNG call and restores raw before returning. */
static int closing_sample(closing_api api, int vocab, int id, float boost,
                          int inside, int prior_thinking, float *raw, float *policy,
                          float temperature, int top_k, float top_p, float min_p,
                          uint64_t *rng, double *delta) {
    *delta = 0;
    if (!closing_active(boost,inside,prior_thinking))
        return api.sample(api.session,temperature,top_k,top_p,min_p,rng);
    if (api.copy(api.session,raw,vocab) != vocab ||
        !closing_adjust(raw,policy,vocab,id,boost,delta)) return -1;
    if (api.set(api.session,policy,vocab)) return -1;
    const int token=api.sample(api.session,temperature,top_k,top_p,min_p,rng);
    if (api.set(api.session,raw,vocab)) return -1;
    return token;
}

/* Same-prefix full-vocabulary P/Q at explicit T, before nucleus filtering. */
static void closing_metrics(double mass, double actual_delta, double temperature,
                            int selected_close, double *kl, double *dnll) {
    overthink_policy_metrics(mass,-actual_delta/temperature,selected_close,kl,dnll);
}
#endif
