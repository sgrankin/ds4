#ifndef REASONING_BUDGET_POLICY_H
#define REASONING_BUDGET_POLICY_H

#include <stdint.h>

#define REASONING_CLOSE_TOKEN 128822

typedef struct {
    int limit;
    int forced;
} reasoning_budget_policy;

typedef int (*reasoning_native_sample)(void *, uint64_t *);

static int reasoning_budget_init(reasoning_budget_policy *policy, int limit) {
    if (!policy || (limit != 0 && limit != 1024 && limit != 1536)) return 0;
    policy->limit = limit;
    policy->forced = 0;
    return 1;
}

/* A single protocol intervention, without a sampler call or RNG advance.
 * The caller must evaluate the returned close token and verify that the
 * following session step is outside thinking before sampling again. */
static int reasoning_budget_sample(reasoning_budget_policy *policy,
                                  int inside, int reasoning_tokens, int remaining_slots,
                                  reasoning_native_sample sample, void *session,
                                  uint64_t *rng, int *forced_this_step) {
    if (!policy || !sample || !rng || !forced_this_step) return -1;
    *forced_this_step = 0;
    if (policy->limit == 0) return sample(session, rng);
    if ((policy->limit != 1024 && policy->limit != 1536) ||
        (policy->forced != 0 && policy->forced != 1) ||
        (inside != 0 && inside != 1) || reasoning_tokens < 0 || remaining_slots < 0 ||
        (policy->forced && inside)) return -1;
    if (inside && reasoning_tokens >= policy->limit) {
        /* Leave one output slot after the evaluated close. Reject before
         * changing policy state or publishing a last-slot protocol token. */
        if (remaining_slots < 2) return -1;
        policy->forced = 1;
        *forced_this_step = 1;
        return REASONING_CLOSE_TOKEN;
    }
    return sample(session, rng);
}

#endif
