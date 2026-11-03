#include "ds4_resident_route.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void base(float *p, float *bias, uint8_t *resident) {
    memset(bias, 0, 10 * sizeof(*bias));
    memset(resident, 0, 10);
    for (uint32_t i = 0; i < 10; i++) p[i] = 2.0f - 0.1f * i;
    for (uint32_t i = 0; i < 5; i++) resident[i] = 1;
    resident[6] = 1;
}

static void four_to_six(void) {
    const int32_t ids[6] = {0, 1, 2, 3, 4, 5};
    const uint32_t weakest[6] = {4, 5, 2, 3, 1, 0};
    /* Exact binary masses make both the per-slot and cumulative boundaries
     * explicit. Equal contributions retain the original slot order. */
    const float p[12] = {6, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
    const float bias[12] = {0};
    uint8_t resident[12] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    ds4_resident_route_plan plan;
    for (uint32_t max_subs = 4; max_subs <= 6; max_subs++) {
        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            1, 12, 1, max_subs, 1, &plan));
        assert(plan.count == max_subs);
        int32_t final[6]; memcpy(final, ids, sizeof(final));
        float replaced_mass = 0;
        for (uint32_t i = 0; i < plan.count; i++) {
            const ds4_resident_route_change change = plan.changes[i];
            assert(change.slot == weakest[i] && change.old_expert == (uint32_t)ids[weakest[i]]);
            assert(change.new_expert == 6 + i && resident[change.new_expert]);
            replaced_mass += p[change.old_expert] / 16;
            final[change.slot] = (int32_t)change.new_expert;
        }
        assert(plan.total_mass == replaced_mass);
        for (uint32_t i = max_subs; i < 6; i++) assert(final[weakest[i]] == ids[weakest[i]]);
        for (uint32_t i = 0; i < 6; i++) for (uint32_t j = 0; j < i; j++) assert(final[i] != final[j]);

        /* An already resident native slot stays put. It does not expand the
         * original eligible set to include a stronger missing contributor. */
        resident[4] = 1;
        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            1, 12, 1, max_subs, 1, &plan));
        assert(plan.count == max_subs - 1);
        for (uint32_t i = 0; i < plan.count; i++) {
            assert(plan.changes[i].slot == weakest[i + 1]);
            assert(!resident[plan.changes[i].old_expert]);
            assert(plan.changes[i].new_expert >= 6);
        }
        resident[4] = 0;

        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            1, 12, .125f, max_subs, 1, &plan) && plan.count == 4);
        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            1, 12, .12f, max_subs, 1, &plan) && plan.count == 2);
        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            1, 12, 1, max_subs, .125f, &plan) && plan.count == 2 && plan.total_mass == .125f);
        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            1, 9, 1, max_subs, 1, &plan) && plan.count == 3);
        assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
            .01f, 12, 1, max_subs, 1, &plan) && plan.count == 2);
    }
    /* A rejected request clears the enlarged plan, including unused slots. */
    memset(&plan, 0xff, sizeof(plan));
    assert(!ds4_resident_route_choose_plan(ids, p, bias, resident, 12, 6,
        1, 12, 1, 7, 1, &plan));
    const ds4_resident_route_plan zero = {0};
    assert(!memcmp(&plan, &zero, sizeof(plan)));

    /* Normalizing and adding these six f32 contributions rounds above one.
     * MAX_TOTAL_MASS=1 disables the cap; smaller limits still enforce it. */
    const float rounding[12] = {.1f, .2f, .3f, .4f, .5f, .6f,
                               .09f, .09f, .09f, .09f, .09f, .09f};
    assert(ds4_resident_route_choose_plan(ids, rounding, bias, resident, 12, 6,
        1, 12, 1, 6, 1, &plan));
    assert(plan.count == 6 && plan.total_mass > 1 && plan.total_mass < 1.000001f);
    assert(ds4_resident_route_choose_plan(ids, rounding, bias, resident, 12, 6,
        1, 12, 1, 6, .99f, &plan) && plan.count == 5 && plan.total_mass <= .99f);
}

int main(void) {
    four_to_six();
    float p[10], bias[10], weights[6];
    uint8_t resident[10];
    int32_t ids[6] = {0, 1, 2, 3, 4, 5};
    ds4_resident_route_change change;
    base(p, bias, resident);
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, 0, 8, 1.0f / 6, &change));
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .01f, 8, 1.0f / 6, &change));
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));
    assert(change.slot == 5 && change.old_expert == 5 && change.new_expert == 6);
    assert(change.candidate_rank == 7);
    const float old_mass = p[5] / (p[0] + p[1] + p[2] + p[3] + p[4] + p[5]);
    assert(fabsf(change.old_mass - old_mass) < 1e-6f);
    ids[change.slot] = (int32_t)change.new_expert;
    ds4_resident_route_weights(weights, ids, p, 6, 1.5f);
    float sum = 0;
    for (int i = 0; i < 6; i++) {
        assert(weights[i] > 0);
        sum += weights[i];
        for (int j = 0; j < i; j++) assert(ids[i] != ids[j]);
    }
    assert(fabsf(sum - 1.5f) < 1e-6f);
    assert(fabsf(weights[0] / weights[5] - p[0] / p[6]) < 1e-6f);

    /* Learned correction bias can make an early route the weakest actual
     * contributor. Protect contribution rather than assuming slot six. */
    ids[5] = 5;
    p[0] = 1.41f; bias[0] = 1.0f; resident[0] = 0; resident[5] = 1;
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));
    bias[6] = 1.0f;
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));
    assert(change.slot == 0 && change.new_expert == 6);
    bias[6] = 100;
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));
    ids[change.slot] = (int32_t)change.new_expert;
    ds4_resident_route_weights(weights, ids, p, 6, 1.5f);
    /* Bias never enters the mixture coefficient. */
    assert(fabsf(weights[0] / weights[1] - p[6] / p[1]) < 1e-6f);

    base(p, bias, resident);
    ids[0] = 0;
    resident[5] = 1;
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, 1, 8, 1.0f / 6, &change));
    resident[5] = 0; resident[6] = 0; resident[9] = 1;
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, 1, 8, 1.0f / 6, &change));
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, 1, 10, 1.0f / 6, &change));
    assert(change.candidate_rank == 10);
    /* Corrected-score proximity does not imply affinity proximity. A large
     * unbiased coefficient hidden by a negative correction must be rejected. */
    base(p, bias, resident);
    p[6] = 3.0f; bias[6] = -1.55f;
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));
    p[6] = 1.64f; bias[6] = -.19f;
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));
    assert(change.affinity_ratio > 1.0f && change.affinity_ratio <= 1.1f);
    p[6] = 1.66f; bias[6] = -.21f;
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 1.0f / 6, &change));

    /* Protecting five contributors permits nearly 1/6 mass in the sixth.
     * A separate mass cap lets the quality screen abstain on those layers. */
    base(p, bias, resident);
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, .14f, &change));
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, .15f, &change));
    assert(change.old_mass <= .15f);
    assert(!ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 8, 0.0f, &change));
    /* Two original weak misses can share a budget while protecting top four.
     * The first fallback becomes the weakest FINAL route: never replace it. */
    base(p, bias, resident);
    for (int i = 0; i < 4; i++) p[i] = 2.0f;
    p[4] = 1.0f; p[5] = .95f; p[6] = .94f; p[7] = .93f;
    p[8] = .5f; p[9] = .4f;
    resident[4] = 0; resident[5] = 0; resident[7] = 1;
    ds4_resident_route_plan plan;
    assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 2, .20f, &plan));
    assert(plan.count == 2 && plan.total_mass <= .20f);
    assert(plan.changes[0].old_expert == 5 && plan.changes[0].new_expert == 6);
    assert(plan.changes[1].old_expert == 4 && plan.changes[1].new_expert == 7);
    const int32_t shuffled[6] = {5, 0, 4, 1, 2, 3};
    ds4_resident_route_plan reordered;
    assert(ds4_resident_route_choose_plan(shuffled, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 2, .20f, &reordered));
    assert(reordered.count == 2 && reordered.changes[0].slot == 0 && reordered.changes[1].slot == 2);
    int32_t final[6];
    memcpy(final, ids, sizeof(final));
    for (uint32_t i = 0; i < plan.count; i++) final[plan.changes[i].slot] = (int32_t)plan.changes[i].new_expert;
    for (int i = 0; i < 4; i++) assert(final[i] == ids[i]);
    for (int i = 0; i < 6; i++) for (int j = 0; j < i; j++) assert(final[i] != final[j]);
    ds4_resident_route_weights(weights, final, p, 6, 1.5f);
    sum = 0; for (int i = 0; i < 6; i++) sum += weights[i];
    assert(fabsf(sum - 1.5f) < 1e-6f);
    assert(fabsf(weights[4] / weights[5] - p[7] / p[6]) < 1e-6f);
    assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 2, .19f, &plan) && plan.count == 1);
    assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 7, 1.0f / 6, 2, .20f, &plan) && plan.count == 1);
    resident[7] = 0;
    assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 2, .20f, &plan) && plan.count == 1);
    resident[5] = 1;
    assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 2, .20f, &plan) && plan.count == 1);
    assert(plan.changes[0].old_expert == 4);
    assert(!ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 1, .20f, &plan));
    resident[5] = 0;
    assert(ds4_resident_route_choose(ids, p, bias, resident, 10, 6, .1f, 10, 1.0f / 6, &change));
    assert(ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 1, 0.0f, &plan));
    assert(plan.count == 1 && !memcmp(&change, &plan.changes[0], sizeof(change)));
    /* Selected prefill freezes residency before any row admits experts.
     * Reuse the existing planner for later rows even after live residency
     * changes; a cold snapshot must leave both native rows untouched. */
    base(p, bias, resident);
    memcpy(ids, (int32_t[]){0, 1, 2, 3, 4, 5}, sizeof(ids));
    uint8_t frozen[10];
    memcpy(frozen, resident, sizeof(frozen));
    ds4_resident_route_plan first, later;
    assert(ds4_resident_route_choose_plan(ids, p, bias, frozen, 10, 6,
        .1f, 10, 1.0f / 6, 3, .50f, &first));
    resident[5] = 1; resident[6] = 0; /* A demand load and eviction. */
    assert(!ds4_resident_route_choose_plan(ids, p, bias, resident, 10, 6,
        .1f, 10, 1.0f / 6, 3, .50f, &later));
    assert(ds4_resident_route_choose_plan(ids, p, bias, frozen, 10, 6,
        .1f, 10, 1.0f / 6, 3, .50f, &later));
    assert(first.count == later.count && first.changes[0].new_expert == later.changes[0].new_expert);
    memset(frozen, 0, sizeof(frozen));
    assert(!ds4_resident_route_choose_plan(ids, p, bias, frozen, 10, 6,
        .1f, 10, 1.0f / 6, 3, .50f, &later));
    puts("PASS: bounded unique routes through six substitutions, top-contribution protection, two-sided affinity and mass limits, two-original-miss budget, one-slot parity, normalized weights, frozen prefill residency");
    return 0;
}
