#include "ds4_resident_route.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static uint32_t state = 0x31a787c5u;
static uint32_t random_u32(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

int main(void) {
    const uint32_t total = 384, used = 6;
    float p[384];
    uint8_t resident[6];
    int32_t ids[6];
    float weights[6];
    for (uint32_t trial = 0; trial < 50000; trial++) {
        for (uint32_t i = 0; i < total; i++)
            p[i] = (float)(1 + random_u32() % 10000) / 1024.0f;
        for (uint32_t i = 0; i < used; i++) {
            bool duplicate;
            do {
                ids[i] = (int32_t)(random_u32() % total);
                duplicate = false;
                for (uint32_t j = 0; j < i; j++) duplicate |= ids[i] == ids[j];
            } while (duplicate);
            resident[i] = random_u32() & 1;
        }
        const float cap = trial & 1 ? .10f : .16f;
        uint32_t weakest = 0;
        float sum = 0;
        bool any_other_resident = false;
        for (uint32_t i = 0; i < used; i++) {
            sum += p[ids[i]];
            if (p[ids[i]] < p[ids[weakest]]) weakest = i;
        }
        for (uint32_t i = 0; i < used; i++)
            any_other_resident |= i != weakest && resident[i];
        const bool expected = !resident[weakest] && any_other_resident &&
            p[ids[weakest]] / sum <= cap;
        ds4_resident_drop_change change;
        const bool chosen = ds4_resident_route_choose_drop(ids, p, resident,
            total, used, cap, &change);
        assert(chosen == expected);
        if (!chosen) continue;
        assert(change.slot == weakest && change.old_expert == (uint32_t)ids[weakest]);
        assert(change.alias_slot != weakest && resident[change.alias_slot]);
        assert(change.alias_expert == (uint32_t)ids[change.alias_slot]);
        assert(change.old_mass <= cap);
        int32_t final[6];
        memcpy(final, ids, sizeof(final));
        final[change.slot] = (int32_t)change.alias_expert;
        ds4_resident_route_drop_weights(weights, ids, p, used, change.slot, 1.5f);
        assert(weights[weakest] == 0.0f && !signbit(weights[weakest]));
        float final_sum = 0;
        for (uint32_t i = 0; i < used; i++) {
            final_sum += weights[i];
            if (i == weakest) continue;
            assert(final[i] == ids[i] && weights[i] > 0);
            for (uint32_t j = 0; j < used; j++) {
                if (j == weakest) continue;
                assert(fabsf(weights[i] / weights[j] - p[ids[i]] / p[ids[j]]) <
                    3e-5f * fmaxf(1.0f, p[ids[i]] / p[ids[j]]));
            }
        }
        assert(fabsf(final_sum - 1.5f) < 1e-6f);
        /* A resident weakest never permits dropping a stronger missing route. */
        resident[weakest] = 1;
        assert(!ds4_resident_route_choose_drop(ids, p, resident,
            total, used, cap, &change));
    }
    assert(!ds4_resident_route_choose_drop(NULL, p, resident, total, used, .1f,
        &(ds4_resident_drop_change){0}));
    const int32_t repeated[6] = {0,1,2,3,4,4};
    assert(!ds4_resident_route_choose_drop(repeated, p, resident, total, used, 1.0f,
        &(ds4_resident_drop_change){0}));
    puts("PASS: 50000 native-order cases, original weakest only, five original positive contributions, resident zero-slot alias, mass guard, normalized affinity ratios");
    return 0;
}
