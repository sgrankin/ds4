/* Frozen bfcde97a header, SHA256
 * a21dbe771153513a5625cfa14940164978cf5574b1a14c5d313ceaf5cec9f37c.
 * Differential checks use the original algorithm, including floating bits. */
#define ds4_resident_route_layer_metrics old_ds4_resident_route_layer_metrics
#define ds4_resident_route_metrics old_ds4_resident_route_metrics
#define ds4_resident_route_metrics_read old_ds4_resident_route_metrics_read
#define ds4_resident_route_change old_ds4_resident_route_change
#define ds4_resident_route_plan old_ds4_resident_route_plan
#define ds4_resident_drop_change old_ds4_resident_drop_change
#define ds4_resident_route_weakest old_ds4_resident_route_weakest
#define ds4_resident_route_choose old_ds4_resident_route_choose
#define ds4_resident_route_weak_pair old_ds4_resident_route_weak_pair
#define ds4_resident_route_choose_plan old_ds4_resident_route_choose_plan
#define ds4_resident_route_choose_drop old_ds4_resident_route_choose_drop
#define ds4_resident_route_drop_weights old_ds4_resident_route_drop_weights
#define ds4_resident_route_weights old_ds4_resident_route_weights
#include "fixtures/resident_route_bfcde97a.h"
#undef ds4_resident_route_layer_metrics
#undef ds4_resident_route_metrics
#undef ds4_resident_route_metrics_read
#undef ds4_resident_route_change
#undef ds4_resident_route_plan
#undef ds4_resident_drop_change
#undef ds4_resident_route_weakest
#undef ds4_resident_route_choose
#undef ds4_resident_route_weak_pair
#undef ds4_resident_route_choose_plan
#undef ds4_resident_route_choose_drop
#undef ds4_resident_route_drop_weights
#undef ds4_resident_route_weights
#undef DS4_RESIDENT_ROUTE_H
#include "ds4_resident_route.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static uint32_t rng_state = UINT32_C(0xd5441337);
static uint32_t next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float unit(void) { return (float)(next() & 1023) / 1024.0f; }

static void compare_plan(const ds4_resident_route_plan *p,
                         const old_ds4_resident_route_plan *old) {
    assert(p->count == old->count);
    assert(!memcmp(&p->total_mass, &old->total_mass, sizeof(float)));
    for (uint32_t i = 0; i < 2; i++) {
        assert(p->changes[i].slot == old->changes[i].slot);
        assert(p->changes[i].old_expert == old->changes[i].old_expert);
        assert(p->changes[i].new_expert == old->changes[i].new_expert);
        assert(p->changes[i].candidate_rank == old->changes[i].candidate_rank);
        assert(!memcmp(&p->changes[i].old_mass, &old->changes[i].old_mass, sizeof(float)));
        assert(!memcmp(&p->changes[i].score_gap, &old->changes[i].score_gap, sizeof(float)));
        assert(!memcmp(&p->changes[i].affinity_ratio, &old->changes[i].affinity_ratio, sizeof(float)));
    }
    const ds4_resident_route_change zero = {0};
    assert(!memcmp(&p->changes[2], &zero, sizeof(zero)));
}

static void choose_native(int32_t *ids, const float *p, const float *bias, uint32_t total) {
    for (uint32_t i = 0; i < 6; i++) ids[i] = -1;
    for (uint32_t e = 0; e < total; e++) {
        for (uint32_t i = 0; i < 6; i++) if (ids[i] < 0 || p[e]+bias[e] > p[ids[i]]+bias[ids[i]]) {
            for (uint32_t j = 5; j > i; j--) ids[j] = ids[j-1];
            ids[i] = (int32_t)e;
            break;
        }
    }
}

static void synthetic(void) {
    int32_t ids[6] = {0, 1, 2, 3, 4, 5};
    float p[10] = {.25f, .20f, .18f, .16f, .12f, .09f, .089f, .119f, .159f, .01f};
    float bias[10] = {0, 0, 0, -.04f, -.02f, 0, 0, -.031f, -.072f, 0};
    uint8_t resident[10] = {1, 1, 1, 0, 0, 0, 1, 1, 1, 0};
    ds4_resident_route_plan plan;
    assert(ds4_resident_route_choose_plan(ids,p,bias,resident,10,6,.30f,10,.20f,3,.40f,&plan));
    assert(plan.count == 3 && plan.total_mass <= .40f);
    assert(plan.changes[0].old_expert == 5 && plan.changes[0].new_expert == 6);
    assert(plan.changes[1].old_expert == 4 && plan.changes[1].new_expert == 7);
    assert(plan.changes[2].old_expert == 3 && plan.changes[2].new_expert == 8);
    assert(ds4_resident_route_choose_plan(ids,p,bias,resident,10,6,.30f,10,.20f,3,.35f,&plan));
    assert(plan.count == 2);
    assert(ds4_resident_route_choose_plan(ids,p,bias,resident,10,6,.30f,10,.20f,3,.20f,&plan));
    assert(plan.count == 1);
    resident[5] = resident[4] = 1;
    assert(ds4_resident_route_choose_plan(ids,p,bias,resident,10,6,.30f,10,.20f,3,.40f,&plan));
    assert(plan.count == 1 && plan.changes[0].old_expert == 3);
    uint32_t slots[3]; float masses[3];
    assert(ds4_resident_route_weak_triple(ids,p,10,6,.20f,slots,masses) == 3);
    assert(slots[0] == 5 && slots[1] == 4 && slots[2] == 3);
    assert(!ds4_resident_route_choose_plan(ids,p,bias,resident,10,6,.30f,6,.20f,3,.40f,&plan));
    assert(!ds4_resident_route_choose_plan(ids,p,bias,resident,10,5,.30f,10,.20f,3,.40f,&plan));
    ids[0] = ids[1];
    assert(!ds4_resident_route_choose_plan(ids,p,bias,resident,10,6,.30f,10,.20f,3,.40f,&plan));
    ids[0] = 0;
    for (uint32_t i = 0; i < 10; i++) p[i] = 1.0f;
    assert(ds4_resident_route_weak_triple(ids,p,10,6,.20f,slots,masses) == 3);
    assert(slots[0] == 0 && slots[1] == 1 && slots[2] == 2);
}

int main(void) {
    synthetic();
    uint64_t parity_plans = 0, triples = 0, third_only = 0;
    uint64_t histogram[4] = {0};
    for (uint32_t trial = 0; trial < 50000; trial++) {
        float p[384], bias[384], weights[6], old_weights[6];
        uint8_t resident[384]; int32_t ids[6];
        for (uint32_t e = 0; e < 384; e++) {
            p[e] = .001f + unit(); bias[e] = unit() - .5f;
            resident[e] = (next() & 3) != 0;
        }
        choose_native(ids,p,bias,384);
        /* Include deterministic index permutations and exact affinity ties. */
        if (trial % 7 == 0) {
            const int32_t temporary = ids[0]; ids[0] = ids[5]; ids[5] = temporary;
        }
        const float gap = trial % 5 == 0 ? .30f : unit();
        const float mass = trial % 5 == 0 ? .20f : unit();
        const float total_mass = trial % 5 == 0 ? .40f : unit();
        const uint32_t rank = trial % 5 == 0 ? 48 : 7 + next() % 378;
        for (uint32_t max_subs = 1; max_subs <= 2; max_subs++) {
            ds4_resident_route_plan plan; old_ds4_resident_route_plan old;
            const bool a = ds4_resident_route_choose_plan(ids,p,bias,resident,384,6,
                gap,rank,mass,max_subs,total_mass,&plan);
            const bool b = old_ds4_resident_route_choose_plan(ids,p,bias,resident,384,6,
                gap,rank,mass,max_subs,total_mass,&old);
            assert(a == b); compare_plan(&plan,&old); parity_plans += a;
            int32_t final[6]; memcpy(final,ids,sizeof(final));
            for (uint32_t i = 0; i < plan.count; i++) final[plan.changes[i].slot] = (int32_t)plan.changes[i].new_expert;
            ds4_resident_route_weights(weights,final,p,6,1.5f);
            old_ds4_resident_route_weights(old_weights,final,p,6,1.5f);
            assert(!memcmp(weights,old_weights,sizeof(weights)));
        }
        ds4_resident_route_plan plan;
        const bool chosen = ds4_resident_route_choose_plan(ids,p,bias,resident,384,6,
            gap,rank,mass,3,total_mass,&plan);
        assert(chosen == (plan.count != 0) && plan.count <= 3);
        histogram[plan.count]++; triples += plan.count == 3;
        assert(plan.total_mass <= total_mass + 1e-6f);
        uint32_t slots[3]; float masses[3];
        const uint32_t n_weak = ds4_resident_route_weak_triple(ids,p,384,6,mass,slots,masses);
        int32_t final[6]; memcpy(final,ids,sizeof(final));
        float accounted = 0.0f;
        for (uint32_t i = 0; i < plan.count; i++) {
            const ds4_resident_route_change c = plan.changes[i];
            bool weak = false;
            for (uint32_t w = 0; w < n_weak; w++) weak |= c.slot == slots[w];
            assert(weak && c.old_expert == (uint32_t)ids[c.slot]);
            assert(!resident[c.old_expert] && resident[c.new_expert]);
            assert(c.old_mass <= mass && c.candidate_rank <= rank);
            assert(c.affinity_ratio >= 1.0f-gap-1e-6f && c.affinity_ratio <= 1.0f+gap+1e-6f);
            assert(c.score_gap <= gap + 1e-6f);
            for (uint32_t j = 0; j < 6; j++) assert(c.new_expert != (uint32_t)ids[j]);
            for (uint32_t j = 0; j < i; j++) assert(c.slot != plan.changes[j].slot && c.new_expert != plan.changes[j].new_expert);
            accounted += c.old_mass; final[c.slot] = (int32_t)c.new_expert;
            third_only += n_weak == 3 && i == 0 && c.slot == slots[2];
        }
        assert(!memcmp(&accounted,&plan.total_mass,sizeof(float)));
        for (uint32_t i = 0; i < 6; i++) {
            bool weak = false;
            for (uint32_t w = 0; w < n_weak; w++) weak |= i == slots[w];
            if (!weak) assert(final[i] == ids[i]);
            for (uint32_t j = 0; j < i; j++) assert(final[i] != final[j]);
        }
        ds4_resident_route_weights(weights,final,p,6,1.5f);
        float sum = 0; for (uint32_t i = 0; i < 6; i++) sum += weights[i];
        assert(fabsf(sum-1.5f) < 1e-5f);
    }
    assert(parity_plans > 0 && triples > 0 && third_only > 0);
    printf("PASS:100000 one/two-slot differential plans and weight bits;50000 triple invariants; parity_plans=%llu triple_plans=%llu third_only=%llu histogram=%llu,%llu,%llu,%llu\n",
        (unsigned long long)parity_plans,(unsigned long long)triples,(unsigned long long)third_only,
        (unsigned long long)histogram[0],(unsigned long long)histogram[1],
        (unsigned long long)histogram[2],(unsigned long long)histogram[3]);
    return 0;
}
