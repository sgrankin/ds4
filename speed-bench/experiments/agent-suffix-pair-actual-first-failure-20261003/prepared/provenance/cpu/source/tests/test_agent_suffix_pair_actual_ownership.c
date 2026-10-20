/* Model-free exact private-hook ledger seam. No native core/object/model links.
 * Uses pinned ds4.h public allocation types and a minimal documented worker.
 * sizeof(worker/cfg) here is fixture-specific; this is not a native ABI or
 * global allocator/peak proof. */
#include "ds4.h"
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {uint64_t padding;} fixture_config;
typedef struct {
    ds4_session_snapshot state,pair_tape;
    ds4_tokens prefix;
} agent_context_frontier;
typedef struct {
    pthread_mutex_t mu;
    fixture_config *cfg;
    ds4_session *session;
    agent_context_frontier context_frontiers[2];
    int context_frontier_count;
    ds4_tokens transcript;
    char *out;
    size_t out_cap;
} agent_worker;
static uint64_t mock_ring=41;
uint64_t ds4_session_suffix_pair_tape_bytes(const ds4_session *s){(void)s;return mock_ring;}
static agent_worker *current;
static int transfer_after_unlock;
static uint64_t locks,unlocks;
static void fixture_mutex_lock(pthread_mutex_t *mu){assert(!pthread_mutex_lock(mu));locks++;}
static void fixture_mutex_unlock(pthread_mutex_t *mu){
    assert(!pthread_mutex_unlock(mu));unlocks++;
    if(transfer_after_unlock){
        transfer_after_unlock=0;
        /* Same ownership transfer/reset under mu as real worker_consume;
         * then free immediately to catch accidental later dereference. */
        assert(!pthread_mutex_lock(&current->mu));
        char *owned=current->out;current->out=NULL;current->out_cap=0;
        assert(!pthread_mutex_unlock(&current->mu));free(owned);
    }
}
#define pthread_mutex_lock fixture_mutex_lock
#define pthread_mutex_unlock fixture_mutex_unlock
#include "speed-bench/agent_suffix_pair_actual_ownership.h"
#undef pthread_mutex_lock
#undef pthread_mutex_unlock

static void init(agent_worker *w,fixture_config *cfg){
    *w=(agent_worker){0};assert(!pthread_mutex_init(&w->mu,NULL));w->cfg=cfg;
    w->context_frontier_count=2;
    w->context_frontiers[0].state=(ds4_session_snapshot){.ptr=malloc(19),.len=11,.cap=19};
    w->context_frontiers[0].pair_tape=(ds4_session_snapshot){.ptr=malloc(23),.len=13,.cap=23};
    w->context_frontiers[0].prefix=(ds4_tokens){.v=calloc(5,sizeof(int)),.len=3,.cap=5};
    w->context_frontiers[1].state=(ds4_session_snapshot){.ptr=malloc(29),.len=17,.cap=29};
    w->context_frontiers[1].pair_tape=(ds4_session_snapshot){.ptr=malloc(31),.len=19,.cap=31};
    w->context_frontiers[1].prefix=(ds4_tokens){.v=calloc(7,sizeof(int)),.len=5,.cap=7};
    w->transcript=(ds4_tokens){.v=calloc(11,sizeof(int)),.len=9,.cap=11};
    assert(w->context_frontiers[0].state.ptr&&w->context_frontiers[0].pair_tape.ptr&&w->context_frontiers[0].prefix.v&&w->context_frontiers[1].state.ptr&&w->context_frontiers[1].pair_tape.ptr&&w->context_frontiers[1].prefix.v&&w->transcript.v);
}
static uint64_t independent_base(void){return 19+23+5*sizeof(int)+29+31+7*sizeof(int)+11*sizeof(int)+41+sizeof(agent_worker)+sizeof(fixture_config);}
static void finish(agent_worker *w){
    for(int i=0;i<2;i++){free(w->context_frontiers[i].state.ptr);free(w->context_frontiers[i].pair_tape.ptr);free(w->context_frontiers[i].prefix.v);}
    free(w->transcript.v);free(w->out);assert(!pthread_mutex_destroy(&w->mu));
}
static void single_cases(void){
    fixture_config cfg={0};agent_worker w;init(&w,&cfg);current=&w;
    assert(pair_actual_owned(&w)==independent_base());assert(locks==1&&unlocks==1);
    w.out=malloc(37);w.out_cap=37;assert(w.out);
    uint64_t before=pair_actual_owned(&w);assert(before==independent_base()+37);
    transfer_after_unlock=1;uint64_t transferred=pair_actual_owned(&w);
    assert(transferred==independent_base()+37);assert(!w.out&&w.out_cap==0);
    assert(pair_actual_owned(&w)==independent_base());
    /* Same identity/same cap must count once; inconsistent caps fail closed. */
    w.out=(char*)w.context_frontiers[0].state.ptr;w.out_cap=19;
    assert(pair_actual_owned(&w)==independent_base());w.out_cap=20;
    assert(pair_actual_owned(&w)==UINT64_MAX);w.out=NULL;w.out_cap=0;
    w.out_cap=37;assert(pair_actual_owned(&w)==UINT64_MAX);w.out_cap=0;
    mock_ring=UINT64_MAX;assert(pair_actual_owned(&w)==UINT64_MAX);mock_ring=41;
    w.transcript.cap=-1;assert(pair_actual_owned(&w)==UINT64_MAX);w.transcript.cap=11;
    assert(locks==unlocks);finish(&w);
}

typedef struct {agent_worker *w;char *a,*b;atomic_int stop;} race;
static void *ui_transfer(void *arg){
    race *r=arg;
    while(!atomic_load_explicit(&r->stop,memory_order_relaxed)){
        assert(!pthread_mutex_lock(&r->w->mu));r->w->out=NULL;r->w->out_cap=0;assert(!pthread_mutex_unlock(&r->w->mu));
        assert(!pthread_mutex_lock(&r->w->mu));r->w->out=r->a;r->w->out_cap=17;assert(!pthread_mutex_unlock(&r->w->mu));
        assert(!pthread_mutex_lock(&r->w->mu));r->w->out=r->b;r->w->out_cap=43;assert(!pthread_mutex_unlock(&r->w->mu));
    }
    return NULL;
}
static void stress(void){
    fixture_config cfg={0};agent_worker w;init(&w,&cfg);current=&w;
    race r={.w=&w,.a=malloc(17),.b=malloc(43)};assert(r.a&&r.b);atomic_init(&r.stop,0);
    pthread_t ui;assert(!pthread_create(&ui,NULL,ui_transfer,&r));uint64_t base=independent_base();
    for(int i=0;i<50000;i++){uint64_t got=pair_actual_owned(&w);assert(got==base||got==base+17||got==base+43);}
    atomic_store_explicit(&r.stop,1,memory_order_relaxed);assert(!pthread_join(ui,NULL));
    w.out=NULL;w.out_cap=0;free(r.a);free(r.b);assert(locks==unlocks);finish(&w);
}
int main(void){single_cases();stress();printf("private ownership CPU seam: PASS50000 concurrent snapshots; exact cap/dedup/transfer/free/overflow controls; no model/native objects\n");return 0;}
