/* Private actual-task ownership ledger; shared with model-free CPU fixture.
 * Expects the real caller or a documented minimal worker fixture type.
 * No engine/model/forward/allocator behavior is introduced. */
#ifndef AGENT_SUFFIX_PAIR_ACTUAL_OWNERSHIP_H
#define AGENT_SUFFIX_PAIR_ACTUAL_OWNERSHIP_H
typedef struct {const void *ptr;uint64_t cap;} pair_actual_allocation;
static bool pair_actual_add(pair_actual_allocation *a,int *n,const void *p,uint64_t cap){
    if(!cap)return true;if(!p||*n>=32)return false;
    for(int i=0;i<*n;i++)if(a[i].ptr==p)return a[i].cap==cap;
    a[(*n)++]=(pair_actual_allocation){p,cap};return true;
}
static uint64_t pair_actual_owned(agent_worker *w){
    pair_actual_allocation a[32];int n=0;uint64_t sum=0;
    for(int i=0;i<w->context_frontier_count;i++){
        agent_context_frontier *f=&w->context_frontiers[i];
        if(f->prefix.cap<0||!pair_actual_add(a,&n,f->state.ptr,f->state.cap)||
           !pair_actual_add(a,&n,f->pair_tape.ptr,f->pair_tape.cap)||
           !pair_actual_add(a,&n,f->prefix.v,(uint64_t)f->prefix.cap*sizeof(int)))return UINT64_MAX;
    }
    if(w->transcript.cap<0||!pair_actual_add(a,&n,w->transcript.v,(uint64_t)w->transcript.cap*sizeof(int)))return UINT64_MAX;
    /* Other buffers are stable on this single worker callback. Output is
     * transferred/freed by the UI under mu. Read pointer+cap and deduplicate
     * while mu still protects their owner; never dereference the output.
     * After unlock only recorded caps are read, so a subsequent transfer can
     * conservatively leave the former output cap charged in this snapshot.
     * Buffers already transferred to the UI and global observed peak are not
     * this worker-owned ledger's scope. */
    pthread_mutex_lock(&w->mu);
    const void *out=w->out;uint64_t out_cap=w->out_cap;
    bool output_valid=pair_actual_add(a,&n,out,out_cap);
    pthread_mutex_unlock(&w->mu);
    if(!output_valid)return UINT64_MAX;
    for(int i=0;i<n;i++){if(a[i].cap>UINT64_MAX-sum)return UINT64_MAX;sum+=a[i].cap;}
    uint64_t ring=ds4_session_suffix_pair_tape_bytes(w->session);
    if(ring>UINT64_MAX-sum)return UINT64_MAX;
    sum+=ring;uint64_t structs=sizeof(*w)+sizeof(*w->cfg);if(structs>UINT64_MAX-sum)return UINT64_MAX;
    return sum+structs;
}
#endif
