/* Private SOURCE/UNCOMPILED. Included after checked native pending finish.
 * All adapter callbacks execute on the owner under reader metadata trylock.
 * Worker reaches ONLY pread into the already reserved shared slab slot.
 * No Metal allocation, cache_wait_inflight, pool join, pread or mlock in hooks.
 */
#include "ds4_original_deadline.c" /* literal 87ac worker, same production TU */

typedef struct {
    const void *owner;
    pthread_t thread;
    ds4_od_model model;
    ds4_od_reader *reader;
    uint64_t serial,model_map_size;
    int bound_fd;
    __strong id<MTLBuffer> lease_buffer;
    uint64_t lease_base;
    uint32_t lease_slot;
    ds4_od_expert lease_expert;
    ds4_od_ticket ticket;
    ds4_gpu_stream_expert_table current_table;
    int32_t candidate[6];
    int ticket_valid,window_active,deadline_finished,closing,closed_stats_valid;
    ds4_od_stats closed_stats;
} ds4_od_native_context;
static ds4_od_native_context g_original_deadline;
static uint64_t g_original_deadline_generation;

static int ds4_od_native_owner(const void *owner) {
    return g_original_deadline.reader && owner==g_original_deadline.owner &&
        pthread_equal(g_original_deadline.thread,pthread_self());
}
static int ds4_od_native_range(id<MTLBuffer> b,uint64_t off,uint64_t n) {
    return b && [b contents] && n && off<=(uint64_t)[b length] &&
        n<=(uint64_t)[b length]-off && off<=(uint64_t)NSUIntegerMax;
}
static int ds4_od_native_slot(id<MTLBuffer> gate,NSUInteger gi,
        id<MTLBuffer> up,NSUInteger ui,id<MTLBuffer> down,NSUInteger di,
        uint64_t gb,uint64_t db,uint32_t *slot) {
    return gate && gate==up && gate==down && gb==UINT64_C(3041280) &&
        db==UINT64_C(3870720) && gi<=NSUIntegerMax-gb*2 &&
        ui==gi+gb && di==gi+gb*2 &&
        ds4_od_native_range(gate,gi,g_stream_expert_cache_slab_slot_bytes) &&
        ds4_od_native_range(up,ui,gb) && ds4_od_native_range(down,di,db) &&
        ds4_gpu_stream_expert_slab_slot_for_buffer(gate,gi,slot);
}
static int ds4_od_native_addresses(uint32_t layer,uint32_t expert,
        id<MTLBuffer> buffer,uint64_t base,int compare_existing) {
    if(layer>=40 || expert>=384 || base>NSUIntegerMax-UINT64_C(6082560)) return 0;
    id<MTLBuffer> a[3]={g_stream_expert_cache_gate_addr_buffers[layer],
        g_stream_expert_cache_up_addr_buffers[layer],g_stream_expert_cache_down_addr_buffers[layer]};
    const uint64_t inner[3]={base,base+3041280,base+6082560};
    for(unsigned k=0;k<3;k++) {
        if(!ds4_od_native_range(a[k],0,384*sizeof(uint64_t))) return 0;
        uint64_t gpu=ds4_gpu_buffer_address(buffer,(NSUInteger)inner[k]);
        if(!gpu) return 0;
        uint64_t saved;memcpy(&saved,(uint8_t *)[a[k] contents]+expert*sizeof(uint64_t),8);
        if(compare_existing && saved!=gpu) return 0;
    }
    return 1;
}
static int ds4_od_native_seen(uint8_t *bits,uint32_t slot,uint32_t cap) {
    if(slot>=cap) return -1;
    const uint8_t mask=(uint8_t)(1u<<(slot&7u));
    if(bits[slot>>3]&mask) return 1;
    bits[slot>>3]|=mask;return 0;
}
/* Independent physical identity census. Pending/cache aliases count once;
 * duplicate valid owners and original/cache aliases are fatal contradictions.
 * Bounded3.75KiB stack; scans up to30720 native records, cost is NOT free. */
static int ds4_od_native_occupied(void *v,ds4_od_occupied *out) {
    ds4_od_native_context *c=v;
    if(!c || c!=&g_original_deadline || !out ||
       !pthread_equal(c->thread,pthread_self())) return 0;
    const uint32_t cap=ds4_gpu_stream_expert_cache_configured_budget();
    if(!cap || cap>DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) return 0;
    uint8_t bits[(DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES+7)/8]={0};
    ds4_od_occupied o={.capacity=cap};
    for(uint32_t l=0;l<DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;l++)
      for(uint32_t e=0;e<DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;e++) {
        ds4_gpu_stream_expert_cache_entry *p=&g_stream_expert_cache[l][e];uint32_t slot;
        if(!p->valid) continue;
        if(!p->slab_backed || !ds4_od_native_slot(p->gate_buffer,p->gate_inner,
             p->up_buffer,p->up_inner,p->down_buffer,p->down_inner,
             p->gate_expert_bytes,p->down_expert_bytes,&slot) || slot!=p->slab_slot ||
             ds4_od_native_seen(bits,slot,cap)!=0) return 0;
        o.valid++;
      }
    if(o.valid!=g_stream_expert_cache_entry_count) return 0;
    ds4_gpu_stream_expert_pending_load *p=&g_stream_expert_pending_load;
    for(uint32_t i=0;i<DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;i++) {
        if(!p->gate_bufs[i] && !p->up_bufs[i] && !p->down_bufs[i]) continue;
        uint32_t slot;
        if(!ds4_od_native_slot(p->gate_bufs[i],p->gate_inners[i],p->up_bufs[i],
           p->up_inners[i],p->down_bufs[i],p->down_inners[i],p->gate_expert_bytes,
           p->down_expert_bytes,&slot)) return 0;
        const int seen=ds4_od_native_seen(bits,slot,cap);if(seen<0) return 0;
        if(!seen) o.mandatory_detached++;
    }
    if(g_original_deadline_reserved_slots) {
        if(g_original_deadline_reserved_slots!=1 || !c->lease_buffer ||
           ds4_od_native_seen(bits,c->lease_slot,cap)!=0) return 0;
        o.original_leases=1;
    }
    if((uint64_t)o.valid+o.mandatory_detached+o.original_leases>cap) return 0;
    if(c->closing && !o.original_leases && c->reader) {
        /* Close's final occupied callback is after join, under reader.mu,
         * after release bookkeeping. The literal reader TU is included above;
         * no trylock reentry or race, and no worker fields read on hot paths. */
        ds4_od_reader *r=c->reader;
        c->closed_stats=r->stats;c->closed_stats.state=DS4_OD_FREE;
        c->closed_stats.worker_present=c->closed_stats.worker_active=0;
        c->closed_stats.contention=atomic_load_explicit(&r->contention,memory_order_relaxed);
        c->closed_stats.deadline_tries=atomic_load_explicit(&r->deadline_attempts,memory_order_relaxed);
        c->closed_stats.deadline_misses=atomic_load_explicit(&r->deadline_misses,memory_order_relaxed);
        c->closed_stats_valid=1;
    }
    *out=o;return 1;
}
static int ds4_od_native_expert_equal(const ds4_od_expert *a,const ds4_od_expert *b) {
    return a && b && od_expert_equal(a,b); /* semantic fields, never struct padding */
}
static int ds4_od_native_ids(const int32_t ids[6]) {
    if(!ids) return 0;
    for(unsigned i=0;i<6;i++) {
        if(ids[i]<0 || ids[i]>=384) return 0;
        for(unsigned j=0;j<i;j++) if(ids[i]==ids[j]) return 0;
    }
    return 1;
}
static int ds4_od_native_expert(ds4_od_native_context *c,
        const ds4_gpu_stream_expert_table *t,uint32_t id,ds4_od_expert *out) {
    if(!c || c!=&g_original_deadline || !t || !out || t->model_map!=(const void *)c->model.map_cookie ||
       t->model_size!=g_model_map_size || t->layer>=40 || t->n_total_expert!=384 ||
       id>=384 || t->gate_expert_bytes!=c->model.gate_bytes ||
       t->down_expert_bytes!=c->model.down_bytes) return 0;
    ds4_od_expert x={.model=c->model,.layer=t->layer,.original=id};
    const uint64_t base[3]={t->gate_offset,t->up_offset,t->down_offset};
    for(unsigned k=0;k<3;k++) {
        const uint64_t n=k==2?c->model.down_bytes:c->model.gate_bytes;
        if(base[k]>c->model.mapped_bytes || id>(c->model.mapped_bytes-base[k])/n) return 0;
        x.offset[k]=base[k]+id*n;
        if(n>c->model.mapped_bytes-x.offset[k]) return 0;
    }
    *out=x;return 1;
}
static int ds4_od_native_exact(const ds4_od_expert *x) {
    if(!x || x->layer>=40 || x->original>=384) return 0;
    ds4_gpu_stream_expert_cache_entry *e=&g_stream_expert_cache[x->layer][x->original];uint32_t slot;
    return ds4_gpu_stream_expert_cache_entry_matches(e,(void *)x->model.map_cookie,
        g_model_map_size,x->offset[0],x->offset[1],x->offset[2],x->model.gate_bytes,
        x->model.down_bytes) && ds4_od_native_slot(e->gate_buffer,e->gate_inner,
        e->up_buffer,e->up_inner,e->down_buffer,e->down_inner,e->gate_expert_bytes,
        e->down_expert_bytes,&slot) && slot==e->slab_slot &&
        ds4_od_native_addresses(x->layer,x->original,e->gate_buffer,e->gate_inner,1);
}
static int ds4_od_native_pending_alias(uint32_t slot) {
    ds4_gpu_stream_expert_pending_load *p=&g_stream_expert_pending_load;
    for(uint32_t i=0;i<DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;i++) {
        uint32_t actual;
        if(!p->gate_bufs[i] && !p->up_bufs[i] && !p->down_bufs[i]) continue;
        if(!ds4_od_native_slot(p->gate_bufs[i],p->gate_inners[i],p->up_bufs[i],
            p->up_inners[i],p->down_bufs[i],p->down_inners[i],p->gate_expert_bytes,
            p->down_expert_bytes,&actual)) return 1; /* Unsafe/unknown owner. */
        if(actual==slot) return 1;
    }
    return 0;
}
static int ds4_od_native_reserve(void *v,const ds4_od_expert *x,
        const ds4_od_protection *protect,ds4_od_lease *out) {
    ds4_od_native_context *c=v;ds4_od_occupied o;
    if(!c || c!=&g_original_deadline || !x || !protect || !out ||
       !ds4_od_native_owner(c->owner) || g_original_deadline_reserved_slots ||
       g_stream_expert_pending_load.active) return 0;
    if(x->layer>=40 || x->original>=384 || !od_model_equal(&x->model,&c->model) ||
       !od_protection_valid(x,protect)) return -1;
    for(unsigned k=0;k<3;k++) {
        const uint64_t n=k==2?c->model.down_bytes:c->model.gate_bytes;
        if(x->offset[k]>c->model.mapped_bytes || n>c->model.mapped_bytes-x->offset[k]) return -1;
        for(unsigned j=0;j<k;j++) {
            const uint64_t m=j==2?c->model.down_bytes:c->model.gate_bytes;
            if(x->offset[k]<x->offset[j]+m && x->offset[j]<x->offset[k]+n) return -1;
        }
    }
    if(!ds4_od_native_occupied(c,&o)) return -1;
    if(ds4_od_native_exact(x)) return 0;
    ds4_gpu_stream_expert_cache_entry *target=&g_stream_expert_cache[x->layer][x->original];
    if(target->valid) return 0; /* Foreign identity is never destructively replaced. */
    uint32_t vl=UINT32_MAX,ve=UINT32_MAX,lowest=UINT32_MAX;uint64_t oldest=UINT64_MAX;
    for(uint32_t l=0;l<DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;l++)
      for(uint32_t e=0;e<DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;e++) {
        ds4_gpu_stream_expert_cache_entry *p=&g_stream_expert_cache[l][e];uint32_t slot;
        if(!p->valid || !p->slab_backed || ds4_gpu_stream_expert_cache_entry_inflight(p) ||
           ds4_gpu_stream_expert_cache_entry_protected(l,e,protect->layer,protect->ids,protect->count) ||
           !ds4_od_native_slot(p->gate_buffer,p->gate_inner,p->up_buffer,p->up_inner,
             p->down_buffer,p->down_inner,p->gate_expert_bytes,p->down_expert_bytes,&slot) ||
           slot!=p->slab_slot || slot>=o.capacity ||
           !g_stream_expert_cache_slab_slot_locked[slot] || ds4_od_native_pending_alias(slot)) continue;
        uint32_t hot=g_stream_expert_cache_route_hotness[l][e];
        if(hot<lowest || (hot==lowest && p->last_used<oldest)) {
            lowest=hot;oldest=p->last_used;vl=l;ve=e;
        }
      }
    if(vl==UINT32_MAX) return 0;
    ds4_gpu_stream_expert_cache_entry *p=&g_stream_expert_cache[vl][ve];
    if(!ds4_od_native_addresses(x->layer,x->original,p->gate_buffer,p->gate_inner,0) ||
       g_stream_expert_cache_slab_slot_bytes>DS4_OD_MAX_PAYLOAD) return 0;
    if(!ds4_od_native_addresses(vl,ve,p->gate_buffer,p->gate_inner,1) ||
       !g_stream_expert_cache_layer_count[vl] || !g_stream_expert_cache_entry_count ||
       p->logical_bytes>g_stream_expert_cache_bytes ||
       g_stream_expert_cache_evictions==UINT64_MAX ||
       g_stream_expert_cache_layer_evictions[vl]==UINT64_MAX ||
       g_stream_expert_cache_buffer_reuses==UINT64_MAX) return -1;
    /* Validate first, then adopt BEFORE metadata detach. No later refusal.
     * Retained lock and slab identity remain unchanged; slot never enters free list.
     * Native eviction advice is intentionally absent from this metadata-only hook. */
    c->lease_buffer=p->gate_buffer;c->lease_base=p->gate_inner;c->lease_slot=p->slab_slot;c->lease_expert=*x;
    *out=(ds4_od_lease){.owner_handle=c,.base=(unsigned char *)[c->lease_buffer contents]+c->lease_base,
        .capacity=g_stream_expert_cache_slab_slot_bytes,.inner={0,x->model.gate_bytes,x->model.gate_bytes*2},
        .physical_slot=c->lease_slot,.retained_lock=1};
    g_original_deadline_reserved_slots=1;
    const uint64_t logical=p->logical_bytes;
    ds4_gpu_stream_expert_cache_zero_addr_slot(vl,ve);
    *p=(ds4_gpu_stream_expert_cache_entry){0};
    g_stream_expert_cache_layer_count[vl]--;g_stream_expert_cache_entry_count--;
    g_stream_expert_cache_bytes-=logical;g_stream_expert_cache_evictions++;
    g_stream_expert_cache_layer_evictions[vl]++;g_stream_expert_cache_buffer_reuses++;
    return 1;
}
static int ds4_od_native_lease_matches(ds4_od_native_context *c,const ds4_od_lease *l) {
    return l && l->owner_handle==c && g_original_deadline_reserved_slots==1 &&
        c->lease_buffer && l->physical_slot==c->lease_slot && l->retained_lock==1 &&
        l->capacity==g_stream_expert_cache_slab_slot_bytes &&
        l->base==(unsigned char *)[c->lease_buffer contents]+c->lease_base &&
        l->inner[0]==0 && l->inner[1]==c->model.gate_bytes && l->inner[2]==c->model.gate_bytes*2;
}
static int ds4_od_native_install(void *v,const ds4_od_expert *x,
        const ds4_od_protection *protect,const ds4_od_lease *l) {
    ds4_od_native_context *c=v;ds4_od_occupied o;
    if(!c || c!=&g_original_deadline || !ds4_od_native_owner(c->owner) ||
       !ds4_od_native_lease_matches(c,l) || !ds4_od_native_occupied(c,&o) ||
       !x || !protect || !ds4_od_native_expert_equal(x,&c->lease_expert)) return -1;
    if(ds4_od_native_exact(x)) return DS4_OD_INSTALL_ALREADY_NATIVE;
    ds4_gpu_stream_expert_cache_entry *e=&g_stream_expert_cache[x->layer][x->original];
    if(e->valid || g_stream_expert_pending_load.active ||
       ds4_od_native_pending_alias(c->lease_slot)) return 0;
    if(!ds4_od_native_addresses(x->layer,x->original,c->lease_buffer,c->lease_base,0)) return 0;
    /* target invalid + preexisting addr storage => native installer does not
     * wait or allocate. Reader mutex publishes completed FD writes before here. */
    e=ds4_gpu_stream_expert_cache_install_loaded((void *)x->model.map_cookie,g_model_map_size,
        x->layer,x->original,x->offset[0],x->offset[1],x->offset[2],x->model.gate_bytes,
        x->model.down_bytes,c->lease_buffer,c->lease_buffer,c->lease_buffer,
        (NSUInteger)c->lease_base,(NSUInteger)(c->lease_base+x->model.gate_bytes),
        (NSUInteger)(c->lease_base+x->model.gate_bytes*2));
    if(!e) {ds4_gpu_stream_expert_cache_zero_addr_slot(x->layer,x->original);return -1;}
    g_original_deadline_reserved_slots=0;c->lease_buffer=nil;c->lease_base=0;
    return DS4_OD_INSTALL_TRANSFERRED;
}
static void ds4_od_native_release(void *v,const ds4_od_lease *l) {
    ds4_od_native_context *c=v;ds4_od_occupied o;
    if(!c || c!=&g_original_deadline || !ds4_od_native_owner(c->owner) ||
       !ds4_od_native_lease_matches(c,l) || !ds4_od_native_occupied(c,&o) ||
       ds4_od_native_pending_alias(c->lease_slot)) abort();
    for(uint32_t i=0;i<g_stream_expert_cache_free_slot_count;i++)
        if(g_stream_expert_cache_free_slots[i]==c->lease_slot) abort();
    /* Reader close has already joined; tick/future release only READY/FAILED.
     * Locked slab goes back to existing free list, no unlock/new allocation. */
    ds4_gpu_stream_expert_slab_push_free_slot(c->lease_slot);
    g_original_deadline_reserved_slots=0;c->lease_buffer=nil;c->lease_base=0;
}
int ds4_gpu_original_deadline_close(const void *owner) {
    ds4_od_native_context *c=&g_original_deadline;
    if(!c->reader) return 1;
    if(!ds4_od_native_owner(owner)) return 0;
    ds4_od_reader *r=c->reader;c->closing=1;c->closed_stats_valid=0;
    /* The accepted close can consume r even on its final occupancy refusal.
     * Fatal after failure: retain all native resources, never retry the pointer. */
    if(!ds4_od_close(r)) abort();
    c->reader=NULL;
    if(!c->closed_stats_valid) abort();
    const ds4_od_stats *s=&c->closed_stats;
    fprintf(stderr,"ds4: ORIGINAL_DEADLINE generation=%" PRIu64 " requests=%" PRIu64
        " started=%" PRIu64 " completed=%" PRIu64 " failed=%" PRIu64
        " promoted=%" PRIu64 " future=%" PRIu64 " expired=%" PRIu64
        " deadline_tries=%" PRIu64 " deadline_misses=%" PRIu64
        " logical_requested=%" PRIu64 " pread_calls=%" PRIu64
        " pread_requested=%" PRIu64 " pread_returned=%" PRIu64
        " reader_host=%" PRIu64 " adapter_reserve=65536 payload_peak=%" PRIu64
        " overflow=%d\n",c->model.generation,s->requests,s->started,s->completed,s->failed,
        s->promoted,s->future_used,s->expired,s->deadline_tries,s->deadline_misses,
        s->logical_requested,s->pread_calls,s->pread_requested,s->pread_returned,
        s->known_host_peak,s->original_payload_peak,s->overflow);
    c->owner=NULL;c->ticket_valid=c->window_active=c->deadline_finished=c->closing=0;
    g_original_deadline_enabled=0;g_original_deadline_protect_count=0;
    return 1;
}
int ds4_gpu_original_deadline_close_model(const void *map,uint64_t size,int fd) {
    ds4_od_native_context *c=&g_original_deadline;
    if(!c->reader || map!=(void *)c->model.map_cookie ||
       size!=c->model_map_size || fd!=c->bound_fd) return 1;
    return ds4_gpu_original_deadline_close(c->owner);
}
static int ds4_gpu_original_deadline_shutdown_current(void) {
    return ds4_gpu_original_deadline_close(g_original_deadline.owner);
}
/* Native model_unmap_engram truncates the CPU mapping to m->size. GPU views
 * can cover only disjoint non-expert spans: their summed bytes/first offset
 * are not the mapped-prefix bound for absolute FD reads. */
static int ds4_od_native_binding(const void *owner,const void *map,uint64_t size,int fd) {
    return owner && map && size && map==g_model_map_ptr && size==g_model_map_size &&
        fd>=0 && fd==g_model_fd && g_ssd_streaming_mode &&
        ds4_gpu_stream_expert_slab_enabled() && ds4_gpu_stream_expert_combined_buffer_enabled();
}
int ds4_gpu_original_deadline_token_begin(const void *owner,const void *map,uint64_t size,int fd) {
    ds4_od_native_context *c=&g_original_deadline;
    if(!ds4_od_native_binding(owner,map,size,fd)) return 0;
    if(c->reader && !ds4_od_native_owner(owner)) return 0;
    if(!c->reader) {
        struct stat st;
        if(fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<=0 ||
           (uint64_t)st.st_size<size || g_original_deadline_generation==UINT64_MAX) return 0;
        c->owner=owner;c->thread=pthread_self();c->serial=0;c->closed_stats_valid=0;
        c->bound_fd=fd;c->model_map_size=size;
        c->model=(ds4_od_model){.owner=(uint64_t)(uintptr_t)owner,.generation=++g_original_deadline_generation,
            .device=(uint64_t)st.st_dev,.inode=(uint64_t)st.st_ino,.whole_bytes=(uint64_t)st.st_size,
            .mapped_bytes=size,.map_cookie=(uintptr_t)map,
            .gate_bytes=3041280,.down_bytes=3870720,.layers=40,.experts=384,.gate_type=16,.down_type=10};
        const ds4_od_adapter adapter={c,ds4_od_native_reserve,ds4_od_native_install,
            ds4_od_native_release,ds4_od_native_occupied};
        if(!ds4_od_open(&c->reader,&c->model,fd,&adapter)) {c->owner=NULL;return 0;}
        /* Setup-only blocking metadata lock; never at the binding deadline.
         * 64KiB covers page-rounded static
         * adapter context plus bounded census/owner call stack, no new GPU
         * wrapper/storage. Reader's actual configured stack/guard are separate. */
        if(pthread_mutex_lock(&c->reader->mu)) abort();
        const uint64_t reader_host=c->reader->stats.known_host_peak;
        if(pthread_mutex_unlock(&c->reader->mu)) abort();
        const long page=sysconf(_SC_PAGESIZE);
        const uint64_t metadata=page>0 ? ((sizeof(*c)+128u+(uint64_t)page-1u)/
            (uint64_t)page)*(uint64_t)page : UINT64_MAX;
        if(page<=0 || metadata>65536-8192 || reader_host>DS4_OD_HOST_CAP-65536) {
            if(!ds4_gpu_original_deadline_close(owner)) abort();
            return 0;
        }
        g_original_deadline_enabled=1;
    }
    if(c->window_active || c->serial==UINT64_MAX) return 0;
    c->serial++;return ds4_od_try_tick(c->reader,c->serial)>=0;
}
int ds4_gpu_original_deadline_future_selected(const void *owner,
        const ds4_gpu_stream_expert_table *t,const int32_t native[6]) {
    ds4_od_native_context *c=&g_original_deadline;if(!ds4_od_native_owner(owner) || !ds4_od_native_ids(native)) return 0;
    ds4_od_protection p={.layer=t?t->layer:0,.count=6};memcpy(p.ids,native,6*sizeof(int32_t));
    for(unsigned i=0;i<6;i++) {
        if(native[i]<0) return 0;ds4_od_expert x;
        if(!ds4_od_native_expert(c,t,(uint32_t)native[i],&x) ||
           ds4_od_try_future(c->reader,&x,&p,c->serial)<0) return 0;
    }
    return 1;
}
int ds4_gpu_original_deadline_arm_selected(const void *owner,
        const ds4_gpu_stream_expert_table *t,const int32_t native[6],const int32_t candidate[6]) {
    ds4_od_native_context *c=&g_original_deadline;
    if(!ds4_od_native_owner(owner) || !t || !ds4_od_native_ids(native) ||
       !ds4_od_native_ids(candidate) || c->window_active) return -1;
    unsigned changes=0,slot=0;for(unsigned i=0;i<6;i++)if(native[i]!=candidate[i]){changes++;slot=i;}
    c->ticket_valid=0;c->deadline_finished=0;c->current_table=*t;memcpy(c->candidate,candidate,sizeof c->candidate);
    if(!changes) return 0;if(changes!=1) return -1;
    ds4_od_request q={.serial=c->serial,.changed_slot=slot};
    if(!ds4_od_native_expert(c,t,(uint32_t)native[slot],&q.expert)) return -1;
    memcpy(q.native,native,sizeof q.native);memcpy(q.candidate,candidate,sizeof q.candidate);
    c->window_active=1;g_original_deadline_protect_layer=t->layer;
    memcpy(g_original_deadline_protect_ids,native,sizeof q.native);
    g_original_deadline_protect_ids[6]=candidate[slot];g_original_deadline_protect_count=7;
    const int status=ds4_od_try_arm(c->reader,&q,&c->ticket);
    c->ticket_valid=status==1;return status;
}
int ds4_gpu_original_deadline_finish_selected(const void *owner,int32_t final_ids[6],int *promoted) {
    ds4_od_native_context *c=&g_original_deadline;
    if(!ds4_od_native_owner(owner) || !final_ids || !promoted || c->deadline_finished) return 0;
    /* No replacement => preserve ordinary pending-load/kernel partition.
     * Native/lookup/noop controls never create a replacement window. */
    if(!c->window_active) {
        c->deadline_finished=1;memcpy(final_ids,c->candidate,sizeof c->candidate);
        *promoted=0;return 1;
    }
    /* Same mandatory native pool, checked and moved before FIRST routed gate/up
     * only for replacement windows (including reader contention declines).
     * This can sacrifice existing resident/miss overlap; measure that cost. */
    if(g_stream_expert_pending_load.active && !ds4_gpu_stream_expert_pending_load_matches(
        c->current_table.model_map,c->current_table.model_size,c->current_table.layer,c->candidate,
        384,6,c->model.gate_bytes,c->model.down_bytes)) return 0;
    if(!ds4_gpu_stream_expert_pending_load_finish(NULL)) return 0;
    ds4_od_occupied o;if(!ds4_od_native_occupied(c,&o)) return 0;
    ds4_od_decision d={0};memcpy(d.final_ids,c->candidate,sizeof d.final_ids);
    if(c->ticket_valid && ds4_od_try_deadline(c->reader,&c->ticket,&d)<0) return 0;
    c->deadline_finished=1;memcpy(final_ids,d.final_ids,sizeof d.final_ids);*promoted=d.promoted;return 1;
}
int ds4_gpu_original_deadline_bind_ready(const void *owner) {
    ds4_od_native_context *c=&g_original_deadline;
    if(!ds4_od_native_owner(owner) || !c->deadline_finished) return 0;
    c->window_active=0;c->ticket_valid=0;g_original_deadline_protect_count=0;return 1;
}
int ds4_gpu_original_deadline_stats(const void *owner,ds4_od_stats *out) {
    ds4_od_native_context *c=&g_original_deadline;
    return out && ds4_od_native_owner(owner) && ds4_od_try_stats(c->reader,out);
}
