#define _POSIX_C_SOURCE 200809L
#include "ds4_original_deadline.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <stdatomic.h>

struct ds4_od_reader {
    pthread_mutex_t mu;
    pthread_cond_t wake;
    pthread_t thread;
    pthread_t owner_thread;
    ds4_od_model model;
    ds4_od_adapter adapter;
    ds4_od_expert expert;
    ds4_od_lease lease;
    ds4_od_request token_request;
    ds4_od_stats stats;
    uint64_t job, transaction, request_serial, last_serial;
    uint64_t page_size;
    _Atomic uint64_t contention, deadline_attempts, deadline_misses;
    int fd, closing, sealed, state, worker_started;
#ifdef DS4_OD_TESTING
    ssize_t (*read_fn)(void *, int, void *, size_t, off_t);
    void *read_context;
#endif
};
_Static_assert(sizeof(ds4_od_reader) <= 16u * 1024u, "owned metadata reserve");
_Static_assert(sizeof(off_t) >= 8, "pread offsets require 64 bits");

static void od_add(ds4_od_reader *r, uint64_t *v, uint64_t n) {
    if (UINT64_MAX - *v < n) { *v = UINT64_MAX; r->stats.overflow = 1; }
    else *v += n;
}
static int od_model_equal(const ds4_od_model *a, const ds4_od_model *b) {
    return a->owner == b->owner && a->generation == b->generation &&
        a->device == b->device && a->inode == b->inode &&
        a->whole_bytes == b->whole_bytes && a->mapped_bytes == b->mapped_bytes &&
        a->map_cookie == b->map_cookie && a->gate_bytes == b->gate_bytes &&
        a->down_bytes == b->down_bytes && a->layers == b->layers &&
        a->experts == b->experts && a->gate_type == b->gate_type &&
        a->down_type == b->down_type;
}
static int od_model_valid(const ds4_od_model *m) {
    return m && m->owner && m->generation && m->map_cookie &&
        m->whole_bytes && m->whole_bytes <= INT64_MAX &&
        m->mapped_bytes && m->mapped_bytes <= m->whole_bytes &&
        m->layers && m->layers <= 80 && m->experts >= 7 && m->experts <= 384 &&
        m->gate_bytes && m->down_bytes && m->gate_type && m->down_type &&
        m->down_bytes <= DS4_OD_MAX_PAYLOAD &&
        m->gate_bytes <= (DS4_OD_MAX_PAYLOAD - m->down_bytes) / 2;
}
static uint64_t od_length(const ds4_od_expert *e, unsigned i) {
    return i == 2 ? e->model.down_bytes : e->model.gate_bytes;
}
static int od_expert_valid(ds4_od_reader *r, const ds4_od_expert *e) {
    if (!e || !od_model_equal(&r->model, &e->model) ||
        e->layer >= e->model.layers || e->original >= e->model.experts) return 0;
    for (unsigned i = 0; i < 3; i++) {
        uint64_t n = od_length(e, i);
        if (e->offset[i] > e->model.mapped_bytes ||
            n > e->model.mapped_bytes - e->offset[i]) return 0;
        for (unsigned j = 0; j < i; j++)
            if (e->offset[i] < e->offset[j] + od_length(e,j) &&
                e->offset[j] < e->offset[i] + n) return 0;
    }
    return 1;
}
static int od_expert_equal(const ds4_od_expert *a, const ds4_od_expert *b) {
    return od_model_equal(&a->model,&b->model) && a->layer == b->layer &&
        a->original == b->original && !memcmp(a->offset,b->offset,sizeof a->offset);
}
static int od_protection_valid(const ds4_od_expert *e, const ds4_od_protection *p) {
    if (!p || p->layer != e->layer || p->count < 6 || p->count > 7) return 0;
    int original = 0;
    for (unsigned i=0;i<p->count;i++) {
        if (p->ids[i] < 0 || (uint32_t)p->ids[i] >= e->model.experts) return 0;
        original |= (uint32_t)p->ids[i] == e->original;
        for (unsigned j=0;j<i;j++) if (p->ids[i] == p->ids[j]) return 0;
    }
    return original;
}
static int od_request_valid(ds4_od_reader *r, const ds4_od_request *q,
        ds4_od_protection *p) {
    if (!q || !od_expert_valid(r,&q->expert) || q->changed_slot >= 6) return 0;
    memset(p,0,sizeof *p); p->layer=q->expert.layer; p->count=7;
    for (unsigned i=0;i<6;i++) {
        if (q->native[i] < 0 || q->candidate[i] < 0 ||
            (uint32_t)q->native[i] >= q->expert.model.experts ||
            (uint32_t)q->candidate[i] >= q->expert.model.experts) return 0;
        if ((i != q->changed_slot && q->native[i] != q->candidate[i]) ||
            (i == q->changed_slot && q->native[i] == q->candidate[i])) return 0;
        for (unsigned j=0;j<i;j++)
            if (q->native[i] == q->native[j] || q->candidate[i] == q->candidate[j]) return 0;
        p->ids[i]=q->native[i];
    }
    p->ids[6]=q->candidate[q->changed_slot];
    return (uint32_t)q->native[q->changed_slot] == q->expert.original &&
        od_protection_valid(&q->expert,p);
}
static int od_occupied(ds4_od_reader *r, unsigned expected_leases) {
    ds4_od_occupied o;
    memset(&o,0,sizeof o);
    if (!r->adapter.occupied(r->adapter.context,&o) || !o.capacity ||
        o.original_leases != expected_leases) return 0;
    uint64_t sum=(uint64_t)o.valid+o.mandatory_detached+o.original_leases;
    return sum <= o.capacity;
}
static int od_lease_valid(ds4_od_reader *r,const ds4_od_expert *e,const ds4_od_lease *l) {
    uint64_t logical=e->model.gate_bytes*2+e->model.down_bytes;
    ds4_od_occupied occupied;
    memset(&occupied,0,sizeof occupied);
    if (!r->adapter.occupied(r->adapter.context,&occupied) ||
        l->physical_slot>=occupied.capacity) return 0;
    if (!l->owner_handle || !l->base || l->retained_lock != 1 ||
        l->capacity < logical || l->capacity > DS4_OD_MAX_PAYLOAD ||
        !r->page_size || l->capacity%r->page_size || l->inner[0] ||
        l->inner[1]!=e->model.gate_bytes || l->inner[2]!=e->model.gate_bytes*2) return 0;
    for (unsigned i=0;i<3;i++) {
        uint64_t n=od_length(e,i);
        if (l->inner[i] > l->capacity || n > l->capacity-l->inner[i]) return 0;
        for (unsigned j=0;j<i;j++)
            if (l->inner[i] < l->inner[j]+od_length(e,j) &&
                l->inner[j] < l->inner[i]+n) return 0;
    }
    return 1;
}
static void od_release(ds4_od_reader *r) {
    if (r->lease.owner_handle) {
        r->adapter.release(r->adapter.context,&r->lease);
        od_add(r,&r->stats.released,1);
        memset(&r->lease,0,sizeof r->lease);
    }
    r->state=DS4_OD_FREE;
}
static int od_tick_locked(ds4_od_reader *r, uint64_t serial) {
    if (serial < r->last_serial) return 0;
    r->last_serial=serial;
    if ((r->state == DS4_OD_READY || r->state == DS4_OD_FAILED) &&
        serial-r->request_serial >= DS4_OD_READY_AGE) {
        od_add(r,&r->stats.expired,1); od_release(r);
        if (!od_occupied(r,0)) return -1;
    }
    return 1;
}
static int od_lock(ds4_od_reader *r) {
    if (!r) return 0;
    return pthread_mutex_trylock(&r->mu) == 0;
}
static int od_owner(ds4_od_reader *r) {
    return r && pthread_equal(r->owner_thread,pthread_self());
}
static void od_atomic_increment(_Atomic uint64_t *v) {
    uint64_t n=atomic_load_explicit(v,memory_order_relaxed);
    while (n!=UINT64_MAX && !atomic_compare_exchange_weak_explicit(v,&n,n+1,
            memory_order_relaxed,memory_order_relaxed)) {}
}
static void *od_worker(void *context) {
    ds4_od_reader *r=context;
    pthread_mutex_lock(&r->mu);
    for (;;) {
        while (!r->closing && (r->state != DS4_OD_READING || r->worker_started))
            pthread_cond_wait(&r->wake,&r->mu);
        if (r->closing) break;
        ds4_od_expert expert=r->expert;
        ds4_od_lease lease=r->lease;
        r->worker_started=1; od_add(r,&r->stats.started,1);
#ifdef DS4_OD_TESTING
        ssize_t (*read_fn)(void *,int,void *,size_t,off_t)=r->read_fn;
        void *read_context=r->read_context;
#endif
        pthread_mutex_unlock(&r->mu);
        uint64_t calls=0,requested=0,returned=0;
        int ok=1,counter_overflow=0;
        for (unsigned i=0;i<3 && ok;i++) {
            uint64_t done=0,n=od_length(&expert,i);
            while (done<n) {
                size_t want=(size_t)(n-done);
                if (calls==UINT64_MAX || UINT64_MAX-requested<want) {
                    ok=0;counter_overflow=1;break;
                }
                calls++; requested+=want;
                ssize_t got;
#ifdef DS4_OD_TESTING
                if (read_fn) got=read_fn(read_context,r->fd,
                    lease.base+lease.inner[i]+done,want,(off_t)(expert.offset[i]+done));
                else
#endif
                got=pread(r->fd,lease.base+lease.inner[i]+done,want,
                    (off_t)(expert.offset[i]+done));
                if (got<0 && errno==EINTR) continue;
                if (got<=0 || (uint64_t)got>n-done) { ok=0; break; }
                done+=(uint64_t)got; returned+=(uint64_t)got;
            }
        }
        /* Final payload access occurred before this lock/publication. No worker
         * cache callback, Metal call, ownership release or later payload write. */
        pthread_mutex_lock(&r->mu);
        if (counter_overflow) r->stats.overflow=1;
        od_add(r,&r->stats.pread_calls,calls);
        od_add(r,&r->stats.pread_requested,requested);
        od_add(r,&r->stats.pread_returned,returned);
        if (ok) { r->state=DS4_OD_READY; od_add(r,&r->stats.completed,1); }
        else { r->state=DS4_OD_FAILED; od_add(r,&r->stats.failed,1); }
        pthread_cond_broadcast(&r->wake);
    }
    pthread_mutex_unlock(&r->mu);
    return NULL;
}

int ds4_od_open(ds4_od_reader **out, const ds4_od_model *m, int fd,
        const ds4_od_adapter *a) {
    if (!out || !od_model_valid(m) || !a || !a->reserve || !a->install ||
        !a->release || !a->occupied) return 0;
    *out=NULL;
    ds4_od_reader *r=calloc(1,sizeof *r);
    if (!r) return 0;
    r->fd=fcntl(fd,F_DUPFD_CLOEXEC,0);
    if (r->fd<0) { free(r); return 0; }
    struct stat s;
    if (fstat(r->fd,&s) || !S_ISREG(s.st_mode) || s.st_size<0 ||
        (uint64_t)s.st_dev!=m->device || (uint64_t)s.st_ino!=m->inode ||
        (uint64_t)s.st_size!=m->whole_bytes) {close(r->fd);free(r);return 0;}
    r->model=*m; r->adapter=*a; r->sealed=1;
    r->owner_thread=pthread_self();
    atomic_init(&r->contention,0);atomic_init(&r->deadline_attempts,0);
    atomic_init(&r->deadline_misses,0);
    if (pthread_mutex_init(&r->mu,NULL)) { close(r->fd);free(r);return 0; }
    if (pthread_cond_init(&r->wake,NULL)) {
        pthread_mutex_destroy(&r->mu);close(r->fd);free(r);return 0;
    }
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) goto fail;
    long page=sysconf(_SC_PAGESIZE);
    size_t stack=0,guard=0;
    int good=page>0 && page<=64*1024 &&
        !pthread_attr_setstacksize(&attr,DS4_OD_STACK_BYTES) &&
        !pthread_attr_setguardsize(&attr,(size_t)page) &&
        !pthread_attr_getstacksize(&attr,&stack) &&
        !pthread_attr_getguardsize(&attr,&guard) &&
        stack==DS4_OD_STACK_BYTES && guard<=64*1024 &&
        sizeof *r+stack+guard<=DS4_OD_HOST_CAP;
    r->stats.metadata_bytes=sizeof *r; r->stats.stack_bytes=stack;
    r->stats.guard_bytes=guard; r->stats.known_host_peak=sizeof *r+stack+guard;
    r->page_size=page>0 ? (uint64_t)page:0;
    if (!good || !od_occupied(r,0) || pthread_create(&r->thread,&attr,od_worker,r)) {
        pthread_attr_destroy(&attr);goto fail;
    }
    pthread_attr_destroy(&attr);*out=r;return 1;
fail:
    pthread_cond_destroy(&r->wake);pthread_mutex_destroy(&r->mu);
    close(r->fd);free(r);return 0;
}
int ds4_od_close(ds4_od_reader *r) {
    if (!r) return 1;
    if (!od_owner(r)) return 0;
    pthread_mutex_lock(&r->mu);r->closing=1;
    pthread_cond_broadcast(&r->wake);pthread_mutex_unlock(&r->mu);
    if (pthread_join(r->thread,NULL)) return 0;
    pthread_mutex_lock(&r->mu);od_release(r);
    int good=od_occupied(r,0);pthread_mutex_unlock(&r->mu);
    close(r->fd);pthread_cond_destroy(&r->wake);pthread_mutex_destroy(&r->mu);
    free(r);return good;
}
int ds4_od_try_arm(ds4_od_reader *r,const ds4_od_request *q,ds4_od_ticket *ticket) {
    ds4_od_protection p;
    if (!od_owner(r) || !ticket || !od_request_valid(r,q,&p)) return DS4_OD_DECLINED;
    if (!od_lock(r)) {od_atomic_increment(&r->contention);return DS4_OD_DECLINED;}
    int result=DS4_OD_DECLINED;
    if (r->closing || r->stats.overflow || !r->sealed) goto end;
    int tick=od_tick_locked(r,q->serial);
    if (tick<0) { result=DS4_OD_ERROR;goto end; }
    if (!tick) goto end;
    if (r->job==UINT64_MAX || r->transaction==UINT64_MAX) { result=DS4_OD_ERROR;goto end; }
    if (r->state==DS4_OD_FAILED) { od_release(r); if (!od_occupied(r,0)) {result=-1;goto end;} }
    if (r->state!=DS4_OD_FREE) {
        if (!od_expert_equal(&r->expert,&q->expert) ||
            q->serial-r->request_serial>=DS4_OD_READY_AGE) {
            od_add(r,&r->stats.busy,1);goto end;
        }
        od_add(r,&r->stats.reused,1);
    } else {
        if (!od_occupied(r,0)) { result=-1;goto end; }
        ds4_od_lease lease;memset(&lease,0,sizeof lease);
        int reserve=r->adapter.reserve(r->adapter.context,&q->expert,&p,&lease);
        if (reserve<0) {
            /* A fatal adapter result may have detached one recoverable lease.
             * Adopt it before returning; no job/ticket is published and the
             * worker never reads it. Owner teardown must release it once. */
            if (lease.owner_handle) {
                r->lease=lease;r->expert=q->expert;r->request_serial=q->serial;
                r->state=DS4_OD_FAILED;
                if (r->stats.original_payload_peak<lease.capacity)
                    r->stats.original_payload_peak=lease.capacity;
            }
            result=-1;goto end;
        }
        if (!reserve) {
            if (lease.owner_handle || !od_occupied(r,0)) {
                r->lease=lease;od_release(r);result=-1;goto end;
            }
            od_add(r,&r->stats.reserve_declines,1);goto end;
        }
        r->lease=lease;
        if (!od_lease_valid(r,&q->expert,&lease) || !od_occupied(r,1)) {
            od_release(r);result=-1;goto end;
        }
        r->expert=q->expert;r->request_serial=q->serial;r->job++;
        r->worker_started=0;r->state=DS4_OD_READING;
        od_add(r,&r->stats.requests,1);
        od_add(r,&r->stats.logical_requested,q->expert.model.gate_bytes*2+q->expert.model.down_bytes);
        if (r->stats.original_payload_peak<lease.capacity) r->stats.original_payload_peak=lease.capacity;
    }
    r->transaction++;r->token_request=*q;r->sealed=0;
    *ticket=(ds4_od_ticket){r->job,r->transaction,q->serial};
    pthread_cond_signal(&r->wake);result=DS4_OD_ACCEPTED;
end:
    pthread_mutex_unlock(&r->mu);return result;
}
static int od_install(ds4_od_reader *r,const ds4_od_protection *p) {
    if (!od_occupied(r,1) || !od_lease_valid(r,&r->expert,&r->lease)) return -1;
    int result=r->adapter.install(r->adapter.context,&r->expert,p,&r->lease);
    if (result==DS4_OD_INSTALL_ALREADY_NATIVE) {
        od_release(r);od_add(r,&r->stats.duplicate_discarded,1);
        return od_occupied(r,0) ? 1:-1;
    }
    if (result==DS4_OD_INSTALL_TRANSFERRED) {
        memset(&r->lease,0,sizeof r->lease);r->state=DS4_OD_FREE;
        if (!od_occupied(r,0)) return -1;
        return 1;
    }
    if (!result) od_add(r,&r->stats.install_declines,1);
    if (!od_occupied(r,1)) return -1;
    return result ? -1:0;
}
int ds4_od_try_deadline(ds4_od_reader *r,const ds4_od_ticket *t,ds4_od_decision *out) {
    if (!od_owner(r) || !t || !out || r->sealed || t->job!=r->job ||
        t->transaction!=r->transaction || t->serial!=r->token_request.serial) return -1;
    /* Only owner writes these transaction fields; worker never reads them.
     * Seal BEFORE trylock so contention cannot permit a second deadline. */
    r->sealed=1;od_atomic_increment(&r->deadline_attempts);
    ds4_od_decision d;memset(&d,0,sizeof d);
    memcpy(d.final_ids,r->token_request.candidate,sizeof d.final_ids);
    if (!od_lock(r)) {
        od_atomic_increment(&r->contention);od_atomic_increment(&r->deadline_misses);
        *out=d;return 0;
    }
    int result=-1;
    if (r->closing || r->stats.overflow) goto end;
    if (r->state==DS4_OD_READY && t->serial>=r->request_serial &&
        t->serial-r->request_serial<DS4_OD_READY_AGE) {
        ds4_od_protection p;
        if (!od_request_valid(r,&r->token_request,&p)) goto end;
        result=od_install(r,&p);
        if (result<0) goto end;
        if (result) {
            d.promoted=1;memcpy(d.final_ids,r->token_request.native,sizeof d.final_ids);
            od_add(r,&r->stats.promoted,1);
        }
    } else result=0;
    if (!d.promoted) od_atomic_increment(&r->deadline_misses);
    *out=d;
end:
    pthread_mutex_unlock(&r->mu);return result;
}
int ds4_od_try_future(ds4_od_reader *r,const ds4_od_expert *e,
        const ds4_od_protection *p,uint64_t serial) {
    if (!od_owner(r) || !od_expert_valid(r,e) || !od_protection_valid(e,p)) return 0;
    if (!od_lock(r)) {od_atomic_increment(&r->contention);return 0;}
    int result=0;
    if (r->closing || !r->sealed) goto end;
    if (r->stats.overflow) {result=-1;goto end;}
    int tick=od_tick_locked(r,serial);
    if (tick<0) {result=-1;goto end;}
    if (!tick || r->state!=DS4_OD_READY ||
        !od_expert_equal(&r->expert,e)) goto end;
    result=od_install(r,p);
    if (result>0) od_add(r,&r->stats.future_used,1);
end:
    pthread_mutex_unlock(&r->mu);return result;
}
int ds4_od_try_tick(ds4_od_reader *r,uint64_t serial) {
    if (!od_owner(r)) return 0;
    if (!od_lock(r)) {od_atomic_increment(&r->contention);return 0;}
    int result=(r->closing || !r->sealed) ? 0:od_tick_locked(r,serial);
    pthread_mutex_unlock(&r->mu);return result;
}
int ds4_od_try_stats(ds4_od_reader *r,ds4_od_stats *stats) {
    if (!stats || !od_lock(r)) return 0;
    *stats=r->stats;stats->state=r->state;stats->worker_present=!r->closing;
    stats->worker_active=r->state==DS4_OD_READING;
    stats->contention=atomic_load_explicit(&r->contention,memory_order_relaxed);
    stats->deadline_tries=atomic_load_explicit(&r->deadline_attempts,memory_order_relaxed);
    stats->deadline_misses=atomic_load_explicit(&r->deadline_misses,memory_order_relaxed);
    pthread_mutex_unlock(&r->mu);return 1;
}
#ifdef DS4_OD_TESTING
int ds4_od_test_read_hook(ds4_od_reader *r,
        ssize_t (*fn)(void *,int,void *,size_t,off_t),void *context) {
    if (!od_owner(r) || !od_lock(r)) return 0;
    int good=!r->stats.requests && !r->closing;
    if (good) {r->read_fn=fn;r->read_context=context;}
    pthread_mutex_unlock(&r->mu);return good;
}
#endif
