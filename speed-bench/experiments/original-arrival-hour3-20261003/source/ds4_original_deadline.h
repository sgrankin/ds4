#ifndef DS4_ORIGINAL_DEADLINE_H
#define DS4_ORIGINAL_DEADLINE_H
/* Unintegrated portable ownership prototype. No core/Metal caller uses it. */
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#define DS4_OD_MAX_PAYLOAD (UINT64_C(16) * 1024 * 1024)
#define DS4_OD_HOST_CAP (UINT64_C(512) * 1024)
#ifndef DS4_OD_READY_AGE
#define DS4_OD_READY_AGE UINT64_C(80)
#endif
#define DS4_OD_STACK_BYTES (256u * 1024u)

enum ds4_od_result { DS4_OD_ERROR = -1, DS4_OD_DECLINED = 0, DS4_OD_ACCEPTED = 1 };
enum ds4_od_install { DS4_OD_INSTALL_DECLINED=0, DS4_OD_INSTALL_TRANSFERRED=1,
    DS4_OD_INSTALL_ALREADY_NATIVE=2 };
enum ds4_od_state { DS4_OD_FREE, DS4_OD_READING, DS4_OD_READY, DS4_OD_FAILED };

typedef struct {
    uint64_t owner, generation, device, inode, whole_bytes, mapped_bytes;
    uintptr_t map_cookie;                 /* comparison only, never dereferenced */
    uint64_t gate_bytes, down_bytes;
    uint32_t layers, experts, gate_type, down_type;
} ds4_od_model;

typedef struct {
    ds4_od_model model;
    uint64_t offset[3];                    /* gate, up, down absolute FD offsets */
    uint32_t layer, original;
} ds4_od_expert;

typedef struct {
    ds4_od_expert expert;
    uint64_t serial;                       /* owner-monotonic, not context position */
    uint32_t changed_slot;
    int32_t native[6], candidate[6];
} ds4_od_request;

typedef struct {
    uint32_t layer, count;
    int32_t ids[7];                        /* protection only; kernels stay six */
} ds4_od_protection;

typedef struct {
    void *owner_handle;                    /* only owner callbacks may inspect */
    unsigned char *base;                   /* existing shared native slab contents */
    uint64_t capacity, inner[3];
    uint32_t physical_slot;
    int retained_lock;
} ds4_od_lease;

typedef struct {
    uint32_t capacity, valid, mandatory_detached, original_leases;
} ds4_od_occupied;

/* Called only by the engine owner under the prototype's short metadata lock.
 * Every callback must be allocation/read/wait-free, must not reenter reader,
 * and must exclude inflight/pending buffers. reserve validates before detach;
 * DECLINED is mutation-free. install0 is mutation-free, install1 transfers slot
 * ownership exactly once. install2 validates an existing exact native entry
 * without mutation; reader then releases its redundant lease. Negative is a
 * fatal adapter error, before binding, with lease ownership retained. reserve
 * may return negative after detaching at most one slot ONLY if its output has
 * a nonnull owner_handle and a complete bounded, release-safe lease identifying
 * every transferred resource. A negative empty output must be mutation-free.
 * The reader adopts that lease without starting a job; caller must stop before
 * binding and close the reader. release must safely accept the completed lease
 * even when reserve reported a fatal error; no output ownership may be hidden.
 * Physical counts must deduplicate pending/cache aliases independently. */
typedef struct {
    void *context;
    int (*reserve)(void *, const ds4_od_expert *, const ds4_od_protection *, ds4_od_lease *);
    int (*install)(void *, const ds4_od_expert *, const ds4_od_protection *, const ds4_od_lease *);
    void (*release)(void *, const ds4_od_lease *);
    int (*occupied)(void *, ds4_od_occupied *);
} ds4_od_adapter;

typedef struct {
    uint64_t job, transaction, serial;
} ds4_od_ticket;

typedef struct {
    int promoted;
    int32_t final_ids[6];
} ds4_od_decision;

typedef struct {
    uint64_t requests, started, reused, busy, reserve_declines, contention;
    uint64_t completed, failed, deadline_tries, deadline_misses, promoted;
    uint64_t future_used, expired, released, install_declines, duplicate_discarded;
    uint64_t logical_requested, pread_calls, pread_requested, pread_returned;
    uint64_t metadata_bytes, stack_bytes, guard_bytes, known_host_peak;
    uint64_t original_payload_peak;
    int overflow, state, worker_present, worker_active;
} ds4_od_stats;

typedef struct ds4_od_reader ds4_od_reader;

/* Setup only. Validate and duplicate already bound regular FD; configure bounded
 * stack/guard and create ONE persistent reader. Does not parse/open a model. */
int ds4_od_open(ds4_od_reader **out, const ds4_od_model *model, int bound_fd,
        const ds4_od_adapter *adapter);
/* Teardown only. Joins BEFORE release/close/free. Nonowner/join failure retains
 * the live reader. After successful join, cleanup consumes it even if the final
 * adapter occupancy check reports failure; the pointer is then invalid. */
int ds4_od_close(ds4_od_reader *reader);

int ds4_od_try_arm(ds4_od_reader *reader, const ds4_od_request *request,
        ds4_od_ticket *ticket);
/* ONE try after checked mandatory selected finish, before all routed gate/up.
 * Accepted or declined writes a sealed decision; error leaves output unchanged.
 * No coefficients or GPU tensors are reachable through this interface. */
int ds4_od_try_deadline(ds4_od_reader *reader, const ds4_od_ticket *ticket,
        ds4_od_decision *decision);
int ds4_od_try_future(ds4_od_reader *reader, const ds4_od_expert *expert,
        const ds4_od_protection *protect, uint64_t serial);
int ds4_od_try_tick(ds4_od_reader *reader, uint64_t serial);
int ds4_od_try_stats(ds4_od_reader *reader, ds4_od_stats *stats);

#ifdef DS4_OD_TESTING
/* CPU fixture seam only; set before first arm. Runs outside metadata lock.
 * Normal source build always calls pread directly. No native model use. */
int ds4_od_test_read_hook(ds4_od_reader *reader,
        ssize_t (*read_fn)(void *, int, void *, size_t, off_t), void *context);
#endif
#endif
