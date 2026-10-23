#ifndef DS4_STATE_CACHE_H
#define DS4_STATE_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Host-only DS4.1 checkpoint storage. One caller serializes operations (the
 * server inference lock is sufficient); a directory has one process owner.
 * Sessions keep their own mutable GPU state. All stored chunks are immutable.
 * Limits count resident chunk bytes and physical disk bytes respectively;
 * metadata and caller-owned materializations are reported/accounted separately.
 */
typedef struct ds4_state_cache ds4_state_cache;
typedef struct ds4_state_cache_ref ds4_state_cache_ref;

/* On-disk DS4SC01 kind at header offset 16. Values 0/1 retain the original
 * exact/approximate encoding; older readers reject the new replay kind. */
typedef enum {
    DS4_STATE_CACHE_NATIVE = 0,
    DS4_STATE_CACHE_SUFFIX = 1,
    DS4_STATE_CACHE_REPLAY = 2,
} ds4_state_cache_kind;

typedef struct {
    uint8_t *state;
    uint64_t state_bytes;
    uint8_t *prerope;
    uint64_t prerope_bytes;
    uint32_t ctx_size, tokens;
    ds4_state_cache_kind kind;
    bool approximate;
} ds4_state_cache_blob;

typedef struct {
    uint64_t hot_bytes, ssd_bytes, metadata_bytes;
    uint64_t endpoints, chunks, pinned_endpoints;
    uint64_t hot_reads, disk_reads, endpoint_evictions;
} ds4_state_cache_stats;

/* profile identifies model weights/tokenizer and the fixed forward policy.
 * The caller supplies it; rendered text is never an exact-prefix key.
 * 0 hot bytes means disk-only. The directory must have an existing parent.
 */
ds4_state_cache *ds4_state_cache_open(const char *dir, uint64_t hot_bytes,
                                     uint64_t ssd_bytes, char *err, size_t errlen);

/* Borrow inputs only until return. Store DSV4 as native, DSSD as suffix state,
 * or DRP1 as resumable decoder replay. Suffix states require their original
 * pre-RoPE blob; replay states cannot carry one or serve as suffix sources.
 * A successful put
 * is committed to SSD. Pages are shared by byte hash, not by mutable pointers.
 * key is a stable 40-character identifier for this immutable source.
 */
int ds4_state_cache_put_kind(ds4_state_cache *cache, const char *profile,
                             const void *state, uint64_t state_bytes,
                             const void *prerope, uint64_t prerope_bytes,
                             ds4_state_cache_kind kind, char key[41],
                             char *err, size_t errlen);
/* Compatibility entry point: false selects native, true selects suffix state. */
int ds4_state_cache_put(ds4_state_cache *cache, const char *profile,
                        const void *state, uint64_t state_bytes,
                        const void *prerope, uint64_t prerope_bytes,
                        bool approximate, char key[41],
                        char *err, size_t errlen);

/* Returned references pin the endpoint and its chunks against eviction.
 * Prefix lookup returns the longest captured native or replay prefix of ids[0:n].
 * The profile must include replay K/selected policy; the core replay loader also
 * validates that policy. A replay result remains approximate after restoration.
 * Suffix lookup returns a candidate whose endpoint ends in all supplied ids;
 * source_start identifies that suffix in the stored original token history.
 * The caller still checks relocation, fresh-tail and latest-user eligibility.
 */
ds4_state_cache_ref *ds4_state_cache_get(ds4_state_cache *cache, const char *key);
ds4_state_cache_ref *ds4_state_cache_find_prefix(ds4_state_cache *cache,
                                               const char *profile,
                                               uint32_t ctx_size,
                                               const int *ids, uint32_t n,
                                               uint32_t *matched_tokens);
ds4_state_cache_ref *ds4_state_cache_find_suffix(ds4_state_cache *cache,
                                               const char *profile,
                                               uint32_t ctx_size,
                                               const int *ids, uint32_t n,
                                               uint32_t *source_start);

/* Restart discovery: initialize cursor[0]=0, iterate key order, and release
 * every returned reference. ids are borrowed immutable bytes owned by that
 * reference, valid until release. Native and suffix sources appear; resumable
 * replay states are excluded because they cannot be relocated.
 * the caller matches its anchor and checks full suffix geometry. No graph or
 * large state blob is materialized merely to enumerate token histories.
 */
ds4_state_cache_ref *ds4_state_cache_next_candidate(ds4_state_cache *cache,
                                                  const char *profile,
                                                  uint32_t ctx_size,
                                                  char cursor[41],
                                                  const int **ids,
                                                  uint32_t *n,
                                                  bool *approximate);

/* Fresh owned bytes, valid after release/reset/overwrite of any session.
 * Native state loads through existing payload APIs; replay state requires its
 * dedicated loader. A suffix caller rebinds
 * its engine pointer after checking profile/context; no pointer is persisted.
 */
int ds4_state_cache_materialize(ds4_state_cache_ref *ref,
                                ds4_state_cache_blob *out,
                                char *err, size_t errlen);
/* Admit caller-owned allocation before materializing a source. */
void ds4_state_cache_ref_sizes(const ds4_state_cache_ref *ref,
                               uint64_t *state_bytes, uint64_t *prerope_bytes);
void ds4_state_cache_blob_free(ds4_state_cache_blob *blob);
void ds4_state_cache_release(ds4_state_cache_ref *ref);
void ds4_state_cache_get_stats(const ds4_state_cache *cache,
                               ds4_state_cache_stats *out);
/* Refuses close with outstanding references; release them and retry. */
int ds4_state_cache_close(ds4_state_cache *cache, char *err, size_t errlen);

#endif
