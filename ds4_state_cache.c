#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "ds4_state_cache.h"
#include "ds4_kvstore.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>

#define SC_HEADER 96u
#define SC_RECORD 64u
#define SC_PAGE 2048u
#define SC_PRE_PAGE 65536u
#define SC_VOCAB 129280u
#define SC_MAX_MANIFEST (8u * 1024u * 1024u)

typedef struct sc_chunk {
    char key[41];
    uint64_t bytes, used;
    uint32_t refs, pins;
    bool present;
    uint8_t *hot;
    struct sc_chunk *next;
} sc_chunk;

typedef struct {
    char key[41];
    uint32_t which;
    uint64_t offset, bytes;
} sc_segment;

typedef struct sc_endpoint {
    char key[41], prefix[41];
    char *profile;
    uint32_t ctx, tokens, nseg, pins;
    bool approximate;
    uint64_t state_bytes, pre_bytes, manifest_bytes, used;
    sc_segment *seg;
    struct sc_endpoint *next;
} sc_endpoint;

struct ds4_state_cache {
    char *dir;
    int lock_fd;
    uint64_t hot_limit, disk_limit, hot_bytes, disk_bytes, clock;
    uint64_t lease_bytes;
    uint64_t hot_reads, disk_reads, evictions;
    sc_chunk *chunks;
    sc_endpoint *endpoints;
};

struct ds4_state_cache_ref {
    ds4_state_cache *cache;
    sc_endpoint *endpoint;
    int *ids;
};

static int sc_error(char *err, size_t len, const char *message) {
    if (err && len) snprintf(err, len, "%s", message);
    return -1;
}

static uint32_t sc_u32(const uint8_t *p) {
    return ds4_kvstore_le_get32(p);
}

static uint64_t sc_u64(const uint8_t *p) {
    return sc_u32(p) | ((uint64_t)sc_u32(p + 4) << 32);
}

static void sc_put64(uint8_t *p, uint64_t n) {
    ds4_kvstore_le_put32(p, (uint32_t)n);
    ds4_kvstore_le_put32(p + 4, (uint32_t)(n >> 32));
}

static bool sc_key(const char *s) {
    if (!s || strlen(s) != 40) return false;
    for (int i = 0; i < 40; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return false;
    return true;
}

static char *sc_path(const ds4_state_cache *c, const char *key, const char *ext) {
    size_t n = strlen(c->dir) + strlen(key) + strlen(ext) + 10;
    char *p = malloc(n);
    if (p) snprintf(p, n, "%s/ds4sc-%s.%s", c->dir, key, ext);
    return p;
}

static bool sc_atomic(const char *dir, const char *path, const void *p, size_t n) {
    size_t z = strlen(dir) + 30;
    char *temp = malloc(z);
    if (!temp) return false;
    snprintf(temp, z, "%s/.ds4sc-tmp-XXXXXX", dir);
    int fd = mkstemp(temp);
    FILE *fp = fd < 0 ? NULL : fdopen(fd, "wb");
    bool ok = fp && fwrite(p, 1, n, fp) == n && !fflush(fp) && !fsync(fd);
    if (fp) {
        if (fclose(fp)) ok = false;
    } else if (fd >= 0) close(fd);
    if (ok) ok = rename(temp, path) == 0;
    if (!ok) unlink(temp);
    free(temp);
    return ok;
}

static sc_chunk *sc_chunk_find(ds4_state_cache *c, const char *key) {
    for (sc_chunk *p = c->chunks; p; p = p->next)
        if (!strcmp(p->key, key)) return p;
    return NULL;
}

static sc_endpoint *sc_endpoint_find(ds4_state_cache *c, const char *key) {
    for (sc_endpoint *p = c->endpoints; p; p = p->next)
        if (!strcmp(p->key, key)) return p;
    return NULL;
}

static uint64_t sc_state_bytes(uint32_t n) {
    uint64_t raw = n < 128 ? n : 128;
    uint64_t size = 52u + ((uint64_t)n + SC_VOCAB) * 4u + 40u * raw * 512u * 4u;
    for (unsigned i = 0; i < 4; i++) {
        size += (uint64_t)(n / (i < 3 ? 2u : 1u)) * (512u + 128u) * 4u;
        if (i < 3 && (n & 1u)) size += 2u * 512u * 4u;
    }
    return size;
}

static bool sc_state_valid(const uint8_t *p, uint64_t bytes, bool approximate,
                            uint32_t *ctx, uint32_t *tokens) {
    if (!p || bytes < 52 || bytes > SIZE_MAX) return false;
    if (sc_u32(p) != (approximate ? UINT32_C(0x44535344) : UINT32_C(0x34565344)) ||
        sc_u32(p + 4) != 2) return false;
    uint32_t h[13];
    for (unsigned i = 0; i < 13; i++) h[i] = sc_u32(p + i * 4u);
    if (!h[2] || h[2] > 1048576 || !h[7] || h[7] >= h[2] || h[3] != 1 ||
        h[4] != 128 || h[5] != 128 || h[6] != h[2] + 1u || h[8] != 40 ||
        h[9] != 512 || h[10] != 128 || h[11] != SC_VOCAB || h[12] != 0x413431u ||
        sc_state_bytes(h[7]) != bytes) return false;
    for (uint32_t i = 0; i < h[7]; i++)
        if (sc_u32(p + 52u + (uint64_t)i * 4u) >= SC_VOCAB) return false;
    *ctx = h[2]; *tokens = h[7];
    return true;
}

static bool sc_prefix(const char *profile, uint32_t ctx, const void *tokens,
                       uint32_t n, bool native_ids, char key[41]) {
    size_t plen = strlen(profile);
    uint64_t bytes = plen + 1u + 8u + (uint64_t)n * 4u;
    if (bytes > SIZE_MAX) return false;
    uint8_t *p = malloc((size_t)bytes);
    if (!p) return false;
    memcpy(p, profile, plen + 1u);
    ds4_kvstore_le_put32(p + plen + 1u, ctx);
    ds4_kvstore_le_put32(p + plen + 5u, n);
    uint8_t *dst = p + plen + 9u;
    if (native_ids) {
        const int *ids = tokens;
        for (uint32_t i = 0; i < n; i++) {
            if (ids[i] < 0 || (uint32_t)ids[i] >= SC_VOCAB) { free(p); return false; }
            ds4_kvstore_le_put32(dst + (uint64_t)i * 4u, (uint32_t)ids[i]);
        }
    } else memcpy(dst, tokens, (size_t)n * 4u);
    ds4_kvstore_sha1_bytes_hex(p, (size_t)bytes, key);
    free(p);
    return true;
}

static bool sc_add(sc_endpoint *e, const uint8_t *data, uint32_t which,
                    uint64_t offset, uint64_t bytes) {
    if (!bytes) return true;
    sc_segment *p = realloc(e->seg, (size_t)(e->nseg + 1u) * sizeof(*p));
    if (!p) return false;
    e->seg = p;
    sc_segment *s = &p[e->nseg++];
    s->which = which; s->offset = offset; s->bytes = bytes;
    ds4_kvstore_sha1_bytes_hex(data + offset, (size_t)bytes, s->key);
    return true;
}

static bool sc_pages(sc_endpoint *e, const uint8_t *p, uint32_t which,
                      uint64_t *offset, uint64_t bytes, uint64_t page_bytes) {
    while (bytes) {
        uint64_t n = bytes < page_bytes ? bytes : page_bytes;
        if (!sc_add(e, p, which, *offset, n)) return false;
        *offset += n; bytes -= n;
    }
    return true;
}

static bool sc_split(sc_endpoint *e, const uint8_t *p, const uint8_t *pre) {
    uint64_t off = 0, bytes;
    if (!sc_pages(e, p, 0, &off, 52u, 52u) ||
        !sc_pages(e, p, 0, &off, (uint64_t)e->tokens * 4u, SC_PAGE * 4u) ||
        !sc_pages(e, p, 0, &off, SC_VOCAB * 4u, SC_VOCAB * 4u)) return false;
    uint64_t raw = e->tokens < 128 ? e->tokens : 128;
    if (!sc_pages(e, p, 0, &off, 40u * raw * 512u * 4u, raw * 512u * 4u)) return false;
    for (unsigned i = 0; i < 4; i++) {
        uint32_t ratio = i < 3 ? 2u : 1u;
        bytes = (uint64_t)(e->tokens / ratio) * 512u * 4u;
        if (!sc_pages(e, p, 0, &off, bytes, (SC_PAGE / ratio) * 512u * 4u)) return false;
        bytes = (uint64_t)(e->tokens / ratio) * 128u * 4u;
        if (!sc_pages(e, p, 0, &off, bytes, (SC_PAGE / ratio) * 128u * 4u)) return false;
        if (i < 3 && (e->tokens & 1u) &&
            !sc_pages(e, p, 0, &off, 2u * 512u * 4u, 512u * 4u)) return false;
    }
    if (off != e->state_bytes) return false;
    off = 0;
    return !e->pre_bytes || sc_pages(e, pre, 1, &off, e->pre_bytes, SC_PRE_PAGE);
}

static uint8_t *sc_manifest(sc_endpoint *e) {
    size_t plen = strlen(e->profile);
    e->manifest_bytes = SC_HEADER + plen + (uint64_t)e->nseg * SC_RECORD;
    if (e->manifest_bytes > SC_MAX_MANIFEST) return NULL;
    uint8_t *p = calloc(1, (size_t)e->manifest_bytes);
    if (!p) return NULL;
    memcpy(p, "DS4SC01", 7);
    ds4_kvstore_le_put32(p + 8, e->ctx);
    ds4_kvstore_le_put32(p + 12, e->tokens);
    ds4_kvstore_le_put32(p + 16, e->approximate ? 1u : 0u);
    ds4_kvstore_le_put32(p + 20, (uint32_t)plen);
    sc_put64(p + 24, e->state_bytes); sc_put64(p + 32, e->pre_bytes);
    ds4_kvstore_le_put32(p + 40, e->nseg);
    memcpy(p + 48, e->prefix, 41);
    memcpy(p + SC_HEADER, e->profile, plen);
    for (uint32_t i = 0; i < e->nseg; i++) {
        const sc_segment *s = &e->seg[i];
        uint8_t *r = p + SC_HEADER + plen + (uint64_t)i * SC_RECORD;
        ds4_kvstore_le_put32(r, s->which);
        ds4_kvstore_le_put32(r + 4, (uint32_t)s->bytes);
        sc_put64(r + 8, s->offset);
        memcpy(r + 16, s->key, 41);
    }
    ds4_kvstore_sha1_bytes_hex(p, (size_t)e->manifest_bytes, e->key);
    return p;
}

static void sc_endpoint_free(sc_endpoint *e) {
    if (!e) return;
    free(e->profile); free(e->seg); free(e);
}

static sc_endpoint *sc_parse_manifest(const uint8_t *p, uint64_t bytes, const char *key) {
    if (bytes < SC_HEADER || bytes > SC_MAX_MANIFEST || memcmp(p, "DS4SC01\0", 8)) return NULL;
    uint32_t plen = sc_u32(p + 20), nseg = sc_u32(p + 40);
    if (!plen || plen > 255 || !nseg ||
        bytes != SC_HEADER + plen + (uint64_t)nseg * SC_RECORD ||
        p[88] != 0 || sc_u32(p + 16) > 1) return NULL;
    char digest[41]; ds4_kvstore_sha1_bytes_hex(p, (size_t)bytes, digest);
    if (strcmp(digest, key)) return NULL;
    sc_endpoint *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    memcpy(e->key, key, 41); memcpy(e->prefix, p + 48, 41);
    e->ctx = sc_u32(p + 8); e->tokens = sc_u32(p + 12);
    e->approximate = sc_u32(p + 16) != 0;
    e->state_bytes = sc_u64(p + 24); e->pre_bytes = sc_u64(p + 32);
    e->manifest_bytes = bytes; e->nseg = nseg;
    e->profile = calloc(1, (size_t)plen + 1u);
    e->seg = calloc(nseg, sizeof(*e->seg));
    if (!e->profile || !e->seg || !sc_key(e->prefix) || !e->ctx || e->ctx > 1048576 ||
        !e->tokens || e->tokens >= e->ctx || e->state_bytes != sc_state_bytes(e->tokens) ||
        e->pre_bytes > SIZE_MAX || (e->approximate && !e->pre_bytes)) goto bad;
    memcpy(e->profile, p + SC_HEADER, plen);
    if (strlen(e->profile) != plen) goto bad;
    uint64_t cursor[2] = {0, 0};
    for (uint32_t i = 0; i < nseg; i++) {
        const uint8_t *r = p + SC_HEADER + plen + (uint64_t)i * SC_RECORD;
        sc_segment *s = &e->seg[i];
        s->which = sc_u32(r); s->bytes = sc_u32(r + 4); s->offset = sc_u64(r + 8);
        memcpy(s->key, r + 16, 41);
        if (s->which > 1 || !s->bytes || s->bytes > 4u * 1024u * 1024u ||
            !sc_key(s->key) || s->offset != cursor[s->which]) goto bad;
        uint64_t total = s->which ? e->pre_bytes : e->state_bytes;
        if (s->bytes > total - cursor[s->which]) goto bad;
        cursor[s->which] += s->bytes;
    }
    if (cursor[0] != e->state_bytes || cursor[1] != e->pre_bytes) goto bad;
    return e;
bad:
    sc_endpoint_free(e); return NULL;
}

static void sc_chunk_remove(ds4_state_cache *c, sc_chunk *chunk, bool erase) {
    sc_chunk **p = &c->chunks;
    while (*p && *p != chunk) p = &(*p)->next;
    if (*p) *p = chunk->next;
    if (chunk->hot) { c->hot_bytes -= chunk->bytes; free(chunk->hot); }
    if (chunk->present) {
        c->disk_bytes -= chunk->bytes;
        if (erase) {
            char *path = sc_path(c, chunk->key, "chunk");
            if (path) { unlink(path); free(path); }
        }
    }
    free(chunk);
}

static void sc_drop_segments(ds4_state_cache *c, sc_endpoint *e, uint32_t count, bool erase) {
    for (uint32_t i = 0; i < count; i++) {
        sc_chunk *p = sc_chunk_find(c, e->seg[i].key);
        if (p && --p->refs == 0 && p->pins == 0) sc_chunk_remove(c, p, erase);
    }
}

static bool sc_register(ds4_state_cache *c, sc_endpoint *e, bool from_disk) {
    uint32_t done = 0;
    for (; done < e->nseg; done++) {
        sc_segment *s = &e->seg[done];
        sc_chunk *p = sc_chunk_find(c, s->key);
        if (p && p->bytes != s->bytes) break;
        if (!p) {
            if (from_disk) {
                char *path = sc_path(c, s->key, "chunk");
                struct stat st;
                bool valid = path && !stat(path, &st) && S_ISREG(st.st_mode) &&
                             st.st_size >= 0 && (uint64_t)st.st_size == s->bytes;
                free(path);
                if (!valid) break;
            }
            p = calloc(1, sizeof(*p));
            if (!p) break;
            memcpy(p->key, s->key, 41); p->bytes = s->bytes; p->present = from_disk;
            p->next = c->chunks; c->chunks = p;
            if (from_disk) c->disk_bytes += p->bytes;
        }
        p->refs++;
    }
    if (done == e->nseg) return true;
    sc_drop_segments(c, e, done, false);
    return false;
}

static void sc_evict_endpoint(ds4_state_cache *c, sc_endpoint *e) {
    sc_endpoint **p = &c->endpoints;
    while (*p && *p != e) p = &(*p)->next;
    if (*p) *p = e->next;
    char *path = sc_path(c, e->key, "manifest");
    if (path) { unlink(path); free(path); }
    c->disk_bytes -= e->manifest_bytes;
    sc_drop_segments(c, e, e->nseg, true);
    sc_endpoint_free(e); c->evictions++;
}

static bool sc_disk_room(ds4_state_cache *c, uint64_t extra) {
    if (extra > c->disk_limit) return false;
    while (c->disk_bytes > c->disk_limit - extra) {
        sc_endpoint *old = NULL;
        for (sc_endpoint *e = c->endpoints; e; e = e->next)
            if (!e->pins && (!old || e->used < old->used)) old = e;
        if (!old) return false;
        sc_evict_endpoint(c, old);
    }
    return true;
}

static void sc_make_hot(ds4_state_cache *c, sc_chunk *p, const uint8_t *data) {
    if (p->hot || p->bytes > c->hot_limit) return;
    while (c->hot_bytes > c->hot_limit - p->bytes) {
        sc_chunk *old = NULL;
        for (sc_chunk *q = c->chunks; q; q = q->next)
            if (q->hot && !q->pins && (!old || q->used < old->used)) old = q;
        if (!old) return;
        free(old->hot); old->hot = NULL; c->hot_bytes -= old->bytes;
    }
    p->hot = malloc((size_t)p->bytes);
    if (p->hot) {
        memcpy(p->hot, data, (size_t)p->bytes);
        c->hot_bytes += p->bytes; p->used = ++c->clock;
    }
}

static bool sc_read_chunk(ds4_state_cache *c, sc_chunk *p, uint8_t *out) {
    if (!p || !p->present) return false;
    p->used = ++c->clock;
    if (p->hot) {
        memcpy(out, p->hot, (size_t)p->bytes); c->hot_reads++; return true;
    }
    char *path = sc_path(c, p->key, "chunk");
    FILE *fp = path ? fopen(path, "rb") : NULL;
    bool ok = fp && fread(out, 1, (size_t)p->bytes, fp) == p->bytes && fgetc(fp) == EOF;
    if (fp) fclose(fp);
    free(path);
    if (!ok) return false;
    char key[41]; ds4_kvstore_sha1_bytes_hex(out, (size_t)p->bytes, key);
    if (strcmp(key, p->key)) return false;
    c->disk_reads++; sc_make_hot(c, p, out); return true;
}

static ds4_state_cache_ref *sc_acquire(ds4_state_cache *c, sc_endpoint *e) {
    ds4_state_cache_ref *ref = malloc(sizeof(*ref));
    if (!ref) return NULL;
    ref->cache = c; ref->endpoint = e; e->pins++; e->used = ++c->clock;
    ref->ids = NULL; c->lease_bytes += sizeof(*ref);
    for (uint32_t i = 0; i < e->nseg; i++) sc_chunk_find(c, e->seg[i].key)->pins++;
    char *path = sc_path(c, e->key, "manifest");
    if (path) { utime(path, NULL); free(path); }
    return ref;
}

static bool sc_name_key(const char *name, const char *ext, char key[41]) {
    if (strlen(name) != 6u + 40u + 1u + strlen(ext) ||
        strncmp(name, "ds4sc-", 6) || strcmp(name + 47, ext) || name[46] != '.') return false;
    memcpy(key, name + 6, 40); key[40] = 0;
    return sc_key(key);
}

ds4_state_cache *ds4_state_cache_open(const char *dir, uint64_t hot_bytes,
                                     uint64_t ssd_bytes, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    if (!dir || !*dir) { sc_error(err, errlen, "missing state-cache directory"); return NULL; }
    if (mkdir(dir, 0700) && errno != EEXIST) {
        sc_error(err, errlen, "cannot create state-cache directory"); return NULL;
    }
    ds4_state_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->lock_fd = -1; c->dir = strdup(dir); c->hot_limit = hot_bytes; c->disk_limit = ssd_bytes;
    if (!c->dir) goto fail;
    size_t z = strlen(dir) + 20; char *lock = malloc(z);
    if (!lock) goto fail;
    snprintf(lock, z, "%s/.ds4sc.lock", dir);
    c->lock_fd = open(lock, O_RDWR | O_CREAT | O_CLOEXEC, 0600); free(lock);
    if (c->lock_fd < 0 || flock(c->lock_fd, LOCK_EX | LOCK_NB)) {
        sc_error(err, errlen, "state-cache directory already owned or unavailable"); goto fail;
    }
    DIR *dp = opendir(dir);
    if (!dp) goto fail;
    struct dirent *entry;
    while ((entry = readdir(dp))) {
        char key[41];
        if (!sc_name_key(entry->d_name, "manifest", key)) continue;
        char *path = sc_path(c, key, "manifest"); struct stat st;
        uint8_t *bytes = NULL; sc_endpoint *e = NULL;
        if (path && !stat(path, &st) && S_ISREG(st.st_mode) &&
            st.st_size >= SC_HEADER && st.st_size <= SC_MAX_MANIFEST) {
            FILE *fp = fopen(path, "rb"); bytes = malloc((size_t)st.st_size);
            if (fp && bytes && fread(bytes, 1, (size_t)st.st_size, fp) == (size_t)st.st_size)
                e = sc_parse_manifest(bytes, (uint64_t)st.st_size, key);
            if (fp) fclose(fp);
        }
        free(bytes);
        if (e && sc_register(c, e, true)) {
            e->used = st.st_mtime > 0 ? (uint64_t)st.st_mtime : 0;
            if (e->used > c->clock) c->clock = e->used;
            e->next = c->endpoints; c->endpoints = e; c->disk_bytes += e->manifest_bytes;
        } else {
            sc_endpoint_free(e);
            if (path) unlink(path); /* Incomplete cache publication is a miss. */
        }
        free(path);
    }
    rewinddir(dp);
    while ((entry = readdir(dp))) {
        char key[41];
        if (sc_name_key(entry->d_name, "chunk", key) && !sc_chunk_find(c, key)) {
            char *path = sc_path(c, key, "chunk");
            if (path) { unlink(path); free(path); }
        } else if (!strncmp(entry->d_name, ".ds4sc-tmp-", 11)) {
            size_t z = strlen(dir) + strlen(entry->d_name) + 2u;
            char *path = malloc(z);
            if (path) { snprintf(path, z, "%s/%s", dir, entry->d_name); unlink(path); free(path); }
        }
    }
    closedir(dp);
    if (!sc_disk_room(c, 0)) goto fail;
    return c;
fail:
    if (err && errlen && !*err) sc_error(err, errlen, "cannot open state cache");
    ds4_state_cache_close(c, NULL, 0); return NULL;
}

int ds4_state_cache_put(ds4_state_cache *c, const char *profile,
                        const void *state, uint64_t state_bytes,
                        const void *pre, uint64_t pre_bytes,
                        bool approximate, char key[41], char *err, size_t errlen) {
    if (!c || !profile || !*profile || strlen(profile) > 255 || !key ||
        pre_bytes > SIZE_MAX || (pre_bytes && !pre) || (approximate && !pre_bytes))
        return sc_error(err, errlen, "invalid state-cache put");
    sc_endpoint *e = calloc(1, sizeof(*e)); uint8_t *manifest = NULL;
    if (!e) return sc_error(err, errlen, "state-cache allocation failed");
    bool registered = false;
    if (!sc_state_valid(state, state_bytes, approximate, &e->ctx, &e->tokens)) goto invalid;
    e->profile = strdup(profile); e->state_bytes = state_bytes; e->pre_bytes = pre_bytes;
    e->approximate = approximate;
    if (!e->profile || !sc_prefix(profile, e->ctx, (const uint8_t *)state + 52,
                                  e->tokens, false, e->prefix) ||
        !sc_split(e, state, pre) || !(manifest = sc_manifest(e))) goto failed;
    sc_endpoint *existing = sc_endpoint_find(c, e->key);
    if (existing) {
        existing->used = ++c->clock; memcpy(key, e->key, 41);
        free(manifest); sc_endpoint_free(e); return 0;
    }
    /* An endpoint must itself fit, even if all other cache entries disappear. */
    uint64_t own = e->manifest_bytes;
    for (uint32_t i = 0; i < e->nseg; i++) {
        bool first = true;
        for (uint32_t j = 0; j < i; j++)
            if (!strcmp(e->seg[j].key, e->seg[i].key)) { first = false; break; }
        if (first) own += e->seg[i].bytes;
    }
    if (own > c->disk_limit) goto budget;
    if (!sc_register(c, e, false)) goto failed;
    registered = true;
    uint64_t missing = e->manifest_bytes;
    for (sc_chunk *p = c->chunks; p; p = p->next)
        if (!p->present) missing += p->bytes;
    /* Registered references protect shared pages while old endpoints evict. */
    if (!sc_disk_room(c, missing)) goto budget;
    for (uint32_t i = 0; i < e->nseg; i++) {
        sc_segment *s = &e->seg[i]; sc_chunk *p = sc_chunk_find(c, s->key);
        const uint8_t *data = s->which ? pre : state;
        if (!p->present) {
            char *path = sc_path(c, p->key, "chunk");
            bool ok = path && sc_atomic(c->dir, path, data + s->offset, (size_t)s->bytes);
            free(path);
            if (!ok) goto failed;
            p->present = true; c->disk_bytes += p->bytes;
        }
    }
    char *path = sc_path(c, e->key, "manifest");
    bool ok = path && sc_atomic(c->dir, path, manifest, (size_t)e->manifest_bytes);
    free(path);
    if (!ok) goto failed;
    e->used = ++c->clock; e->next = c->endpoints; c->endpoints = e;
    c->disk_bytes += e->manifest_bytes;
    for (uint32_t i = 0; i < e->nseg; i++) {
        sc_segment *s = &e->seg[i]; const uint8_t *data = s->which ? pre : state;
        sc_make_hot(c, sc_chunk_find(c, s->key), data + s->offset);
    }
    memcpy(key, e->key, 41); free(manifest); return 0;
invalid:
    sc_error(err, errlen, "not a complete DS4.1 payload of the requested kind"); goto cleanup;
budget:
    sc_error(err, errlen, "state-cache SSD budget is occupied or endpoint exceeds it"); goto cleanup;
failed:
    sc_error(err, errlen, "state-cache allocation or atomic publication failed");
cleanup:
    if (registered) sc_drop_segments(c, e, e->nseg, true);
    free(manifest); sc_endpoint_free(e); return -1;
}

ds4_state_cache_ref *ds4_state_cache_get(ds4_state_cache *c, const char *key) {
    if (!c || !sc_key(key)) return NULL;
    sc_endpoint *e = sc_endpoint_find(c, key);
    return e ? sc_acquire(c, e) : NULL;
}

ds4_state_cache_ref *ds4_state_cache_find_prefix(ds4_state_cache *c, const char *profile,
                                               uint32_t ctx, const int *ids, uint32_t n,
                                               uint32_t *matched_tokens) {
    if (matched_tokens) *matched_tokens = 0;
    if (!c || !profile || !ids) return NULL;
    sc_endpoint *best = NULL;
    for (sc_endpoint *e = c->endpoints; e; e = e->next) {
        if (e->approximate || e->ctx != ctx || e->tokens > n || strcmp(e->profile, profile) ||
            (best && best->tokens >= e->tokens)) continue;
        char key[41];
        if (sc_prefix(profile, ctx, ids, e->tokens, true, key) && !strcmp(key, e->prefix)) best = e;
    }
    if (!best) return NULL;
    ds4_state_cache_ref *ref = sc_acquire(c, best);
    if (ref && matched_tokens) *matched_tokens = best->tokens;
    return ref;
}

static bool sc_token_tail(ds4_state_cache *c, sc_endpoint *e, const int *ids, uint32_t n) {
    uint64_t start = 52u + (uint64_t)(e->tokens - n) * 4u;
    uint64_t stop = 52u + (uint64_t)e->tokens * 4u;
    for (uint32_t i = 0; i < e->nseg; i++) {
        sc_segment *s = &e->seg[i];
        if (s->which || s->offset >= stop || s->offset + s->bytes <= start) continue;
        uint8_t *buf = malloc((size_t)s->bytes);
        if (!buf) return false;
        bool ok = sc_read_chunk(c, sc_chunk_find(c, s->key), buf);
        uint64_t a = s->offset > start ? s->offset : start;
        uint64_t b = s->offset + s->bytes < stop ? s->offset + s->bytes : stop;
        for (uint64_t off = a; ok && off < b; off += 4u) {
            int id = ids[(off - start) / 4u];
            ok = id >= 0 && (uint32_t)id < SC_VOCAB && sc_u32(buf + off - s->offset) == (uint32_t)id;
        }
        free(buf);
        if (!ok) return false;
    }
    return true;
}

ds4_state_cache_ref *ds4_state_cache_find_suffix(ds4_state_cache *c, const char *profile,
                                               uint32_t ctx, const int *ids, uint32_t n,
                                               uint32_t *source_start) {
    if (source_start) *source_start = 0;
    if (!c || !profile || !ids || !n) return NULL;
    sc_endpoint *best = NULL;
    for (sc_endpoint *e = c->endpoints; e; e = e->next)
        if (e->ctx == ctx && e->tokens >= n && !strcmp(e->profile, profile) &&
            (!best || e->used > best->used) && sc_token_tail(c, e, ids, n)) best = e;
    if (!best) return NULL;
    ds4_state_cache_ref *ref = sc_acquire(c, best);
    if (ref && source_start) *source_start = best->tokens - n;
    return ref;
}

static bool sc_ref_tokens(ds4_state_cache_ref *ref) {
    sc_endpoint *e = ref->endpoint;
    ref->ids = malloc((size_t)e->tokens * sizeof(*ref->ids));
    if (!ref->ids) return false;
    ref->cache->lease_bytes += (uint64_t)e->tokens * sizeof(*ref->ids);
    const uint64_t start = 52u, stop = 52u + (uint64_t)e->tokens * 4u;
    for (uint32_t i = 0; i < e->nseg; i++) {
        sc_segment *s = &e->seg[i];
        if (s->which || s->offset >= stop || s->offset + s->bytes <= start) continue;
        uint8_t *buf = malloc((size_t)s->bytes);
        if (!buf) return false;
        bool ok = sc_read_chunk(ref->cache, sc_chunk_find(ref->cache, s->key), buf);
        uint64_t a = s->offset > start ? s->offset : start;
        uint64_t b = s->offset + s->bytes < stop ? s->offset + s->bytes : stop;
        for (uint64_t off = a; ok && off < b; off += 4u) {
            uint32_t id = sc_u32(buf + off - s->offset);
            if (id >= SC_VOCAB) ok = false;
            else ref->ids[(off - start) / 4u] = (int)id;
        }
        free(buf);
        if (!ok) return false;
    }
    return true;
}

ds4_state_cache_ref *ds4_state_cache_next_candidate(ds4_state_cache *c,
                                                  const char *profile,
                                                  uint32_t ctx,
                                                  char cursor[41],
                                                  const int **ids, uint32_t *n,
                                                  bool *approximate) {
    if (ids) *ids = NULL;
    if (n) *n = 0;
    if (!c || !profile || !cursor || (*cursor && !sc_key(cursor))) return NULL;
    sc_endpoint *next = NULL;
    for (sc_endpoint *e = c->endpoints; e; e = e->next)
        if (e->ctx == ctx && !strcmp(e->profile, profile) && strcmp(e->key, cursor) > 0 &&
            (!next || strcmp(e->key, next->key) < 0)) next = e;
    if (!next) return NULL;
    ds4_state_cache_ref *ref = sc_acquire(c, next);
    if (!ref) return NULL;
    memcpy(cursor, next->key, 41);
    if (!sc_ref_tokens(ref)) { ds4_state_cache_release(ref); return NULL; }
    if (ids) *ids = ref->ids;
    if (n) *n = next->tokens;
    if (approximate) *approximate = next->approximate;
    return ref;
}

void ds4_state_cache_ref_sizes(const ds4_state_cache_ref *ref,
                               uint64_t *state_bytes, uint64_t *prerope_bytes) {
    if (state_bytes) *state_bytes = ref ? ref->endpoint->state_bytes : 0;
    if (prerope_bytes) *prerope_bytes = ref ? ref->endpoint->pre_bytes : 0;
}

int ds4_state_cache_materialize(ds4_state_cache_ref *ref, ds4_state_cache_blob *out,
                                char *err, size_t errlen) {
    if (!ref || !out) return sc_error(err, errlen, "missing state-cache reference/output");
    ds4_state_cache_blob blob = {0}; sc_endpoint *e = ref->endpoint;
    blob.state = malloc((size_t)e->state_bytes);
    if (e->pre_bytes) blob.prerope = malloc((size_t)e->pre_bytes);
    if (!blob.state || (e->pre_bytes && !blob.prerope)) goto failed;
    blob.state_bytes = e->state_bytes; blob.prerope_bytes = e->pre_bytes;
    blob.ctx_size = e->ctx; blob.tokens = e->tokens; blob.approximate = e->approximate;
    for (uint32_t i = 0; i < e->nseg; i++) {
        sc_segment *s = &e->seg[i];
        uint8_t *p = s->which ? blob.prerope : blob.state;
        if (!sc_read_chunk(ref->cache, sc_chunk_find(ref->cache, s->key), p + s->offset)) goto failed;
    }
    uint32_t ctx, n;
    if (!sc_state_valid(blob.state, blob.state_bytes, blob.approximate, &ctx, &n) ||
        ctx != e->ctx || n != e->tokens) goto failed;
    *out = blob; return 0;
failed:
    ds4_state_cache_blob_free(&blob);
    return sc_error(err, errlen, "state-cache materialization failed or chunk is corrupt");
}

void ds4_state_cache_blob_free(ds4_state_cache_blob *blob) {
    if (!blob) return;
    free(blob->state); free(blob->prerope); memset(blob, 0, sizeof(*blob));
}

void ds4_state_cache_release(ds4_state_cache_ref *ref) {
    if (!ref) return;
    sc_endpoint *e = ref->endpoint;
    for (uint32_t i = 0; i < e->nseg; i++) sc_chunk_find(ref->cache, e->seg[i].key)->pins--;
    e->pins--;
    ref->cache->lease_bytes -= sizeof(*ref) + (ref->ids ? (uint64_t)e->tokens * sizeof(*ref->ids) : 0u);
    free(ref->ids); free(ref);
}

void ds4_state_cache_get_stats(const ds4_state_cache *c, ds4_state_cache_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!c) return;
    out->hot_bytes = c->hot_bytes; out->ssd_bytes = c->disk_bytes;
    out->metadata_bytes = sizeof(*c) + strlen(c->dir) + 1u + c->lease_bytes;
    out->hot_reads = c->hot_reads; out->disk_reads = c->disk_reads;
    out->endpoint_evictions = c->evictions;
    for (sc_chunk *p = c->chunks; p; p = p->next) {
        out->chunks++; out->metadata_bytes += sizeof(*p);
    }
    for (sc_endpoint *e = c->endpoints; e; e = e->next) {
        out->endpoints++; out->pinned_endpoints += e->pins != 0;
        out->metadata_bytes += sizeof(*e) + strlen(e->profile) + 1u + (uint64_t)e->nseg * sizeof(*e->seg);
    }
}

int ds4_state_cache_close(ds4_state_cache *c, char *err, size_t errlen) {
    if (!c) return 0;
    for (sc_endpoint *e = c->endpoints; e; e = e->next)
        if (e->pins) return sc_error(err, errlen, "state-cache references remain pinned");
    while (c->endpoints) {
        sc_endpoint *e = c->endpoints; c->endpoints = e->next; sc_endpoint_free(e);
    }
    while (c->chunks) sc_chunk_remove(c, c->chunks, false);
    if (c->lock_fd >= 0) { flock(c->lock_fd, LOCK_UN); close(c->lock_fd); }
    free(c->dir); free(c); return 0;
}
