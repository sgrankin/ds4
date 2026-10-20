/* Standalone lossless MXFP4 scale codec, matching Python's MXS3P001 format.
 * libc only; no engine/GPU dependencies. Define MXS3_NO_MAIN to reuse codec.
 * Native input: [E8M0 scale, 16 unchanged E2M1 code bytes] per 32 weights.
 * Each group: tag0 + native blocks, or tag1..8 + sorted dictionary + packed
 * little-endian three-bit indices + concatenated unchanged code bytes.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MXS3_HEADER = 20, MXS3_GROUP = 4096 };
static const uint8_t mxs3_magic[8] = {'M','X','S','3','P','0','0','1'};
typedef struct {
    size_t groups, raw_groups, dictionary_bytes, index_bytes;
} mxs3_stats;

static void mxs3_put_le(uint8_t *out, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) out[i] = (uint8_t)(value >> (8u*i));
}
static uint64_t mxs3_get_le(const uint8_t *in, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; i++) value |= (uint64_t)in[i] << (8u*i);
    return value;
}

size_t mxs3_pack_bound(size_t bytes, uint32_t group) {
    if (bytes % 17u || !group) return 0;
    size_t blocks = bytes/17u, groups = blocks/group + (blocks%group != 0);
    if (bytes > SIZE_MAX-MXS3_HEADER || groups > (SIZE_MAX-MXS3_HEADER-bytes)/9u) return 0;
    return MXS3_HEADER + bytes + groups*9u;
}

int mxs3_pack(const uint8_t *raw, size_t bytes, uint32_t group,
              uint8_t *out, size_t capacity, size_t *written, mxs3_stats *stats) {
    size_t bound = mxs3_pack_bound(bytes, group);
    if (!raw || !out || !written || !bound || capacity < bound) return -1;
    size_t blocks = bytes/17u, pos = MXS3_HEADER;
    mxs3_stats st = {0};
    memcpy(out, mxs3_magic, 8);
    mxs3_put_le(out+8, blocks, 8); mxs3_put_le(out+16, group, 4);
    for (size_t first = 0; first < blocks;) {
        size_t count = blocks-first < group ? blocks-first : group;
        const uint8_t *src = raw+first*17u;
        uint8_t present[256] = {0}, dictionary[8], lookup[256] = {0};
        unsigned kinds = 0;
        for (size_t i = 0; i < count; i++) present[src[i*17u]] = 1;
        for (unsigned v = 0; v < 256; v++) if (present[v]) {
            if (kinds < 8) { dictionary[kinds] = (uint8_t)v; lookup[v] = (uint8_t)kinds; }
            kinds++;
        }
        st.groups++;
        if (kinds > 8) {
            out[pos++] = 0; memcpy(out+pos, src, count*17u); pos += count*17u;
            st.raw_groups++; first += count; continue;
        }
        out[pos++] = (uint8_t)kinds;
        memcpy(out+pos, dictionary, kinds); pos += kinds; st.dictionary_bytes += kinds;
        size_t index_bytes = (count*3u+7u)/8u;
        uint8_t *indices = out+pos;
        memset(indices, 0, index_bytes); pos += index_bytes; st.index_bytes += index_bytes;
        for (size_t i = 0; i < count; i++) {
            size_t bit = i*3u;
            unsigned value = lookup[src[i*17u]], shift = (unsigned)(bit%8u);
            indices[bit/8u] |= (uint8_t)(value << shift);
            if (shift > 5) indices[bit/8u+1u] |= (uint8_t)(value >> (8u-shift));
            memcpy(out+pos+i*16u, src+i*17u+1u, 16);
        }
        pos += count*16u; first += count;
    }
    *written = pos;
    if (stats) *stats = st;
    return 0;
}

int mxs3_unpack(const uint8_t *packed, size_t bytes, uint8_t *out,
                size_t capacity, size_t *written) {
    if (!packed || !out || !written || bytes < MXS3_HEADER ||
        memcmp(packed, mxs3_magic, 8)) return -1;
    uint64_t block64 = mxs3_get_le(packed+8, 8);
    uint32_t group = (uint32_t)mxs3_get_le(packed+16, 4);
    if (!group || block64 > SIZE_MAX/17u || block64*17u > capacity) return -1;
    size_t blocks = (size_t)block64, pos = MXS3_HEADER;
    for (size_t first = 0; first < blocks;) {
        size_t count = blocks-first < group ? blocks-first : group;
        if (pos == bytes) return -1;
        unsigned kinds = packed[pos++];
        if (!kinds) {
            if (count*17u > bytes-pos) return -1;
            memcpy(out+first*17u, packed+pos, count*17u); pos += count*17u;
            first += count; continue;
        }
        if (kinds > 8 || kinds > bytes-pos) return -1;
        const uint8_t *dictionary = packed+pos; pos += kinds;
        size_t index_bytes = (count*3u+7u)/8u;
        if (index_bytes > bytes-pos) return -1;
        const uint8_t *indices = packed+pos; pos += index_bytes;
        if (count*16u > bytes-pos) return -1;
        const uint8_t *codes = packed+pos;
        for (size_t i = 0; i < count; i++) {
            size_t bit = i*3u;
            unsigned shift = (unsigned)(bit%8u), value = indices[bit/8u] >> shift;
            if (shift > 5) value |= (unsigned)indices[bit/8u+1u] << (8u-shift);
            value &= 7u;
            if (value >= kinds) return -1;
            out[(first+i)*17u] = dictionary[value];
            memcpy(out+(first+i)*17u+1u, codes+i*16u, 16);
        }
        pos += count*16u; first += count;
    }
    if (pos != bytes) return -1;
    *written = blocks*17u;
    return 0;
}

#ifndef MXS3_NO_MAIN
static volatile uint64_t observed;
static void *(*volatile observed_memcpy)(void *, const void *, size_t) = memcpy;
static double now_seconds(void) {
    struct timespec t;
#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
#else
    if (!timespec_get(&t, TIME_UTC)) abort();
#endif
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9;
}
static uint64_t digest(const uint8_t *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) h = (h ^ p[i])*UINT64_C(1099511628211);
    return h;
}
static uint8_t *load_file(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END)) return NULL;
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    uint8_t *p = malloc((size_t)n ? (size_t)n : 1u);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f); *bytes = (size_t)n; return p;
}
static int sanity(void) {
    uint8_t raw[19*17], packed[19*17+128], restored[sizeof(raw)];
    const unsigned groups[] = {3, 11, 19}, alphabets[] = {7, 8, 9};
    for (unsigned c = 0; c < 3; c++) {
        size_t n = c == 0 ? 7u : c == 1 ? 11u : 19u, packed_n = 0, restored_n = 0;
        for (size_t i = 0; i < n; i++) {
            raw[i*17u] = (uint8_t)(i%alphabets[c]);
            for (unsigned j = 1; j < 17; j++) raw[i*17u+j] = (uint8_t)(i*17u+j);
        }
        mxs3_stats st;
        if (mxs3_pack(raw,n*17u,groups[c],packed,sizeof(packed),&packed_n,&st) ||
            mxs3_unpack(packed,packed_n,restored,sizeof(restored),&restored_n) ||
            restored_n != n*17u || memcmp(raw,restored,restored_n) ||
            (c == 2 && st.raw_groups != 1)) return -1;
    }
    return 0;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s ITERATIONS SAMPLE.mxfp4 ...\n",argv[0]); return 2; }
    char *end; errno = 0; unsigned long iterations = strtoul(argv[1],&end,10);
    if (errno || !iterations || *end || iterations > 10000) return 2;
    if (sanity()) { fprintf(stderr,"codec boundary/fallback check failed\n"); return 1; }
    printf("sample,input_bytes,packed_bytes,iterations,encode_seconds,decode_seconds,memcpy_seconds,encode_MiB_s,decode_MiB_s,memcpy_MiB_s,roundtrip_exact,python_packet_exact,raw_groups\n");
    for (int a = 2; a < argc; a++) {
        size_t n = 0, pn = 0, rn = 0, reference_n = 0;
        uint8_t *raw = load_file(argv[a],&n);
        size_t bound = mxs3_pack_bound(n,MXS3_GROUP);
        uint8_t *packed = malloc(bound), *restored = malloc(n);
        char *reference_path = malloc(strlen(argv[a])+7u);
        if (!raw || !bound || !packed || !restored || !reference_path) return 1;
        strcpy(reference_path,argv[a]); strcat(reference_path,".mxs3");
        uint8_t *reference = load_file(reference_path,&reference_n);
        mxs3_stats st;
        if (!reference || mxs3_pack(raw,n,MXS3_GROUP,packed,bound,&pn,&st) ||
            pn != reference_n || memcmp(packed,reference,pn) ||
            mxs3_unpack(packed,pn,restored,n,&rn) || rn != n || memcmp(raw,restored,n)) {
            fprintf(stderr,"initial exact roundtrip/Python format check failed: %s\n",argv[a]); return 1;
        }
        double t = now_seconds();
        for (unsigned long i = 0; i < iterations; i++) {
            if (mxs3_pack(raw,n,MXS3_GROUP,packed,bound,&pn,NULL)) return 1;
            observed ^= packed[(i*17u)%pn];
        }
        double encode_s = now_seconds()-t;
        t = now_seconds();
        for (unsigned long i = 0; i < iterations; i++) {
            if (mxs3_unpack(packed,pn,restored,n,&rn)) return 1;
            observed ^= restored[(i*17u)%rn];
        }
        double decode_s = now_seconds()-t;
        if (rn != n || memcmp(raw,restored,n)) return 1;
        t = now_seconds();
        for (unsigned long i = 0; i < iterations; i++) {
            observed_memcpy(restored,raw,n); observed ^= restored[(i*17u)%n];
        }
        double copy_s = now_seconds()-t;
        if (memcmp(raw,restored,n)) return 1;
        observed ^= digest(restored,n) ^ digest(packed,pn);
        double mib = (double)n*(double)iterations/1048576.0;
        printf("%s,%zu,%zu,%lu,%.9f,%.9f,%.9f,%.3f,%.3f,%.3f,true,true,%zu\n",
               argv[a],n,pn,iterations,encode_s,decode_s,copy_s,mib/encode_s,mib/decode_s,mib/copy_s,st.raw_groups);
        free(reference); free(reference_path); free(restored); free(packed); free(raw);
    }
    fprintf(stderr,"boundary/fallback3 PASS; all byte/format checks PASS; observed=%llu\n",(unsigned long long)observed);
    return 0;
}
#endif
