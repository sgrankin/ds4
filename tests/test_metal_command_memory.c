/* Exercise the C entry points without an Objective-C caller's autorelease pool. */
#include "ds4_gpu.h"
#include <malloc/malloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int exchange(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    (void)layer; (void)gate;
    return seq == __atomic_add_fetch((uint64_t *)ud, 1, __ATOMIC_RELAXED);
}

static int batch_exchange(void *ud, uint32_t layer, uint32_t rows, uint64_t seq) {
    return rows == 1 && exchange(ud, layer, 1, seq);
}

static int big_exchange(void *ud, uint32_t layer, uint64_t seq,
                         const void *out, void *in, uint64_t bytes) {
    if (bytes != 1024 || !exchange(ud, layer, 1, seq)) return 0;
    memcpy(in, out, (size_t)bytes);
    return 1;
}

static int check_tp_memory(int mode) {
    ds4_gpu_tensor *a = NULL, *slab = NULL, *out = NULL, *in = NULL;
    uint64_t exchanges = 0;
    int rc = 1;
    CHECK(ds4_gpu_init());
    a = ds4_gpu_tensor_alloc(1024);
    in = ds4_gpu_tensor_alloc(1024);
    slab = ds4_gpu_tensor_alloc(8192);
    CHECK(a && in && slab);
    CHECK(ds4_gpu_tensor_fill_f32(slab, 0, 2048));
    out = ds4_gpu_tensor_view(slab, 4096, 1024);
    CHECK(out);
    CHECK(ds4_gpu_tp_init(0, slab, 0, 4096, 1024, exchange, &exchanges));
    if (mode == 1) ds4_gpu_tp_set_session_batch_mode(1);
    ds4_gpu_tp_set_batch_exchange(batch_exchange);
    ds4_gpu_tp_set_big_exchange(big_exchange);
    float input[256], output[256];
    for (int j = 0; j < 256; j++) input[j] = j / 4.0f;
    CHECK(ds4_gpu_tensor_write(a, 0, input, sizeof(input)));
    size_t baseline = 0;
    for (int i = 0; i < 8192; i++) {
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_add_tensor(out, a, a, 256));
        if (mode == 3) CHECK(ds4_gpu_tp_big_gate_encode(0, 1, out, in, 1024));
        else if (mode == 2) CHECK(ds4_gpu_tp_batch_gate_encode(0, 1));
        else CHECK(ds4_gpu_tp_gate_encode(0, 0));
        CHECK(ds4_gpu_end_commands());
        CHECK(!ds4_gpu_tp_failed());
        CHECK(ds4_gpu_tensor_read(mode == 3 ? in : out, 0, output, sizeof(output)));
        for (int j = 0; j < 256; j++) CHECK(output[j] == 2 * input[j]);
        if (i % 1024 == 1023) {
            malloc_statistics_t stats;
            malloc_zone_statistics(NULL, &stats);
            if (!baseline) baseline = stats.size_in_use;
            fprintf(stderr, "Metal TP mode %d: %d gates, heap %.3f MiB\n",
                    mode, i + 1, stats.size_in_use / 1048576.0);
            CHECK(stats.size_in_use <= baseline + 8u * 1024u * 1024u);
        }
    }
    CHECK(__atomic_load_n(&exchanges, __ATOMIC_RELAXED) == 8192);
    puts("Metal TP payloads, ordered loopback exchanges and bounded command heap: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tp_shutdown();
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(slab);
    ds4_gpu_tensor_free(in);
    ds4_gpu_tensor_free(a);
    ds4_gpu_cleanup();
    return rc;
}

/* The real public resource actions: nested offsets, retained aliases, empty
 * operations, caller-owned blits, and duplicate-free validation before ARC. */
static int check_resource_boundaries(void) {
    ds4_gpu_tensor *base = NULL, *view = NULL, *nested = NULL;
    ds4_gpu_tensor *empty = NULL, *destination = NULL;
    int rc = 1;
    const unsigned char value[] = {11, 22, 33, 44};
    unsigned char actual[8] = {0};
    CHECK(ds4_gpu_init());
    CHECK(!ds4_gpu_tensor_alloc(0));
    CHECK(!ds4_gpu_tensor_view(NULL, 0, 0));
    CHECK(!ds4_gpu_tensor_bytes(NULL) && !ds4_gpu_tensor_contents(NULL));
    CHECK(!ds4_gpu_tensor_read(NULL, 0, NULL, 0));
    CHECK(!ds4_gpu_tensor_write(NULL, 0, NULL, 0));
    base = ds4_gpu_tensor_alloc_managed(32);
    destination = ds4_gpu_tensor_alloc(8);
    CHECK(base && destination);
    CHECK(ds4_gpu_tensor_fill_f32(base, 0, 8));
    CHECK(!ds4_gpu_tensor_fill_f32(base, 0, 9));
    CHECK(ds4_gpu_tensor_write(destination, 0, actual, sizeof actual));
    CHECK(!ds4_gpu_tensor_view(base, 33, 0));
    CHECK(!ds4_gpu_tensor_view(base, 31, 2));
    empty = ds4_gpu_tensor_view(base, 32, 0);
    view = ds4_gpu_tensor_view(base, 3, 16);
    CHECK(empty && view && !ds4_gpu_tensor_bytes(empty));
    nested = ds4_gpu_tensor_view(view, 5, sizeof value);
    CHECK(nested && ds4_gpu_tensor_bytes(nested) == sizeof value);
    CHECK(ds4_gpu_tensor_write(nested, 0, value, sizeof value));
    CHECK(ds4_gpu_tensor_read(base, 8, actual, sizeof value));
    CHECK(!memcmp(value, actual, sizeof value));
    CHECK(ds4_gpu_tensor_write(base, 32, NULL, 0));
    CHECK(ds4_gpu_tensor_read(empty, 0, NULL, 0));
    CHECK(!ds4_gpu_tensor_write(base, 33, NULL, 0));
    CHECK(!ds4_gpu_tensor_read(nested, 0, NULL, 1));
    ds4_gpu_tensor_free(empty); empty = NULL;
    CHECK(ds4_gpu_tensor_copy(destination, 8, nested, sizeof value, 0));
    CHECK(!ds4_gpu_tensor_copy(destination, 9, nested, 0, 0));
    CHECK(!ds4_gpu_tensor_copy(destination, 0, nested, 0, 1));
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_copy(destination, 2, nested, 1, 3));
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(destination, 0, actual, sizeof actual));
    CHECK(actual[0] == 0 && actual[1] == 0 && actual[2] == 22 &&
          actual[3] == 33 && actual[4] == 44 && actual[5] == 0);
    ds4_gpu_tensor_free(base); base = NULL;
    ds4_gpu_tensor_free(view); view = NULL;
    CHECK(ds4_gpu_tensor_read(nested, 0, actual, sizeof value));
    CHECK(!memcmp(actual, value, sizeof value));
    ds4_gpu_tensor *retired = nested;
    ds4_gpu_tensor_free(nested); nested = NULL;
    ds4_gpu_tensor_free(retired); /* must not dereference the retired object */
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(empty);
    ds4_gpu_tensor_free(nested);
    ds4_gpu_tensor_free(view);
    ds4_gpu_tensor_free(base);
    ds4_gpu_tensor_free(destination);
    return rc;
}

/* These public actions use real Q8 weights and real retained tensor views.
 * The test-only stage shim calls the existing private stage functions; it
 * neither fabricates scheduler state nor models a routed expert controller. */
extern int ds4_gpu_test_parallel_ffn_stages(const ds4_gpu_tensor *routed_mid);

static ds4_gpu_tensor *scheduling_upload(const void *data, uint64_t bytes) {
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    if (tensor && data && !ds4_gpu_tensor_write(tensor, 0, data, bytes)) {
        ds4_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

static int check_parallel_scheduling(void) {
    enum { D = 32, H = 64, ROW = 34, MATRIX = H * ROW, BYTES = 3 * MATRIX };
    unsigned char *model = MAP_FAILED;
    ds4_gpu_tensor *x = NULL, *gate = NULL, *up = NULL, *mid = NULL;
    ds4_gpu_tensor *out = NULL, *ids = NULL;
    float input[D], expected_gate[H], expected_up[H], expected_mid[H];
    float expected_out[D], actual[H];
    const int32_t selected[] = {0, 2, 5, 7, 11, 13};
    int rc = 1;
    CHECK(ds4_gpu_init());
    model = mmap(NULL, BYTES, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(model != MAP_FAILED);
    /* Q8_0 wire blocks: an exact half scale and signed quantized bytes. */
    for (int block = 0; block < BYTES / ROW; block++) {
        const uint16_t scale = 0x2800; /* 1/32 */
        memcpy(model + block * ROW, &scale, sizeof scale);
        for (int lane = 0; lane < 32; lane++)
            model[block * ROW + 2 + lane] = (unsigned char)(int8_t)((block + lane) % 7 - 3);
    }
    for (int i = 0; i < D; i++) input[i] = (i - 15) / 32.0f;
    x = scheduling_upload(input, sizeof input);
    gate = scheduling_upload(NULL, sizeof expected_gate);
    up = scheduling_upload(NULL, sizeof expected_up);
    mid = scheduling_upload(NULL, sizeof expected_mid);
    out = scheduling_upload(NULL, sizeof expected_out);
    ids = scheduling_upload(selected, sizeof selected);
    CHECK(x && gate && up && mid && out && ids);
    CHECK(ds4_gpu_set_model_map(model, BYTES));
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate, up, mid,
        model, BYTES, 0, MATRIX, D, H, x, 7.0f));
    CHECK(ds4_gpu_matmul_q8_0_tensor(out, model, BYTES, 2 * MATRIX, H, D, mid, 1));
    CHECK(ds4_gpu_tensor_read(gate, 0, expected_gate, sizeof expected_gate));
    CHECK(ds4_gpu_tensor_read(up, 0, expected_up, sizeof expected_up));
    CHECK(ds4_gpu_tensor_read(mid, 0, expected_mid, sizeof expected_mid));
    CHECK(ds4_gpu_tensor_read(out, 0, expected_out, sizeof expected_out));

    CHECK(ds4_gpu_begin_commands());
    CHECK(!ds4_gpu_parallel_ffn_finish());
    CHECK(ds4_gpu_parallel_ffn_start(gate, up, mid, out,
        model, BYTES, 0, MATRIX, 2 * MATRIX, D, H, x, 7.0f));
    CHECK(!ds4_gpu_parallel_ffn_start(gate, up, mid, out,
        model, BYTES, 0, MATRIX, 2 * MATRIX, D, H, x, 7.0f));
    ds4_gpu_parallel_ffn_abort();
    ds4_gpu_parallel_ffn_abort();
    CHECK(!ds4_gpu_parallel_ffn_finish());
    CHECK(ds4_gpu_parallel_ffn_start_sliced(gate, up, mid, out,
        model, BYTES, 0, MATRIX, 2 * MATRIX, D, H, 32, 32, x, 7.0f));
    CHECK(!ds4_gpu_parallel_ffn_finish()); /* incomplete join clears its owner */
    CHECK(ds4_gpu_parallel_ffn_start_split(gate, up, mid, out,
        model, BYTES, 0, MATRIX, 2 * MATRIX, D, H, x, 7.0f,
        ids, 0, 2, 16, 6, 0));
    ds4_gpu_parallel_ffn_abort(); /* local split admission, no peer transport */
    CHECK(ds4_gpu_parallel_ffn_start(gate, up, mid, out,
        model, BYTES, 0, MATRIX, 2 * MATRIX, D, H, x, 7.0f));
    CHECK(ds4_gpu_test_parallel_ffn_stages(mid));
    CHECK(ds4_gpu_parallel_ffn_finish());
    CHECK(!ds4_gpu_parallel_ffn_finish());
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(gate, 0, actual, sizeof expected_gate));
    CHECK(!memcmp(actual, expected_gate, sizeof expected_gate));
    CHECK(ds4_gpu_tensor_read(up, 0, actual, sizeof expected_up));
    CHECK(!memcmp(actual, expected_up, sizeof expected_up));
    CHECK(ds4_gpu_tensor_read(mid, 0, actual, sizeof expected_mid));
    CHECK(!memcmp(actual, expected_mid, sizeof expected_mid));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof expected_out));
    CHECK(!memcmp(actual, expected_out, sizeof expected_out));
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_add_tensor(out, x, x, D)); /* stale concurrent state is gone */
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof expected_out));
    for (int i = 0; i < D; i++) CHECK(actual[i] == input[i] * 2);
    puts("Metal parallel FFN real stages, abort and incomplete-join reset: PASS");
    rc = 0;
done:
    ds4_gpu_parallel_ffn_abort();
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); /* retire mapped views before releasing mmap backing */
    if (model != MAP_FAILED) munmap(model, BYTES);
    return rc;
}

static int check_deferred_kv_scheduling(void) {
    enum { Q = 256, KV = 128, CAP = 3, VALUES = Q + KV + CAP * KV };
    float weights[Q + KV], q_input[Q], kv_input[KV];
    float expected[VALUES], actual[VALUES];
    ds4_gpu_tensor *q = NULL, *kv = NULL, *q_out = NULL, *kv_out = NULL;
    ds4_gpu_tensor *raw = NULL, *independent = NULL;
    void *model = MAP_FAILED;
    int rc = 1;
    CHECK(ds4_gpu_init());
    model = mmap(NULL, sizeof weights, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(model != MAP_FAILED);
    for (int i = 0; i < Q + KV; i++) weights[i] = 1.0f + (i % 5) / 16.0f;
    memcpy(model, weights, sizeof weights);
    for (int i = 0; i < Q; i++) q_input[i] = (i % 23 - 11) / 8.0f;
    for (int i = 0; i < KV; i++) kv_input[i] = (i % 19 - 9) / 8.0f;
    q = scheduling_upload(q_input, sizeof q_input);
    kv = scheduling_upload(kv_input, sizeof kv_input);
    q_out = scheduling_upload(NULL, sizeof q_input);
    kv_out = scheduling_upload(NULL, sizeof kv_input);
    raw = scheduling_upload(NULL, CAP * sizeof kv_input);
    independent = scheduling_upload(NULL, sizeof q_input);
    CHECK(q && kv && q_out && kv_out && raw && independent);
    CHECK(ds4_gpu_set_model_map(model, sizeof weights));
    CHECK(!ds4_gpu_kv_norm_task_pending());
    CHECK(ds4_gpu_kv_norm_task_flush());
    CHECK(!ds4_gpu_kv_norm_task_begin_concurrent());
    for (int mode = 0; mode < 3; mode++) {
        CHECK(ds4_gpu_tensor_fill_f32(raw, 123.0f, CAP * KV));
        CHECK(ds4_gpu_begin_commands());
        if (mode) ds4_gpu_dsv4_qkv_norm_defer_kv_next();
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
            q_out, q, model, sizeof weights, 0, Q, kv_out, kv, Q * sizeof(float), KV,
            raw, CAP, 1, 64, 7, 4096, 10000.0f, 1.0f, 1.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        CHECK(ds4_gpu_kv_norm_task_pending() == (mode != 0));
        if (mode == 1) CHECK(ds4_gpu_kv_norm_task_flush());
        if (mode == 2) {
            CHECK(ds4_gpu_kv_norm_task_begin_concurrent());
            CHECK(!ds4_gpu_kv_norm_task_pending());
            CHECK(ds4_gpu_add_tensor(independent, q, q, Q));
            ds4_gpu_kv_norm_task_end_concurrent();
            ds4_gpu_kv_norm_task_end_concurrent();
        }
        CHECK(!ds4_gpu_kv_norm_task_pending());
        CHECK(ds4_gpu_kv_norm_task_flush());
        CHECK(!ds4_gpu_kv_norm_task_begin_concurrent());
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(q_out, 0, actual, sizeof q_input));
        CHECK(ds4_gpu_tensor_read(kv_out, 0, actual + Q, sizeof kv_input));
        CHECK(ds4_gpu_tensor_read(raw, 0, actual + Q + KV, CAP * sizeof kv_input));
        for (int i = 0; i < KV; i++) {
            CHECK(actual[Q + KV + i] == 123.0f);
            CHECK(actual[Q + 3 * KV + i] == 123.0f);
        }
        if (!mode) memcpy(expected, actual, sizeof expected);
        else CHECK(!memcmp(actual, expected, sizeof expected));
    }
    CHECK(ds4_gpu_tensor_read(independent, 0, actual, sizeof q_input));
    for (int i = 0; i < Q; i++) CHECK(actual[i] == q_input[i] * 2);
    puts("Metal fused q/KV versus deferred flush/concurrent publication: PASS");
    rc = 0;
done:
    ds4_gpu_kv_norm_task_end_concurrent();
    if (ds4_gpu_kv_norm_task_pending()) ds4_gpu_kv_norm_task_flush();
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(independent); ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(kv_out); ds4_gpu_tensor_free(q_out);
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(q);
    ds4_gpu_cleanup();
    if (model != MAP_FAILED) munmap(model, sizeof weights);
    return rc;
}

int main(int argc, char **argv) {
    if (check_resource_boundaries()) return 1;
    if (argc == 2 && !strcmp(argv[1], "parallel-ffn")) return check_parallel_scheduling();
    if (argc == 2 && !strcmp(argv[1], "deferred-kv")) return check_deferred_kv_scheduling();
    if (argc == 2 && (!strcmp(argv[1], "exact-cache-evict") ||
                      !strcmp(argv[1], "exact-cache-unlimited"))) {
        const int evict = !strcmp(argv[1], "exact-cache-evict");
        int ok = ds4_gpu_test_exact_view_lifetimes(evict);
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2) {
        const char *modes[] = {"row", "session", "batch", "big"};
        for (int i = 0; i < 4; i++)
            if (!strcmp(argv[1], modes[i])) return check_tp_memory(i);
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [row|session|batch|big|exact-cache-evict|exact-cache-unlimited|parallel-ffn|deferred-kv]\n", argv[0]);
        return 2;
    }
    ds4_gpu_tensor *a = NULL, *b = NULL;
    int rc = 1;
    CHECK(ds4_gpu_init());
    a = ds4_gpu_tensor_alloc(1024);
    b = ds4_gpu_tensor_alloc(1024);
    CHECK(a && b);
    float input[256], output[256];
    for (int i = 0; i < 256; i++) input[i] = (float)i / 4.0f;
    CHECK(ds4_gpu_tensor_write(a, 0, input, sizeof(input)));
    size_t baseline = 0;
    for (int i = 0; i < 8192; i++) {
        CHECK(ds4_gpu_begin_commands());
        for (int j = 0; j < 8; j++) {
            CHECK(ds4_gpu_add_tensor(b, a, a, 256));
            CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, sizeof(input)));
            if (j == 3 && i % 2) CHECK(ds4_gpu_flush_commands());
        }
        CHECK(ds4_gpu_add_tensor(b, a, a, 256));
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(b, 0, output, sizeof(output)));
        for (int j = 0; j < 256; j++) CHECK(output[j] == 2.0f * input[j]);
        if (i % 1024 == 1023) {
            malloc_statistics_t stats;
            malloc_zone_statistics(NULL, &stats);
            if (!baseline) baseline = stats.size_in_use;
            fprintf(stderr, "Metal copies: %d batches, heap %.3f MiB\n",
                    i + 1, stats.size_in_use / 1048576.0);
            /* Leave room for driver caches, but not per-copy retained objects.
             * Before the fix this grows by about 35 MiB after warmup. */
            CHECK(stats.size_in_use <= baseline + 8u * 1024u * 1024u);
        }
    }
    puts("Metal copy contents, asynchronous flush and bounded command heap: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(b);
    ds4_gpu_cleanup();
    return rc;
}
