/* Model-free residual oracle: first publish the full residual with the old
 * kernels, then independently project/add. This rejects the former block hook. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static float *values(size_t n, unsigned seed) {
    float *out = malloc(n * sizeof(float));
    if (!out) abort();
    for (size_t i = 0; i < n; ++i)
        out[i] = ((int)((i * 1664525u + seed * 1013904223u) % 1023u) - 511) / 1024.f;
    return out;
}

static float *directions(unsigned width) {
    float *out = values((size_t)3 * width, 17);
    for (unsigned layer = 0; layer < 3; ++layer) {
        double norm = 0;
        for (unsigned i = 0; i < width; ++i) norm += (double)out[(size_t)layer * width + i] * out[(size_t)layer * width + i];
        for (unsigned i = 0; i < width; ++i) out[(size_t)layer * width + i] /= (float)sqrt(norm);
    }
    return out;
}

static ds4_gpu_tensor *upload(const float *data, size_t n) {
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n * sizeof(float));
    if (!out || !ds4_gpu_tensor_write(out, 0, data, n * sizeof(float))) abort();
    return out;
}

static int compare(ds4_gpu_tensor *reference, ds4_gpu_tensor *actual, size_t n,
                   float tolerance, bool exact, double *maximum) {
    float *a = malloc(n * 4), *b = malloc(n * 4);
    CHECK(a && b);
    CHECK(ds4_gpu_tensor_read(reference, 0, a, n * 4) && ds4_gpu_tensor_read(actual, 0, b, n * 4));
    size_t different = 0;
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        CHECK(isfinite(a[i]) && isfinite(b[i]));
        double delta = fabs((double)a[i] - b[i]);
        if (delta > worst) worst = delta;
        different += memcmp(a + i, b + i, 4) != 0;
    }
    free(a); free(b);
    if (worst > tolerance || (exact && different)) {
        fprintf(stderr, "steering mismatch: n=%zu different=%zu max=%.9g tolerance=%g exact=%d\n", n, different, worst, tolerance, exact);
        return 1;
    }
    if (worst > *maximum) *maximum = worst;
    return 0;
}

/* Project reference uses the established standalone kernel. Addition is a
 * plain CPU offset, independent of every new fused shader. */
static int reference(ds4_gpu_tensor *x, ds4_gpu_tensor *bank, const float *dirs,
                     unsigned width, unsigned rows, float scale, int mode) {
    if (scale == 0) return 0;
    if (mode == 0) { CHECK(ds4_gpu_directional_steering_project_tensor(x, bank, 2, width, rows, scale)); return 0; }
    CHECK(ds4_gpu_end_commands());
    const size_t n = (size_t)width * rows;
    float *values = malloc(n * sizeof(float));
    CHECK(values && ds4_gpu_tensor_read(x, 0, values, n * sizeof(float)));
    for (size_t i = 0; i < n; ++i) values[i] += scale * dirs[2 * width + i % width];
    CHECK(ds4_gpu_tensor_write(x, 0, values, n * sizeof(float)));
    free(values);
    CHECK(ds4_gpu_begin_commands());
    return 0;
}

static int expand(unsigned width, unsigned rows, bool split, bool bf16,
                  float scale, bool alias, int mode, double *maximum) {
    const size_t n = (size_t)width * rows, hc = n * 4;
    float *block = values(n, 1), *residual = values(hc, 2);
    float *post = values((size_t)rows * 4, 3), *comb = values((size_t)rows * 16, 4);
    float *parts = values((size_t)rows * 24, 5), *dirs = directions(width);
    // Every old residual carries a known component along the bank direction.
    // Steering only the new block cannot remove this component.
    for (size_t i = 0; i < hc; ++i) residual[i] += (4 + (i / width) % 4) * dirs[2 * width + i % width];
    for (unsigned row = 0; row < rows; ++row) {
        memcpy(parts + (size_t)row * 24 + 4, post + (size_t)row * 4, 16);
        memcpy(parts + (size_t)row * 24 + 8, comb + (size_t)row * 16, 64);
    }
    ds4_gpu_tensor *t[] = {
        upload(block, n), upload(block, n), upload(residual, hc), upload(residual, hc),
        upload(residual, hc), upload(residual, hc), upload(post, (size_t)rows * 4),
        upload(comb, (size_t)rows * 16), upload(parts, (size_t)rows * 24), upload(dirs, (size_t)width * 3),
    };
    ds4_gpu_tensor *a = t[alias ? 2 : 4], *b = t[alias ? 3 : 5];
    CHECK(ds4_gpu_begin_commands());
    if (split) CHECK(ds4_gpu_hc_expand_split_tensor(a, t[0], t[2], t[8], width, 4));
    else CHECK(ds4_gpu_hc_expand_tensor(a, t[0], t[2], t[6], t[7], width, 4));
    if (bf16) CHECK(ds4_gpu_dsv41_quantize(a, width * 4, rows, DS4_V41_BF16));
    CHECK(!reference(a, t[9], dirs, width, rows * 4, scale, mode));
    if (bf16) CHECK(ds4_gpu_dsv41_quantize(a, width * 4, rows, DS4_V41_BF16));
    if (split) CHECK(ds4_gpu_hc_expand_split_steering_tensor(b, t[1], t[3], t[8], t[9], 2, width, 4, scale, bf16, mode));
    else CHECK(ds4_gpu_hc_expand_steering_tensor(b, t[1], t[3], t[6], t[7], t[9], 2, width, 4, scale, mode));
    CHECK(ds4_gpu_end_commands());
    CHECK(!compare(t[0], t[1], n, 0.00001f, scale == 0, maximum));
    CHECK(!compare(a, b, hc, bf16 ? 0.00390625f : 0.00001f, scale == 0, maximum));
    for (unsigned i = 0; i < sizeof(t) / sizeof(*t); ++i) ds4_gpu_tensor_free(t[i]);
    free(block); free(residual); free(post); free(comb); free(parts); free(dirs);
    return 0;
}

static int qwen(unsigned rows, float scale, unsigned shared_source, int mode, double *maximum) {
    const unsigned width = 2560, slots = 3, stride = 4;
    const size_t n = (size_t)width * rows, hc = n * 4;
    float *block = values(n, 11), *residual = values(hc, 12), *inj = values((size_t)rows * 128, 13);
    float *partial = values(n * stride, 14), *weights = values((size_t)rows * slots, 15);
    float *gate = values(rows, 16), *shared = values(n, 17), *dirs = directions(width);
    ds4_gpu_tensor *t[] = {
        upload(block, n), upload(block, n), upload(residual, hc), upload(residual, hc),
        upload(inj, (size_t)rows * 128), upload(partial, n * stride), upload(weights, (size_t)rows * slots),
        upload(gate, rows), upload(shared, n), upload(dirs, (size_t)width * 3),
    };
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_qwen4_hc_combine_tensor(t[2], t[0], t[4], rows, width, 4));
    CHECK(!reference(t[2], t[9], dirs, width, rows * 4, scale, mode));
    CHECK(ds4_gpu_qwen4_hc_combine_steering_tensor(t[3], t[1], t[4], t[9], 2, rows, width, 4, scale, mode));
    CHECK(ds4_gpu_end_commands());
    CHECK(!compare(t[0], t[1], n, 0.00001f, scale == 0, maximum));
    CHECK(!compare(t[2], t[3], hc, 0.00001f, scale == 0, maximum));
    CHECK(ds4_gpu_tensor_write(t[2], 0, residual, hc * 4) && ds4_gpu_tensor_write(t[3], 0, residual, hc * 4));
    const ds4_gpu_tensor *g = shared_source ? t[7] : NULL;
    const ds4_gpu_tensor *s = shared_source == 2 ? t[8] : NULL;
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_qwen4_moe_reduce_tensor(t[0], t[5], t[6], g, s, t[2], t[4], rows, slots, stride, width, 4));
    CHECK(!reference(t[2], t[9], dirs, width, rows * 4, scale, mode));
    CHECK(ds4_gpu_qwen4_moe_reduce_steering_tensor(t[1], t[5], t[6], g, s, t[3], t[4], t[9], 2, rows, slots, stride, width, 4, scale, mode));
    CHECK(ds4_gpu_end_commands());
    CHECK(!compare(t[0], t[1], n, 0.00001f, true, maximum));
    CHECK(!compare(t[2], t[3], hc, 0.00001f, scale == 0, maximum));
    for (unsigned i = 0; i < sizeof(t) / sizeof(*t); ++i) ds4_gpu_tensor_free(t[i]);
    free(block); free(residual); free(inj); free(partial); free(weights); free(gate); free(shared); free(dirs);
    return 0;
}

static int residual_case(unsigned rows, float scale, bool with_shared, unsigned alias,
                         const void *model, uint64_t model_bytes, int mode, double *maximum) {
    const unsigned width = 6144;
    const size_t n = (size_t)width * rows;
    float *block = values(n, 21), *residual = values(n, 22), *shared = values(n, 23), *dirs = directions(width);
    ds4_gpu_tensor *t[] = {upload(block, n), upload(block, n), upload(residual, n), upload(residual, n),
        upload(shared, n), upload(dirs, (size_t)width * 3), upload(block, n), upload(block, n)};
    ds4_gpu_tensor *a = alias == 1 ? t[0] : alias == 2 ? t[2] : t[6];
    ds4_gpu_tensor *b = alias == 1 ? t[1] : alias == 2 ? t[3] : t[7];
    CHECK(ds4_gpu_begin_commands());
    if (with_shared) CHECK(ds4_gpu_add_tensor(t[0], t[0], t[4], n));
    CHECK(ds4_gpu_add_tensor(a, t[2], t[0], n));
    CHECK(!reference(a, t[5], dirs, width, rows, scale, mode));
    CHECK(ds4_gpu_residual_steering_tensor(b, t[1], with_shared ? t[4] : NULL, t[3], t[5], 2, width, rows, scale, mode));
    CHECK(ds4_gpu_end_commands());
    CHECK(!compare(a, b, n, 0.00001f, scale == 0, maximum));
    // The normalized path publishes both the residual sum and weighted norm.
    // Compare against the existing scalar add+RMSNorm at every row.
    if (!with_shared && alias == 0) {
        CHECK(ds4_gpu_tensor_write(t[0], 0, block, n * 4) && ds4_gpu_tensor_write(t[1], 0, block, n * 4));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_add_tensor(t[6], t[3], t[0], n));
        CHECK(!reference(t[6], t[5], dirs, width, rows, scale, mode));
        for (unsigned row = 0; row < rows; ++row) {
            const uint64_t off = (uint64_t)row * width * 4, bytes = (uint64_t)width * 4;
            ds4_gpu_tensor *sum = ds4_gpu_tensor_view(t[6], off, bytes), *norm = ds4_gpu_tensor_view(t[2], off, bytes);
            CHECK(sum && norm);
            if (scale == 0) {
                ds4_gpu_tensor *x = ds4_gpu_tensor_view(t[0], off, bytes), *r = ds4_gpu_tensor_view(t[3], off, bytes);
                CHECK(x && r && ds4_gpu_add_rms_norm_weight_tensor(norm, sum, r, x, model, model_bytes, 16, width, 1e-6f));
                ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(r);
            } else CHECK(ds4_gpu_rms_norm_weight_tensor(norm, sum, model, model_bytes, 16, width, 1e-6f));
            ds4_gpu_tensor_free(sum); ds4_gpu_tensor_free(norm);
        }
        CHECK(ds4_gpu_residual_norm_steering_tensor(t[4], t[7], t[1], t[3], t[5], model, model_bytes, 16, 2, width, rows, scale, 1e-6f, mode));
        CHECK(ds4_gpu_end_commands());
        CHECK(!compare(t[6], t[7], n, 0.00001f, scale == 0, maximum));
        CHECK(!compare(t[2], t[4], n, 0.00001f, scale == 0, maximum));
    }
    for (unsigned i = 0; i < sizeof(t) / sizeof(*t); ++i) ds4_gpu_tensor_free(t[i]);
    free(block); free(residual); free(shared); free(dirs);
    return 0;
}

static int transform_case(unsigned rows, float scale, int mode, double *maximum) {
    const unsigned width = 5120;
    const size_t n = (size_t)width * rows;
    float *raw = values(n, 42), *dirs = directions(width);
    ds4_gpu_tensor *a = upload(raw, n), *b = upload(raw, n), *bank = upload(dirs, width * 3);
    CHECK(ds4_gpu_begin_commands());
    CHECK(!reference(a, bank, dirs, width, rows, scale, mode));
    CHECK(ds4_gpu_directional_steering_tensor(b, bank, 2, width, rows, scale, mode));
    CHECK(ds4_gpu_end_commands());
    CHECK(!compare(a, b, n, .00001f, scale == 0, maximum));
    ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(bank);
    free(raw); free(dirs);
    return 0;
}

static double seconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

/* Wall time includes command encoding and its final GPU wait. No public GPU
 * timestamp API exists. Fixed tensors remove expert-selection and SSD effects.
 * Each of the 40 blocks starts identical and is used once per arm, since active
 * steering mutates it; resetting inputs happens outside the timed command. */
static int timings(void) {
    CHECK(ds4_gpu_init());
    const unsigned width = 5120, sublayers = 40, shapes[] = {1, 256};
    for (unsigned shape = 0; shape < 2; ++shape) {
        const unsigned rows = shapes[shape];
        const size_t n = (size_t)width * rows;
        float *raw = values(n, 31), *residual = values(n * 4, 32);
        float *split = values((size_t)rows * 24, 33), *dirs = directions(width);
        ds4_gpu_tensor *blocks[40];
        for (unsigned i = 0; i < sublayers; ++i) blocks[i] = upload(raw, n);
        ds4_gpu_tensor *r = upload(residual, n * 4), *out = upload(residual, n * 4);
        ds4_gpu_tensor *mix = upload(split, (size_t)rows * 24), *bank = upload(dirs, (size_t)width * 3);
        // One warmup bracket, then three individual ABBA brackets.
        for (unsigned repeat = 0; repeat <= 3; ++repeat) for (unsigned arm = 0; arm < 4; ++arm) {
            const bool active = arm == 1 || arm == 2;
            for (unsigned i = 0; i < sublayers; ++i)
                CHECK(ds4_gpu_tensor_write(blocks[i], 0, raw, n * sizeof(float)));
            const double start = seconds();
            CHECK(ds4_gpu_begin_commands());
            for (unsigned i = 0; i < sublayers; ++i) {
                if (active) CHECK(ds4_gpu_hc_expand_split_steering_tensor(out, blocks[i], r, mix, bank, 2, width, 4, .2f, true, 0));
                else {
                    CHECK(ds4_gpu_hc_expand_split_tensor(out, blocks[i], r, mix, width, 4));
                    CHECK(ds4_gpu_dsv41_quantize(out, width * 4, rows, DS4_V41_BF16));
                }
            }
            CHECK(ds4_gpu_end_commands());
            const double elapsed = seconds() - start;
            if (repeat) printf("hc_timing rows=%u repeat=%u arm=%u active=%d sublayers=%u elapsed_ms=%.6f us_per_sublayer=%.6f\n",
                rows, repeat, arm, active, sublayers, elapsed * 1e3, elapsed * 1e6 / sublayers);
        }
        for (unsigned i = 0; i < sublayers; ++i) ds4_gpu_tensor_free(blocks[i]);
        ds4_gpu_tensor_free(r); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(mix); ds4_gpu_tensor_free(bank);
        free(raw); free(residual); free(split); free(dirs);
    }
    ds4_gpu_cleanup();
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--timings")) return timings();
    if (argc != 1) { fprintf(stderr, "usage: %s [--timings]\n", argv[0]); return 2; }
    CHECK(ds4_gpu_init());
    const unsigned widths[] = {3072, 4096, 5120}, rows[] = {1, 3, 17};
    const float scales[] = {-0.7f, 0, 1};
    double maximum = 0;
    unsigned cases = 0;
    for (int mode = 0; mode < 2; ++mode) for (unsigned w = 0; w < 3; ++w) for (unsigned r = 0; r < 3; ++r)
        for (unsigned s = 0; s < 3; ++s) for (unsigned layout = 0; layout < 3; ++layout) {
            CHECK(!expand(widths[w], rows[r], layout != 0, layout == 2, scales[s], cases % 2 != 0, mode, &maximum));
            ++cases;
        }
    for (int mode = 0; mode < 2; ++mode) for (unsigned r = 0; r < 3; ++r) for (unsigned s = 0; s < 3; ++s)
        for (unsigned shared = 0; shared < 3; ++shared) {
            CHECK(!qwen(rows[r], scales[s], shared, mode, &maximum));
            ++cases;
        }
    const size_t page = (size_t)sysconf(_SC_PAGESIZE), model_bytes = ((6144 * 4 + 16 + page - 1) / page) * page;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, model_bytes));
    memset(model, 0, model_bytes);
    float *weight = (float *)((char *)model + 16);
    for (unsigned i = 0; i < 6144; ++i) weight[i] = .5f + (i % 53) / 53.f;
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    for (int mode = 0; mode < 2; ++mode) for (unsigned r = 0; r < 3; ++r) for (unsigned s = 0; s < 3; ++s)
        for (unsigned shared = 0; shared < 2; ++shared) for (unsigned alias = 0; alias < 3; ++alias) {
            CHECK(!residual_case(rows[r], scales[s], shared != 0, alias, model, model_bytes, mode, &maximum));
            ++cases;
        }
    for (int mode = 0; mode < 2; ++mode) for (unsigned r = 0; r < 3; ++r) for (unsigned s = 0; s < 3; ++s) {
        CHECK(!transform_case(rows[r], scales[s], mode, &maximum));
        ++cases;
    }
    ds4_gpu_cleanup();
    free(model);
    printf("steered HC: PASS %u cases, maximum absolute drift %.9g; zero-scale exact\n", cases, maximum);
    return 0;
}
