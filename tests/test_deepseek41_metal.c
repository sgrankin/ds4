#include "ds4_gpu.h"
#ifdef __APPLE__
#include "metal/bridge_types.h"
#include "metal/attention_batch.h"
#include "metal/attention_output.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 7919;
static float random_value(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 65537) - 32768) / 8192.0f;
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, 4);
    return value;
}

static float nearest(float value, int fp4) {
    const float fp4_values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best_value = 0, best_error = INFINITY;
    int best = 0;
    for (int i = 0; i < (fp4 ? 8 : 127); i++) {
        const float v = fp4 ? fp4_values[i] : i < 8 ? ldexpf(i, -9) :
                        ldexpf(1.0f + (i & 7) / 8.0f, (i >> 3) - 7);
        const float error = fabsf(fabsf(value) - v);
        if (error < best_error || (error == best_error && !(i & 1) && (best & 1))) {
            best = i; best_value = v; best_error = error;
        }
    }
    return copysignf(best_value, value);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_managed(bytes);
    if (t && data && !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static int check_router(void) {
    enum { ROWS = 129, MAX_EXPERTS = 384, USED = 6 };
    const size_t bias_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(bias_bytes >= MAX_EXPERTS * sizeof(float));
    CHECK(posix_memalign((void **)&bias, bias_bytes, bias_bytes) == 0);
    memset(bias, 0, bias_bytes);
    float *logits = malloc(ROWS * MAX_EXPERTS * sizeof(float));
    int32_t tokens[ROWS] = {0};
    CHECK(bias && logits);
    for (unsigned e = 0; e < MAX_EXPERTS; e++) bias[e] = (e % 7) * 0.125f;
    CHECK(ds4_gpu_set_model_map(bias, bias_bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *p = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *ids = upload(NULL, ROWS * USED * sizeof(int32_t));
    ds4_gpu_tensor *weights = upload(NULL, ROWS * USED * sizeof(float));
    ds4_gpu_tensor *tok = upload(tokens, sizeof(tokens));
    CHECK(x && p && ids && weights && tok);
    const unsigned counts[] = {1, 3, 17, ROWS};
    for (unsigned n = 128; n <= MAX_EXPERTS; n += 128) {
        const unsigned USED = n == 128 ? 3 : 6;
        for (unsigned mode = 0; mode < 3; mode++) {
            for (unsigned i = 0; i < ROWS * n; i++) logits[i] = mode == 1 ? 0 :
                mode == 2 ? (i % 2 ? -80 : 40) : random_value() * 8;
            CHECK(ds4_gpu_tensor_write(x, 0, logits, ROWS * n * sizeof(float)));
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                CHECK(ds4_gpu_router_select_batch_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, 0, true, false,
                    x, tok, n, USED, 1.5f, rows));
                CHECK(ds4_gpu_synchronize());
                const int32_t *selected = ds4_gpu_tensor_contents(ids);
                const float *actual = ds4_gpu_tensor_contents(weights);
                const float *probs = ds4_gpu_tensor_contents(p);
                CHECK(selected && actual && probs);
                for (unsigned t = 0; t < rows; t++) {
                    double ref[MAX_EXPERTS];
                    int best[USED];
                    for (unsigned k = 0; k < USED; k++) best[k] = -1;
                    for (unsigned e = 0; e < n; e++) {
                        const double v = logits[t * n + e];
                        ref[e] = sqrt(v > 20 ? v : v < -20 ? exp(v) : log1p(exp(v)));
                        if (!(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e])))
                            fprintf(stderr, "router n=%u mode=%u rows=%u row=%u expert=%u logit=%.9g actual=%.9g ref=%.9g\n",
                                    n, mode, rows, t, e, v, probs[t * n + e], ref[e]);
                        CHECK(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e]));
                        const double score = ref[e] + bias[e];
                        for (unsigned k = 0; k < USED; k++) {
                            if (best[k] < 0 || score > ref[best[k]] + bias[best[k]]) {
                                for (unsigned j = USED - 1; j > k; j--) best[j] = best[j - 1];
                                best[k] = (int)e;
                                break;
                            }
                        }
                    }
                    double sum = 0;
                    for (unsigned k = 0; k < USED; k++) {
                        const int id = selected[t * USED + k];
                        CHECK(id >= 0 && id < (int)n);
                        for (unsigned j = 0; j < k; j++) CHECK(id != selected[t * USED + j]);
#ifdef __APPLE__
                        /* Bitonic selection does not promise CUDA's lower-ID
                         * tie break. Still require the correct top-k scores. */
                        CHECK(fabs((ref[id] + bias[id]) - (ref[best[k]] + bias[best[k]])) < 3e-6);
#else
                        CHECK(id == best[k]);
#endif
                        sum += ref[id];
                    }
                    for (unsigned k = 0; k < USED; k++) {
                        CHECK(fabs(actual[t * USED + k] - 1.5 * ref[selected[t * USED + k]] /
                              fmax(sum, 0x1p-14)) < 3e-6);
                    }
                }
                int32_t first_ids[USED]; float first_weights[USED];
                memcpy(first_ids, selected, sizeof(first_ids));
                memcpy(first_weights, actual, sizeof(first_weights));
                CHECK(ds4_gpu_router_select_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, n, USED, 1.5f,
                    0, 0, true, false, x));
                CHECK(ds4_gpu_synchronize());
                const int32_t *scalar_ids = ds4_gpu_tensor_contents(ids);
                const float *scalar_weights = ds4_gpu_tensor_contents(weights);
#ifdef __APPLE__
                for (unsigned k = 0; k < USED; k++) {
                    CHECK(scalar_ids[k] >= 0 && scalar_ids[k] < (int32_t)n);
                    CHECK(probs[first_ids[k]] + bias[first_ids[k]] ==
                          probs[scalar_ids[k]] + bias[scalar_ids[k]]);
                    CHECK(fabsf(first_weights[k] - scalar_weights[k]) < 3e-6f);
                }
#else
                CHECK(!memcmp(first_ids, scalar_ids, sizeof(first_ids)));
                CHECK(!memcmp(first_weights, scalar_weights, sizeof(first_weights)));
#endif
            }
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(tok);
    ds4_gpu_cleanup();
    free(bias); free(logits);
    CHECK(ds4_gpu_init());
    fprintf(stderr, "256/384-expert routing, ties/extremes and scalar/batch oracle: PASS\n");
    return 1;
}

#ifdef __APPLE__
static int check_resident_router(void) {
    enum { N = 384, K = 6 };
    const size_t model_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(posix_memalign((void **)&bias, model_bytes, model_bytes) == 0);
    memset(bias, 0, model_bytes);
    float logits[N];
    for (unsigned e = 0; e < N; ++e) {
        logits[e] = (float)((e * 37) % 193) / 16.0f - 6.0f;
        bias[e] = (float)(e % 7) * 0.125f;
    }
    CHECK(ds4_gpu_set_model_map(bias, model_bytes));
    ds4_gpu_tensor *x = upload(logits, sizeof(logits));
    ds4_gpu_tensor *p = upload(NULL, sizeof(logits));
    ds4_gpu_tensor *ids = upload(NULL, K * sizeof(int32_t));
    ds4_gpu_tensor *weights = upload(NULL, K * sizeof(float));
    ds4_gpu_tensor *trace = upload(NULL, 32);
    CHECK(x && p && ids && weights && trace);
    for (unsigned n = 256; n <= N; n += 128) {
        CHECK(ds4_gpu_dsv41_router_one(ids, weights, p, x, bias, model_bytes, 0, n, K, 1.5f));
        CHECK(ds4_gpu_synchronize());
        int32_t native[K]; float native_weights[K], native_probs[N];
        CHECK(ds4_gpu_tensor_read(ids, 0, native, sizeof(native)));
        CHECK(ds4_gpu_tensor_read(weights, 0, native_weights, sizeof(native_weights)));
        CHECK(ds4_gpu_tensor_read(p, 0, native_probs, n * sizeof(float)));
        for (unsigned mode = 0; mode < 3; ++mode) {
            uint8_t resident[N] = {0};
            if (mode == 0) memset(resident, 1, n);
            else {
                unsigned count = 0;
                if (mode == 1) for (unsigned k = 0; k < K; k += 2) {
                    resident[native[k]] = 1;
                    ++count;
                }
                for (unsigned e = 0; e < n && count < K; ++e) {
                    int original = 0;
                    for (unsigned k = 0; k < K; ++k) original |= native[k] == (int32_t)e;
                    if (!original) { resident[e] = 1; ++count; }
                }
                CHECK(count == K);
            }
            CHECK(ds4_gpu_tensor_write(ids, 0, native, sizeof(native)));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_resident_route_one(ids, p, bias, model_bytes, 0, n, K, resident, trace));
            CHECK(ds4_gpu_end_commands());
            int32_t actual[K]; float actual_weights[K], actual_probs[N]; uint32_t details[8];
            CHECK(ds4_gpu_tensor_read(ids, 0, actual, sizeof(actual)));
            CHECK(ds4_gpu_tensor_read(weights, 0, actual_weights, sizeof(actual_weights)));
            CHECK(ds4_gpu_tensor_read(p, 0, actual_probs, n * sizeof(float)));
            CHECK(ds4_gpu_tensor_read(trace, 0, details, sizeof(details)));
            CHECK(!memcmp(native_weights, actual_weights, sizeof(native_weights)));
            CHECK(!memcmp(native_probs, actual_probs, n * sizeof(float)));
            CHECK(!memcmp(native, details, sizeof(native)));
            unsigned changes = 0; float total = 0, changed = 0;
            for (unsigned k = 0; k < K; ++k) {
                CHECK(isfinite(actual_weights[k]));
                CHECK(actual[k] >= 0 && actual[k] < (int32_t)n && resident[actual[k]]);
                for (unsigned j = 0; j < k; ++j) CHECK(actual[k] != actual[j]);
                total += native_probs[native[k]];
                if (resident[native[k]]) CHECK(actual[k] == native[k]);
                else { ++changes; changed += native_probs[native[k]]; }
            }
            float traced_mass; memcpy(&traced_mass, &details[7], sizeof(traced_mass));
            CHECK(details[6] == changes);
            CHECK(fabsf(traced_mass - changed / fmaxf(total, 0x1p-14f)) < 1e-6f);
            if (mode == 0) CHECK(!memcmp(native, actual, sizeof(native)));
            if (mode == 1) CHECK(changes == 3);
            if (mode == 2) CHECK(changes == 6);
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(trace);
    ds4_gpu_cleanup(); free(bias);
    CHECK(ds4_gpu_init());
    fprintf(stderr, "Resident router: all-resident identity, partial/all replacements, unique IDs and unchanged finite weights PASS\n");
    return 1;
}
#endif

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++) source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_read(t, 0, actual, N * sizeof(float)));
        for (int start = 0; start < N; start += block) {
            float amax = 0, scale = 1;
            for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP8_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f))));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            if (mode == DS4_V41_FP4_E4M3) scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                float expected = bf16(source[start + i]);
                if (mode) expected = bf16(nearest(expected / scale, mode != 1) * scale);
                if (memcmp(&expected, actual + start + i, 4)) {
                    fprintf(stderr, "quantization mode=%d index=%d: %.9g != %.9g\n",
                            mode, start + i, actual[start + i], expected);
                    return 0;
                }
            }
        }
    }
    CHECK(ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(t, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 32, 1, (ds4_v41_activation_format)4));
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 BF16/FP8/FP4 round trips: exact\n");
    return 1;
}

static double monotonic_seconds(void);

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint32_t bf16_input(size_t i) {
    const uint32_t low[] = {0, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
    if (i < 65536u * 5u) return ((uint32_t)(i / 5u) << 16u) | low[i % 5u];
    return (uint32_t)i * 2654435761u;
}

static int check_bf16_linear(void) {
    const struct { uint32_t width, rows, offset; } shapes[] = {
        {1, 1, 0}, {3, 1, 0}, {5, 1, 0}, {24, 1, 0}, {31, 3, 0},
        {32, 7, 1}, {129, 3, 4}, {1023, 1, 0}, {1024, 1, 0}, {1025, 1, 0},
        {65536u * 5u, 1, 0}, {5120, 1, 0}, {20480, 512, 0},
        {32768, 4096, 0}, {32768, 8192, 0}
    };
    const unsigned modes[] = {0, 1, 1, 0};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const size_t count = (size_t)shapes[s].width * shapes[s].rows;
        const size_t offset = shapes[s].offset;
        ds4_gpu_tensor *storage = upload(NULL, (count + 8) * sizeof(uint32_t));
        CHECK(storage);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(storage,
            offset * sizeof(uint32_t), count * sizeof(uint32_t));
        CHECK(view);
        uint32_t *bits = ds4_gpu_tensor_contents(storage);
        CHECK(bits);
        for (unsigned pass = 0; pass < 4; pass++) {
            const unsigned mode = modes[pass];
            if (mode) unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
            else setenv("DS4_METAL_DISABLE_V41_LINEAR_BF16", "1", 1);
            for (size_t i = 0; i < count + 8; i++) bits[i] = 0x12345678;
            for (size_t i = 0; i < count; i++) bits[offset + i] = bf16_input(i);
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 5; repeat++) {
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_dsv41_quantize(view, shapes[s].width, shapes[s].rows, DS4_V41_BF16));
                CHECK(ds4_gpu_synchronize());
                if (repeat) elapsed += (monotonic_seconds() - begin) * 250;
            }
            for (size_t i = 0; i < count; i++) CHECK(bits[offset + i] == bf16_bits(bf16_input(i)));
            for (size_t i = 0; i < offset; i++) CHECK(bits[i] == 0x12345678);
            for (size_t i = offset + count; i < count + 8; i++) CHECK(bits[i] == 0x12345678);
            fprintf(stderr, "BF16 width=%u rows=%u offset=%zu mode=%u pass=%u %.3f ms: exact\n",
                shapes[s].width, shapes[s].rows, offset, mode, pass, elapsed);
        }
        ds4_gpu_tensor_free(view);
        ds4_gpu_tensor_free(storage);
    }
    unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
    return 1;
}

#ifdef __APPLE__
static int check_hc_scaled(void) {
    enum { WIDTH = 20480, OUT = 24, ROWS = 8192 };
    const size_t weight_bytes = WIDTH * OUT * sizeof(_Float16);
    const size_t page = (size_t)getpagesize();
    const size_t mapped_bytes = (weight_bytes + page - 1) / page * page;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, mapped_bytes));
    _Float16 *weights = model;
    for (size_t i = 0; i < WIDTH * OUT; i++) weights[i] = (_Float16)(random_value() / 64);
    CHECK(ds4_gpu_set_model_map(model, mapped_bytes));
    ds4_gpu_tensor *x = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *norm = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *scales = upload(NULL, (ROWS + 1) * 4);
    ds4_gpu_tensor *out = upload(NULL, (OUT * ROWS + 1) * 4);
    ds4_gpu_tensor *ref = upload(NULL, (OUT * ROWS + 1) * 4);
    CHECK(x && norm && scales && out && ref);
    float *input = ds4_gpu_tensor_contents(x), *y = ds4_gpu_tensor_contents(out);
    float *expected = ds4_gpu_tensor_contents(ref), *s = ds4_gpu_tensor_contents(scales);
    for (size_t i = 0; i < (size_t)WIDTH * ROWS; i++)
        input[i] = bf16(random_value() * ((i / WIDTH) % 7 ? 1 : 0x1p-16f));
    const unsigned rows[] = {9, 31, 32, 33, 127, 512, 2048, 4096, 8192};
    for (size_t shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        const uint32_t n = rows[shape];
        double elapsed[2] = {0};
        for (unsigned repeat = 0; repeat < 7; repeat++) {
            y[n * OUT] = expected[n * OUT] = s[n] = 12345;
            for (unsigned j = 0; j < 2; j++) {
                const unsigned mode = j ^ (repeat & 1);
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                if (mode) {
                    CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(out, n >= 512u ? scales : norm,
                        model, mapped_bytes, 0, WIDTH, OUT, x, n, 1e-6f));
                } else {
                    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(norm, x, WIDTH, n, 1e-6f));
                    CHECK(ds4_gpu_matmul_f16_tensor(ref, model, mapped_bytes, 0,
                        WIDTH, OUT, norm, n));
                }
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - begin) * (1000.0 / 6);
            }
            CHECK(!memcmp(y, expected, n * OUT * 4));
            CHECK(y[n * OUT] == 12345 && expected[n * OUT] == 12345 && s[n] == 12345);
        }
        fprintf(stderr, "HC scaled rows=%u: exact %.3f -> %.3f ms\n", n, elapsed[0], elapsed[1]);
    }
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); free(model);
    return 1;
}

#endif

static int check_engram(void) {
    enum { D = 5120, ROWS = 5, N = ROWS * 4 * D };
    float *x = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    float *kv = malloc(ROWS * 5 * D * sizeof(float));
    float *qw = malloc(4 * D * sizeof(float)), *kw = malloc(4 * D * sizeof(float));
    uint8_t mask[] = {1, 1, 0, 1, 0};
    CHECK(x && actual && kv && qw && kw);
    for (int i = 0; i < N; i++) x[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 5 * D; i++) kv[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) { qw[i] = random_value(); kw[i] = random_value(); }
    memset(x, 0, D * sizeof(float));
    memset(kv + D, 0, D * sizeof(float));
    ds4_gpu_tensor *xt = upload(x, N * sizeof(float));
    ds4_gpu_tensor *kt = upload(kv, ROWS * 5 * D * sizeof(float));
    ds4_gpu_tensor *qwt = upload(qw, 4 * D * sizeof(float));
    ds4_gpu_tensor *kwt = upload(kw, 4 * D * sizeof(float));
    ds4_gpu_tensor *mt = upload(mask, sizeof(mask));
    CHECK(xt && kt && qwt && kwt && mt);
    size_t rounded_differently = 0;
    double error2 = 0, norm2 = 0;
    for (int masked = 0; masked < 2; masked++) {
        CHECK(ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, masked ? mt : NULL, D, ROWS, 1e-20f));
        CHECK(ds4_gpu_tensor_read(xt, 0, actual, N * sizeof(float)));
        for (int row = 0; row < ROWS; row++) for (int h = 0; h < 4; h++) {
            double dot = 0, h2 = 0, k2 = 0;
            for (int i = 0; i < D; i++) {
                const double a = x[(row * 4 + h) * D + i], b = kv[(row * 5 + h) * D + i];
                h2 += a * a; k2 += b * b;
                dot += a * (float)(qw[h * D + i] * kw[h * D + i]) * b;
            }
            dot /= sqrt(h2 / D + 1e-20) * sqrt(k2 / D + 1e-20) * sqrt(D);
            double gate = 1 / (1 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
            if (masked && !mask[row]) gate = 0;
            for (int i = 0; i < D; i++) {
                const int off = (row * 4 + h) * D + i;
                const float expected = bf16(x[off] + (float)gate * kv[(row * 5 + 4) * D + i]);
                const double error = actual[off] - expected;
                CHECK(isfinite(actual[off]));
                CHECK(fabs(error) <= fmax(1e-6, fabs(expected) / 128));
                if (masked && !mask[row]) CHECK(actual[off] == x[off]);
                rounded_differently += actual[off] != expected;
                error2 += error * error; norm2 += (double)expected * expected;
            }
        }
    }
    CHECK(rounded_differently < N / 1000);
    CHECK(sqrt(error2 / norm2) < 1e-4);
    fprintf(stderr, "V4.1 Engram gate: %zu BF16 boundary differences, relative RMS %.8g\n",
            rounded_differently, sqrt(error2 / norm2));
#ifdef __APPLE__
    {   /* the decode step's gate over 8 SIMD groups (unmasked, width 5120, <= 16 rows) gives the 32-lane bits */
        float *spread = malloc(N * sizeof(float));
        CHECK(spread);
        ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
        for (int off = 0; off < 2; off++) {
            const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_engram_add_tg256");
            ds4_gpu_test_set_v41_off(off ? DS4_GPU_V41_SPLIT_REDUCE : 0);
            const int ok = ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)) &&
                ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, NULL, D, ROWS, 1e-20f);
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok && ds4_gpu_tensor_read(xt, 0, off ? actual : spread, N * sizeof(float)));
            CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_engram_add_tg256") == runs + !off);
        }
        CHECK(!memcmp(actual, spread, N * sizeof(float)));
        free(spread);
        fprintf(stderr, "V4.1 Engram gate over 8 SIMD groups == 32 lanes, bitwise PASS\n");
    }
#endif
    CHECK(!ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, mt, D, ROWS + 1, 1e-20f));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qwt);
    ds4_gpu_tensor_free(kwt); ds4_gpu_tensor_free(mt);
    free(x); free(actual); free(kv); free(qw); free(kw);
    return 1;
}

static int check_rope_stride(void) {
    enum { WIDTH = 512, HEADS = 2, ROWS = 129, N = WIDTH * HEADS * ROWS };
    ds4_gpu_tensor *a = upload(NULL, (N + 1u) * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, (N + 1u) * sizeof(float));
    CHECK(a && b);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    const uint32_t starts[] = {0, 126, 32766, 1048318};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (size_t j = 0; j < N; j++) x[j] = y[j] = bf16(random_value());
            x[N] = y[N] = 12345;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, starts[i], 2, kind, inverse));
            for (uint32_t row = 0; row < ROWS; row++) {
                ds4_gpu_tensor *v = ds4_gpu_tensor_view(b, (uint64_t)row * WIDTH * HEADS * 4u,
                                                       WIDTH * HEADS * 4u);
                CHECK(v && ds4_gpu_dsv41_rope(v, WIDTH, HEADS, 1, starts[i] + row * 2u, kind, inverse));
                ds4_gpu_tensor_free(v);
            }
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(x, y, (N + 1u) * sizeof(float)) && x[N] == 12345);
        }
    }
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, 0, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 1048320, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS + 1, 0, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, UINT32_MAX, true, false));
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 strided RoPE: exact row parity through 1M positions, guards PASS\n");
    return 1;
}

#ifdef __APPLE__
/* The fused RoPE, quantize and store kernel against the three separate steps. */
static int check_rope_quantize(void) {
    enum { WIDTH = 512, TRIALS = 300 };
    ds4_gpu_tensor *a = upload(NULL, WIDTH * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, WIDTH * sizeof(float));
    ds4_gpu_tensor *c = upload(NULL, (WIDTH + 1u) * sizeof(float));
    CHECK(a && b && c);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    float *z = ds4_gpu_tensor_contents(c);
    for (uint32_t t = 0; t < TRIALS; t++) {
        const ds4_v41_activation_format format = (ds4_v41_activation_format)(1 + t % 3);
        const uint32_t width = format == DS4_V41_FP4_E8M0 ? 128 : WIDTH;
        const bool compressed = (t / 3) & 1;
        const uint32_t pos = (uint32_t)(fabsf(random_value()) * 15000);
        for (uint32_t i = 0; i < width; i++) x[i] = y[i] = ldexpf(random_value(), (int)(t % 11) - 5);
        z[width] = 12345;
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope(a, width, 1, 1, pos, compressed, false));
        CHECK(ds4_gpu_dsv41_quantize(a, width, 1, format));
        CHECK(ds4_gpu_dsv41_rope_quantize(b, c, 0, width, pos, compressed, format));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(x, z, width * sizeof(float)) && z[width] == 12345);
    }
    ds4_gpu_tensor_free(c); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 fused RoPE quantize: bitwise match with the separate steps PASS\n");
    return 1;
}
#else
static int check_rope_quantize(void) { return 1; }
#endif

#ifdef __APPLE__
/* Single-Mac decode's deferred hyper-connection weights against the original order, bitwise, on random weights and
 * fresh inputs per draw. Attention: the collapse, then the weights, query and KV row in one dispatch, against the
 * block input, the roped query and the KV row's rope and store. FFN: the collapse, then the block input into scratch,
 * against the block input. With DS4_GPU_V41_DEFERRED_HC off the collapse declines. */
static int check_deferred_hc(void) {
    const uint64_t qo = 0, qb = 32768ull * 40 * 34, ho = qo + qb, hb = 24ull * 20480 * 2;
    const uint64_t so = ho + hb, bo = so + 64, no = bo + 128, wb = no + 5120 * 4;
    char *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), wb) == 0);
    for (uint64_t i = 0; i < qb / 34; i++) {
        uint8_t *block = (uint8_t *)model + qo + i * 34;
        const uint16_t d = (uint16_t)(0x1c00 + (int)(fabsf(random_value()) * 128));
        memcpy(block, &d, 2);
        for (int j = 0; j < 32; j++) block[2 + j] = (uint8_t)(int8_t)(random_value() * 31);
    }
    uint16_t *fn = (uint16_t *)(model + ho);
    for (uint64_t i = 0; i < hb / 2; i++)
        fn[i] = (uint16_t)((random_value() < 0 ? 0x8000 : 0) | (0x2000 + (int)(fabsf(random_value()) * 1536)));
    float *scale = (float *)(model + so), *base = (float *)(model + bo), *nw = (float *)(model + no);
    scale[0] = 0.04f; scale[1] = 0.03f; scale[2] = 0.02f;
    for (int i = 0; i < 24; i++) base[i] = random_value() / 32;
    for (int i = 0; i < 5120; i++) nw[i] = 1.0f + random_value() / 32;
    CHECK(ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *res = upload(NULL, 20480 * 4), *pre = upload(NULL, 16), *qr = upload(NULL, 1280 * 4);
    ds4_gpu_tensor *kv = upload(NULL, 512 * 4), *scratch = upload(NULL, 5120 * 4);
    enum { X, NORM, MIX, SPLIT, Q, KV_ROW, OUTS };
    const uint32_t widths[OUTS] = {5120, 5120, 24, 24, 32768, 512};
    ds4_gpu_tensor *out[2][OUTS];
    CHECK(res && pre && qr && kv && scratch);
    for (int a = 0; a < 2; a++)
        for (int i = 0; i < OUTS; i++) CHECK((out[a][i] = upload(NULL, widths[i] * 4)) != NULL);
    float *r = ds4_gpu_tensor_contents(res), *p = ds4_gpu_tensor_contents(pre);
    float *x = ds4_gpu_tensor_contents(qr), *k = ds4_gpu_tensor_contents(kv);
    const float eps = 1e-6f;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (uint32_t draw = 0; draw < 6; draw++) {
        const int ffn = draw & 1;
        const uint32_t pos = draw * 104729u % 1048576u;
        const bool compressed = (draw >> 1) & 1;
        for (int i = 0; i < 20480; i++) r[i] = random_value() * (draw % 3 == 2 ? 16.0f : 1.0f);
        for (int i = 0; i < 4; i++) p[i] = random_value() / 4;
        for (int i = 0; i < 1280; i++) x[i] = random_value();
        for (int i = 0; i < 512; i++) k[i] = random_value();
        for (int a = 0; a < 2; a++)
            for (int i = 0; i < OUTS; i++) memset(ds4_gpu_tensor_contents(out[a][i]), 0xff, widths[i] * 4);
        ds4_gpu_tensor **o = out[0], **d = out[1];
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_hc_block_input_rows(o[MIX], o[X], o[NORM], o[SPLIT], res, pre, model, wb, ho, so, bo, no,
                                                20480, 24, 5120, 4, 20, eps, eps, 1, 4));
        if (!ffn) {
            CHECK(ds4_gpu_dsv41_project_q(o[Q], model, wb, qo, 1280, 32768, qr, pos, compressed));
            CHECK(ds4_gpu_dsv41_rope_quantize(kv, o[KV_ROW], 0, 512, pos, compressed, DS4_V41_FP8_E8M0));
        }
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        const uint64_t collapses = ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse");
        const uint64_t queries = ds4_gpu_test_kernel_runs("kernel_dsv41_query_deferred_hc");
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_deferred_collapse(d[X], d[NORM], res, pre, model, wb, no, eps));
        if (ffn) {
            CHECK(ds4_gpu_dsv41_hc_block_input_rows(d[MIX], d[X], scratch, d[SPLIT], res, pre, model, wb, ho, so, bo,
                                                    no, 20480, 24, 5120, 4, 20, eps, eps, 1, 4));
        } else {
            CHECK(ds4_gpu_dsv41_query_deferred_hc(d[Q], d[MIX], d[SPLIT], qr, res, model, wb, qo, ho, so, bo, pos,
                                                  compressed, 20, eps, eps, d[X], d[NORM], pre, no, kv, d[KV_ROW], 0));
        }
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse") == collapses + 1);
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_query_deferred_hc") == queries + !ffn);
        for (int i = 0; i < (ffn ? Q : OUTS); i++)
            CHECK(!memcmp(ds4_gpu_tensor_contents(o[i]), ds4_gpu_tensor_contents(d[i]), widths[i] * 4));
        if (ffn) CHECK(!memcmp(ds4_gpu_tensor_contents(o[NORM]), ds4_gpu_tensor_contents(scratch), 5120 * 4));
    }
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_DEFERRED_HC);
    const int declined = !ds4_gpu_dsv41_deferred_collapse(out[1][X], out[1][NORM], res, pre, model, wb, no, eps);
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    CHECK(declined);
    for (int a = 0; a < 2; a++)
        for (int i = 0; i < OUTS; i++) ds4_gpu_tensor_free(out[a][i]);
    ds4_gpu_tensor_free(scratch); ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(qr);
    ds4_gpu_tensor_free(pre); ds4_gpu_tensor_free(res);
    free(model);
    fprintf(stderr, "V4.1 deferred HC weights (attention and FFN) == original order, bitwise PASS\n");
    return 1;
}
#else
static int check_deferred_hc(void) { return 1; }
#endif

static int check_pool(void) {
    enum { D = 512, ROWS = 257, PAIRS = ROWS / 2 };
    float *kv = malloc(ROWS * D * sizeof(float)), *scores = malloc(ROWS * D * sizeof(float));
    float *got = malloc(PAIRS * D * sizeof(float)), *reference = malloc(PAIRS * D * sizeof(float));
    CHECK(kv && scores && got && reference);
    for (int i = 0; i < ROWS * D; i++) { kv[i] = random_value(); scores[i] = random_value() * 25; }
    ds4_gpu_tensor *kt = upload(kv, ROWS * D * sizeof(float));
    ds4_gpu_tensor *st = upload(scores, ROWS * D * sizeof(float));
    ds4_gpu_tensor *pk = upload(NULL, D * sizeof(float)), *ps = upload(NULL, D * sizeof(float));
    ds4_gpu_tensor *out = upload(NULL, PAIRS * D * sizeof(float));
    CHECK(kt && st && pk && ps && out);
    const uint32_t chunks[] = {257, 1, 2, 3, 17, 127, 128, 129};
    for (size_t c = 0; c < sizeof(chunks) / sizeof(*chunks); c++) {
        CHECK(ds4_gpu_tensor_fill_f32(pk, NAN, D));
        CHECK(ds4_gpu_tensor_fill_f32(ps, NAN, D));
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t start = 0; start < ROWS;) {
            uint32_t n = chunks[c] < ROWS - start ? chunks[c] : ROWS - start;
            uint32_t pairs = (n + (start & 1u)) / 2;
            ds4_gpu_tensor *k = ds4_gpu_tensor_view(kt, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *s = ds4_gpu_tensor_view(st, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *o = pairs ? ds4_gpu_tensor_view(out, (uint64_t)(start / 2) * D * 4,
                                                         (uint64_t)pairs * D * 4) : NULL;
            CHECK(k && s && (!pairs || o));
            CHECK(ds4_gpu_dsv41_pool2(o, k, s, pk, ps, D, n, start));
            ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(o);
            start += n;
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(out, 0, got, PAIRS * D * sizeof(float)));
        if (!c) {
            memcpy(reference, got, PAIRS * D * sizeof(float));
            for (int p = 0; p < PAIRS; p++) for (int i = 0; i < D; i++) {
                const int a = 2 * p * D + i, b = a + D;
                const double gate = 1 / (1 + exp((double)scores[b] - scores[a]));
                const float expected = bf16((float)(kv[a] * gate + kv[b] * (1 - gate)));
                CHECK(isfinite(got[p * D + i]));
                CHECK(fabsf(got[p * D + i] - expected) <= fmaxf(1e-6f, fabsf(expected) / 128));
            }
        } else CHECK(!memcmp(reference, got, PAIRS * D * sizeof(float)));
        float tail[D];
        CHECK(ds4_gpu_tensor_read(pk, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, kv + (ROWS - 1) * D, sizeof(tail)));
        CHECK(ds4_gpu_tensor_read(ps, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, scores + (ROWS - 1) * D, sizeof(tail)));
    }
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_pool2(out, kt, st, pk, ps, D, ROWS - 1, 0));
    CHECK(ds4_gpu_end_commands());
    CHECK(!memcmp(ds4_gpu_tensor_contents(pk), kv + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(ps), scores + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(out), reference, PAIRS * D * sizeof(float)));
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(ps); ds4_gpu_tensor_free(out);
    free(kv); free(scores); free(got); free(reference);
    fprintf(stderr, "V4.1 pair pooling: exact across eight chunk sizes\n");
    return 1;
}

typedef struct { float score; uint32_t index; } candidate;
static int candidate_desc(const void *a, const void *b) {
    const candidate *x = a, *y = b;
    return x->score > y->score ? -1 : x->score < y->score ? 1 :
           x->index < y->index ? -1 : x->index > y->index;
}

static int check_candidates(void) {
    const uint32_t widths[] = {1, 7, 8, 9, 127, 16385, 17017};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, rows = 17;
        const uint32_t top = blocks < 2048 ? blocks : 2048;
        float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
        float *maxima = malloc((size_t)blocks * rows * 4);
        candidate *sorted = malloc(blocks * sizeof(candidate));
        uint8_t *kept = malloc(blocks);
        CHECK(scores && got && maxima && sorted && kept);
        /* Unique finite values avoid unspecified top-k tie ordering. */
        for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
            scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
        ds4_gpu_tensor *s = upload(NULL, (size_t)n * rows * 4);
        ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
        ds4_gpu_tensor *t = upload(NULL, (size_t)top * rows * 4);
        ds4_gpu_tensor *m = upload(NULL, (size_t)blocks * rows * 4);
        CHECK(s && b && t && m);
        for (uint32_t ratio = 1; ratio <= 2; ratio++) for (int late = 0; late < 2; late++) {
            const uint32_t start = late ? n * ratio - 1 : 0;
            CHECK(ds4_gpu_tensor_write(s, 0, scores, (size_t)n * rows * 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, ratio));
            CHECK(ds4_gpu_indexer_topk_tensor(t, b, blocks, rows, top));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(m, t, blocks, rows, top));
            CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, ratio, 0));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(b, 0, maxima, (size_t)blocks * rows * 4));
            CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
            for (uint32_t r = 0; r < rows; r++) {
                uint32_t visible = (start + r + 1) / ratio;
                if (visible > n) visible = n;
                for (uint32_t j = 0; j < blocks; j++) {
                    float best = -INFINITY;
                    for (uint32_t i = j * 8; i < (j + 1) * 8 && i < visible; i++)
                        best = fmaxf(best, scores[(size_t)r * n + i]);
                    if (visible && j == (visible - 1) / 8) best = INFINITY;
                    CHECK(maxima[(size_t)r * blocks + j] == best);
                    sorted[j] = (candidate){best, j};
                }
                qsort(sorted, blocks, sizeof(*sorted), candidate_desc);
                memset(kept, 0, blocks);
                for (uint32_t j = 0; j < top; j++)
                    if (sorted[j].score > -INFINITY) kept[sorted[j].index] = 1;
                for (uint32_t i = 0; i < n; i++) {
                    float expected = i < visible && kept[i / 8] ? scores[(size_t)r * n + i] : -INFINITY;
                    CHECK(got[(size_t)r * n + i] == expected);
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, UINT32_MAX, 1));
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, 0, 0));
        CHECK(!ds4_gpu_dsv41_candidate_filter(s, m, n, rows + 1, 0, 1, 0));
        CHECK(blocks == 1 || !ds4_gpu_dsv41_candidate_filter(s, m, n, rows, 0, 1, blocks - 1));
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(t); ds4_gpu_tensor_free(m);
        free(scores); free(got); free(maxima); free(sorted); free(kept);
    }
    fprintf(stderr, "V4.1 causal candidate blocks and filtering: exact\n");
    return 1;
}

/* Batched block top-k, strided masks and filtering against the per-row calls. */
static int check_candidates_batch(uint32_t n, uint32_t rows, uint32_t start) {
    const uint32_t blocks = (n + 7) / 8, stride = blocks + 3;
    float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
    float *mask = malloc((size_t)stride * rows * 4);
    CHECK(scores && got && mask);
    for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
        scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
    for (size_t i = 0; i < (size_t)stride * rows; i++) mask[i] = 7.0f;
    ds4_gpu_tensor *s = upload(scores, (size_t)n * rows * 4);
    ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
    ds4_gpu_tensor *t = upload(NULL, (size_t)2048 * rows * 4);
    ds4_gpu_tensor *ref = upload(NULL, (size_t)2048 * rows * 4);
    ds4_gpu_tensor *m = upload(mask, (size_t)stride * rows * 4);
    CHECK(s && b && t && ref && m);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, 1));
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t visible = (start + r + 8) / 8;
        if (visible > blocks) visible = blocks;
        ds4_gpu_tensor *br = ds4_gpu_tensor_view(b, (size_t)r * blocks * 4, (size_t)visible * 4);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(ref, (size_t)r * 2048 * 4, 2048 * 4);
        CHECK(br && out && ds4_gpu_indexer_topk_tensor(out, br, visible, 1, 2048));
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(br);
    }
    CHECK(ds4_gpu_dsv41_candidate_topk_batch(t, b, blocks, rows, start));
    CHECK(ds4_gpu_dsv41_candidate_mask_batch(m, t, blocks, rows, stride));
    CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, 1, stride));
    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    const int32_t *ids = ds4_gpu_tensor_contents(t), *expected = ds4_gpu_tensor_contents(ref);
    CHECK(ids && expected && !memcmp(ids, expected, (size_t)2048 * rows * 4));
    CHECK(ds4_gpu_tensor_read(m, 0, mask, (size_t)stride * rows * 4));
    CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t visible = start + r + 1;
        if (visible > n) visible = n;
        for (uint32_t j = 0; j < blocks; j++) {
            bool kept = false;
            for (uint32_t k = 0; k < 2048 && !kept; k++) kept = ids[(size_t)r * 2048 + k] == (int32_t)j;
            CHECK(mask[(size_t)r * stride + j] == (kept ? 0.0f : -INFINITY));
        }
        for (uint32_t j = blocks; j < stride; j++) CHECK(mask[(size_t)r * stride + j] == 7.0f);
        for (uint32_t i = 0; i < n; i++) {
            const float expect = i < visible && mask[(size_t)r * stride + i / 8] == 0.0f ?
                scores[(size_t)r * n + i] : -INFINITY;
            CHECK(got[(size_t)r * n + i] == expect);
        }
    }
    CHECK(!ds4_gpu_dsv41_candidate_topk_batch(t, b, blocks, rows, 16376));
    CHECK(!ds4_gpu_dsv41_candidate_mask_batch(m, t, blocks, rows, blocks - 1));
    ds4_gpu_tensor_free(m); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(t);
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(s);
    free(scores); free(got); free(mask);
    fprintf(stderr, "V4.1 batched candidates n=%u start=%u: exact\n", n, start);
    return 1;
}

static int check_sparse_gather(void) {
    const uint32_t sizes[] = {1, 3, 511, 512, 513, 8193, 17017};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        const uint32_t rows = sizes[i], selected = rows < 512 ? rows : 512;
        ds4_gpu_tensor *source = upload(NULL, (uint64_t)rows * 512 * 4);
        ds4_gpu_tensor *ids = upload(NULL, selected * 4);
        ds4_gpu_tensor *out = upload(NULL, (uint64_t)selected * 512 * 4);
        CHECK(source && ids && out);
        float *input = ds4_gpu_tensor_contents(source);
        int32_t *indices = ds4_gpu_tensor_contents(ids);
        for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < 512; c++)
            input[(size_t)r * 512 + c] = (float)r + (float)c / 512.0f;
        for (uint32_t r = 0; r < selected; r++) indices[r] = (int32_t)(rows - 1u - r);
        CHECK(ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected));
        const float *result = ds4_gpu_tensor_contents(out);
        for (uint32_t r = 0; r < selected; r++)
            CHECK(!memcmp(result + r * 512, input + (size_t)indices[r] * 512, 512 * 4));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows + 1, selected));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected + 1));
        ds4_gpu_tensor_free(source); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    }
    fprintf(stderr, "V4.1 bounded sparse KV gather: exact\n");
    return 1;
}

#ifdef __APPLE__
static int check_hc_collapse_norm(void) {
    enum { MAX_WIDTH = 8192, MAX_ROWS = 17, GUARD = 16 };
    const unsigned widths[] = {5120, 8192}, rows_list[] = {1, 5, 17};
    const unsigned strides[] = {16, 96};
    const size_t page = (size_t)getpagesize();
    const size_t model_bytes = ((MAX_WIDTH * sizeof(float) + page - 1) / page) * page;
    float *norm_weight = NULL;
    CHECK(posix_memalign((void **)&norm_weight, page, model_bytes) == 0);
    CHECK(ds4_gpu_set_model_map(norm_weight, model_bytes));
    const size_t raw_bytes = (size_t)MAX_ROWS * 4 * MAX_WIDTH * sizeof(float);
    const size_t storage_bytes = ((size_t)MAX_ROWS * MAX_WIDTH + 2 * GUARD) * sizeof(float);
    float *values = malloc(raw_bytes);
    float pre[MAX_ROWS * 24];
    CHECK(values);
    ds4_gpu_tensor *residual = upload(NULL, raw_bytes);
    ds4_gpu_tensor *weights = upload(NULL, sizeof(pre));
    ds4_gpu_tensor *storage[4];
    for (unsigned i = 0; i < 4; i++) storage[i] = upload(NULL, storage_bytes);
    CHECK(residual && weights && storage[0] && storage[1] && storage[2] && storage[3]);
    for (unsigned wi = 0; wi < 2; wi++) for (unsigned ri = 0; ri < 3; ri++)
    for (unsigned si = 0; si < 2; si++) for (unsigned pattern = 0; pattern < 2; pattern++) {
        const unsigned width = widths[wi], rows = rows_list[ri], stride = strides[si];
        const size_t count = (size_t)width * rows, bytes = count * sizeof(float);
        for (size_t i = 0; i < count * 4; i++)
            values[i] = pattern ? copysignf(0, i & 1 ? -1 : 1) : bf16(random_value() * 3);
        for (unsigned i = 0; i < width; i++) norm_weight[i] = pattern ? (i & 1 ? -1 : 1) : 1 + random_value() * .125f;
        for (unsigned row = 0; row < rows; row++) {
            for (unsigned j = 0; j < stride / 4; j++) pre[row * (stride / 4) + j] = 1000 + j;
            for (unsigned h = 0; h < 4; h++)
                pre[row * (stride / 4) + h] = row ? .125f * (h + 1) : (h == 0 ? 1 : 0);
        }
        CHECK(ds4_gpu_tensor_write(residual, 0, values, count * 4 * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(weights, 0, pre, rows * stride));
        ds4_gpu_tensor *out[4];
        for (unsigned i = 0; i < 4; i++) {
            uint8_t *guard = ds4_gpu_tensor_contents(storage[i]);
            CHECK(guard); memset(guard, 0xa5, storage_bytes);
            out[i] = ds4_gpu_tensor_view(storage[i], GUARD * sizeof(float), bytes);
            CHECK(out[i]);
        }
        CHECK((stride == 96 ? ds4_gpu_hc_weighted_sum_split_tensor(out[0], residual, weights, width, 4) :
            ds4_gpu_hc_weighted_sum_tensor(out[0], residual, weights, width, 4)));
        CHECK(ds4_gpu_dsv41_quantize(out[0], width, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(out[1], out[0], norm_weight, model_bytes, 0, width, rows, 1e-6f));
        CHECK(ds4_gpu_dsv41_quantize(out[1], width, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_dsv41_hc_collapse_norm(out[2], out[3], residual, weights,
            norm_weight, model_bytes, 0, width, rows, stride, 1e-6f));
        CHECK(ds4_gpu_synchronize());
        for (unsigned result = 0; result < 2; result++) {
            const float *old = ds4_gpu_tensor_contents(out[result]);
            const float *fused = ds4_gpu_tensor_contents(out[result + 2]);
            CHECK(old && fused);
            size_t differences = 0, first = 0, zeros = 0;
            for (size_t i = 0; i < count; i++) {
                uint32_t old_bits, fused_bits;
                memcpy(&old_bits, old + i, 4); memcpy(&fused_bits, fused + i, 4);
                CHECK(isfinite(old[i]) && isfinite(fused[i]) && !(old_bits & 0xffff) && !(fused_bits & 0xffff));
                if (old[i] == 0) zeros++;
                if (old_bits != fused_bits) { if (!differences) first = i; differences++; }
            }
            fprintf(stderr, "HC_COLLAPSE_NORM width=%u rows=%u stride=%u pattern=%u result=%s bit_differences=%zu zeros=%zu first=%zu old=%a new=%a\n",
                width, rows, stride, pattern, result ? "norm" : "collapse", differences, zeros, first, old[first], fused[first]);
            CHECK(!differences);
        }
        CHECK(!memcmp(ds4_gpu_tensor_contents(residual), values, count * 4 * sizeof(float)));
        CHECK(!memcmp(ds4_gpu_tensor_contents(weights), pre, rows * stride));
        for (unsigned i = 0; i < 4; i++) {
            const uint8_t *guard = ds4_gpu_tensor_contents(storage[i]);
            for (size_t j = 0; j < GUARD * sizeof(float); j++) CHECK(guard[j] == 0xa5);
            for (size_t j = GUARD * sizeof(float) + bytes; j < storage_bytes; j++) CHECK(guard[j] == 0xa5);
            ds4_gpu_tensor_free(out[i]);
        }
    }
    for (unsigned i = 0; i < 4; i++) ds4_gpu_tensor_free(storage[i]);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(residual);
    CHECK(ds4_gpu_synchronize());
    ds4_gpu_cleanup(); free(norm_weight); free(values);
    fprintf(stderr, "V4.1 supplied-PRE HC collapse/BF16/RMSNorm/BF16: scalar/batch, stride16/96, outputs/bounds exact\n");
    return 1;
}

static uint32_t main_fp4_bits(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits)); return bits;
}

static float adjacent_bf16(float value, int delta);

static int check_main_kv_fp4(void) {
    enum { WIDTH = 512, ROWS = 129, KEPT = 5, PREFIX = 3, N = WIDTH * ROWS };
    const int32_t picked[KEPT] = {128, 0, -1, 37, 8};
    const int32_t identity[KEPT] = {0, 1, 2, 3, 4};
    float *values = malloc(N * sizeof(float));
    float *expected = malloc(N * sizeof(float));
    float *actual = malloc(N * sizeof(float));
    CHECK(values && expected && actual);
    const uint64_t row_bytes = 288, storage_bytes = (PREFIX + ROWS + 1u) * row_bytes;
    ds4_gpu_tensor *source = upload(NULL, N * sizeof(float));
    ds4_gpu_tensor *rounded = upload(NULL, N * sizeof(float));
    ds4_gpu_tensor *storage = upload(NULL, storage_bytes);
    ds4_gpu_tensor *cache = ds4_gpu_tensor_view(storage, PREFIX * row_bytes, ROWS * row_bytes);
    ds4_gpu_tensor *repacked = upload(NULL, ROWS * row_bytes);
    ds4_gpu_tensor *decoded = upload(NULL, N * sizeof(float));
    ds4_gpu_tensor *ids = upload(picked, sizeof(picked));
    ds4_gpu_tensor *identity_ids = upload(identity, sizeof(identity));
    ds4_gpu_tensor *gathered = upload(NULL, KEPT * WIDTH * sizeof(float));
    ds4_gpu_tensor *gathered_half = upload(NULL, KEPT * WIDTH * sizeof(uint16_t));
    CHECK(source && rounded && storage && cache && repacked && decoded && ids && identity_ids && gathered && gathered_half);
    CHECK(ds4_gpu_tensor_bytes(cache) == ROWS * row_bytes);
    uint8_t *guard = ds4_gpu_tensor_contents(storage);
    CHECK(guard); memset(guard, 0xa5, (size_t)storage_bytes);
    const unsigned counts[] = {1, 9, 37, ROWS};
    const float alphabet[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (unsigned pattern = 0; pattern < 4; pattern++) {
        for (unsigned i = 0; i < N; i++) {
            const unsigned scale_code = 1 + (i / 16) % 126;
            const float scale = scale_code < 8 ? ldexpf(scale_code, -9) :
                ldexpf(1.0f + (scale_code & 7) / 8.0f, (scale_code >> 3) - 7);
            const float v = pattern == 0 ? scale * alphabet[i % 8] :
                pattern == 1 ? random_value() * .00001f : random_value() * ldexpf(1, (int)((i / 16) % 20) - 10);
            values[i] = pattern == 0 ? copysignf(v, (i / 8) & 1 ? -1 : 1) : v;
            if (pattern == 3) {
                // Amax at and around every adjacent E4M3 scale midpoint;
                // values at and around all E2M1 midpoints. BF16 neighbors
                // expose reciprocal-multiply rounding that random rows miss.
                const float ties[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
                const unsigned group = i / 16, col = i % 16;
                const unsigned code = 1 + (group / 9) % 125;
                const float lo = code < 8 ? ldexpf(code, -9) :
                    ldexpf(1.0f + (code & 7) / 8.0f, (code >> 3) - 7);
                const unsigned next = code + 1;
                const float hi = next < 8 ? ldexpf(next, -9) :
                    ldexpf(1.0f + (next & 7) / 8.0f, (next >> 3) - 7);
                const float maximum = adjacent_bf16(bf16(3 * (lo + hi)), (int)((group / 3) % 3) - 1);
                const float selected_scale = nearest(fmaxf(maximum, .01171875f) / 6, 0);
                float tied = adjacent_bf16(bf16(ties[col % 7] * selected_scale), (int)(group % 3) - 1);
                if (col == 14) tied = 0;
                values[i] = col == 15 ? maximum : copysignf(tied, (col & 1) ? -1.0f : 1.0f);
                if (group == 0) values[i] = col == 0 ? -37.5f : col == 15 ? 45.0f : 0.0f;
            }
        }
        for (unsigned group = 0; group < N / 16; group++) {
            float maximum = 0;
            for (unsigned j = 0; j < 16; j++) maximum = fmaxf(maximum, fabsf(bf16(values[group * 16 + j])));
            const float scale = nearest(fmaxf(maximum, .01171875f) / 6, 0);
            for (unsigned j = 0; j < 16; j++) {
                const float v = bf16(values[group * 16 + j]);
                expected[group * 16 + j] = bf16(copysignf(nearest(fabsf(v) / scale, 1), v) * scale);
            }
        }
        CHECK(ds4_gpu_tensor_write(source, 0, values, N * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(rounded, 0, values, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(rounded, WIDTH, ROWS, DS4_V41_FP4_E4M3));
        CHECK(ds4_gpu_tensor_read(rounded, 0, actual, N * sizeof(float)));
        if (memcmp(expected, actual, N * sizeof(float))) {
            unsigned first = 0, mismatches = 0;
            for (unsigned i = 0; i < N; i++) if (main_fp4_bits(expected[i]) != main_fp4_bits(actual[i])) {
                if (!mismatches) first = i;
                mismatches++;
            }
            float maximum = 0;
            for (unsigned j = first / 16 * 16; j < first / 16 * 16 + 16; j++) maximum = fmaxf(maximum, fabsf(bf16(values[j])));
            const float scale = nearest(fmaxf(maximum, .01171875f) / 6, 0);
            fprintf(stderr, "MAIN_FP4_CPU_REFERENCE_DIFF pattern=%u count=%u first=%u row=%u col=%u original=%a/%08x bf16=%a maximum=%a scale=%a/%08x cpu=%a/%08x plain_gpu=%a/%08x\n",
                pattern, mismatches, first, first / WIDTH, first % WIDTH, values[first], main_fp4_bits(values[first]), bf16(values[first]), maximum, scale, main_fp4_bits(scale), expected[first], main_fp4_bits(expected[first]), actual[first], main_fp4_bits(actual[first]));
            float *decoded_values = malloc(N * sizeof(float));
            CHECK(decoded_values && ds4_gpu_dsv41_main_pack_fp4(cache, source, ROWS) && ds4_gpu_dsv41_main_unpack_fp4(decoded, cache, ROWS));
            CHECK(ds4_gpu_tensor_read(decoded, 0, decoded_values, N * sizeof(float)));
            unsigned packed_first = 0, packed_mismatches = 0;
            for (unsigned i = 0; i < N; i++) if (main_fp4_bits(actual[i]) != main_fp4_bits(decoded_values[i])) {
                if (!packed_mismatches) packed_first = i;
                packed_mismatches++;
            }
            const unsigned row = packed_first / WIDTH, col = packed_first % WIDTH;
            const uint8_t *p = guard + (PREFIX + row) * row_bytes;
            fprintf(stderr, "MAIN_FP4_PACKED_REFERENCE_DIFF pattern=%u count=%u first=%u row=%u col=%u original=%a/%08x plain_gpu=%a/%08x decoded=%a/%08x code_byte=%02x scale_byte=%02x\n",
                pattern, packed_mismatches, packed_first, row, col, values[packed_first], main_fp4_bits(values[packed_first]), actual[packed_first], main_fp4_bits(actual[packed_first]), decoded_values[packed_first], main_fp4_bits(decoded_values[packed_first]), p[col / 2], p[256 + col / 16]);
            free(decoded_values);
            CHECK(!packed_mismatches);
        }
        // Storage must reproduce the canonical GPU quantizer, bit for bit.
        // Its exact divisions must also be used by packed publication: a
        // fast reciprocal can cross either scale or value rounding midpoint.
        memcpy(expected, actual, N * sizeof(float));
        for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const unsigned rows = counts[ci];
            CHECK(ds4_gpu_dsv41_main_pack_fp4(cache, source, rows));
            CHECK(ds4_gpu_dsv41_main_unpack_fp4(decoded, cache, rows));
            CHECK(ds4_gpu_tensor_read(decoded, 0, actual, rows * WIDTH * sizeof(float)));
            if (memcmp(expected, actual, rows * WIDTH * sizeof(float))) {
                unsigned first = 0;
                while (main_fp4_bits(expected[first]) == main_fp4_bits(actual[first])) first++;
                fprintf(stderr, "MAIN_FP4_PUBLICATION_MISMATCH pattern=%u rows=%u row=%u col=%u input=%a canonical=%a packed=%a scale_code=%02x\n",
                    pattern, rows, first / WIDTH, first % WIDTH, values[first], expected[first], actual[first],
                    guard[(PREFIX + first / WIDTH) * row_bytes + 256 + first % WIDTH / 16]);
            }
            CHECK(!memcmp(expected, actual, rows * WIDTH * sizeof(float)));
            CHECK(ds4_gpu_dsv41_main_repack_fp4(repacked, rounded, rows));
            CHECK(ds4_gpu_dsv41_main_unpack_fp4(decoded, repacked, rows));
            CHECK(ds4_gpu_tensor_read(decoded, 0, actual, rows * WIDTH * sizeof(float)));
            CHECK(!memcmp(expected, actual, rows * WIDTH * sizeof(float)));
        }
        CHECK(ds4_gpu_dsv41_main_gather_fp4(gathered, cache, ids, ROWS, KEPT, false));
        CHECK(ds4_gpu_tensor_read(gathered, 0, actual, KEPT * WIDTH * sizeof(float)));
        for (unsigned row = 0; row < KEPT; row++) {
            if (picked[row] < 0) for (unsigned col = 0; col < WIDTH; col++) CHECK(actual[row * WIDTH + col] == 0);
            else CHECK(!memcmp(actual + row * WIDTH, expected + picked[row] * WIDTH, WIDTH * sizeof(float)));
        }
        CHECK(ds4_gpu_dsv41_main_gather_fp4(gathered_half, cache, ids, ROWS, KEPT, true));
        CHECK(ds4_gpu_dsv41_gather_kv_f16(gathered, gathered_half, identity_ids, KEPT, KEPT));
        CHECK(ds4_gpu_tensor_read(gathered, 0, actual, KEPT * WIDTH * sizeof(float)));
        for (unsigned row = 0; row < KEPT; row++) {
            if (picked[row] < 0) for (unsigned col = 0; col < WIDTH; col++) CHECK(actual[row * WIDTH + col] == 0);
            else CHECK(!memcmp(actual + row * WIDTH, expected + picked[row] * WIDTH, WIDTH * sizeof(float)));
        }
    }
    // The observed failure: 37.5 / 7.5 is exactly 5. E2M1 ties choose 4,
    // yielding 30. A fast reciprocal landed above 5 and incorrectly yielded 45.
    // Exercise the same publication boundary inside both fused RoPE kernels
    // and the tail-only path used when relocating suffix cache rows.
    memset(values, 0, WIDTH * sizeof(float));
    values[0] = values[448] = -37.5f;
    values[15] = values[463] = 45.0f;
    CHECK(ds4_gpu_tensor_write(source, 0, values, WIDTH * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(rounded, 0, values, WIDTH * sizeof(float)));
    ds4_gpu_tensor *one = ds4_gpu_tensor_view(rounded, 0, WIDTH * sizeof(float));
    ds4_gpu_tensor *tail = ds4_gpu_tensor_view(source, 0, WIDTH * sizeof(float));
    ds4_gpu_tensor *table = upload(NULL, 512);
    CHECK(one && tail && table);
    CHECK(ds4_gpu_dsv41_quantize(one, WIDTH, 1, DS4_V41_FP4_E4M3));
    CHECK(ds4_gpu_tensor_read(one, 0, expected, WIDTH * sizeof(float)));
    CHECK(expected[0] == -30.0f && expected[448] == -30.0f);
    CHECK(ds4_gpu_dsv41_rope_quantize(tail, decoded, 0, WIDTH, 0, false, DS4_V41_FP4_E4M3));
    CHECK(ds4_gpu_tensor_read(decoded, 0, actual, WIDTH * sizeof(float)));
    CHECK(!memcmp(expected, actual, WIDTH * sizeof(float)));
    CHECK(ds4_gpu_dsv41_rope_coeff_table(table, 0));
    CHECK(ds4_gpu_tensor_write(one, 0, values, WIDTH * sizeof(float)));
    CHECK(ds4_gpu_dsv41_rope_quantize_heads_tab(one, WIDTH, 1, 0, false, DS4_V41_FP4_E4M3, table));
    CHECK(ds4_gpu_tensor_read(one, 0, actual, WIDTH * sizeof(float)));
    CHECK(!memcmp(expected, actual, WIDTH * sizeof(float)));
    CHECK(ds4_gpu_dsv41_prerope_quantize(tail, WIDTH, 1, DS4_V41_FP4_E4M3));
    CHECK(ds4_gpu_tensor_read(tail, 0, actual, WIDTH * sizeof(float)));
    CHECK(!memcmp(values, actual, 448 * sizeof(float)));
    CHECK(!memcmp(expected + 448, actual + 448, 64 * sizeof(float)));
    ds4_gpu_tensor_free(table); ds4_gpu_tensor_free(tail); ds4_gpu_tensor_free(one);
    fprintf(stderr, "V4.1 FP4 midpoint: canonical, fused RoPE, table RoPE and suffix-tail publication agree exactly\n");

    // Real checkpoint regression: original scale2/512, maximum code4. Fresh
    // mode3 quantization of these decoded values incorrectly chooses scale1/512.
    const float legacy_group[16] = {
        .001953125f, -.0078125f, 0, 0, 0, 0, .001953125f, .0078125f,
        -.001953125f, -.0078125f, .001953125f, .0078125f,
        .015625f, .01171875f, .015625f, .015625f,
    };
    for (unsigned group = 0; group < N / 16; group++) {
        const unsigned scale_code = 1 + group % 7;
        const int max4 = (group / 7) & 1;
        const float scale = ldexpf(scale_code, -9);
        for (unsigned j = 0; j < 16; j++) {
            const float magnitude = alphabet[j % (max4 ? 7 : 8)] * scale;
            values[group * 16 + j] = copysignf(magnitude, j & 1 ? -1 : 1);
        }
        values[group * 16 + 15] = (max4 ? 4 : 6) * scale;
    }
    memcpy(values, legacy_group, sizeof(legacy_group));
    CHECK(ds4_gpu_tensor_write(source, 0, values, N * sizeof(float)));
    CHECK(ds4_gpu_dsv41_main_repack_fp4(cache, source, ROWS));
    CHECK(ds4_gpu_dsv41_main_unpack_fp4(decoded, cache, ROWS));
    CHECK(ds4_gpu_tensor_read(decoded, 0, actual, N * sizeof(float)));
    if (memcmp(values, actual, N * sizeof(float))) {
        unsigned first = 0;
        while (first < N && main_fp4_bits(values[first]) == main_fp4_bits(actual[first])) first++;
        fprintf(stderr, "MAIN_FP4_LEGACY_REPACK first=%u expected=%a/%08x actual=%a/%08x\n",
            first, values[first], main_fp4_bits(values[first]), actual[first], main_fp4_bits(actual[first]));
        CHECK(0);
    }
    fprintf(stderr, "V4.1 legacy FP4 repack: checkpoint corpus, all7 subnormal scales with max4/max6 and signed zero exact\n");
    for (uint64_t i = 0; i < PREFIX * row_bytes; i++) CHECK(guard[i] == 0xa5);
    for (uint64_t i = (PREFIX + ROWS) * row_bytes; i < storage_bytes; i++) CHECK(guard[i] == 0xa5);
    ds4_gpu_tensor_free(gathered_half); ds4_gpu_tensor_free(gathered); ds4_gpu_tensor_free(identity_ids);
    ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(decoded); ds4_gpu_tensor_free(repacked);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(storage); ds4_gpu_tensor_free(rounded); ds4_gpu_tensor_free(source);
    free(actual); free(expected); free(values);
    fprintf(stderr, "V4.1 main FP4: exact mode3 values/scales, legacy repack, signed zero, tails, offset guards and F32/F16 gathers\n");
    return 1;
}

static int check_compressed_kv_f16(void) {
    enum { WIDTH = 512, ROWS = 9, KEPT = 4, PREFIX = 3, N = WIDTH * ROWS };
    const float scales[] = {0x1p-9f, 0x1p-6f, .5f, 1.0f, 64.0f, 448.0f};
    const float alphabet[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const int32_t picked[KEPT] = {8, 0, 5, 2};
    const int32_t identity[KEPT] = {0, 1, 2, 3};
    float source[N], quantized[N], actual[KEPT * WIDTH];
    for (int i = 0; i < N; i++) {
        float value = scales[(i / 16) % 6] * alphabet[i % 8];
        source[i] = copysignf(value, (i / 8) & 1 ? -1.0f : 1.0f);
    }
    const uint64_t packed_bytes = (uint64_t)(PREFIX + ROWS + 1) * WIDTH * sizeof(uint16_t);
    const uint64_t row_bytes = WIDTH * sizeof(uint16_t);
    ds4_gpu_tensor *plain = upload(source, sizeof(source));
    ds4_gpu_tensor *packed = upload(NULL, packed_bytes);
    ds4_gpu_tensor *ids = upload(picked, sizeof(picked));
    ds4_gpu_tensor *identity_ids = upload(identity, sizeof(identity));
    ds4_gpu_tensor *out = upload(NULL, sizeof(actual));
    const uint64_t selected_bytes = (PREFIX + KEPT + 1u) * row_bytes;
    ds4_gpu_tensor *selected_storage = upload(NULL, selected_bytes);
    CHECK(plain && packed && ids && identity_ids && out && selected_storage);
    uint8_t *guard = ds4_gpu_tensor_contents(packed);
    CHECK(guard); memset(guard, 0xa5, (size_t)packed_bytes);
    uint8_t *selected_guard = ds4_gpu_tensor_contents(selected_storage);
    CHECK(selected_guard); memset(selected_guard, 0xa5, (size_t)selected_bytes);
    ds4_gpu_tensor *cache = ds4_gpu_tensor_view(packed, PREFIX * row_bytes, ROWS * row_bytes);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_view(selected_storage,
        PREFIX * row_bytes, KEPT * row_bytes);
    CHECK(cache && selected && ds4_gpu_tensor_bytes(cache) == sizeof(source) / 2);
    CHECK(ds4_gpu_dsv41_quantize(plain, WIDTH, ROWS, DS4_V41_FP4_E4M3));
    CHECK(ds4_gpu_tensor_read(plain, 0, quantized, sizeof(quantized)));
    for (int i = 0; i < N; i++) {
        CHECK(isfinite(quantized[i]) && fabsf(quantized[i]) <= 2688.0f);
        CHECK(quantized[i] == 0 || fabsf(quantized[i]) >= 0x1p-10f);
    }
    CHECK(ds4_gpu_tensor_copy_f32_to_f16(packed, PREFIX * row_bytes, plain, 0, N));
    CHECK(ds4_gpu_dsv41_gather_kv_f16(out, cache, ids, ROWS, KEPT));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    for (int row = 0; row < KEPT; row++)
        CHECK(!memcmp(actual + row * WIDTH, quantized + picked[row] * WIDTH, WIDTH * sizeof(float)));
    CHECK(ds4_gpu_dsv41_gather_kv_f16_packed(selected, cache, ids, ROWS, KEPT));
    CHECK(ds4_gpu_dsv41_gather_kv_f16(out, selected, identity_ids, KEPT, KEPT));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    for (int row = 0; row < KEPT; row++) {
        CHECK(!memcmp(selected_guard + (PREFIX + row) * row_bytes,
            guard + (PREFIX + picked[row]) * row_bytes, (size_t)row_bytes));
        CHECK(!memcmp(actual + row * WIDTH, quantized + picked[row] * WIDTH, WIDTH * sizeof(float)));
    }
    CHECK(ds4_gpu_synchronize());
    for (uint64_t i = 0; i < PREFIX * row_bytes; i++) CHECK(guard[i] == 0xa5);
    for (uint64_t i = (PREFIX + ROWS) * row_bytes; i < packed_bytes; i++) CHECK(guard[i] == 0xa5);
    for (uint64_t i = 0; i < PREFIX * row_bytes; i++) CHECK(selected_guard[i] == 0xa5);
    for (uint64_t i = (PREFIX + KEPT) * row_bytes; i < selected_bytes; i++)
        CHECK(selected_guard[i] == 0xa5);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(plain); ds4_gpu_tensor_free(packed);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(selected_storage);
    ds4_gpu_tensor_free(identity_ids); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    fprintf(stderr, "V4.1 mode3 compressed KV F16 storage, offset publication and F32/packed sparse gathers: exact\n");
    return 1;
}

/* Reuse the real scalar attention API: legacy gather expands to F32 before
 * attention stages it as F16; packed gather supplies those F16 bytes directly.
 * All buffers/model-map storage are prepared before warmup and timing. */
static int bench_compressed_kv_f16(void) {
    enum { WIDTH = 512, ROWS = 1537, MAX_KEPT = 512, HEADS = 64, RAW = 128, CALLS = 16 };
    const unsigned kept_counts[] = {128, 512};
    const float alphabet[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const size_t source_values = (size_t)ROWS * WIDTH, head_values = HEADS * WIDTH;
    const size_t page = (size_t)getpagesize();
    float *sinks = NULL;
    CHECK(!posix_memalign((void **)&sinks, page, page));
    for (int h = 0; h < HEADS; h++) sinks[h] = random_value() / 16;
    CHECK(ds4_gpu_set_model_map(sinks, page));
    ds4_gpu_set_quality(false);
    ds4_gpu_tensor *plain = upload(NULL, source_values * sizeof(float));
    ds4_gpu_tensor *packed = upload(NULL, source_values * sizeof(uint16_t));
    ds4_gpu_tensor *ids = upload(NULL, MAX_KEPT * sizeof(int32_t));
    ds4_gpu_tensor *legacy = upload(NULL, MAX_KEPT * WIDTH * sizeof(float));
    ds4_gpu_tensor *selected = upload(NULL, MAX_KEPT * WIDTH * sizeof(uint16_t));
    ds4_gpu_tensor *query = upload(NULL, head_values * sizeof(float));
    ds4_gpu_tensor *raw = upload(NULL, RAW * WIDTH * sizeof(float));
    ds4_gpu_tensor *heads[2] = {upload(NULL, head_values * sizeof(float)),
                               upload(NULL, head_values * sizeof(float))};
    CHECK(plain && packed && ids && legacy && selected && query && raw && heads[0] && heads[1]);
    float *source = ds4_gpu_tensor_contents(plain), *q = ds4_gpu_tensor_contents(query);
    float *r = ds4_gpu_tensor_contents(raw);
    int32_t *picked = ds4_gpu_tensor_contents(ids);
    CHECK(source && q && r && picked);
    for (size_t i = 0; i < source_values; i++)
        source[i] = copysignf(alphabet[i % 8] / 8, (i / 8) & 1 ? -1 : 1);
    for (size_t i = 0; i < head_values; i++) q[i] = bf16(random_value() / 8);
    for (int i = 0; i < RAW * WIDTH; i++) r[i] = bf16(random_value() / 8);
    for (int i = 0; i < MAX_KEPT; i++) picked[i] = ROWS - 1 - (i * 19 % ROWS);
    CHECK(ds4_gpu_tensor_copy_f32_to_f16(packed, 0, plain, 0, source_values));
    puts("source_rows,selected,scope,calls_per_batch,paired_repeats,legacy_ms_per_call,packed_ms_per_call,gather_byte_equal,attention_byte_equal,attention_max_abs");
    for (unsigned k = 0; k < sizeof(kept_counts) / sizeof(*kept_counts); k++) {
        const uint32_t kept = kept_counts[k];
        for (unsigned attention = 0; attention < 2; attention++) {
            double elapsed[2] = {0};
            bool attention_equal = true;
            double attention_max_abs = 0;
            for (unsigned repeat = 0; repeat < 7; repeat++) {
                for (unsigned j = 0; j < 2; j++) {
                    const unsigned mode = j ^ (repeat & 1);
                    const double begin = monotonic_seconds();
                    CHECK(ds4_gpu_begin_commands());
                    for (unsigned call = 0; call < CALLS; call++) {
                        CHECK(mode ? ds4_gpu_dsv41_gather_kv_f16_packed(selected, packed, ids, ROWS, kept)
                                   : ds4_gpu_dsv41_gather_kv_f16(legacy, packed, ids, ROWS, kept));
                        if (attention) CHECK(ds4_gpu_attention_decode_heads_tensor(heads[mode],
                            sinks, page, 0, query, raw, RAW, RAW, 17,
                            mode ? selected : legacy, mode, kept, NULL, 0, HEADS, WIDTH));
                    }
                    CHECK(ds4_gpu_end_commands());
                    if (repeat) elapsed[mode] += (monotonic_seconds() - begin) * (1000.0 / (6 * CALLS));
                }
                if (attention) {
                    const float *before = ds4_gpu_tensor_contents(heads[0]);
                    const float *after = ds4_gpu_tensor_contents(heads[1]);
                    attention_equal = attention_equal && !memcmp(before, after, head_values * sizeof(float));
                    for (size_t i = 0; i < head_values; i++) {
                        CHECK(isfinite(before[i]) && isfinite(after[i]));
                        attention_max_abs = fmax(attention_max_abs, fabs((double)before[i] - after[i]));
                    }
                }
            }
            const uint16_t *packed_source = ds4_gpu_tensor_contents(packed);
            const uint16_t *packed_selected = ds4_gpu_tensor_contents(selected);
            const float *legacy_selected = ds4_gpu_tensor_contents(legacy);
            for (uint32_t i = 0; i < kept; i++) {
                CHECK(!memcmp(packed_selected + i * WIDTH,
                    packed_source + (size_t)picked[i] * WIDTH, WIDTH * sizeof(uint16_t)));
                CHECK(!memcmp(legacy_selected + i * WIDTH,
                    source + (size_t)picked[i] * WIDTH, WIDTH * sizeof(float)));
            }
            printf("%u,%u,%s,%u,6,%.9f,%.9f,1,%d,%.12g\n", (unsigned)ROWS, kept,
                attention ? "gather_attention" : "gather", (unsigned)CALLS,
                elapsed[0], elapsed[1], attention ? attention_equal : -1, attention_max_abs);
        }
    }
    ds4_gpu_tensor_free(plain); ds4_gpu_tensor_free(packed); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(legacy); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(query);
    ds4_gpu_tensor_free(raw); ds4_gpu_tensor_free(heads[0]); ds4_gpu_tensor_free(heads[1]);
    ds4_gpu_cleanup(); free(sinks);
    return 1;
}
#endif

#ifdef __APPLE__
/* Random Q8_0 weights [rows][cols] at page-aligned memory (scales around 2^-8). */
static void *q8_weights(uint32_t rows, uint32_t cols, uint64_t *bytes) {
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    *bytes = (uint64_t)rows * cols / 32 * sizeof(q8_block);
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), *bytes) != 0) return NULL;
    q8_block *w = model;
    for (uint64_t i = 0; i < *bytes / sizeof(*w); i++) {
        w[i].d = (uint16_t)(0x1c00 + (int)(random_value() * 8192) % 1024);
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 127);
    }
    return model;
}

/* The standalone short-Q8 query preserves the generic projection's accumulation,
 * BF16 rounding and RoPE order. An explicit NSG override must retain generic dispatch. */
static int check_short_query(void) {
    enum { WIDTH = 1280, OUTPUTS = 32768 };
    const uint32_t positions[] = {0, 1, 8191, 1048575};
    const char *candidate = "kernel_dsv41_query_short_q8";
    const char *generic = "kernel_mul_mv_q8_0_f32_bf16_rope";
    const char *old_short = getenv("DS4_METAL_V41_SHORT_Q8_QUERY");
    const char *old_nsg = getenv("DS4_METAL_Q8_MV_NSG");
    char *saved_short = old_short ? strdup(old_short) : NULL;
    char *saved_nsg = old_nsg ? strdup(old_nsg) : NULL;
    CHECK((!old_short || saved_short) && (!old_nsg || saved_nsg));
    CHECK(unsetenv("DS4_METAL_Q8_MV_NSG") == 0);
    uint64_t wb = 0;
    void *model = q8_weights(OUTPUTS, WIDTH, &wb);
    CHECK(model && ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *x = upload(NULL, WIDTH * sizeof(float));
    ds4_gpu_tensor *out[2] = {upload(NULL, OUTPUTS * sizeof(float)),
                              upload(NULL, OUTPUTS * sizeof(float))};
    CHECK(x && out[0] && out[1]);
    float *input = ds4_gpu_tensor_contents(x);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (uint32_t compressed = 0; compressed < 2; compressed++) {
        for (uint32_t p = 0; p < sizeof(positions) / sizeof(positions[0]); p++) {
            for (uint32_t i = 0; i < WIDTH; i++) input[i] = random_value();
            for (uint32_t mode = 0; mode < 2; mode++) {
                CHECK(setenv("DS4_METAL_V41_SHORT_Q8_QUERY", mode ? "1" : "0", 1) == 0);
                const uint64_t short_runs = ds4_gpu_test_kernel_runs(candidate);
                const uint64_t generic_runs = ds4_gpu_test_kernel_runs(generic);
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_dsv41_project_q(out[mode], model, wb, 0, WIDTH, OUTPUTS,
                    x, positions[p], compressed));
                CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
                CHECK(ds4_gpu_test_kernel_runs(candidate) == short_runs + mode);
                CHECK(ds4_gpu_test_kernel_runs(generic) == generic_runs + !mode);
            }
            CHECK(!memcmp(ds4_gpu_tensor_contents(out[0]), ds4_gpu_tensor_contents(out[1]),
                OUTPUTS * sizeof(float)));
        }
    }
    /* Even the generic default value, when explicit, takes precedence over opt-in. */
    CHECK(setenv("DS4_METAL_Q8_MV_NSG", "4", 1) == 0);
    const uint64_t short_runs = ds4_gpu_test_kernel_runs(candidate);
    const uint64_t generic_runs = ds4_gpu_test_kernel_runs(generic);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_project_q(out[1], model, wb, 0, WIDTH, OUTPUTS, x, 1048575, true));
    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    CHECK(ds4_gpu_test_kernel_runs(candidate) == short_runs);
    CHECK(ds4_gpu_test_kernel_runs(generic) == generic_runs + 1);
    CHECK(!memcmp(ds4_gpu_tensor_contents(out[0]), ds4_gpu_tensor_contents(out[1]),
        OUTPUTS * sizeof(float)));
    ds4_gpu_test_set_flags(0);
    if (saved_short) { CHECK(setenv("DS4_METAL_V41_SHORT_Q8_QUERY", saved_short, 1) == 0); free(saved_short); }
    else CHECK(unsetenv("DS4_METAL_V41_SHORT_Q8_QUERY") == 0);
    if (saved_nsg) { CHECK(setenv("DS4_METAL_Q8_MV_NSG", saved_nsg, 1) == 0); free(saved_nsg); }
    else CHECK(unsetenv("DS4_METAL_Q8_MV_NSG") == 0);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out[0]); ds4_gpu_tensor_free(out[1]);
    free(model);
    fprintf(stderr, "V4.1 short-Q8 query: bitwise parity at 0/1/8191/1048575, both frequency modes; activation and NSG override PASS\n");
    return 1;
}

/* Exercise the pre-M5 exact kernels on any Apple GPU. Hardware selection
 * stays in the native owner; these checks compare the actual shader results,
 * including an odd final token and output bounds, against scalar dispatches. */
static int check_low_projection_rows(void) {
    enum { WIDTH = 256, RANK = 16, GROUPS = 4, COUNT = 2, ROWS = 9 };
    uint64_t wb = 0;
    void *model = q8_weights(GROUPS * RANK, WIDTH, &wb);
    CHECK(model && ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *x = upload(NULL, ROWS * GROUPS * WIDTH * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * COUNT * RANK * 4);
    ds4_gpu_tensor *got = upload(NULL, (ROWS * COUNT * RANK + 1) * 4);
    CHECK(x && ref && got);
    float *input = ds4_gpu_tensor_contents(x);
    for (int i = 0; i < ROWS * GROUPS * WIDTH; ++i) input[i] = random_value();
    const uint32_t counts[] = {1, 2, 8, 9};
    for (uint32_t first = 0; first <= COUNT; first += COUNT) {
        for (uint32_t row = 0; row < ROWS; ++row) {
            ds4_gpu_tensor *src = ds4_gpu_tensor_view(x, (row * GROUPS + first) * WIDTH * 4, COUNT * WIDTH * 4);
            ds4_gpu_tensor *dst = ds4_gpu_tensor_view(ref, row * COUNT * RANK * 4, COUNT * RANK * 4);
            CHECK(src && dst && ds4_gpu_attention_output_low_q8_tensor(dst, model, wb,
                first * RANK * (WIDTH / 32 * 34), WIDTH, RANK, COUNT, src));
            ds4_gpu_tensor_free(src); ds4_gpu_tensor_free(dst);
        }
        for (uint32_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
            const uint32_t rows = counts[i], values = rows * COUNT * RANK;
            ((float *)ds4_gpu_tensor_contents(got))[values] = 12345;
            CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(got, model, wb, 0,
                WIDTH, RANK, GROUPS, first, COUNT, x, rows) && ds4_gpu_synchronize());
            CHECK(!memcmp(ds4_gpu_tensor_contents(got), ds4_gpu_tensor_contents(ref), values * 4));
            CHECK(((float *)ds4_gpu_tensor_contents(got))[values] == 12345);
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(got);
    free(model);
    fprintf(stderr, "V4.1 grouped low projection rows, strided heads and group windows: bitwise PASS\n");
    return 1;
}

static int check_kslice_rows(void) {
    enum { WIDTH = 512, COUNT = 256, OUTPUTS = 40, ROWS = 33 };
    uint64_t wb = 0;
    void *model = q8_weights(OUTPUTS, WIDTH, &wb);
    CHECK(model && ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *x = upload(NULL, ROWS * COUNT * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * OUTPUTS * 4);
    ds4_gpu_tensor *got = upload(NULL, (ROWS * OUTPUTS + 1) * 4);
    ds4_gpu_tensor *rounded = upload(NULL, COUNT * 4);
    CHECK(x && ref && got && rounded);
    float *input = ds4_gpu_tensor_contents(x);
    for (int i = 0; i < ROWS * COUNT; ++i) input[i] = random_value();
    const char *old_nsg = getenv("DS4_METAL_Q8_MV_NSG");
    char *saved_nsg = old_nsg ? strdup(old_nsg) : NULL;
    CHECK(setenv("DS4_METAL_Q8_MV_NSG", "2", 1) == 0);
    for (uint32_t first = 0; first <= COUNT; first += COUNT) {
        for (uint32_t row = 0; row < ROWS; row++) {
            ds4_gpu_tensor *out = ds4_gpu_tensor_view(ref, row * OUTPUTS * 4, OUTPUTS * 4);
            CHECK(out && ds4_gpu_matmul_q8_0_kslice_tensor(out, model, wb, 0,
                WIDTH, first, COUNT, OUTPUTS, x, row * COUNT));
            ds4_gpu_tensor_free(out);
        }
        const char *names[] = {"kernel_mul_mv_q8_0_f32", "kernel_mul_mv_q8_0_f32_rows8",
                               "kernel_mul_mv_q8_0_f32_rows8_tok2"};
        for (uint32_t mode = 0; mode < 3; ++mode) {
            ((float *)ds4_gpu_tensor_contents(got))[ROWS * OUTPUTS] = 12345;
            uint64_t offset = 0;
            ds4_metal_model_buffer *w = ds4_metal_retain_model_range(model, wb, 0, wb, &offset);
            ds4_metal_output_launch plan = {0};
            plan.kind = DS4_OUT_KSLICE_MV;
            plan.pipeline = ds4_metal_routed_pipeline(1, names[mode], 2);
            plan.weights = (ds4_metal_attention_view){w, offset + first / 32 * 34, wb};
            plan.input = ds4_metal_attention_tensor_view(x);
            plan.output = ds4_metal_attention_tensor_view(got);
            ds4_metal_mv_args *a = &plan.args.vector;
            a->ne00 = a->ne10 = COUNT; a->ne01 = a->ne0 = OUTPUTS;
            a->ne02 = a->ne12 = a->r2 = a->r3 = 1;
            a->ne11 = a->ne1 = ROWS;
            a->nb00 = 34; a->nb01 = WIDTH / 32 * 34;
            a->nb02 = a->nb03 = a->nb01 * OUTPUTS;
            a->nb10 = 4; a->nb11 = COUNT * 4;
            a->nb12 = a->nb13 = ROWS * COUNT * 4;
            a->nr0 = mode ? 8 : 2;
            plan.groups_x = (OUTPUTS + a->nr0 - 1) / a->nr0;
            plan.groups_y = mode == 2 ? (ROWS + 1) / 2 : ROWS;
            plan.groups_z = 1; plan.threads_x = 32; plan.threads_y = 2;
            plan.shared_bytes = mode ? 1024 : 256;
            int owned = 0;
            void *command = ds4_zig_metal_command_acquire(&owned);
            CHECK(command && plan.pipeline && w &&
                ds4_zig_metal_command_release(command, owned,
                    ds4_metal_output_encode(command, &plan), "K-slice fixture") && ds4_gpu_synchronize());
            CHECK(!memcmp(ds4_gpu_tensor_contents(got), ds4_gpu_tensor_contents(ref), ROWS * OUTPUTS * 4));
            CHECK(((float *)ds4_gpu_tensor_contents(got))[ROWS * OUTPUTS] == 12345);
            ds4_metal_release_model_buffer(w);
            ds4_metal_release_model_buffer(plan.input.buffer);
            ds4_metal_release_model_buffer(plan.output.buffer);
        }
        float *round = ds4_gpu_tensor_contents(rounded);
        for (int i = 0; i < COUNT; ++i) round[i] = bf16(input[i]);
        CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(ref, model, wb, 0,
            WIDTH, first, COUNT, OUTPUTS, rounded, 0));
        CHECK(ds4_gpu_begin_commands() && ds4_gpu_matmul_q8_0_kslice_bf16in_tensor(got,
            model, wb, 0, WIDTH, first, COUNT, OUTPUTS, x) && ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(!memcmp(ds4_gpu_tensor_contents(got), ds4_gpu_tensor_contents(ref), OUTPUTS * 4));
    }
    if (saved_nsg) { CHECK(setenv("DS4_METAL_Q8_MV_NSG", saved_nsg, 1) == 0); free(saved_nsg); }
    else CHECK(unsetenv("DS4_METAL_Q8_MV_NSG") == 0);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(got); ds4_gpu_tensor_free(rounded);
    free(model);
    fprintf(stderr, "V4.1 K-slice batched rows, rows8/tok2 and BF16 input: bitwise PASS\n");
    return 1;
}

/* Run one call as built, where kernel must run once, then with the V4.1 paths in off; the outputs must match. */
#define LITERAL_PAIR(off, kernel, out, bytes, call) do { \
        const uint64_t runs_ = ds4_gpu_test_kernel_runs(kernel); \
        CHECK(ds4_gpu_begin_commands() && (call) && ds4_gpu_end_commands() && ds4_gpu_synchronize()); \
        CHECK(ds4_gpu_test_kernel_runs(kernel) == runs_ + 1); \
        void *lit_ = malloc(bytes); \
        CHECK(lit_); \
        memcpy(lit_, ds4_gpu_tensor_contents(out), bytes); \
        ds4_gpu_test_set_v41_off(off); \
        const int gen_ok_ = ds4_gpu_begin_commands() && (call) && ds4_gpu_end_commands() && ds4_gpu_synchronize(); \
        ds4_gpu_test_set_v41_off(0); \
        CHECK(gen_ok_ && ds4_gpu_test_kernel_runs(kernel) == runs_ + 1); \
        CHECK(!memcmp(lit_, ds4_gpu_tensor_contents(out), bytes)); \
        free(lit_); \
    } while (0)

/* One RoPE table per decode token. From the same attention partials: the reduce that writes the heads inverse-roped
 * and bf16-rounded, then the plain low (literal and generic), gives the plain reduce and inline-rope low's bits; the
 * indexer query's one-dispatch rope + quantize gives rope then quantize. Raw and compressed, up to the 1M limit. */
static int check_rope_tab(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, NROWS = 64, DV = 512, NWG = 32, QN = 32 * 128 };
    const uint64_t hb = (uint64_t)NROWS * DV * 4, lb = (uint64_t)GROUPS * RANK * 4;
    const uint64_t tn = (uint64_t)NROWS * (DV * NWG + 2u * NWG);
    uint64_t wb = 0;
    void *model = q8_weights(GROUPS * RANK, GROUP, &wb);
    float *t = malloc(tn * 4), *qref = malloc(QN * 4);
    CHECK(model && t && qref && ds4_gpu_set_model_map(model, wb));
    for (uint64_t i = 0; i < (uint64_t)NROWS * DV * NWG; i++) t[i] = random_value();
    for (uint64_t i = (uint64_t)NROWS * DV * NWG; i < tn; i += 2) {
        t[i] = fabsf(random_value()) + 0.25f;   /* S */
        t[i + 1] = random_value();              /* M */
    }
    ds4_gpu_tensor *tmp = upload(t, tn * 4), *plain = upload(NULL, hb), *once = upload(NULL, hb);
    ds4_gpu_tensor *q = upload(NULL, QN * 4), *tab = upload(NULL, 512);
    ds4_gpu_tensor *low[3] = {upload(NULL, lb), upload(NULL, lb), upload(NULL, lb)};
    CHECK(tmp && plain && once && q && tab && low[0] && low[1] && low[2]);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint32_t positions[] = {0, 1, 8191, 1048575};
    for (size_t pi = 0; pi < sizeof(positions) / sizeof(*positions); pi++) for (int compressed = 0; compressed < 2; compressed++) {
        const uint32_t pos = positions[pi];
        const uint64_t reduce = ds4_gpu_test_kernel_runs("kernel_dsv41_flash_attn_vec_reduce_heads");
        const uint64_t literal = ds4_gpu_test_kernel_runs("kernel_dsv4_attn_out_low_q8_0_f32_static");
        CHECK(ds4_gpu_begin_commands() && ds4_gpu_dsv41_rope_coeff_table(tab, pos));
        CHECK(ds4_gpu_dsv41_flash_attn_reduce_check(plain, tmp, NROWS, 0, NULL, false));
        CHECK(ds4_gpu_dsv41_flash_attn_reduce_check(once, tmp, NROWS, 1, tab, compressed));
        CHECK(ds4_gpu_dsv41_attention_low_plain(low[1], model, wb, 0, GROUP, RANK, GROUPS, once));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        /* the indexer query from the plain heads */
        CHECK(ds4_gpu_tensor_write(q, 0, ds4_gpu_tensor_contents(plain), QN * 4) && ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope(q, 128, 32, 1, pos, compressed, false));
        CHECK(ds4_gpu_dsv41_quantize(q, 128, 32, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_tensor_read(q, 0, qref, QN * 4));
        CHECK(ds4_gpu_tensor_write(q, 0, ds4_gpu_tensor_contents(plain), QN * 4) && ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope_quantize_heads_tab(q, 128, 32, pos, compressed, DS4_V41_FP4_E8M0, tab));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize() && !memcmp(qref, ds4_gpu_tensor_contents(q), QN * 4));
        /* the inline-rope low on the plain heads, and the generic plain low on the transformed ones */
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_attention_low(low[0], model, wb, 0, GROUP, RANK, GROUPS, plain, pos, compressed));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_LITERAL_SHAPES);
        const int ok = ds4_gpu_dsv41_attention_low_plain(low[2], model, wb, 0, GROUP, RANK, GROUPS, once);
        ds4_gpu_test_set_v41_off(0);
        CHECK(ok && ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(!memcmp(ds4_gpu_tensor_contents(low[0]), ds4_gpu_tensor_contents(low[1]), lb));
        CHECK(!memcmp(ds4_gpu_tensor_contents(low[0]), ds4_gpu_tensor_contents(low[2]), lb));
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_flash_attn_vec_reduce_heads") == reduce + 1);
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv4_attn_out_low_q8_0_f32_static") == literal + 1);
    }
    ds4_gpu_tensor *tensors[] = {tmp, plain, once, q, tab, low[0], low[1], low[2]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(t); free(qref); free(model);
    fprintf(stderr, "V4.1 RoPE table: transformed heads + plain low == inline rope low; indexer query rope + quantize "
            "in one dispatch == two PASS\n");
    return 1;
}

/* Exact verifier rows retain each standalone matvec's reduction, including
 * input/output BF16 boundaries, across the row-grouped dispatch. */
static int check_dense_bf16_rows(void) {
    enum { IN = 512, OUT = 67, ROWS = 8 };
    uint64_t weight_bytes = 0;
    void *model = q8_weights(OUT, IN, &weight_bytes);
    CHECK(model && ds4_gpu_set_model_map(model, weight_bytes));
    float values[ROWS * IN];
    for (size_t i = 0; i < ROWS * IN; i++) values[i] = random_value();
    ds4_gpu_tensor *input = upload(values, sizeof(values)), *reference_input = upload(values, sizeof(values));
    ds4_gpu_tensor *actual = upload(NULL, ROWS * OUT * 4), *expected = upload(NULL, ROWS * OUT * 4);
    CHECK(input && reference_input && actual && expected);
    for (uint32_t rows = 1; rows <= ROWS; rows++) for (int round_input = 0; round_input < 2; round_input++) {
        CHECK(ds4_gpu_tensor_write(reference_input, 0, values, sizeof(values)));
        CHECK(ds4_gpu_begin_commands());
        if (round_input) CHECK(ds4_gpu_dsv41_quantize(reference_input, IN, rows, DS4_V41_BF16));
        for (uint32_t row = 0; row < rows; row++) {
            ds4_gpu_tensor *x = ds4_gpu_tensor_view(reference_input, row * IN * 4, IN * 4);
            ds4_gpu_tensor *out = ds4_gpu_tensor_view(expected, row * OUT * 4, OUT * 4);
            CHECK(x && out);
            CHECK(ds4_gpu_matmul_q8_0_tensor(out, model, weight_bytes, 0, IN, OUT, x, 1));
            CHECK(ds4_gpu_dsv41_quantize(out, OUT, 1, DS4_V41_BF16));
            ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out);
        }
        CHECK(round_input ? ds4_gpu_matmul_q8_0_bf16io_tensor(actual, model, weight_bytes, 0, IN, OUT, input, rows) :
                            ds4_gpu_matmul_q8_0_bf16_tensor(actual, model, weight_bytes, 0, IN, OUT, input, rows));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(!memcmp(ds4_gpu_tensor_contents(actual), ds4_gpu_tensor_contents(expected), rows * OUT * 4));
    }
    const size_t page = (size_t)getpagesize();
    const size_t half_bytes = ((size_t)IN * OUT * 2 + page - 1) / page * page;
    _Float16 *half = NULL;
    CHECK(posix_memalign((void **)&half, page, half_bytes) == 0);
    for (size_t i = 0; i < IN * OUT; i++) half[i] = (_Float16)(random_value() / 32);
    CHECK(ds4_gpu_set_model_map(half, half_bytes));
    for (uint32_t rows = 1; rows <= ROWS; rows++) {
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t row = 0; row < rows; row++) {
            ds4_gpu_tensor *x = ds4_gpu_tensor_view(input, row * IN * 4, IN * 4);
            ds4_gpu_tensor *out = ds4_gpu_tensor_view(expected, row * OUT * 4, OUT * 4);
            CHECK(x && out && ds4_gpu_matmul_f16_tensor(out, half, half_bytes, 0, IN, OUT, x, 1));
            ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out);
        }
        CHECK(ds4_gpu_dsv41_projection_rows(actual, half, half_bytes, 0, IN, OUT, rows, input));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(!memcmp(ds4_gpu_tensor_contents(actual), ds4_gpu_tensor_contents(expected), rows * OUT * 4));
    }
    ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(reference_input);
    ds4_gpu_tensor_free(actual); ds4_gpu_tensor_free(expected);
    free(model); free(half);
    fprintf(stderr, "V4.1 Q8/F16 exact verifier rows and BF16 boundaries: PASS\n");
    return 1;
}

/* The V4.1 decode calls that take literal-shape kernels give the generic kernels' bits. */
static int check_literal_shapes(void) {
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    {   /* attention output-B into the HC streams: 8192 -> 5120, one row, no add */
        uint64_t wb = 0;
        void *model = q8_weights(5120, 8192, &wb);
        CHECK(model && ds4_gpu_set_model_map(model, wb));
        ds4_gpu_tensor *x = upload(NULL, 8192 * 4), *res = upload(NULL, 4 * 5120 * 4), *split = upload(NULL, 24 * 4);
        ds4_gpu_tensor *out = upload(NULL, 4 * 5120 * 4);
        CHECK(x && res && split && out);
        float *h = ds4_gpu_tensor_contents(x), *r = ds4_gpu_tensor_contents(res), *sp = ds4_gpu_tensor_contents(split);
        for (int i = 0; i < 8192; i++) h[i] = random_value();
        for (int i = 0; i < 4 * 5120; i++) r[i] = random_value();
        for (int i = 0; i < 24; i++) sp[i] = random_value() / 4.0f;
        LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_static", out, 4 * 5120 * 4,
                     ds4_gpu_dsv41_matmul_expand_rows(out, model, wb, 0, 8192, 5120, x, NULL, res, split, 4, 1));
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(res); ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(out);
        free(model);
    }
    {   /* the serial FFN join: shared + routed into the HC streams, one token, bf16 rounding */
        ds4_gpu_tensor *blk = upload(NULL, 5120 * 4), *add = upload(NULL, 5120 * 4), *res = upload(NULL, 4 * 5120 * 4);
        ds4_gpu_tensor *split = upload(NULL, 24 * 4), *out = upload(NULL, 4 * 5120 * 4);
        CHECK(blk && add && res && split && out);
        float *b = ds4_gpu_tensor_contents(blk), *a = ds4_gpu_tensor_contents(add);
        float *r = ds4_gpu_tensor_contents(res), *sp = ds4_gpu_tensor_contents(split);
        for (int i = 0; i < 5120; i++) { b[i] = random_value(); a[i] = random_value(); }
        for (int i = 0; i < 4 * 5120; i++) r[i] = random_value();
        for (int i = 0; i < 24; i++) sp[i] = random_value() / 4.0f;
        LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_dsv4_hc_expand4_serial_add_bf16", out, 4 * 5120 * 4,
                     ds4_gpu_hc_expand_split_add_bf16_tensor(out, blk, add, res, split, 5120, 4));
        ds4_gpu_tensor_free(blk); ds4_gpu_tensor_free(add); ds4_gpu_tensor_free(res);
        ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(out);
    }
    {   /* the attention q_a/kv pair: 5120 -> 1280 and 5120 -> 512, one row, bf16 outputs */
        uint64_t wb = 0;
        void *model = q8_weights(1280 + 512, 5120, &wb);
        CHECK(model && ds4_gpu_set_model_map(model, wb));
        const uint64_t off_b = 1280ull * (5120 / 32) * 34;
        ds4_gpu_tensor *x = upload(NULL, 5120 * 4), *qa = upload(NULL, 1280 * 4), *kv = upload(NULL, 512 * 4);
        CHECK(x && qa && kv);
        float *h = ds4_gpu_tensor_contents(x);
        for (int i = 0; i < 5120; i++) h[i] = random_value();
        for (int i = 0; i < 2; i++)
            LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv", i ? kv : qa,
                         i ? 512 * 4 : 1280 * 4, ds4_gpu_dsv41_project_pair_q8(qa, kv, model, wb, 0, off_b, 5120, 1280, 512, x));
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(qa); ds4_gpu_tensor_free(kv);
        free(model);
    }
    if (ds4_gpu_device_is_pre_m5_apple_silicon()) {
        /* MoE gate/up of one token, 6 of 384 MXFP4 experts (5120 -> 2304), into the down projection. Only the routed
         * experts are written: up reads the gate experts one row on, down (5120 x 1224 bytes) from byte 0. */
        const uint64_t row = 2720, expert = 2304 * row, page = (uint64_t)getpagesize();
        const uint64_t map_bytes = (384 * expert + row + page - 1) / page * page;
        unsigned char *map = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        CHECK(map != MAP_FAILED && ds4_gpu_set_model_map(map, map_bytes));
        const int32_t sel[6] = {3, 77, 150, 200, 301, 383};
        const float weights[6] = {0.3f, 0.25f, 0.2f, 0.12f, 0.08f, 0.05f};
        for (int e = 0; e < 6; e++)
            for (uint64_t i = (uint64_t)sel[e] * expert; i < (uint64_t)(sel[e] + 1) * expert + row; i++) {
                (void)random_value();
                map[i] = i % 17 ? (unsigned char)(seed >> 5) : (unsigned char)(118 + seed % 8);   /* e8m0 scales */
            }
        ds4_gpu_tensor *x = upload(NULL, 5120 * 4), *st = upload(sel, sizeof(sel)), *wt = upload(weights, sizeof(weights));
        ds4_gpu_tensor *t[5];
        for (int i = 0; i < 5; i++) t[i] = upload(NULL, 6 * 5120 * 4);
        CHECK(x && st && wt && t[0] && t[1] && t[2] && t[3] && t[4]);
        float *h = ds4_gpu_tensor_contents(x);
        for (int i = 0; i < 5120; i++) h[i] = random_value();
        LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_v41_f32",
                     t[2], 6 * 2304 * 4, ds4_gpu_routed_moe_one_tensor(t[4], t[0], t[1], t[2], t[3], map, map_bytes,
                     0, row, 0, 39, 39, expert, row, expert, 1224, 5120, 2304, 5120, st, wt, 384, 6, 10.0f, x, NULL,
                     0, true));
        for (int i = 0; i < 5; i++) ds4_gpu_tensor_free(t[i]);
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
        /* map stays: the model views may wrap it */
    }
    fprintf(stderr, "V4.1 literal shapes (output-B HC, serial HC join, q_a/kv pair, MoE gate/up) == generic PASS\n");
    return 1;
}

/* The HC collapse over 32 + 10 groups against one group, bitwise, fresh inputs on every call: the collapsed row and
 * its RMS norm. */
static int check_split_collapse(void) {
    const uint64_t wb = 65536;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
    ds4_gpu_tensor *res = upload(NULL, 4 * 5120 * 4), *pre = upload(NULL, 16);
    ds4_gpu_tensor *x[2] = {upload(NULL, 5120 * 4), upload(NULL, 5120 * 4)};
    ds4_gpu_tensor *n[2] = {upload(NULL, 5120 * 4), upload(NULL, 5120 * 4)};
    CHECK(res && pre && x[0] && x[1] && n[0] && n[1] && ds4_gpu_set_model_map(model, wb));
    float *w = model, *r = ds4_gpu_tensor_contents(res), *p = ds4_gpu_tensor_contents(pre);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int call = 0; call < 8; call++) {
        for (int i = 0; i < 5120; i++) w[i] = 1.0f + random_value() / 8.0f;
        for (int i = 0; i < 4 * 5120; i++) r[i] = random_value() * (call % 4 == 3 ? 64.0f : 1.0f);
        for (int i = 0; i < 4; i++) p[i] = random_value();
        const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse_part");
        for (int off = 0; off < 2; off++) {
            ds4_gpu_test_set_v41_off(off ? DS4_GPU_V41_SPLIT_REDUCE : 0);
            const int ok = ds4_gpu_begin_commands() && ds4_gpu_dsv41_deferred_collapse(x[off], n[off], res, pre,
                model, wb, 0, 1e-6f) && ds4_gpu_end_commands() && ds4_gpu_synchronize();
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok);
        }
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse_part") == runs + 1);
        CHECK(!memcmp(ds4_gpu_tensor_contents(x[0]), ds4_gpu_tensor_contents(x[1]), 5120 * 4));
        CHECK(!memcmp(ds4_gpu_tensor_contents(n[0]), ds4_gpu_tensor_contents(n[1]), 5120 * 4));
    }
    ds4_gpu_tensor_free(res); ds4_gpu_tensor_free(pre);
    for (int i = 0; i < 2; i++) { ds4_gpu_tensor_free(x[i]); ds4_gpu_tensor_free(n[i]); }
    free(model);
    fprintf(stderr, "V4.1 split HC collapse == one group, bitwise PASS\n");
    return 1;
}
#else
static int check_short_query(void) { return 1; }
static int check_low_projection_rows(void) { return 1; }
static int check_kslice_rows(void) { return 1; }
static int check_rope_tab(void) { return 1; }
static int check_literal_shapes(void) { return 1; }
static int check_dense_bf16_rows(void) { return 1; }
static int check_split_collapse(void) { return 1; }
#endif

static int check_attention_output(bool large) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120 };
    const uint32_t ROWS = large ? 8192u : 513u;
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = 0x2000; /* Exact 1/128 scale and binary-fraction inputs. */
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int)(random_value() * 8192) % 8;
    }
    float *x = malloc(xb), *reference = malloc(ob), *reference_low = malloc(lb);
    CHECK(x && reference && reference_low);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = (int)(random_value() * 8192) % 51 / 256.0f;
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(xt && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out, (uint64_t)r * OUT * 4, OUT * 4);
        CHECK(xr && lr && yr);
        CHECK(ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, xr));
        CHECK(ds4_gpu_dsv41_quantize(lr, GROUPS * RANK, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(yr, model, a_bytes + b_bytes,
            a_bytes, GROUPS * RANK, OUT, lr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(out, 0, reference, ob));
    CHECK(ds4_gpu_tensor_read(low, 0, reference_low, lb));
    const uint32_t sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 63, 64, 65, 257, 512, 513, 8191, 8192};
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        const uint32_t rows = sizes[n];
        if (rows > ROWS) break;
        CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
        CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
        CHECK(ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
            0, a_bytes, xt, rows));
        const float *got = ds4_gpu_tensor_contents(out);
        const float *got_low = ds4_gpu_tensor_contents(low);
        CHECK(!memcmp(got_low, reference_low, (uint64_t)rows * GROUPS * RANK * 4));
        for (uint64_t i = 0; i < (uint64_t)rows * OUT; i++) {
            if (!isfinite(got[i]) || fabsf(got[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i])))
                fprintf(stderr, "attention output rows=%u index=%llu actual=%.9g reference=%.9g\n",
                    rows, (unsigned long long)i, got[i], reference[i]);
            CHECK(isfinite(got[i]) && fabsf(got[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        }
        if (rows < ROWS) CHECK(isnan(got[(uint64_t)rows * OUT]));
        fprintf(stderr, "V4.1 batched Q8 output, BF16 boundary, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, xt, ROWS));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, xt, ROWS + 1));
#ifndef DS4_ROCM_BUILD
    ds4_gpu_tensor *packed = upload(NULL, xb / 2);
    ds4_gpu_tensor *partial[2] = {upload(NULL, ob), upload(NULL, ob)};
    CHECK(packed && partial[0] && partial[1]);
    const uint32_t tp_sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 512, 513};
    const uint32_t q_half = GROUPS / 2 * GROUP, low_half = GROUPS / 2 * RANK;
    for (size_t n = 0; n < sizeof(tp_sizes) / sizeof(*tp_sizes); n++) {
        const uint32_t rows = tp_sizes[n];
        for (uint32_t rank = 0; rank < 2; rank++) {
            float *px = ds4_gpu_tensor_contents(packed);
            for (uint32_t t = 0; t < rows; t++)
                memcpy(px + (size_t)t * q_half, x + (size_t)t * 2 * q_half + rank * q_half,
                       q_half * sizeof(float));
            CHECK(ds4_gpu_tensor_fill_f32(partial[rank], NAN, ob / 4));
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_dsv41_attention_output_tp_batch(partial[rank], low,
                model, a_bytes + b_bytes, 0, a_bytes, packed, rows, rank));
            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(partial[rank]);
            for (uint32_t t = 0; t < rows; t++) {
                CHECK(!memcmp(got_low + (size_t)t * low_half,
                    reference_low + (size_t)t * 2 * low_half + rank * low_half,
                    low_half * sizeof(float)));
                /* Independent double sums sample both ends and interior output
                 * rows, including rank one's strided input slice. */
                const uint32_t columns[] = {0, 1, 31, 64, 997, 4095, OUT - 1};
                for (size_t c = 0; c < sizeof(columns) / sizeof(*columns); c++) {
                    const uint32_t col = columns[c];
                    const q8_block *bw = (const q8_block *)((const char *)model + a_bytes) +
                        (size_t)col * 2 * low_half / 32 + rank * low_half / 32;
                    double sum = 0;
                    double rounding_bound = 0;
#ifndef __APPLE__
                    /* CUDA's established Q8 dot first rounds each activation
                     * block to signed bytes. TP must keep that same boundary. */
                    for (uint32_t k = 0; k < low_half; k += 32) {
                        const float *v = reference_low + (size_t)t * 2 * low_half + rank * low_half + k;
                        float amax = 0;
                        for (uint32_t j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(v[j]));
                        const float scale = amax / 127.0f;
                        const float inv = scale ? 1.0f / scale : 0;
                        int dot = 0;
                        for (uint32_t j = 0; j < 32; j++) {
                            dot += bw[k / 32].qs[j] * (int)lrintf(v[j] * inv);
                            /* The fast GPU reciprocal can approach an exact
                             * half-integer from the other side. Bound only
                             * those ambiguous bins, not arbitrary dot error. */
                            const double q = amax ? fabs((double)v[j] * 127.0 / amax) : 0;
                            if (q - floor(q) == 0.5)
                                rounding_bound += abs(bw[k / 32].qs[j]) * (double)scale / 128.0;
                        }
                        sum += dot * (double)scale / 128.0;
                    }
#else
                    for (uint32_t k = 0; k < low_half; k++)
                        sum += bw[k / 32].qs[k % 32] / 128.0 *
                            reference_low[(size_t)t * 2 * low_half + rank * low_half + k];
#endif
                    if (!isfinite(got[(size_t)t * OUT + col]) ||
                        fabs(got[(size_t)t * OUT + col] - sum) > rounding_bound + 2e-5 * (1 + fabs(sum)))
                        fprintf(stderr, "TP output rank=%u rows=%u row=%u col=%u actual=%.9g oracle=%.9g tie_bound=%.9g\n",
                            rank, rows, t, col, got[(size_t)t * OUT + col], sum, rounding_bound);
                    CHECK(isfinite(got[(size_t)t * OUT + col]) &&
                        fabs(got[(size_t)t * OUT + col] - sum) <= rounding_bound + 2e-5 * (1 + fabs(sum)));
                }
            }
            CHECK(isnan(got_low[(size_t)rows * low_half]));
            if (rows < ROWS) CHECK(isnan(got[(size_t)rows * OUT]));
        }
        const float *a = ds4_gpu_tensor_contents(partial[0]);
        const float *b = ds4_gpu_tensor_contents(partial[1]);
        for (size_t i = 0; i < (size_t)rows * OUT; i++)
            CHECK(isfinite(a[i]) && isfinite(b[i]) &&
                fabsf(a[i] + b[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        fprintf(stderr, "V4.1 TP Q8 output, both ranks, BF16 and double oracle, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, 1, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, packed, 1, 1));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, ROWS + 1, 0));
    ds4_gpu_tensor_free(partial[0]); ds4_gpu_tensor_free(partial[1]); ds4_gpu_tensor_free(packed);
#else
    fprintf(stderr, "V4.1 TP output projection: skipped (ROCm has no network TP)\n");
#endif
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(reference_low); free(model);
    return 1;
}

#ifdef __APPLE__
static int check_indexer_candidates(void) {
    enum { HEADS = 32, DIM = 128, MAX_BLOCKS = 2048 };
    const uint32_t shapes[] = {37, 8192, 32768, 32771, 65536, 262144};
    float query[HEADS * DIM], weights[HEADS];
    for (size_t i = 0; i < HEADS * DIM; i++) query[i] = bf16(random_value());
    for (size_t i = 0; i < HEADS; i++) weights[i] = bf16(random_value() / 8);
    ds4_gpu_tensor *q = upload(query, sizeof(query));
    ds4_gpu_tensor *w = upload(weights, sizeof(weights));
    CHECK(q && w);
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const uint32_t rows = shapes[shape], blocks = (rows + 7) / 8;
        const uint32_t count = blocks < MAX_BLOCKS ? blocks : MAX_BLOCKS;
        const uint32_t top = rows < 512 ? rows : 512;
        int32_t *ids = malloc(count * sizeof(*ids));
        float *mask = malloc(blocks * sizeof(*mask));
        float *keys = malloc((size_t)rows * DIM * sizeof(*keys));
        CHECK(ids && mask && keys);
        for (uint32_t i = 0; i < blocks; i++) mask[i] = -INFINITY;
        const uint32_t stride = blocks / count;
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t block = i + 1 == count ? blocks - 1 : i * stride;
            // Reverse order prevents compact positions from accidentally
            // replacing the global-ID tie ordering. Include the partial tail.
            ids[count - i - 1] = (int32_t)block;
            mask[block] = 0;
        }
        ds4_gpu_tensor *k = upload(NULL, (size_t)rows * DIM * sizeof(float));
        ds4_gpu_tensor *b = upload(ids, count * sizeof(*ids));
        ds4_gpu_tensor *m = upload(mask, blocks * sizeof(*mask));
        ds4_gpu_tensor *reference = upload(NULL, (size_t)rows * sizeof(float));
        ds4_gpu_tensor *storage = upload(NULL, (size_t)(rows + 4) * sizeof(float));
        ds4_gpu_tensor *candidate = ds4_gpu_tensor_view(storage, 0, (size_t)rows * sizeof(float));
        ds4_gpu_tensor *reference_ids = upload(NULL, top * sizeof(int32_t));
        ds4_gpu_tensor *candidate_ids = upload(NULL, top * sizeof(int32_t));
        CHECK(k && b && m && reference && storage && candidate && reference_ids && candidate_ids);
        float *guard = ds4_gpu_tensor_contents(storage);
        for (int pattern = 0; pattern < 2; pattern++) {
            for (size_t i = 0; i < (size_t)rows * DIM; i++)
                keys[i] = pattern ? 0.5f : bf16(random_value());
            CHECK(ds4_gpu_tensor_write(k, 0, keys, (size_t)rows * DIM * sizeof(float)));
            for (uint32_t i = rows; i < rows + 4; i++) guard[i] = -12345.0f;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_glm_indexer_score_one_tensor(reference, q, w, k, rows, HEADS, DIM, 1.0f / 64, false));
            CHECK(ds4_gpu_dsv41_candidate_filter(reference, m, rows, 1, rows - 1, 1, 0));
            CHECK(ds4_gpu_indexer_topk_tensor(reference_ids, reference, rows, 1, top));
            CHECK(ds4_gpu_dsv41_indexer_score_candidates(candidate, q, w, k, b, rows, count));
            CHECK(ds4_gpu_dsv41_candidate_filter(candidate, m, rows, 1, rows - 1, 1, 0));
            CHECK(ds4_gpu_indexer_topk_tensor(candidate_ids, candidate, rows, 1, top));
            CHECK(ds4_gpu_end_commands());
            const float *expected = ds4_gpu_tensor_contents(reference);
            const float *actual = ds4_gpu_tensor_contents(candidate);
            for (uint32_t row = 0; row < rows; row++) {
                if (mask[row / 8] == 0) {
                    CHECK(isfinite(actual[row]));
                    if (memcmp(&actual[row], &expected[row], sizeof(float)))
                        fprintf(stderr, "candidate score rows=%u pattern=%d row=%u actual=%.9g expected=%.9g\n",
                            rows, pattern, row, actual[row], expected[row]);
                    CHECK(!memcmp(&actual[row], &expected[row], sizeof(float)));
                } else CHECK(isinf(actual[row]) && actual[row] < 0);
            }
            CHECK(!memcmp(ds4_gpu_tensor_contents(candidate_ids), ds4_gpu_tensor_contents(reference_ids), top * sizeof(int32_t)));
            for (uint32_t i = rows; i < rows + 4; i++) CHECK(guard[i] == -12345.0f);
            fprintf(stderr, "V4.1 candidate scores rows=%u blocks=%u pattern=%s: bit-exact scores/IDs, tail guard PASS\n",
                rows, count, pattern ? "ties" : "finite");
            if (pattern || rows < 8192 || rows == 32771) continue;
            // Compare the full production score+filter+top-k chain. Alternate
            // order, warm both paths, and keep all timings out of correctness.
            double elapsed[2] = {0};
            for (int repeat = 0; repeat < 9; repeat++) for (int turn = 0; turn < 2; turn++) {
                const int mode = turn ^ (repeat & 1);
                const double start = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                CHECK(mode ? ds4_gpu_dsv41_indexer_score_candidates(candidate, q, w, k, b, rows, count) :
                    ds4_gpu_glm_indexer_score_one_tensor(candidate, q, w, k, rows, HEADS, DIM, 1.0f / 64, false));
                CHECK(ds4_gpu_dsv41_candidate_filter(candidate, m, rows, 1, rows - 1, 1, 0));
                CHECK(ds4_gpu_indexer_topk_tensor(candidate_ids, candidate, rows, 1, top));
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - start) * (1000.0 / 8);
            }
            fprintf(stderr, "V4.1 candidate score/filter/topk rows=%u candidates=%u full_ms=%.6f candidate_ms=%.6f\n",
                rows, count * 8, elapsed[0], elapsed[1]);
        }
        ds4_gpu_tensor_free(candidate_ids); ds4_gpu_tensor_free(reference_ids);
        ds4_gpu_tensor_free(candidate); ds4_gpu_tensor_free(storage);
        ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(m);
        ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(k);
        free(keys); free(mask); free(ids);
    }
    CHECK(!ds4_gpu_dsv41_indexer_score_candidates(q, q, w, q, q, 1, 0));
    ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(q);
    return 1;
}
#endif

static int check_indexer_batch(void) {
    enum { KEYS = 1025, ROWS = 33, HEADS = 32, DIM = 128 };
    ds4_gpu_tensor *qt = upload(NULL, (size_t)ROWS * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)KEYS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)ROWS * HEADS * sizeof(float));
    ds4_gpu_tensor *st = upload(NULL, (size_t)(ROWS * KEYS + 1) * sizeof(float));
    const uint64_t packed_bytes = ds4_gpu_dsv41_indexer_packed_bytes(KEYS, ROWS);
    ds4_gpu_tensor *pt = upload(NULL, (size_t)packed_bytes + 4);
    CHECK(qt && kt && wt && st && pt);
    ds4_gpu_tensor *reference = upload(NULL, (size_t)KEYS * sizeof(float));
    CHECK(reference);
    float *q = ds4_gpu_tensor_contents(qt), *k = ds4_gpu_tensor_contents(kt);
    float *w = ds4_gpu_tensor_contents(wt), *s = ds4_gpu_tensor_contents(st);
    CHECK(q && k && w && s);
    const bool packed_available = ds4_gpu_dsv41_tensor_ops_available();
    for (uint32_t range = 0; range < 3; range++) {
        for (size_t i = 0; i < (size_t)ROWS * HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)KEYS * DIM; i++)
            k[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, KEYS, DS4_V41_FP4_E8M0));
        if (range == 2) { q[0] = 1.0001f; k[67u * DIM] = 1.0003f; }
        uint32_t *packed_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(pt) + packed_bytes);
        *packed_guard = 0xabcdef01;
        if (packed_available) CHECK(ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS));
        CHECK(*packed_guard == 0xabcdef01);
        const uint32_t counts[] = {1, 7, 8, 9, 31, 32, 33};
        for (uint32_t ratio = 1; ratio <= 2; ratio++) {
            for (uint32_t early = 0; early <= 1; early++) {
                for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                    const uint32_t rows = counts[ci];
                    const uint32_t start = early ? 0 : KEYS * ratio - rows;
                    for (uint32_t mode = 0; mode < (packed_available ? 3u : 1u); mode++) {
                        const uint32_t offset = mode == 2 && rows < ROWS ? 1u : 0u;
                        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(qt, (uint64_t)offset * HEADS * DIM * 4u,
                            (uint64_t)rows * HEADS * DIM * 4u);
                        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wt, (uint64_t)offset * HEADS * 4u,
                            (uint64_t)rows * HEADS * 4u);
                        CHECK(qv && wv);
                        s[(size_t)rows * KEYS] = 12345;
                        CHECK(mode ? ds4_gpu_dsv41_indexer_scores_packed(st, qv, wv, kt, pt,
                            KEYS, rows, start, ratio, ROWS, offset) :
                            ds4_gpu_dsv41_indexer_scores_batch(st, qv, wv, kt, KEYS, rows, start, ratio));
                        CHECK(ds4_gpu_synchronize());
                        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv);
                        CHECK(s[(size_t)rows * KEYS] == 12345);
                        CHECK(*packed_guard == 0xabcdef01);
#ifdef __APPLE__
                        if (mode == 0 && rows >= 8) {
                            const size_t bytes = (size_t)rows * KEYS * sizeof(float);
                            float *mma = malloc(bytes);
                            CHECK(mma);
                            memcpy(mma, s, bytes);
                            CHECK(setenv("DS4_METAL_DISABLE_V41_INDEX_MMA", "1", 1) == 0);
                            qv = ds4_gpu_tensor_view(qt, 0, (uint64_t)rows * HEADS * DIM * 4u);
                            wv = ds4_gpu_tensor_view(wt, 0, (uint64_t)rows * HEADS * 4u);
                            const int ok = qv && wv && ds4_gpu_dsv41_indexer_scores_batch(st,
                                qv, wv, kt, KEYS, rows, start, ratio) && ds4_gpu_synchronize();
                            CHECK(unsetenv("DS4_METAL_DISABLE_V41_INDEX_MMA") == 0);
                            CHECK(ok && !memcmp(mma, s, bytes));
                            CHECK(s[(size_t)rows * KEYS] == 12345);
                            ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv); free(mma);
                        }
#endif
#ifndef __APPLE__
                        /* Compare the CUDA warp scorer with the original
                         * shared-memory reduction, not only a loose oracle. */
                        ds4_gpu_set_quality(true);
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            if (!visible) continue;
                            ds4_gpu_tensor *qr = ds4_gpu_tensor_view(qt,
                                (uint64_t)t * HEADS * DIM * 4u, HEADS * DIM * 4u);
                            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt,
                                (uint64_t)t * HEADS * 4u, HEADS * 4u);
                            CHECK(qr && wr && ds4_gpu_glm_indexer_score_one_tensor(
                                reference, qr, wr, kt, visible, HEADS, DIM, 1.0f / 64.0f, false));
                            CHECK(ds4_gpu_synchronize());
                            CHECK(!memcmp(s + (size_t)t * KEYS,
                                ds4_gpu_tensor_contents(reference), visible * sizeof(float)));
                            ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(wr);
                        }
                        ds4_gpu_set_quality(false);
#endif
                        /* The scalar public API was formerly tested only on
                         * CUDA. Check it against this same double oracle. */
                        const uint32_t scalar_rows = (start + 1u) / ratio;
                        const bool check_scalar = rows == 1 && mode == 0 && scalar_rows != 0;
                        if (check_scalar) {
                            CHECK(ds4_gpu_glm_indexer_score_one_tensor(reference,
                                qt, wt, kt, scalar_rows, HEADS, DIM, 1.0f / 64.0f, false));
                            CHECK(ds4_gpu_synchronize());
                        }
                        double worst = 0;
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            for (uint32_t j = 0; j < KEYS; j++) {
                                const float actual = s[(size_t)t * KEYS + j];
                                if (j >= visible) { CHECK(actual == -INFINITY); continue; }
                                double expected = 0, magnitude = 0;
                                for (uint32_t h = 0; h < HEADS; h++) {
                                    double dot = 0;
                                    for (uint32_t d = 0; d < DIM; d++)
                                        dot += (double)q[((size_t)(t + offset) * HEADS + h) * DIM + d] *
                                               k[(size_t)j * DIM + d];
                                    const double term = fmax(dot / 64.0, 0) * w[(t + offset) * HEADS + h];
                                    expected += term;
                                    magnitude += fabs(term);
                                }
                                CHECK(isfinite(actual));
                                const double error = fabs(actual - expected) / fmax(magnitude, 1);
                                worst = fmax(worst, error);
                                CHECK(error < 0.00001);
                                if (check_scalar) {
                                    const float *scalar = ds4_gpu_tensor_contents(reference);
                                    CHECK(isfinite(scalar[j]) && fabs(scalar[j] - expected) / fmax(magnitude, 1) < 0.00001);
                                }
                            }
                        }
                        fprintf(stderr, "V4.1 index scores mode=%u range=%u ratio=%u start=%u rows=%u error=%.9g\n",
                                mode, range, ratio, start, rows, worst);
                    }
                }
            }
        }
    }
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 4));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, UINT32_MAX, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS + 1, 0, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, KEYS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_pack(st, qt, kt, KEYS, ROWS));
    CHECK(!ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS + 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 0, ROWS, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, ROWS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, UINT32_MAX, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, st,
        KEYS, ROWS, 0, 1, ROWS, 0));
    ds4_gpu_tensor_free(pt);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 causal batched index scores: double-precision oracle PASS\n");
    return 1;
}

static double monotonic_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int check_embedding(void) {
    enum { VOCAB = 19, WIDTH = 5120, HC = 4, ROWS = 257 };
    const size_t row_bytes = WIDTH * HC * sizeof(float);
    const size_t weight_bytes = VOCAB * WIDTH * sizeof(uint16_t);
    uint16_t *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), weight_bytes) == 0);
    for (unsigned i = 0; i < VOCAB * WIDTH; i++)
        model[i] = (uint16_t)(0x2000u | (i % 1024u) | ((i & 1u) ? 0x8000u : 0));
    int32_t ids[ROWS];
    for (unsigned i = 0; i < ROWS; i++) ids[i] = (int32_t)((i * 7u) % VOCAB);
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *tokens = upload(ids, sizeof(ids));
    ds4_gpu_tensor *out = upload(NULL, ROWS * row_bytes + sizeof(float));
    ds4_gpu_tensor *ref = upload(NULL, VOCAB * row_bytes);
    CHECK(tokens && out && ref);
    for (unsigned token = 0; token < VOCAB; token++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(ref, token * row_bytes, row_bytes);
        CHECK(row && ds4_gpu_embed_token_hc_tensor(row, model, weight_bytes, 0,
                                                   VOCAB, token, WIDTH, HC));
        ds4_gpu_tensor_free(row);
    }
    const unsigned counts[] = {1, 7, 127, 128, ROWS};
    const float marker = 12345;
    float *actual = malloc(ROWS * row_bytes + sizeof(float));
    float *expected = malloc(VOCAB * row_bytes);
    CHECK(actual && expected && ds4_gpu_tensor_read(ref, 0, expected, VOCAB * row_bytes));
    for (unsigned i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        const unsigned rows = counts[i];
        CHECK(ds4_gpu_tensor_write(out, rows * row_bytes, &marker, sizeof(marker)));
        CHECK(ds4_gpu_embed_tokens_hc_tensor(out, tokens, model, weight_bytes, 0,
                                             VOCAB, rows, WIDTH, HC));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, rows * row_bytes + sizeof(float)));
        CHECK(actual[rows * WIDTH * HC] == marker);
        for (unsigned row = 0; row < rows; row++) {
            CHECK(!memcmp(actual + row * WIDTH * HC, expected + ids[row] * WIDTH * HC, row_bytes));
            for (unsigned hc = 0; hc < HC; hc++) for (unsigned d = 0; d < WIDTH; d++) {
                const unsigned w = (unsigned)ids[row] * WIDTH + d;
                const float value = (1.0f + (w % 1024u) / 1024.0f) / 128.0f * ((w & 1u) ? -1 : 1);
                CHECK(actual[(row * HC + hc) * WIDTH + d] == value);
            }
        }
        fprintf(stderr, "V4.1 batched embedding rows=%u: exact scalar and independent oracle PASS\n", rows);
    }
    free(expected); free(actual);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tokens);
    ds4_gpu_cleanup(); free(model);
    CHECK(ds4_gpu_init());
    return 1;
}

static int check_index_projection(void) {
    enum { MAX_ROWS = 4096 };
    const uint32_t widths[] = {1280, 5120, 5120, 512}, outputs[] = {4096, 32, 512, 128};
    const uint32_t counts[] = {1, 31, 64, 65, 513, 2048, MAX_ROWS};
    for (uint32_t shape = 0; shape < 4; shape++) {
        const uint32_t width = widths[shape], output = outputs[shape];
        const size_t weight_bytes = (size_t)width * output * sizeof(uint16_t);
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), weight_bytes) == 0);
        uint16_t *weights = model;
        for (size_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
            const int value = (int)(random_value() * 8192);
            weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) |
                                    (value < 0 ? 0x8000u : 0));
        }
        CHECK(ds4_gpu_set_model_map(model, weight_bytes));
        ds4_gpu_tensor *in = upload(NULL, (size_t)MAX_ROWS * width * sizeof(float));
        ds4_gpu_tensor *out = upload(NULL, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
        ds4_gpu_tensor *ref = upload(NULL, (size_t)MAX_ROWS * output * sizeof(float));
        CHECK(in && out && ref);
        float *x = ds4_gpu_tensor_contents(in), *y = ds4_gpu_tensor_contents(out);
        const float *expected = ds4_gpu_tensor_contents(ref);
        for (size_t i = 0; i < (size_t)MAX_ROWS * width; i++) x[i] = bf16(random_value());
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t rows = counts[ci];
            y[(size_t)rows * output] = 12345;
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < rows; t++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in, (size_t)t * width * 4, width * 4);
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)t * output * 4, output * 4);
                CHECK(xr && yr && ds4_gpu_matmul_f16_tensor(yr, model, weight_bytes, 0, width, output, xr, 1));
                ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
            }
            CHECK(ds4_gpu_end_commands());
            const double start = monotonic_seconds();
            CHECK(ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows, in));
            CHECK(ds4_gpu_synchronize());
            const double seconds = monotonic_seconds() - start;
            CHECK(y[(size_t)rows * output] == 12345);
            double error = 0, norm = 0;
            for (size_t i = 0; i < (size_t)rows * output; i++) {
                CHECK(isfinite(y[i]) && isfinite(expected[i]));
                const double d = (double)y[i] - expected[i];
                error += d * d; norm += (double)expected[i] * expected[i];
            }
            const double relative = sqrt(error / fmax(norm, 1e-30));
            fprintf(stderr, "V4.1 exact F16 projection width=%u output=%u rows=%u RMS=%.9g %.3f ms\n",
                width, output, rows, relative, seconds * 1000);
            CHECK(memcmp(y, expected, (size_t)rows * output * sizeof(float)) == 0);
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t o = j * (output / 8), t = rows - 1;
                double sum = 0, magnitude = 0;
                for (uint32_t k = 0; k < width; k++) {
                    const uint16_t w = weights[(size_t)o * width + k];
                    const float value = (1.0f + (w & 1023) / 1024.0f) / 128.0f *
                                        (w & 0x8000 ? -1 : 1);
                    const double term = (double)value * x[(size_t)t * width + k];
                    sum += term; magnitude += fabs(term);
                }
                CHECK(fabs(y[(size_t)t * output + o] - sum) < 0.000001 * fmax(magnitude, 1));
            }
        }
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes - 1, 0, width, output, 1, in));
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, MAX_ROWS + 1, in));
        ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in);
        ds4_gpu_cleanup(); free(model);
        CHECK(ds4_gpu_init());
    }
    return 1;
}

typedef struct { float value; uint32_t id; } topk_entry;

static int compare_topk_entry(const void *a, const void *b) {
    const topk_entry *x = a, *y = b;
    if (x->value > y->value) return -1;
    if (x->value < y->value) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int check_general_topk(void) {
    const uint32_t widths[] = {1, 2, 31, 127, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 3;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *selected = upload(NULL, ((size_t)width * rows + 1) * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && selected && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        uint32_t *ids = ds4_gpu_tensor_contents(selected);
        const uint32_t counts[] = {1, 2, 31, 128, 511, 512, 513, 2048, width};
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    const float value = pattern == 0 ? random_value() :
                        pattern == 1 ? (float)(i % 7) :
                        pattern == 2 ? -INFINITY : (i & 1 ? -0.0f : 0.0f);
                    const float v = pattern == 1 && i % 17 == 0 ? INFINITY : value;
                    s[t * width + i] = v;
                    reference[t * width + i] = (topk_entry){v, i};
                }
                qsort(reference + t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                if (k > width) continue;
                ids[rows * k] = 0xabcdef01;
                CHECK(ds4_gpu_indexer_topk_tensor(selected, scores, width, rows, k));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * k] == 0xabcdef01);
                for (uint32_t t = 0; t < rows; t++) {
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const uint32_t id = ids[t * k + i];
                        CHECK(id < width && !seen[id]);
                        seen[id] = true;
#ifdef __APPLE__
                        /* Metal's bitonic order may permute exactly tied keys. */
                        CHECK(s[t * width + id] == reference[t * width + i].value);
#else
                        if (ids[t * k + i] != reference[t * width + i].id) {
                            fprintf(stderr, "top-k width=%u k=%u pattern=%u row=%u rank=%u: %u vs %u\n",
                                width, k, pattern, t, i, ids[t * k + i], reference[t * width + i].id);
                            CHECK(0);
                        }
#endif
                    }
                }
#ifdef __APPLE__
                if ((k & (k - 1u)) == 0) {
                    CHECK(ds4_gpu_sort_i32_rows_asc_tensor(selected, selected, k, rows));
                    CHECK(ds4_gpu_synchronize());
                    CHECK(ids[rows * k] == 0xabcdef01);
                    for (uint32_t t = 0; t < rows; t++) for (uint32_t i = 1; i < k; i++)
                        CHECK(ids[t * k + i - 1] <= ids[t * k + i]);
                }
#endif
            }
        }
        fprintf(stderr, "V4.1 variable-k sort width=%u: independent sorted-value/unique-ID oracle PASS\n", width);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores); free(reference); free(seen);
    }
    return 1;
}

/* The radix-select path against the merge ladder: same values by rank, same
 * ids and order when the selected values are unique, ties by ascending id.
 * For k <= 512 the winners' 512-wide sort gives the 2048-wide sort's ids. */
static int check_topk_select(void) {
    const uint32_t widths[] = {16384, 65536, 131072, 262144};
    const uint32_t counts[] = {1, 512, 2048};
#ifdef __APPLE__
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
#endif
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 2;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *fast = upload(NULL, (size_t)2048 * rows * 4);
        ds4_gpu_tensor *slow = upload(NULL, (size_t)2048 * rows * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && fast && slow && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    /* 0: coarse random values with many ties; 1: unique; 2: unique
                     * with seven of eight blocks masked like the candidate filter;
                     * 3: fewer finite values than 2048, so ties at -inf are selected */
                    const float unique = (float)((i * 7919u + t * 1009u) % 1048573u) - 500000.0f;
                    s[(size_t)t * width + i] = pattern == 0 ? random_value() : pattern == 1 ? unique :
                        pattern == 2 ? (i / 8 % 8 ? -INFINITY : unique) : (i % 1361 ? -INFINITY : unique);
                    reference[(size_t)t * width + i] = (topk_entry){s[(size_t)t * width + i], i};
                }
                qsort(reference + (size_t)t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                const uint32_t n = ki == 1 ? 1u : rows;   /* the scratch changes row layout between calls */
                CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
#ifdef __APPLE__
                const uint64_t sort512 = ds4_gpu_test_kernel_runs("kernel_topk_select_sort512");
#endif
                CHECK(ds4_gpu_dsv41_indexer_topk_tensor(fast, scores, width, n, k));
#ifdef __APPLE__
                CHECK(ds4_gpu_synchronize() && ds4_gpu_test_kernel_runs("kernel_topk_select_sort512") == sort512 + (k <= 512));
                if (k <= 512) {
                    ds4_gpu_test_set_v41_off(DS4_GPU_V41_DECODE_TOPK);
                    const int ok = ds4_gpu_dsv41_indexer_topk_tensor(slow, scores, width, n, k) && ds4_gpu_synchronize();
                    ds4_gpu_test_set_v41_off(0);
                    CHECK(ok && ds4_gpu_test_kernel_runs("kernel_topk_select_sort512") == sort512 + 1);
                    CHECK(!memcmp(ds4_gpu_tensor_contents(fast), ds4_gpu_tensor_contents(slow), (size_t)n * k * 4));
                }
#endif
                CHECK(setenv("DS4_METAL_DISABLE_V41_TOPK_SELECT", "1", 1) == 0);
                CHECK(ds4_gpu_indexer_topk_tensor(slow, scores, width, n, k));
                CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
                CHECK(ds4_gpu_synchronize());
                const int32_t *a = ds4_gpu_tensor_contents(fast), *b = ds4_gpu_tensor_contents(slow);
                for (uint32_t t = 0; t < n; t++) {
                    const float *row = s + (size_t)t * width;
                    const topk_entry *ref = reference + (size_t)t * width;
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const int32_t id = a[t * k + i];
                        CHECK(id >= 0 && (uint32_t)id < width && !seen[id]);
                        seen[id] = true;
                        CHECK(row[id] == ref[i].value);
                        CHECK(row[id] == row[b[t * k + i]]);
                        if (i && row[id] == row[a[t * k + i - 1]]) CHECK(a[t * k + i - 1] < id);
                    }
                    if (pattern == 1 || pattern == 2)
                        CHECK(!memcmp(a + t * k, b + t * k, k * sizeof(*a)));
                }
            }
        }
        fprintf(stderr, "V4.1 radix top-k width=%u: values by rank, unique-ID and tie-order oracle PASS\n", width);
        ds4_gpu_tensor_free(scores); ds4_gpu_tensor_free(fast); ds4_gpu_tensor_free(slow);
        free(reference); free(seen);
    }
    return 1;
}

static int check_causal_topk(void) {
    const uint32_t frontiers[] = {513, 514, 1023, 1024, 1025, 1535, 2047, 2048, 4095, 4096, 8191, 8192,
                                  16383, 32767, 65535};
    const uint32_t counts[] = {1, 2, 31, 32, 33, 127, 128, 129};
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
    for (size_t fi = 0; fi < sizeof(frontiers) / sizeof(*frontiers); fi++) {
        const uint32_t start = frontiers[fi] * ratio - 1u;
        const uint32_t width = (start + 129u) / ratio + 1024u;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * 129u * sizeof(float));
        ds4_gpu_tensor *selected = upload(NULL, (129u * 512u + 1u) * sizeof(int32_t));
        ds4_gpu_tensor *reference = upload(NULL, 129u * 512u * sizeof(int32_t));
        CHECK(scores && selected && reference);
        float *s = ds4_gpu_tensor_contents(scores);
        int32_t *ids = ds4_gpu_tensor_contents(selected);
        const int32_t *ref = ds4_gpu_tensor_contents(reference);
        for (uint32_t pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < 129u; t++) {
                const uint32_t visible = (start + t + 1u) / ratio;
                for (uint32_t j = 0; j < width; j++) {
                    /* Future entries are deliberately attractive. Causality
                     * must come from each row's bounds, not score contents. */
                    float v = pattern == 0 ? random_value() : pattern == 1 ? (float)(j % 7u) : -INFINITY;
                    s[(size_t)t * width + j] = j < visible ? v : 12345.0f;
                }
            }
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const uint32_t rows = counts[ci];
                /* The causal kernels are the ladder's variants and agree with it
                 * tie for tie; the radix path orders ties by id, so the reference
                 * stays on the ladder. */
                CHECK(setenv("DS4_METAL_DISABLE_V41_TOPK_SELECT", "1", 1) == 0);
                CHECK(ds4_gpu_begin_commands());
                for (uint32_t t = 0; t < rows; t++) {
                    const uint32_t visible = (start + t + 1u) / ratio;
                    ds4_gpu_tensor *sr = ds4_gpu_tensor_view(scores, (size_t)t * width * 4u, visible * 4u);
                    ds4_gpu_tensor *out = ds4_gpu_tensor_view(reference, (size_t)t * 512u * 4u, 512u * 4u);
                    CHECK(sr && out && ds4_gpu_indexer_topk_tensor(out, sr, visible, 1, 512));
                    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(sr);
                }
                CHECK(ds4_gpu_end_commands());
                CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
                ids[rows * 512u] = 123456;
                CHECK(ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows, start, ratio));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * 512u] == 123456);
                for (uint32_t t = 0; t < rows; t++) {
                    CHECK(s[(size_t)t * width + width - 1u] == 12345.0f);
                    for (uint32_t k = 0; k < 512u; k++) {
                        if (ids[t * 512u + k] != ref[t * 512u + k]) {
                            fprintf(stderr, "top-k mismatch ratio=%u visible=%u rows=%u pattern=%u row=%u rank=%u: %d vs %d\n",
                                ratio, frontiers[fi], rows, pattern, t, k, ids[t * 512u + k], ref[t * 512u + k]);
                            CHECK(0);
                        }
                        CHECK(ids[t * 512u + k] >= 0 && (uint32_t)ids[t * 512u + k] < (start + t + 1u) / ratio);
                    }
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 130, start, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, UINT32_MAX, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 0, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 512u * ratio - 1u, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, start, 0));
        fprintf(stderr, "V4.1 causal top-k ratio=%u visible=%u: exact IDs including ties PASS\n", ratio, frontiers[fi]);
        ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores);
    }
    }
    return 1;
}


#ifdef __APPLE__
/* The V4.1 decode top-512 of one row (shuffle block sort, staged then fused merge) against itself without the fused
 * merge on random, tied with +inf, -inf-tailed, signed-zero and NaN rows, and that against the plain ladder (V4 Flash's
 * call) where no NaN is present: the same ids in the same order. */
static int check_decode_topk(void) {
    const uint32_t widths[] = {513, 1024, 1025, 10240, 10241, 16383, 65536};
    ds4_gpu_tensor *scores = upload(NULL, 65536u * sizeof(float)), *ids[3];
    for (int i = 0; i < 3; i++) ids[i] = upload(NULL, 513u * sizeof(int32_t));
    CHECK(scores && ids[0] && ids[1] && ids[2]);
    float *s = ds4_gpu_tensor_contents(scores);
    int32_t *a = ds4_gpu_tensor_contents(ids[0]), *b = ds4_gpu_tensor_contents(ids[1]);
    int32_t *c = ds4_gpu_tensor_contents(ids[2]);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    CHECK(setenv("DS4_METAL_DISABLE_V41_TOPK_SELECT", "1", 1) == 0);   /* 65536 stays off the radix path */
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi];
        for (uint32_t pattern = 0; pattern < 5; pattern++) {
            for (uint32_t i = 0; i < width; i++)
                s[i] = pattern == 0 ? random_value() : pattern == 1 ? (i % 17 == 0 ? INFINITY : (float)(i % 7)) :
                       pattern == 2 ? (i % 3 ? -INFINITY : random_value()) : pattern == 3 ? (i & 1 ? -0.0f : 0.0f) :
                       (i % 5 == 0 ? NAN : random_value());
            a[512] = b[512] = c[512] = 0x5eed;
            const uint64_t shuffle = ds4_gpu_test_kernel_runs("kernel_argsort_f32_i32_desc");
            const uint64_t fused = ds4_gpu_test_kernel_runs("kernel_argsort_merge_f32_i32_desc_prefix_fused");
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(ids[0], scores, width, 1, 512) && ds4_gpu_synchronize());
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_UNFUSED_TOPK);
            const int ok = ds4_gpu_dsv41_indexer_topk_tensor(ids[1], scores, width, 1, 512) && ds4_gpu_synchronize();
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok && ds4_gpu_indexer_topk_tensor(ids[2], scores, width, 1, 512) && ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_kernel_runs("kernel_argsort_f32_i32_desc") == shuffle + 3);
            CHECK(ds4_gpu_test_kernel_runs("kernel_argsort_merge_f32_i32_desc_prefix_fused") == fused + (width > 1024));
            CHECK(a[512] == 0x5eed && b[512] == 0x5eed && c[512] == 0x5eed);
            for (uint32_t k = 0; k < 512u; k++) {
                if (a[k] != b[k] || (pattern != 4 && b[k] != c[k])) {
                    fprintf(stderr, "decode top-k width=%u pattern=%u rank=%u: %d, unfused %d, ladder %d\n",
                            width, pattern, k, a[k], b[k], c[k]);
                    CHECK(0);
                }
            }
        }
    }
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
    ds4_gpu_tensor_free(scores);
    for (int i = 0; i < 3; i++) ds4_gpu_tensor_free(ids[i]);
    fprintf(stderr, "V4.1 decode top-512: fused == staged == plain ladder ids, ties included PASS\n");
    return 1;
}
#else
static int check_decode_topk(void) { return 1; }
#endif

/* Rows with at most 512 visible keys select every key; other slots stay. */
static int check_indexer_all(void) {
    const uint32_t rows = 129u;
    ds4_gpu_tensor *selected = upload(NULL, (rows * 512u + 1u) * sizeof(int32_t));
    CHECK(selected);
    int32_t *ids = ds4_gpu_tensor_contents(selected);
    CHECK(ids);
    for (uint32_t ratio = 1; ratio <= 2; ratio++) for (uint32_t start = 0; start <= 300u; start += 300u) {
        for (uint32_t i = 0; i <= rows * 512u; i++) ids[i] = -1;
        CHECK(ds4_gpu_dsv41_indexer_all_batch(selected, rows, start, ratio));
        CHECK(ds4_gpu_synchronize());
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t visible = (start + t + 1u) / ratio;
            for (uint32_t i = 0; i < 512u; i++)
                CHECK(ids[t * 512u + i] == (i < visible ? (int32_t)i : -1));
        }
        CHECK(ids[rows * 512u] == -1);
    }
    CHECK(!ds4_gpu_dsv41_indexer_all_batch(selected, rows, 512u, 1));
    CHECK(!ds4_gpu_dsv41_indexer_all_batch(selected, rows, 0, 0));
    ds4_gpu_tensor_free(selected);
    fprintf(stderr, "V4.1 indexer select-all rows: exact\n");
    return 1;
}

#ifdef __APPLE__
static int compare_score_entry(const void *a, const void *b) {
    const float *x = *(const float *const *)a, *y = *(const float *const *)b;
    return *x < *y ? 1 : *x > *y ? -1 : 0;
}

/* The per-key and the wide single-row index scoring kernels on the same random
 * query and keys: both against a double oracle, then their top-2048 sets, where
 * any difference must be a near-tie at the 2048th score. */
static int check_index_score_wide(void) {
    enum { HEADS = 32, DIM = 128, TOP = 2048, ROUNDS = 8 };
    const uint32_t counts[] = {4096, 131072};
    const uint32_t max_keys = counts[1];
    ds4_gpu_tensor *qt = upload(NULL, (size_t)HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)HEADS * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * sizeof(float));
    ds4_gpu_tensor *sd = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    ds4_gpu_tensor *sw = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    CHECK(qt && wt && kt && sd && sw);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt);
    float *direct = ds4_gpu_tensor_contents(sd), *wide = ds4_gpu_tensor_contents(sw);
    const float **order = malloc((size_t)max_keys * sizeof(*order));
    unsigned char *in_direct = calloc(max_keys, 1), *in_wide = calloc(max_keys, 1);
    CHECK(q && w && k && direct && wide && order && in_direct && in_wide);
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t keys = counts[ci];
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), (int)((i / 32) % 41) - 20);
        for (size_t i = 0; i < (size_t)keys * DIM; i++)
            k[i] = ldexpf(random_value(), (int)((i / 32) % 41) - 20);
        for (size_t i = 0; i < HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, keys, DS4_V41_FP4_E8M0));
        direct[keys] = wide[keys] = 12345;
        double seconds[2] = {0, 0};
        for (int mode = 0; mode < 2; mode++) {
            ds4_gpu_test_set_flags(mode ? DS4_GPU_TEST_INDEX_SCORE_WIDE : DS4_GPU_TEST_INDEX_SCORE_DIRECT);
            for (int round = 0; round < ROUNDS; round++) {
                const double t0 = monotonic_seconds();
                CHECK(ds4_gpu_glm_indexer_score_one_tensor(mode ? sw : sd, qt, wt, kt,
                    keys, HEADS, DIM, 1.0f / 64.0f, false));
                CHECK(ds4_gpu_synchronize());
                if (round) seconds[mode] += monotonic_seconds() - t0;
            }
        }
        ds4_gpu_test_set_flags(0);
        CHECK(direct[keys] == 12345 && wide[keys] == 12345);
        double worst[2] = {0, 0}, gap = 0;
        for (uint32_t j = 0; j < keys; j++) {
            double expected = 0, magnitude = 0;
            for (uint32_t h = 0; h < HEADS; h++) {
                double dot = 0;
                for (uint32_t d = 0; d < DIM; d++)
                    dot += (double)q[(size_t)h * DIM + d] * k[(size_t)j * DIM + d];
                const double term = fmax(dot / 64.0, 0) * w[h];
                expected += term;
                magnitude += fabs(term);
            }
            CHECK(isfinite(direct[j]) && isfinite(wide[j]));
            worst[0] = fmax(worst[0], fabs(direct[j] - expected) / fmax(magnitude, 1));
            worst[1] = fmax(worst[1], fabs(wide[j] - expected) / fmax(magnitude, 1));
            gap = fmax(gap, fabs((double)direct[j] - wide[j]) / fmax(magnitude, 1));
        }
        CHECK(worst[0] < 0.00001 && worst[1] < 0.00001);
        /* top-2048 membership of both kernels */
        for (uint32_t j = 0; j < keys; j++) order[j] = direct + j;
        qsort(order, keys, sizeof(*order), compare_score_entry);
        const float threshold = *order[TOP - 1];
        memset(in_direct, 0, keys);
        for (uint32_t i = 0; i < TOP; i++) in_direct[order[i] - direct] = 1;
        for (uint32_t j = 0; j < keys; j++) order[j] = wide + j;
        qsort(order, keys, sizeof(*order), compare_score_entry);
        memset(in_wide, 0, keys);
        for (uint32_t i = 0; i < TOP; i++) in_wide[order[i] - wide] = 1;
        uint32_t differing = 0;
        for (uint32_t j = 0; j < keys; j++) {
            if (in_direct[j] == in_wide[j]) continue;
            differing++;
            CHECK(fabs(direct[j] - threshold) <= 0.00001 * fmax(fabs(threshold), 1));
        }
        fprintf(stderr, "V4.1 index score wide keys=%u oracle error direct=%.3g wide=%.3g "
                "kernel gap=%.3g top-%d symmetric difference=%u (near-ties) "
                "us per call direct=%.0f wide=%.0f\n",
                keys, worst[0], worst[1], gap, TOP, differing,
                seconds[0] / (ROUNDS - 1) * 1e6, seconds[1] / (ROUNDS - 1) * 1e6);
    }
    free(in_wide); free(in_direct); free(order);
    ds4_gpu_tensor_free(sw); ds4_gpu_tensor_free(sd); ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 wide index score: double-precision oracle PASS\n");
    return 1;
}

/* Candidate-only index rows against the wide row they replace (wide scores,
 * candidate filter, radix top-512): the same ids in the same order, with
 * realistic and wide FP4 block scales, partial newest blocks and ties. */
static int check_index_candidates(void) {
    enum { HEADS = 32, DIM = 128, TOP = 512, ROUNDS = 6 };
    const uint32_t widths[] = {16385, 16391, 20007, 65536, 131072, 131075, 262147};
    const uint32_t max_keys = 262147, max_blocks = (max_keys + 7) / 8;
    ds4_gpu_tensor *qt = upload(NULL, (size_t)HEADS * DIM * 4), *wt = upload(NULL, HEADS * 4);
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * 4);
    ds4_gpu_tensor *full = upload(NULL, (size_t)max_keys * 4), *plain = upload(NULL, (size_t)max_keys * 4);
    ds4_gpu_tensor *source = upload(NULL, (size_t)max_keys * 4), *cand = upload(NULL, 16384 * 4);
    ds4_gpu_tensor *list = upload(NULL, 2048 * 4), *blocks_t = upload(NULL, (size_t)max_blocks * 4);
    ds4_gpu_tensor *picked = upload(NULL, 2048 * 4), *mask = upload(NULL, (size_t)max_blocks * 4);
    ds4_gpu_tensor *old_sel = upload(NULL, TOP * 4), *new_sel = upload(NULL, TOP * 4);
    CHECK(qt && wt && kt && full && plain && source && cand && list && blocks_t && picked && mask &&
          old_sel && new_sel);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt), *s = ds4_gpu_tensor_contents(source);
    const float *p = ds4_gpu_tensor_contents(plain), *c = ds4_gpu_tensor_contents(cand);
    const float *m = ds4_gpu_tensor_contents(mask);
    const uint32_t *l = ds4_gpu_tensor_contents(list);
    const int32_t *a = ds4_gpu_tensor_contents(old_sel), *b = ds4_gpu_tensor_contents(new_sel);
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, count = 16384 - (blocks * 8 - n);
        double seconds[2] = {0, 0};
        uint32_t ties = 0;
        for (int mode = 0; mode < 4; mode++) {
            /* bit 0: block exponents over +-20 instead of +-2; bit 1: ties */
            const int range = mode & 1 ? 20 : 2;
            for (size_t i = 0; i < (size_t)HEADS * DIM; i++)
                q[i] = ldexpf(random_value(), (int)((i / 32) % (2 * range + 1)) - range);
            for (size_t i = 0; i < (size_t)n * DIM; i++)
                k[i] = ldexpf(random_value(), (int)((i / 32 + n) % (2 * range + 1)) - range);
            for (size_t i = 0; i < HEADS; i++) w[i] = random_value() * 0.37139f;
            for (uint32_t i = 0; i < n; i++) s[i] = (float)((i * 7919u + mode) % 1048573u);
            CHECK(ds4_gpu_dsv41_quantize(qt, DIM, HEADS, DS4_V41_FP4_E8M0));
            CHECK(ds4_gpu_dsv41_quantize(kt, DIM, n, DS4_V41_FP4_E8M0));
            CHECK(ds4_gpu_synchronize());
            if (mode & 2) {
                /* copy the best key over 1/16 of all keys: the 512th score ties */
                CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(plain, qt, wt, kt, n, HEADS, DIM,
                    1.0f / 64.0f, false) && ds4_gpu_synchronize());
                uint32_t best = 0;
                for (uint32_t i = 1; i < n; i++) if (p[i] > p[best]) best = i;
                for (uint32_t i = 0; i < n; i += 16)
                    memcpy(k + (size_t)(i ^ 5) % n * DIM, k + (size_t)best * DIM, DIM * 4);
            }
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(blocks_t, source, n, 1, n - 1, 1));
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(picked, blocks_t, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(mask, picked, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(plain, qt, wt, kt, n, HEADS, DIM, 1.0f / 64.0f, false));
            CHECK(ds4_gpu_end_commands());
            ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
            for (int round = 0; round < ROUNDS; round++) {
                /* path 1 builds a zeroed list, 2 reuses it, 3 rebuilds it with the decode top-k off */
                for (int path = 0; path < 4; path++) {
                    const uint64_t reused = ds4_gpu_test_kernel_runs("candidate list reused");
                    if (path == 1 || path == 3) memset(ds4_gpu_tensor_contents(list), 0, 2048 * 4);
                    if (path == 3) ds4_gpu_test_set_v41_off(DS4_GPU_V41_DECODE_TOPK);
                    const double t0 = monotonic_seconds();
                    CHECK(ds4_gpu_begin_commands());
                    if (path) {
                        CHECK(ds4_gpu_dsv41_indexer_candidates_tensor(new_sel, cand, list, qt, wt, kt, mask,
                            n, 1.0f / 64.0f, TOP, path >= 2));
                    } else {
                        CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(full, qt, wt, kt, n, HEADS, DIM,
                            1.0f / 64.0f, false));
                        CHECK(ds4_gpu_dsv41_candidate_filter(full, mask, n, 1, n - 1, 1, 0));
                        CHECK(ds4_gpu_dsv41_indexer_topk_tensor(old_sel, full, n, 1, TOP));
                    }
                    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
                    ds4_gpu_test_set_v41_off(0);
                    if (round && path < 2) seconds[path] += monotonic_seconds() - t0;
                    CHECK(ds4_gpu_test_kernel_runs("candidate list reused") == reused + (path == 2));
                    if (path) CHECK(!memcmp(a, b, TOP * sizeof(*a)));
                }
            }
            /* the list is the mask's candidates in order; each score is the wide one */
            uint32_t at = 0;
            for (uint32_t j = 0; j < blocks; j++) if (m[j] == 0.0f) CHECK(at < 2048 && l[at++] == j);
            CHECK(at == 2048 && l[2047] == blocks - 1);
            for (uint32_t i = 0; i < count; i++) {
                const uint32_t key = l[i / 8] * 8 + i % 8;
                CHECK(key < n && !memcmp(c + i, p + key, sizeof(*c)));
            }
            for (uint32_t i = 0; i < TOP; i++) ties += i && p[a[i]] == p[a[i - 1]];
        }
        fprintf(stderr, "V4.1 candidate index rows keys=%u candidates=%u: ids, order and scores "
                "identical (%u tied neighbours) us per row full=%.0f candidates=%.0f\n",
                n, count, ties, seconds[0] / (4 * (ROUNDS - 1)) * 1e6, seconds[1] / (4 * (ROUNDS - 1)) * 1e6);
    }
    {   /* the engine's record (the list was built on this mask): a row with another mask rebuilds the list */
        const uint32_t n = max_keys, blocks = max_blocks;
        ds4_gpu_tensor *masks[2] = {mask, upload(NULL, (size_t)max_blocks * 4)};
        CHECK(masks[1]);
        int32_t ref[2][TOP];
        for (int i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < n; j++) s[j] = (float)((j * (7919u + 104u * (uint32_t)i) + (uint32_t)i) % 1048573u);
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(blocks_t, source, n, 1, n - 1, 1));
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(picked, blocks_t, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(masks[i], picked, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(full, qt, wt, kt, n, HEADS, DIM, 1.0f / 64.0f, false));
            CHECK(ds4_gpu_dsv41_candidate_filter(full, masks[i], n, 1, n - 1, 1, 0));
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(old_sel, full, n, 1, TOP));
            CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
            memcpy(ref[i], a, sizeof(ref[i]));
        }
        CHECK(memcmp(ref[0], ref[1], sizeof(ref[0])));
        const ds4_gpu_tensor *record = NULL;
        const int steps[] = {0, 0, 1, 1, 0};
        for (size_t i = 0; i < sizeof(steps) / sizeof(*steps); i++) {
            const ds4_gpu_tensor *m_i = masks[steps[i]];
            const uint64_t reused = ds4_gpu_test_kernel_runs("candidate list reused");
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_indexer_candidates_tensor(new_sel, cand, list, qt, wt, kt, m_i, n, 1.0f / 64.0f, TOP,
                record == m_i));
            CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_kernel_runs("candidate list reused") == reused + (record == m_i));
            CHECK(!memcmp(b, ref[steps[i]], sizeof(ref[0])));
            record = m_i;
        }
        ds4_gpu_tensor_free(masks[1]);
    }
    CHECK(!ds4_gpu_dsv41_indexer_candidates_tensor(new_sel, cand, list, qt, wt, kt, mask,
        16384, 1.0f / 64.0f, TOP, 0));
    ds4_gpu_tensor *tensors[] = {qt, wt, kt, full, plain, source, cand, list, blocks_t, picked, mask,
                                 old_sel, new_sel};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    return 1;
}

/* The bf16 key view and its scorers against the wide scorer on the f32 cache: the same bits for every key. Keys pass
 * the FP4_E8M0 quantizer with block scales spread over 41 exponents and over 12 (where summation order shows in the
 * bits); the view converts in two pieces at an odd boundary. The decode row entry reads the view where the wide
 * scorer runs and scores the f32 keys without it. */
static int check_index_score_bf16(void) {
    enum { HEADS = 32, DIM = 128 };
    const uint32_t counts[] = {8200, 65535, 65536, 98303, 131077};
    const uint32_t max_keys = 131077;
    const char *kernels[] = {"kernel_dsv41_indexer_score_rows_split4_bf16", "kernel_dsv41_indexer_score_rows_split2_bf16",
                             "kernel_dsv41_indexer_score_bf16"};
    ds4_gpu_tensor *qt = upload(NULL, (size_t)HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)HEADS * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * sizeof(float));
    ds4_gpu_tensor *vt = upload(NULL, (size_t)max_keys * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *sw = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    ds4_gpu_tensor *sb = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    ds4_gpu_tensor *ft = upload(NULL, sizeof(uint32_t));
    CHECK(qt && wt && kt && vt && sw && sb && ft);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt);
    float *wide = ds4_gpu_tensor_contents(sw), *fast = ds4_gpu_tensor_contents(sb);
    uint32_t *flag = ds4_gpu_tensor_contents(ft);
    for (int spread = 0; spread < 2; spread++) {
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++)
            q[i] = spread == 0 ? ldexpf(random_value(), (int)((i / 32) % 41) - 20) :
                                 ldexpf(random_value(), (int)((i / 32 * 7) % 13) - 6);
        for (size_t i = 0; i < HEADS; i++) w[i] = bf16(random_value());
        for (size_t i = 0; i < (size_t)max_keys * DIM; i++)
            k[i] = spread == 0 ? ldexpf(random_value(), (int)((i / 32) % 41) - 20) :
                                 ldexpf(random_value(), (int)((i / 32 * 5 + i / DIM) % 13) - 6);
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, max_keys, DS4_V41_FP4_E8M0));
        *flag = 0;
        CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 0, 77777, ft));
        CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 77777, max_keys - 77777, ft));
        CHECK(ds4_gpu_synchronize() && *flag == 0);
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t keys = counts[ci];
            const char *kernel = kernels[keys < 10240u ? 0 : keys < 65536u ? 1 : 2];
            wide[keys] = fast[keys] = 12345;
            ds4_gpu_test_set_flags(DS4_GPU_TEST_INDEX_SCORE_WIDE);
            CHECK(ds4_gpu_glm_indexer_score_one_tensor(sw, qt, wt, kt, keys, HEADS, DIM, 1.0f / 64.0f, false));
            ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
            for (int view = 1; view >= 0; view--) {
                const uint64_t runs = ds4_gpu_test_kernel_runs(kernel);
                memset(fast, 0xff, (size_t)keys * sizeof(float));
                CHECK(ds4_gpu_dsv41_indexer_score_row(sb, qt, wt, kt, view ? vt : NULL, keys) && ds4_gpu_synchronize());
                CHECK(ds4_gpu_test_kernel_runs(kernel) == runs + (uint64_t)view);
                CHECK(!memcmp(wide, fast, (size_t)keys * sizeof(float)) && fast[keys] == 12345);
            }
            uint32_t positive = 0, rounded = 0;
            for (uint32_t j = 0; j < keys; j++) positive += wide[j] > 0;
            /* sensitivity: keys whose f32 score is not the exact sum, so order shows */
            for (uint32_t j = 0; ci == 0 && j < 2048; j++) {
                double expected = 0;
                for (uint32_t h = 0; h < HEADS; h++) {
                    double dot = 0;
                    for (uint32_t d = 0; d < DIM; d++)
                        dot += (double)q[(size_t)h * DIM + d] * k[(size_t)j * DIM + d];
                    expected += fmax(dot / 64.0, 0) * w[h];
                }
                rounded += (double)wide[j] != expected;
            }
            CHECK(wide[keys] == 12345 && positive > keys / 16 && (ci != 0 || rounded > 0));
        }
    }
    /* a key that is not bf16-exact raises the flag */
    uint32_t bits;
    memcpy(&bits, &k[(size_t)123456 * DIM + 77], 4);
    bits |= 1u;
    memcpy(&k[(size_t)123456 * DIM + 77], &bits, 4);
    CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 123456, 1, ft) && ds4_gpu_synchronize() && *flag != 0);
    ds4_gpu_tensor_free(ft); ds4_gpu_tensor_free(sb); ds4_gpu_tensor_free(sw);
    ds4_gpu_tensor_free(vt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 bf16 index score (split4, split2, full) == wide scorer bits PASS\n");
    return 1;
}

/* The FP8 copy reader (ds4_gpu_matmul_e4m3_bf16_tensor), with its blend kernels and with the narrow readers, gives
 * the BF16-rounded F16 projection's bits at the Engram (6144 -> 256, two-row blend) and indexer q_b (1280 -> 4096,
 * block-owner blend) shapes: power-of-two 32x32 block scales 2^-15..2^-8 (every weight exact in F16), normal and
 * small codes. The BF16 router's blend reader (5120 -> 384) gives the narrow reader's bits. */
static int check_e4m3_copies(void) {
    const uint32_t ins[2] = {6144, 1280}, outs[2] = {256, 4096};
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int which = 0; which < 2; which++) {
        const uint32_t IN = ins[which], OUT = outs[which];
        const char *blend = which ? "kernel_e4m3_blend_owner_r2_bf16" : "kernel_e4m3_blend_r2_bf16";
        const size_t f16_bytes = (size_t)IN * OUT * 2, code_bytes = (size_t)IN * OUT;
        const size_t nscale = (size_t)(IN / 32) * (OUT / 32), page = (size_t)getpagesize();
        const size_t map_bytes = (f16_bytes + page - 1) / page * page;
        unsigned char *map = NULL, *codes = malloc(code_bytes);
        float *scales = malloc(nscale * sizeof(float)), *x = malloc((size_t)IN * sizeof(float));
        CHECK(posix_memalign((void **)&map, page, map_bytes) == 0 && codes && scales && x);
        memset(map, 0, map_bytes);
        _Float16 *w16 = (_Float16 *)map;
        CHECK(ds4_gpu_set_model_map(map, map_bytes));
        for (uint32_t i = 0; i < IN; i++) x[i] = random_value() / 4;
        ds4_gpu_tensor *xt = upload(x, (size_t)IN * sizeof(float));
        ds4_gpu_tensor *o16 = upload(NULL, OUT * 4), *o8 = upload(NULL, OUT * 4), *o8n = upload(NULL, OUT * 4);
        CHECK(xt && o16 && o8 && o8n);
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 0; i < nscale; i++) { (void)random_value(); scales[i] = ldexpf(1.0f, -15 + (int)(seed % 8u)); }
            for (size_t i = 0; i < code_bytes; i++) {
                (void)random_value();
                unsigned char c = (unsigned char)(seed >> 3);
                if ((c & 0x7f) == 0x7f) c ^= 1;   /* no NaN codes */
                if (pass) c = (unsigned char)(c & 0x87);   /* subnormal and small codes */
                codes[i] = c;
            }
            for (size_t r = 0; r < OUT; r++)
                for (size_t k = 0; k < IN; k++) {
                    const unsigned char c = codes[r * IN + k], a = c & 0x7f;
                    const uint32_t ex = (a >> 3) & 15u, m = a & 7u;
                    float v = a == 0 ? 0.0f : ex == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, (int)ex - 7);
                    if (c & 0x80) v = -v;
                    w16[r * IN + k] = (_Float16)(v * scales[(r / 32) * (IN / 32) + k / 32]);
                }
            ds4_gpu_tensor *ct = upload(codes, code_bytes), *st = upload(scales, nscale * sizeof(float));
            CHECK(ct && st);
            const uint64_t runs = ds4_gpu_test_kernel_runs(blend);
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_project_f16_bf16(o16, map, map_bytes, 0, IN, OUT, xt));
            CHECK(ds4_gpu_matmul_e4m3_bf16_tensor(o8, ct, st, IN, OUT, xt));
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_PLAIN_READERS);
            const int ok = ds4_gpu_matmul_e4m3_bf16_tensor(o8n, ct, st, IN, OUT, xt);
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok && ds4_gpu_end_commands() && ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_kernel_runs(blend) == runs + 1);
            CHECK(!memcmp(ds4_gpu_tensor_contents(o16), ds4_gpu_tensor_contents(o8), OUT * sizeof(float)));
            CHECK(!memcmp(ds4_gpu_tensor_contents(o16), ds4_gpu_tensor_contents(o8n), OUT * sizeof(float)));
            ds4_gpu_tensor_free(ct); ds4_gpu_tensor_free(st);
        }
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(o16); ds4_gpu_tensor_free(o8); ds4_gpu_tensor_free(o8n);
        free(codes); free(scales); free(x);   /* map stays: the model views may wrap it */
    }
    {   /* BF16 router 5120 -> 384 */
        const uint32_t in = 5120, out = 384;
        ds4_gpu_tensor *wt = upload(NULL, (size_t)in * out * 2), *xt = upload(NULL, (size_t)in * 4), *o = upload(NULL, out * 4);
        CHECK(wt && xt && o);
        uint16_t *w = ds4_gpu_tensor_contents(wt);
        float *x = ds4_gpu_tensor_contents(xt);
        for (size_t i = 0; i < (size_t)in * out; i++) {
            const float v = random_value() / 64;
            uint32_t bits;
            memcpy(&bits, &v, 4);
            w[i] = (uint16_t)(bits >> 16);
        }
        for (uint32_t i = 0; i < in; i++) x[i] = random_value();
        LITERAL_PAIR(DS4_GPU_V41_PLAIN_READERS, "kernel_router_blend_r1", o, out * 4,
                     ds4_gpu_matmul_bf16_weights_tensor(o, wt, in, out, xt));
        ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(o);
    }
    fprintf(stderr, "V4.1 FP8 copy reader (blend and narrow) == F16 projection; router blend == narrow PASS\n");
    return 1;
}
#else
static int check_index_score_wide(void) { return 1; }
static int check_index_candidates(void) { return 1; }
static int check_index_score_bf16(void) { return 1; }
static int check_e4m3_copies(void) { return 1; }
#endif

static int check_compact_carry(void) {
    const uint32_t widths[] = {1, 31, 32, 33, 127, 128, 129, 20480};
    const uint32_t rows = 129, offset = 2;
    for (uint32_t format = 0; format <= DS4_V41_CARRY_MASK; format++) {
        for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
            const uint32_t width = widths[wi];
            const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1) / 2 : (width + 31) / 32;
            const size_t count = (size_t)width * rows;
            const size_t packed_bytes = (size_t)(rows + 4) * words * sizeof(uint32_t);
            ds4_gpu_tensor *plain = upload(NULL, (count + 1) * sizeof(float));
            ds4_gpu_tensor *packed = upload(NULL, packed_bytes);
            uint32_t *expected = malloc(count * sizeof(uint32_t));
            CHECK(plain && packed && expected);
            uint32_t *bits = ds4_gpu_tensor_contents(plain);
            unsigned char *storage = ds4_gpu_tensor_contents(packed);
            for (size_t i = 0; i < count; i++) {
                const uint32_t b = (uint32_t)(i * 40503u + 32768u) & 0xffffu;
                expected[i] = format == DS4_V41_CARRY_BF16 ? b << 16 : (b & 1u ? 0xff800000u : 0u);
            }
            memcpy(bits, expected, count * sizeof(uint32_t));
            bits[count] = 0x12345678u;
            memset(storage, 0xa5, packed_bytes);
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, true));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)));
            memset(bits, 0, count * sizeof(uint32_t));
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, false));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)) && bits[count] == 0x12345678u);
            for (size_t i = 0; i < (size_t)offset * words * 4u; i++) CHECK(storage[i] == 0xa5);
            for (size_t i = (size_t)(offset + rows) * words * 4u; i < packed_bytes; i++)
                CHECK(storage[i] == 0xa5);
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, UINT32_MAX, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset + 3, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, 0, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, 0, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, DS4_V41_CARRY_F32, true));
            fprintf(stderr, "V4.1 compact carry format=%u width=%u: exact round-trip and bounds PASS\n", format, width);
            free(expected);
            ds4_gpu_tensor_free(plain);
            ds4_gpu_tensor_free(packed);
        }
    }
    return 1;
}

/* Exercise the full-head and compact TP layouts with the same causal keys.
 * Selected rows are shuffled; include masked future rows at odd frontiers. */
static int check_tp_attention(void) {
    enum { D = 512, H = 64, K = 512, C = 2048 };
    const struct { uint32_t rows, selected; } cases[] = {
        {1, K}, {31, K}, {32, K}, {33, K}, {129, K}, {257, K}, {2048, K},
        {1, 0}, {1, 1}, {2, K}, {6, K}, {8, K},
        {2, 0}, {2, 1}, {6, 0}
    };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    for (unsigned s = 0; s < sizeof(cases) / sizeof(*cases); s++) {
        const uint32_t n = cases[s].rows, nr = n + 127, start = 3073;
        const uint32_t ratio = 1u + s % 2u;
        const size_t nq = (size_t)n * H * D;
        float *q = malloc(nq * 4), *actual = malloc(nq * 4);
        float *compact = malloc(nq * 2), *part = malloc(nq * 2);
        float *raw = malloc((size_t)nr * D * 4), *comp = malloc(C * D * 4);
        int32_t *ids = malloc((size_t)n * K * 4);
        CHECK(q && actual && compact && part && raw && comp && ids);
        for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < nr * D; i++) raw[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t j = 0; j < K; j++)
                ids[t * K + j] = j < cases[s].selected &&
                    (cases[s].selected != K || j % 29)
                    ? (int32_t)((j * 127u + t * 17u) % C) : -1;
        ds4_gpu_tensor *qt = upload(q, nq * 4), *rt = upload(raw, (size_t)nr * D * 4);
        ds4_gpu_tensor *ct = upload(comp, C * D * 4), *it = upload(ids, (size_t)n * K * 4);
        ds4_gpu_tensor *out = upload(NULL, nq * 4), *qp = upload(NULL, nq * 2);
        ds4_gpu_tensor *op = upload(NULL, nq * 2);
        CHECK(qt && rt && ct && it && out && qp && op);
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
            0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, nq * 4));
        double max_split = 0, max_oracle = 0;
        for (uint32_t rank = 0; rank < 2; rank++) {
            for (uint32_t t = 0; t < n; t++)
                memcpy(compact + (size_t)t * H/2 * D,
                    q + ((size_t)t * H + rank * H/2) * D, H/2 * D * 4);
            CHECK(ds4_gpu_tensor_write(qp, 0, compact, nq * 2));
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
            CHECK(ds4_gpu_tensor_read(op, 0, part, nq * 2));
#ifdef __APPLE__
            if (n > 1) {
                ds4_gpu_set_quality(true); /* Existing eight-head kernel. */
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                    rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
                CHECK(!memcmp(part, ds4_gpu_tensor_contents(op), nq * 2));
                ds4_gpu_set_quality(false);
            }
#endif
            if (n == 2048 && rank == 0) {
                const bool controls[] = {true, false, false, true};
                for (unsigned pass = 0; pass < 4; pass++) {
                    ds4_gpu_set_quality(controls[pass]);
                    const double begin = monotonic_seconds();
                    for (int repeat = 0; repeat < 4; repeat++)
                        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op,
                            sinks, getpagesize(), 0, qp, rt, ct, 0, it, n, start,
                            nr, nr, 0, C, K, 128, ratio, H/2, D));
                    CHECK(ds4_gpu_synchronize());
                    fprintf(stderr, "TP indexed attention control=%u %.3f ms\n",
                        controls[pass], (monotonic_seconds() - begin) * 250);
                }
                ds4_gpu_set_quality(false);
            }
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H/2; h++) {
                    const float *got = part + ((size_t)t * H/2 + h) * D;
                    const float *ref = actual + ((size_t)t * H + rank * H/2 + h) * D;
                    for (int d = 0; d < D; d++) {
                        const double err = fabs((double)got[d] - ref[d]);
                        max_split = fmax(max_split, err);
                        CHECK(isfinite(got[d]) && err < 3e-5 * (1 + fabs(ref[d])));
                    }
                    if ((t != 0 && t != n/2 && t + 1 != n) || h % 15) continue;
                    const float *query = compact + ((size_t)t * H/2 + h) * D;
                    double logits[128 + K + 1], values[128 + K + 1];
                    const uint32_t col = (t * 71u + h * 19u) % D;
                    unsigned count = 0;
                    for (uint32_t j = 0; j < 128 + K; j++) {
                        const int32_t id = j < 128 ? (int32_t)(t + j) : ids[t * K + j - 128];
                        if (j >= 128 && (uint32_t)id >= (start + t + 1) / ratio) continue;
                        const float *key = (j < 128 ? raw : comp) + (size_t)id * D;
                        double dot = 0;
                        for (int d = 0; d < D; d++) dot += (double)query[d] * key[d];
                        logits[count] = dot / sqrt(512.0);
                        values[count++] = key[col];
                    }
                    logits[count] = sinks[rank * H/2 + h]; values[count++] = 0;
                    double max = -INFINITY, den = 0, num = 0;
                    for (unsigned j = 0; j < count; j++) max = fmax(max, logits[j]);
                    for (unsigned j = 0; j < count; j++) {
                        const double p = exp(logits[j] - max);
                        den += p; num += p * values[j];
                    }
                    const double err = fabs(got[col] - num / den);
                    max_oracle = fmax(max_oracle, err);
                    CHECK(err < 3e-5 * (1 + fabs(num / den)));
                }
            }
        }
        fprintf(stderr, "V4.1 TP attention rows=%u selected=%u ratio=%u max split=%g oracle=%g: PASS\n",
            n, cases[s].selected, ratio, max_split, max_oracle);
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(ct);
        ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(qp);
        ds4_gpu_tensor_free(op);
        free(q); free(actual); free(compact); free(part); free(raw); free(comp); free(ids);
    }
    ds4_gpu_cleanup();
    free(sinks);
    return 1;
}


#ifdef __APPLE__
/* Compare the packed half4 attention reader with independently expanded rows.
 * The raw ring and selected IDs are identical for every comparison, including
 * the row order chosen by split decode or chronological prefill. */
static int check_fp4_attention(uint32_t benchmark_rows) {
    enum { D = 512, H = 64, K = 512, PACKED_ROW = 288 };
    const bool benchmark = benchmark_rows != 0;
    const struct fp4_attention_case { uint32_t rows, heads; bool quality; } checks[] = {
        {1, 64, false}, /* Split decode. */
        {1, 32, true},  /* Sixteen-row decode staging. */
        {2, 64, false}, {33, 32, false}, /* Dual-head prefill. */
        {2, 32, true}, /* Ordinary eight-head prefill. */
        {3, 64, false} /* Start-zero prefill: raw windows of one to three rows. */
    }, bench[] = {{benchmark_rows, 64, false}};
    const struct fp4_attention_case *cases = benchmark ? bench : checks;
    const size_t case_count = benchmark ? sizeof(bench) / sizeof(*bench) : sizeof(checks) / sizeof(*checks);
    const uint32_t comp_rows = benchmark ? 8192 : 640;
    static const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const char *flag = "DS4_METAL_V41_ATTN_UNIT_SCALE";
    const char *previous = getenv(flag);
    char *saved = previous ? strdup(previous) : NULL;
    CHECK(!previous || saved);
    const char *rb4_flag = "DS4_METAL_V41_PREFILL_ATTN_RB4";
    const char *previous_rb4 = getenv(rb4_flag);
    char *saved_rb4 = previous_rb4 ? strdup(previous_rb4) : NULL;
    CHECK(!previous_rb4 || saved_rb4);
    CHECK(setenv(rb4_flag, "1", 1) == 0);
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (unsigned h = 0; h < H; h++) sinks[h] = (float)(h % 7) * .125f - 2;
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));

    uint8_t *packed = calloc(comp_rows, PACKED_ROW);
    float *expanded = malloc((size_t)comp_rows * D * sizeof(float));
    CHECK(packed && expanded);
    for (uint32_t row = 0; row < comp_rows; row++) {
        uint8_t *p = packed + row * PACKED_ROW;
        for (uint32_t group = 0; group < D / 16; group++) {
            /* All finite nonnegative E4M3 scale codes, including zero and
             * subnormals, occur alongside every signed E2M1 nibble. */
            const uint32_t scale_code = (row * 32 + group) % 127;
            const float scale = scale_code < 8 ? ldexpf(scale_code, -9) :
                ldexpf(1.0f + (scale_code & 7) * .125f, (int)(scale_code >> 3) - 7);
            p[256 + group] = (uint8_t)scale_code;
            for (uint32_t i = 0; i < 16; i++) {
                const uint32_t col = group * 16 + i, code = (i + row) & 15;
                p[col / 2] |= (uint8_t)(code << (4 * (col & 1)));
                const float magnitude = levels[code & 7] * scale;
                uint32_t bits;
                memcpy(&bits, &magnitude, sizeof(bits));
                bits |= (code & 8) << 28;
                memcpy(expanded + row * D + col, &bits, sizeof(bits));
            }
        }
    }
    CHECK(main_fp4_bits(expanded[0]) == 0);
    CHECK(main_fp4_bits(expanded[8]) == 0x80000000u);
    ds4_gpu_tensor *ct = upload(expanded, (size_t)comp_rows * D * sizeof(float));
    ds4_gpu_tensor *pt = upload(packed, (size_t)comp_rows * PACKED_ROW);
    CHECK(ct && pt);
    for (size_t s = 0; s < case_count; s++) {
        const uint32_t n = cases[s].rows, heads = cases[s].heads;
        const bool initial = !benchmark && n == 3;
        const uint32_t nr = initial ? n : n + 127, cap = nr + 7;
        const uint32_t raw_start = initial ? cap - 2 : 11;
        const uint32_t start = initial ? 0 : benchmark ? 24576 : 2301;
        const size_t nq = (size_t)n * heads * D;
        float *q = malloc(nq * sizeof(float));
        float *raw = malloc((size_t)cap * D * sizeof(float));
        float *reference = malloc(nq * sizeof(float));
        int32_t *ids = malloc((size_t)n * K * sizeof(int32_t));
        CHECK(q && raw && reference && ids);
        for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 8192);
        for (size_t i = 0; i < (size_t)cap * D; i++) raw[i] = bf16(random_value() / 16);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t j = 0; j < K; j++)
                /* Timing scans 512 distinct visible keys; correctness cases
                 * retain their negative and future selections. */
                ids[t * K + j] = benchmark ? (int32_t)((j * 127 + t * 17) % (start / 4)) :
                    j < 4 ? (int32_t)j : j % 29 == 0 ? -1 :
                    (int32_t)((j * 127 + t * 17) % comp_rows);
        ds4_gpu_tensor *qt = upload(q, nq * sizeof(float));
        ds4_gpu_tensor *rt = upload(raw, (size_t)cap * D * sizeof(float));
        ds4_gpu_tensor *it = upload(ids, (size_t)n * K * sizeof(int32_t));
        ds4_gpu_tensor *out = upload(NULL, nq * sizeof(float));
        CHECK(qt && rt && it && out);
        ds4_gpu_set_quality(cases[s].quality);
        CHECK(setenv(flag, "0", 1) == 0);
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
            0, qt, rt, ct, 0, it, n, start, nr, cap, raw_start, comp_rows, K, 128, 4, heads, D));
        CHECK(ds4_gpu_tensor_read(out, 0, reference, nq * sizeof(float)));
        for (size_t i = 0; i < nq; i++) CHECK(isfinite(reference[i]));
        const bool candidate_path = n > 1 && !cases[s].quality && ds4_metal_attention_batch_mpp();
        for (unsigned unit_scale = 0; unit_scale < 2; unit_scale++) {
            CHECK(setenv(flag, unit_scale ? "1" : "0", 1) == 0);
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
                0, qt, rt, pt, 2, it, n, start, nr, cap, raw_start, comp_rows, K, 128, 4, heads, D));
            CHECK(ds4_gpu_synchronize());
            const float *actual = ds4_gpu_tensor_contents(out);
            CHECK(actual);
            if (unit_scale && candidate_path) {
                size_t changed = 0, first = nq;
                double max_abs = 0, squared_error = 0;
                for (size_t i = 0; i < nq; i++) {
                    CHECK(isfinite(actual[i]));
                    if (main_fp4_bits(reference[i]) != main_fp4_bits(actual[i])) {
                        if (first == nq) first = i;
                        changed++;
                    }
                    const double error = (double)actual[i] - reference[i];
                    max_abs = fmax(max_abs, fabs(error));
                    squared_error += error * error;
                }
                fprintf(stderr, "FP4 attention rows=%u heads=%u quality=%u unit_scale=1 approximate measured: changed=%zu/%zu max_abs=%.9g rms=%.9g\n",
                    n, heads, cases[s].quality, changed, nq, max_abs, sqrt(squared_error / nq));
                if (first != nq)
                    fprintf(stderr, "  first mismatch index=%zu expected=%08x actual=%08x\n",
                        first, main_fp4_bits(reference[first]), main_fp4_bits(actual[first]));
                continue;
            }
            if (memcmp(reference, actual, nq * sizeof(float))) {
                size_t first = 0;
                while (main_fp4_bits(reference[first]) == main_fp4_bits(actual[first])) first++;
                fprintf(stderr, "FP4 attention rows=%u heads=%u quality=%u unit_scale=%u index=%zu expected=%08x actual=%08x\n",
                    n, heads, cases[s].quality, unit_scale, first,
                    main_fp4_bits(reference[first]), main_fp4_bits(actual[first]));
                CHECK(!memcmp(reference, actual, nq * sizeof(float)));
            }
            fprintf(stderr, "FP4 attention rows=%u heads=%u quality=%u unit_scale=%u %s exact: PASS\n",
                n, heads, cases[s].quality, unit_scale, unit_scale ? "fallback" : "packed control");
        }
        if (benchmark) {
            /* The comparisons above also warm both attention paths. Time the
             * complete indexed-attention calls, including sorting, encoding
             * and completion; this is not isolated shader GPU time. */
            const unsigned repetitions = 4;
            puts("pair,arm,rows,heads,repetitions,seconds");
            for (unsigned pair = 1; pair <= 5; pair++) {
                for (unsigned order = 0; order < 2; order++) {
                    const unsigned unit_scale = pair % 2 ? order : 1 - order;
                    CHECK(setenv(flag, unit_scale ? "1" : "0", 1) == 0);
                    CHECK(ds4_gpu_synchronize());
                    const double begin = monotonic_seconds();
                    for (unsigned repeat = 0; repeat < repetitions; repeat++)
                        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
                            0, qt, rt, pt, 2, it, n, start, nr, cap, raw_start, comp_rows, K, 128, 4, heads, D));
                    CHECK(ds4_gpu_synchronize());
                    const double seconds = monotonic_seconds() - begin;
                    printf("%u,%s,%u,%u,%u,%.9f\n", pair, unit_scale ? "candidate" : "baseline",
                        n, heads, repetitions, seconds);
                    fflush(stdout);
                }
            }
        }
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(rt);
        ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(out);
        free(q); free(raw); free(reference); free(ids);
    }
    CHECK((saved ? setenv(flag, saved, 1) : unsetenv(flag)) == 0);
    free(saved);
    CHECK((saved_rb4 ? setenv(rb4_flag, saved_rb4, 1) : unsetenv(rb4_flag)) == 0);
    free(saved_rb4);
    ds4_gpu_set_quality(false);
    ds4_gpu_tensor_free(ct); ds4_gpu_tensor_free(pt);
    ds4_gpu_cleanup();
    free(sinks); free(packed); free(expanded);
    return 1;
}

static float unpack_index_fp4(const uint8_t *row, uint32_t column) {
    static const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const uint8_t code = (row[column / 2] >> (4 * (column & 1))) & 15;
    const uint32_t scale_bits = (uint32_t)row[64 + column / 32] << 23;
    const uint32_t sign_bits = (code & 8) ? 0x80000000u : 0;
    float scale, sign;
    memcpy(&scale, &scale_bits, 4); memcpy(&sign, &sign_bits, 4);
    return bf16(copysignf(levels[code & 7], sign) * scale);
}

static float adjacent_bf16(float value, int delta) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += delta * 0x10000;
    memcpy(&value, &bits, 4);
    return value;
}

static const uint32_t index_fp4_shapes[] = {37, 128, 256, 1024, 2048, 4096, 8192, 32768, 32771, 65536, 262144};

static int encode_index_score(ds4_gpu_tensor *scores, ds4_gpu_tensor *q, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *rounded, ds4_gpu_tensor *packed, ds4_gpu_tensor *blocks,
        uint32_t rows, uint32_t count, int sparse, int mode) {
    if (mode >= 2) return ds4_gpu_dsv41_indexer_score_simd(scores, q, weights,
        mode & 1 ? packed : rounded, sparse ? blocks : NULL, rows, sparse ? count : 0, mode & 1);
    if (mode == 1) return ds4_gpu_dsv41_indexer_score_fp4(scores, q, weights,
        packed, sparse ? blocks : NULL, rows, sparse ? count : 0);
    return sparse ? ds4_gpu_dsv41_indexer_score_candidates(scores, q, weights, rounded, blocks, rows, count) :
        ds4_gpu_glm_indexer_score_one_tensor(scores, q, weights, rounded, rows, 32, 128, 1.0f / 64, false);
}

static int check_index_simd(ds4_gpu_tensor *q, ds4_gpu_tensor *weights, ds4_gpu_tensor *rounded,
        ds4_gpu_tensor *packed, ds4_gpu_tensor *blocks, ds4_gpu_tensor *sparse_mask,
        uint32_t rows, uint32_t count, int ties) {
    enum { CALLS = 8, REPEATS = 4, MODES = 16 };
    const uint32_t top = rows < 512 ? rows : 512, nblocks = (rows + 7) / 8;
    float *zeros = calloc(nblocks, sizeof(float));
    ds4_gpu_tensor *dense_mask = upload(zeros, (size_t)nblocks * 4);
    ds4_gpu_tensor *reference = upload(NULL, (size_t)rows * 4);
    ds4_gpu_tensor *storage = upload(NULL, (size_t)(rows + 8) * 4);
    ds4_gpu_tensor *actual = ds4_gpu_tensor_view(storage, 16, (size_t)rows * 4);
    ds4_gpu_tensor *reference_ids = upload(NULL, top * 4), *actual_ids = upload(NULL, top * 4);
    CHECK(zeros && dense_mask && reference && storage && actual && reference_ids && actual_ids);
    float *guard = ds4_gpu_tensor_contents(storage);
    CHECK(guard);
    for (unsigned i = 0; i < 4; i++) guard[i] = guard[rows + 4 + i] = -12345.0f;
    for (int sparse = 0; sparse < 2; sparse++) {
        ds4_gpu_tensor *mask = sparse ? sparse_mask : dense_mask;
        CHECK(ds4_gpu_begin_commands());
        CHECK(encode_index_score(reference, q, weights, rounded, packed, blocks, rows, count, 0, 0));
        CHECK(ds4_gpu_dsv41_candidate_filter(reference, mask, rows, 1, rows - 1, 1, 0));
        CHECK(ds4_gpu_indexer_topk_tensor(reference_ids, reference, rows, 1, top));
        CHECK(ds4_gpu_end_commands());
        for (int mode = 0; mode < 4; mode++) {
            CHECK(ds4_gpu_begin_commands());
            CHECK(encode_index_score(actual, q, weights, rounded, packed, blocks, rows, count, sparse, mode));
            CHECK(ds4_gpu_dsv41_candidate_filter(actual, mask, rows, 1, rows - 1, 1, 0));
            CHECK(ds4_gpu_indexer_topk_tensor(actual_ids, actual, rows, 1, top));
            CHECK(ds4_gpu_end_commands());
            const float *a = ds4_gpu_tensor_contents(actual), *b = ds4_gpu_tensor_contents(reference);
            const int32_t *ai = ds4_gpu_tensor_contents(actual_ids), *bi = ds4_gpu_tensor_contents(reference_ids);
            size_t changed = 0, changed_ids = 0;
            double max_abs = 0, max_rel = 0;
            for (uint32_t row = 0; row < rows; row++) if (memcmp(a + row, b + row, 4)) {
                changed++;
                const double error = fabs((double)a[row] - b[row]);
                max_abs = fmax(max_abs, error);
                max_rel = fmax(max_rel, error / fmax(fabs((double)b[row]), 1e-30));
            }
            for (uint32_t i = 0; i < top; i++) changed_ids += ai[i] != bi[i];
            fprintf(stderr, "index SIMD rows=%u %s %s %s pattern=%s changed_scores=%zu changed_ids=%zu max_abs=%.9g max_rel=%.9g\n",
                rows, sparse ? "sparse" : "dense", mode >= 2 ? "simd" : "old", mode & 1 ? "fp4" : "f32",
                ties ? "ties" : "finite", changed, changed_ids, max_abs, max_rel);
            CHECK(changed == 0 && changed_ids == 0);
            for (unsigned i = 0; i < 4; i++) CHECK(guard[i] == -12345.0f && guard[rows + 4 + i] == -12345.0f);
        }
    }
    if (!ties) {
        double elapsed[MODES] = {0};
        for (int repeat = 0; repeat <= REPEATS; repeat++) for (int turn = 0; turn < MODES; turn++) {
            const int point = (turn + repeat * 5) % MODES;
            const int mode = point % 4, sparse = (point / 4) % 2, full = point / 8;
            ds4_gpu_tensor *mask = sparse ? sparse_mask : dense_mask;
            if (getenv("DS4_METAL_CB_TIMES")) {
                fprintf(stderr, "index-simd sample rows=%u repeat=%d selection=%s implementation=%s dtype=%s scope=%s calls=%d\n",
                    rows, repeat, sparse ? "sparse" : "dense", mode >= 2 ? "simd" : "old",
                    mode & 1 ? "fp4" : "f32", full ? "score_filter_topk" : "score", CALLS);
                fflush(stderr);
            }
            const double start = monotonic_seconds();
            CHECK(ds4_gpu_begin_commands());
            for (int call = 0; call < CALLS; call++) {
                CHECK(encode_index_score(actual, q, weights, rounded, packed, blocks, rows, count, sparse, mode));
                if (full) {
                    CHECK(ds4_gpu_dsv41_candidate_filter(actual, mask, rows, 1, rows - 1, 1, 0));
                    CHECK(ds4_gpu_indexer_topk_tensor(actual_ids, actual, rows, 1, top));
                }
            }
            CHECK(ds4_gpu_end_commands());
            if (repeat) elapsed[point] += (monotonic_seconds() - start) * (1000.0 / (REPEATS * CALLS));
        }
        for (int point = 0; point < MODES; point++)
            printf("%u,%s,%s,%s,%s,%.9f\n", rows, (point / 4) % 2 ? "sparse" : "dense",
                point % 4 >= 2 ? "simd" : "old", point & 1 ? "fp4" : "f32",
                point / 8 ? "score_filter_topk" : "score", elapsed[point]);
        fflush(stdout);
    }
    ds4_gpu_tensor_free(actual_ids); ds4_gpu_tensor_free(reference_ids);
    ds4_gpu_tensor_free(actual); ds4_gpu_tensor_free(storage); ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(dense_mask); free(zeros);
    return 1;
}

static int check_index_fp4_batch(void) {
    enum { KEYS = 1025, ROWS = 33, HEADS = 32, DIM = 128, TOP = 16 };
    const size_t query_bytes = (size_t)ROWS * HEADS * DIM * 4;
    const size_t key_bytes = (size_t)KEYS * DIM * 4;
    const size_t score_bytes = (size_t)ROWS * KEYS * 4;
    ds4_gpu_tensor *q = upload(NULL, query_bytes), *keys = upload(NULL, key_bytes);
    ds4_gpu_tensor *weights = upload(NULL, (size_t)ROWS * HEADS * 4);
    ds4_gpu_tensor *packed = upload(NULL, KEYS * 68);
    ds4_gpu_tensor *reference = upload(NULL, score_bytes);
    ds4_gpu_tensor *storage = upload(NULL, score_bytes + 32);
    ds4_gpu_tensor *actual = ds4_gpu_tensor_view(storage, 16, score_bytes);
    ds4_gpu_tensor *reference_ids = upload(NULL, ROWS * TOP * 4), *actual_ids = upload(NULL, ROWS * TOP * 4);
    const uint64_t workspace_bytes = ds4_gpu_dsv41_indexer_packed_bytes(KEYS, ROWS);
    ds4_gpu_tensor *old_workspace = upload(NULL, workspace_bytes + 4), *workspace = upload(NULL, workspace_bytes + 4);
    CHECK(q && keys && weights && packed && reference && storage && actual && reference_ids && actual_ids && old_workspace && workspace);
    float *queries = ds4_gpu_tensor_contents(q), *key_values = ds4_gpu_tensor_contents(keys);
    float *weight_values = ds4_gpu_tensor_contents(weights), *reference_values = ds4_gpu_tensor_contents(reference);
    for (size_t i = 0; i < query_bytes / 4; i++) queries[i] = ldexpf(random_value(), (int)((i / 32) % 21) - 10);
    for (size_t i = 0; i < key_bytes / 4; i++) key_values[i] = ldexpf(random_value(), (int)((i / 32) % 21) - 10);
    for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) weight_values[i] = bf16(random_value());
    uint8_t *guard = ds4_gpu_tensor_contents(storage);
    memset(guard, 0xa5, score_bytes + 32);
    memset(ds4_gpu_tensor_contents(workspace), 0, workspace_bytes + 4);
    memset(ds4_gpu_tensor_contents(old_workspace), 0, workspace_bytes + 4);
    uint32_t *workspace_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(workspace) + workspace_bytes);
    uint32_t *old_workspace_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(old_workspace) + workspace_bytes);
    *workspace_guard = *old_workspace_guard = 0xabcdef01;
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_quantize(q, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
    CHECK(ds4_gpu_dsv41_indexer_pack_fp4(packed, keys, KEYS));
    CHECK(ds4_gpu_dsv41_quantize(keys, DIM, KEYS, DS4_V41_FP4_E8M0));
    CHECK(ds4_gpu_end_commands());
    const bool tensors = ds4_gpu_dsv41_tensor_ops_available();
    if (tensors) {
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_indexer_pack(old_workspace, q, keys, KEYS, ROWS));
        CHECK(ds4_gpu_dsv41_indexer_prepare_fp4(workspace, q, packed, KEYS, ROWS));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(old_workspace), ds4_gpu_tensor_contents(workspace), workspace_bytes));
    }
    const uint32_t counts[] = {1, 7, 8, 33};
    for (uint32_t ratio = 1; ratio <= 2; ratio++) for (uint32_t early = 0; early <= 1; early++) {
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t rows = counts[ci], offset = rows < ROWS ? 1 : 0;
            const uint32_t start = early ? 0 : KEYS * ratio - rows;
            ds4_gpu_tensor *qv = ds4_gpu_tensor_view(q, (uint64_t)offset * HEADS * DIM * 4, (uint64_t)rows * HEADS * DIM * 4);
            ds4_gpu_tensor *wv = ds4_gpu_tensor_view(weights, (uint64_t)offset * HEADS * 4, (uint64_t)rows * HEADS * 4);
            CHECK(qv && wv);
            for (uint32_t mode = 0; mode < (tensors ? 2u : 1u); mode++) {
                uint32_t *actual_words = ds4_gpu_tensor_contents(actual);
                if (rows < ROWS) actual_words[(size_t)rows * KEYS] = 0xabcdef02;
                CHECK(ds4_gpu_begin_commands());
                if (mode) CHECK(ds4_gpu_dsv41_indexer_scores_packed(reference, qv, wv, keys, old_workspace,
                    KEYS, rows, start, ratio, ROWS, offset));
                else for (uint32_t token = 0; token < rows; token++) {
                    ds4_gpu_tensor *qr = ds4_gpu_tensor_view(qv, (uint64_t)token * HEADS * DIM * 4, HEADS * DIM * 4);
                    ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wv, (uint64_t)token * HEADS * 4, HEADS * 4);
                    ds4_gpu_tensor *sr = ds4_gpu_tensor_view(reference, (uint64_t)token * KEYS * 4, KEYS * 4);
                    CHECK(qr && wr && sr && ds4_gpu_glm_indexer_score_one_tensor(sr, qr, wr, keys, KEYS, HEADS, DIM, 1.0f / 64, false));
                    ds4_gpu_tensor_free(sr); ds4_gpu_tensor_free(wr); ds4_gpu_tensor_free(qr);
                }
                CHECK(ds4_gpu_dsv41_indexer_scores_batch_fp4(actual, qv, wv, packed, mode ? workspace : NULL,
                    KEYS, rows, start, ratio, ROWS, offset));
                CHECK(ds4_gpu_end_commands());
                if (!mode) for (uint32_t token = 0; token < rows; token++)
                    for (uint32_t row = (start + token + 1) / ratio; row < KEYS; row++)
                        reference_values[(size_t)token * KEYS + row] = -INFINITY;
                CHECK(!memcmp(ds4_gpu_tensor_contents(actual), reference_values, (size_t)rows * KEYS * 4));
                if (rows < ROWS) CHECK(actual_words[(size_t)rows * KEYS] == 0xabcdef02);
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_indexer_topk_tensor(reference_ids, reference, KEYS, rows, TOP));
                CHECK(ds4_gpu_indexer_topk_tensor(actual_ids, actual, KEYS, rows, TOP));
                CHECK(ds4_gpu_end_commands());
                CHECK(!memcmp(ds4_gpu_tensor_contents(actual_ids), ds4_gpu_tensor_contents(reference_ids), rows * TOP * 4));
                for (size_t i = 0; i < 16; i++) CHECK(guard[i] == 0xa5 && guard[16 + score_bytes + i] == 0xa5);
                CHECK(*workspace_guard == 0xabcdef01 && *old_workspace_guard == 0xabcdef01);
                fprintf(stderr, "FP4 batch index mode=%s rows=%u start=%u ratio=%u offset=%u scores+IDs exact: PASS\n",
                    mode ? "tensor" : "simd", rows, start, ratio, offset);
            }
            ds4_gpu_tensor_free(wv); ds4_gpu_tensor_free(qv);
        }
    }
    ds4_gpu_tensor_free(workspace); ds4_gpu_tensor_free(old_workspace);
    ds4_gpu_tensor_free(actual_ids); ds4_gpu_tensor_free(reference_ids);
    ds4_gpu_tensor_free(actual); ds4_gpu_tensor_free(storage); ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(packed); ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(keys); ds4_gpu_tensor_free(q);
    return 1;
}

static int check_index_fp4(int simd, uint32_t selected_rows) {
    enum { HEADS = 32, DIM = 128, MAX_BLOCKS = 2048 };
    const float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const float ties[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5};
    const int exponents[] = {-120, -10, 0, 10, 80};
    float query[HEADS * DIM], weights[HEADS];
    for (size_t i = 0; i < HEADS * DIM; i++) query[i] = bf16(random_value());
    for (size_t i = 0; i < HEADS; i++) weights[i] = bf16(random_value() / 8);
    ds4_gpu_tensor *q = upload(query, sizeof(query)), *w = upload(weights, sizeof(weights));
    CHECK(q && w && ds4_gpu_dsv41_quantize(q, DIM, HEADS, DS4_V41_FP4_E8M0));
    for (size_t shape = 0; shape < sizeof(index_fp4_shapes) / sizeof(index_fp4_shapes[0]); shape++) {
        const uint32_t rows = index_fp4_shapes[shape], nblocks = (rows + 7) / 8;
        if (selected_rows && rows != selected_rows) continue;
        const uint32_t count = nblocks < MAX_BLOCKS ? nblocks : MAX_BLOCKS;
        const uint32_t top = rows < 512 ? rows : 512;
        const size_t bytes = (size_t)rows * DIM * sizeof(float), packed_bytes = (size_t)rows * 68;
        float *values = malloc(bytes), *mask = malloc(nblocks * sizeof(float));
        int32_t *ids = malloc(count * sizeof(int32_t));
        CHECK(values && mask && ids);
        for (uint32_t row = 0; row < rows; row++) for (uint32_t col = 0; col < DIM; col++) {
            float value = bf16(ldexpf(random_value(), (int)((col / 32 + row) % 17) - 8));
            if (rows == 37) {
                if (row == 0) value = copysignf(levels[col & 7], (col & 8) ? -1.0f : 1.0f);
                if (row == 1) {
                    value = adjacent_bf16(ties[col % 7], (int)((col / 7) % 3) - 1);
                    if (col & 16) value = -value;
                    if (col % 32 == 31) value = 6;
                }
                if (row >= 2 && row < 7) {
                    const float anchor = ldexpf(6, exponents[row - 2]);
                    value = col % 32 == 31 ? adjacent_bf16(anchor, (int)(col / 32) - 1) : anchor / 4;
                }
                if (row == 7) value = copysignf(0.0f, (col & 1) ? -1.0f : 1.0f);
                if (row == 8) value = copysignf(ldexpf(1.0f, (int)(col % 8) - 133), (col & 1) ? -1.0f : 1.0f);
                if (row == 9 || row == 10) value = 0.5f; // Exact cross-row score ties.
            }
            values[(size_t)row * DIM + col] = value;
        }
        for (uint32_t i = 0; i < nblocks; i++) mask[i] = -INFINITY;
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t block = i + 1 == count ? nblocks - 1 : i * (nblocks / count);
            ids[count - i - 1] = block; mask[block] = 0;
        }
        ds4_gpu_tensor *original = upload(values, bytes), *rounded = upload(values, bytes);
        ds4_gpu_tensor *storage = upload(NULL, packed_bytes + 32);
        ds4_gpu_tensor *packed = ds4_gpu_tensor_view(storage, 16, packed_bytes);
        ds4_gpu_tensor *blocks = upload(ids, count * sizeof(*ids)), *mask_tensor = upload(mask, nblocks * sizeof(float));
        ds4_gpu_tensor *reference = upload(NULL, (size_t)rows * 4), *actual = upload(NULL, (size_t)rows * 4);
        ds4_gpu_tensor *reference_ids = upload(NULL, top * 4), *actual_ids = upload(NULL, top * 4);
        CHECK(original && rounded && storage && packed && blocks && mask_tensor && reference && actual && reference_ids && actual_ids);
        uint8_t *raw = ds4_gpu_tensor_contents(storage);
        memset(raw, 0xa5, packed_bytes + 32);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_quantize(rounded, DIM, rows, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_indexer_pack_fp4(packed, original, rows));
        CHECK(ds4_gpu_end_commands());
        const uint8_t *data = ds4_gpu_tensor_contents(packed);
        const float *expected_keys = ds4_gpu_tensor_contents(rounded);
        for (uint32_t row = 0; row < rows; row++) for (uint32_t col = 0; col < DIM; col++) {
            const float value = unpack_index_fp4(data + (size_t)row * 68, col);
            const float expected = expected_keys[(size_t)row * DIM + col];
            if (memcmp(&value, &expected, 4)) {
                fprintf(stderr, "FP4 roundtrip rows=%u row=%u col=%u got=%a expected=%a sf=%u code=%u\n",
                    rows, row, col, value, expected, data[(size_t)row * 68 + 64 + col / 32],
                    (data[(size_t)row * 68 + col / 2] >> (4 * (col & 1))) & 15);
                CHECK(0);
            }
        }
        for (size_t i = 0; i < 16; i++) CHECK(raw[i] == 0xa5 && raw[16 + packed_bytes + i] == 0xa5);
        if (rows == 37) {
            for (uint32_t col = 0; col < DIM; col++)
                CHECK(((data[col / 2] >> (4 * (col & 1))) & 15) == (col & 15));
            for (uint32_t group = 0; group < 4; group++) CHECK(data[64 + group] == 127);
            // row4 uses e=0: group0 is below6, group1 exactly6, group2 above6.
            CHECK(data[4 * 68 + 64] == 127 && data[4 * 68 + 65] == 127 && data[4 * 68 + 66] == 128);
        }
        // Old checkpoint payloads contain rounded F32 values, not original
        // scales. Repacking may change codes/scales but must preserve all bits.
        ds4_gpu_tensor *imported = upload(NULL, packed_bytes);
        ds4_gpu_tensor *decoded_storage = upload(NULL, bytes + 32);
        ds4_gpu_tensor *decoded = ds4_gpu_tensor_view(decoded_storage, 16, bytes);
        CHECK(imported && decoded_storage && decoded);
        uint8_t *decoded_raw = ds4_gpu_tensor_contents(decoded_storage);
        memset(decoded_raw, 0xa5, bytes + 32);
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_indexer_unpack_fp4(decoded, packed, rows));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(decoded), expected_keys, bytes));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_indexer_pack_fp4(imported, rounded, rows));
        CHECK(ds4_gpu_dsv41_indexer_unpack_fp4(decoded, imported, rows));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(decoded), expected_keys, bytes));
        for (size_t i = 0; i < 16; i++) CHECK(decoded_raw[i] == 0xa5 && decoded_raw[16 + bytes + i] == 0xa5);
        // Exercise a nonzero68B source row offset, as bounded state I/O does.
        ds4_gpu_tensor *packed_range = ds4_gpu_tensor_view(imported, 3 * 68, 7 * 68);
        ds4_gpu_tensor *plain_range = ds4_gpu_tensor_view(decoded, 3 * DIM * 4, 7 * DIM * 4);
        CHECK(packed_range && plain_range && ds4_gpu_dsv41_indexer_unpack_fp4(plain_range, packed_range, 7));
        CHECK(ds4_gpu_synchronize());
        CHECK(!memcmp(ds4_gpu_tensor_contents(decoded), expected_keys, bytes));
        ds4_gpu_tensor_free(plain_range); ds4_gpu_tensor_free(packed_range);
        ds4_gpu_tensor_free(decoded); ds4_gpu_tensor_free(decoded_storage); ds4_gpu_tensor_free(imported);
        fprintf(stderr, "FP4 index rounded-state import/unpack rows=%u bit exact incl signedzero/ranges: PASS\n", rows);
        if (simd) {
            CHECK(check_index_simd(q, w, rounded, packed, blocks, mask_tensor, rows, count, 0));
            for (size_t i = 0; i < (size_t)rows * DIM; i++) values[i] = 0.5f;
            CHECK(ds4_gpu_tensor_write(original, 0, values, bytes));
            CHECK(ds4_gpu_tensor_write(rounded, 0, values, bytes));
            CHECK(ds4_gpu_dsv41_indexer_pack_fp4(packed, original, rows));
            CHECK(ds4_gpu_synchronize());
            CHECK(check_index_simd(q, w, rounded, packed, blocks, mask_tensor, rows, count, 1));
        } else for (int sparse = 0; sparse < 2; sparse++) {
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_glm_indexer_score_one_tensor(reference, q, w, rounded, rows, HEADS, DIM, 1.0f / 64, false));
            CHECK(ds4_gpu_dsv41_candidate_filter(reference, mask_tensor, rows, 1, rows - 1, 1, 0));
            CHECK(ds4_gpu_indexer_topk_tensor(reference_ids, reference, rows, 1, top));
            CHECK(ds4_gpu_dsv41_indexer_score_fp4(actual, q, w, packed, sparse ? blocks : NULL, rows, sparse ? count : 0));
            CHECK(ds4_gpu_dsv41_candidate_filter(actual, mask_tensor, rows, 1, rows - 1, 1, 0));
            CHECK(ds4_gpu_indexer_topk_tensor(actual_ids, actual, rows, 1, top));
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(ds4_gpu_tensor_contents(actual), ds4_gpu_tensor_contents(reference), (size_t)rows * 4));
            const float *scores = ds4_gpu_tensor_contents(actual);
            for (uint32_t row = 0; row < rows; row++) if (mask[row / 8] == 0) CHECK(isfinite(scores[row]));
            CHECK(!memcmp(ds4_gpu_tensor_contents(actual_ids), ds4_gpu_tensor_contents(reference_ids), top * 4));
        }
        fprintf(stderr, "FP4 index rows=%u bytes=%zu/%zu: exact roundtrip/dense/candidate scores+IDs, offset guards PASS\n", rows, packed_bytes, bytes);
        if (!simd && rows >= 8192 && rows != 32771) {
            double elapsed[5] = {0};
            for (int repeat = 0; repeat < 7; repeat++) for (int turn = 0; turn < 5; turn++) {
                const int mode = (turn + repeat) % 5;
                const double start = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                if (mode == 0) CHECK(ds4_gpu_dsv41_indexer_pack_fp4(packed, original, rows));
                else {
                    if (mode == 1) CHECK(ds4_gpu_glm_indexer_score_one_tensor(actual, q, w, rounded, rows, HEADS, DIM, 1.0f / 64, false));
                    if (mode == 2) CHECK(ds4_gpu_dsv41_indexer_score_fp4(actual, q, w, packed, NULL, rows, 0));
                    if (mode == 3) CHECK(ds4_gpu_dsv41_indexer_score_candidates(actual, q, w, rounded, blocks, rows, count));
                    if (mode == 4) CHECK(ds4_gpu_dsv41_indexer_score_fp4(actual, q, w, packed, blocks, rows, count));
                    CHECK(ds4_gpu_dsv41_candidate_filter(actual, mask_tensor, rows, 1, rows - 1, 1, 0));
                    CHECK(ds4_gpu_indexer_topk_tensor(actual_ids, actual, rows, 1, top));
                }
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - start) * (1000.0 / 6);
            }
            fprintf(stderr, "FP4 index timing rows=%u pack_ms=%.6f dense_f32_ms=%.6f dense_fp4_ms=%.6f candidates_f32_ms=%.6f candidates_fp4_ms=%.6f\n",
                rows, elapsed[0], elapsed[1], elapsed[2], elapsed[3], elapsed[4]);
        }
        ds4_gpu_tensor_free(actual_ids); ds4_gpu_tensor_free(reference_ids);
        ds4_gpu_tensor_free(actual); ds4_gpu_tensor_free(reference);
        ds4_gpu_tensor_free(mask_tensor); ds4_gpu_tensor_free(blocks);
        ds4_gpu_tensor_free(packed); ds4_gpu_tensor_free(storage);
        ds4_gpu_tensor_free(rounded); ds4_gpu_tensor_free(original);
        free(ids); free(mask); free(values);
    }
    ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(q);
    return 1;
}
#endif

extern int ds4_gpu_test_v41_routed_policy(void);

int main(int argc, char **argv) {
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--hc-collapse-norm")) {
        const int ok = ds4_gpu_init() && check_hc_collapse_norm();
        ds4_gpu_cleanup(); return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--main-fp4")) {
        const int ok = ds4_gpu_init() && check_main_kv_fp4();
        ds4_gpu_cleanup(); return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-fp4-batch")) {
        const int ok = ds4_gpu_init() && check_index_fp4_batch();
        ds4_gpu_cleanup(); return ok ? 0 : 1;
    }
    if ((argc == 2 || argc == 3) && (!strcmp(argv[1], "--index-fp4") || !strcmp(argv[1], "--index-simd"))) {
        const int simd = !strcmp(argv[1], "--index-simd");
        uint32_t selected_rows = 0;
        if (argc == 3) {
            char *end;
            const unsigned long rows = strtoul(argv[2], &end, 10);
            for (size_t i = 0; i < sizeof(index_fp4_shapes) / sizeof(index_fp4_shapes[0]); i++)
                if (rows == index_fp4_shapes[i]) selected_rows = index_fp4_shapes[i];
            if (!argv[2][0] || *end || !selected_rows) {
                fprintf(stderr, "index rows must be one of 37,128,256,1024,2048,4096,8192,32768,32771,65536,262144\n");
                return 1;
            }
        }
        if (simd) puts("rows,selection,implementation,dtype,scope,ms");
        const int ok = ds4_gpu_init() && check_index_fp4(simd, selected_rows);
        ds4_gpu_cleanup(); return ok ? 0 : 1;
    }
#endif
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--index-candidates")) {
        const int ok = ds4_gpu_init() && check_indexer_candidates();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--embedding")) {
        const int ok = ds4_gpu_init() && check_embedding();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--router")) {
        const int ok = ds4_gpu_init() && ds4_gpu_test_v41_routed_policy() && check_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--resident-router")) {
        const int ok = ds4_gpu_init() && check_resident_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--tp-attention")) {
        const int ok = ds4_gpu_init() && check_tp_attention();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc >= 2 && (!strcmp(argv[1], "--fp4-attention") ||
                     !strcmp(argv[1], "--fp4-attention-bench"))) {
        const bool benchmark = !strcmp(argv[1], "--fp4-attention-bench");
        uint32_t rows = benchmark ? 2048 : 0;
        if (argc > (benchmark ? 3 : 2)) {
            fprintf(stderr, "usage: %s --fp4-attention | --fp4-attention-bench [rows]\n", argv[0]);
            return 1;
        }
        if (benchmark && argc == 3) {
            char *end;
            const long parsed = strtol(argv[2], &end, 10);
            if (*end || parsed < 2 || parsed > 8192) {
                fprintf(stderr, "FP4 attention benchmark rows must be between 2 and 8192\n");
                return 1;
            }
            rows = (uint32_t)parsed;
        }
        const int ok = ds4_gpu_init() && check_fp4_attention(rows);
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && (!strcmp(argv[1], "--attention-output") ||
                      !strcmp(argv[1], "--attention-output-large"))) {
        const int ok = ds4_gpu_init() && check_attention_output(
            !strcmp(argv[1], "--attention-output-large"));
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--compressed-kv-f16")) {
        const int ok = ds4_gpu_init() && check_compressed_kv_f16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--compressed-kv-f16-bench")) {
        const int ok = ds4_gpu_init() && check_compressed_kv_f16() && bench_compressed_kv_f16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--hc-scaled")) {
        const int ok = ds4_gpu_init() && check_hc_scaled();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--kslice-rows")) {
        const int ok = ds4_gpu_init() && check_kslice_rows() && check_low_projection_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--bf16-linear")) {
        const int ok = ds4_gpu_init() && check_bf16_linear();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--activation-dispatch")) {
        const int ok = ds4_gpu_init() && check_bf16_linear() && check_engram() && check_pool();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--compact-carry")) {
        const int ok = ds4_gpu_init() && check_compact_carry();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-quantize")) {
        const int ok = ds4_gpu_init() && check_rope_quantize();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--deferred-hc")) {
        const int ok = ds4_gpu_init() && check_deferred_hc();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--short-query")) {
        const int ok = ds4_gpu_init() && check_short_query();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-scores")) {
        const int ok = ds4_gpu_init() && check_indexer_batch();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-score-wide")) {
        const int ok = ds4_gpu_init() && check_index_score_wide();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-candidates")) {
        const int ok = ds4_gpu_init() && check_index_candidates();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-topk")) {
        const int ok = ds4_gpu_init() && check_causal_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--topk-select")) {
        const int ok = ds4_gpu_init() && check_topk_select();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--general-topk")) {
        const int ok = ds4_gpu_init() && check_general_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-projection")) {
        const int ok = ds4_gpu_init() && check_index_projection();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-score-bf16")) {
        const int ok = ds4_gpu_init() && check_index_score_bf16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--e4m3-copies")) {
        const int ok = ds4_gpu_init() && check_e4m3_copies();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--engram")) {
        const int ok = ds4_gpu_init() && check_engram();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-tab")) {
        const int ok = ds4_gpu_init() && check_rope_tab();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--dense-bf16-rows")) {
        const int ok = ds4_gpu_init() && check_dense_bf16_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--literal-shapes")) {
        const int ok = ds4_gpu_init() && check_literal_shapes();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--decode-topk")) {
        const int ok = ds4_gpu_init() && check_decode_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--split-collapse")) {
        const int ok = ds4_gpu_init() && check_split_collapse();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc != 1) return 2;
    int ok = ds4_gpu_init() && ds4_gpu_test_v41_routed_policy() && check_router() && check_quantization() && check_engram() && check_rope_stride() &&
             check_rope_quantize() && check_deferred_hc() && check_short_query() && check_pool() && check_candidates() && check_candidates_batch(16449, 17, 16400) &&
             check_candidates_batch(17017, 17, 17000) && check_sparse_gather() && check_indexer_batch() &&
             check_index_score_wide() && check_index_candidates() && check_embedding() && check_index_projection() && check_general_topk() &&
             check_topk_select() && check_causal_topk() && check_indexer_all() && check_compact_carry() &&
             check_attention_output(false) && check_tp_attention();
    ok = ok && check_dense_bf16_rows() && check_kslice_rows() && check_low_projection_rows();
    ok = ok && check_literal_shapes();
    ok = ok && check_e4m3_copies();
    ok = ok && check_rope_tab();
    ok = ok && check_split_collapse();
    ok = ok && check_index_score_bf16();
    ok = ok && check_decode_topk();
#ifdef __APPLE__
    if (ok) ok = check_main_kv_fp4() && check_compressed_kv_f16() && check_indexer_candidates();
#endif
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
