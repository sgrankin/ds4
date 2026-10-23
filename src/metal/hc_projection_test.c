/* CPU descriptor/ownership oracle for the real native HC controller.
 * Link only the native module and these stubs; no model, ObjC or GPU is used. */
#include "bridge.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ds4_gpu_tensor { uint64_t bytes; };
struct ds4_metal_hc_pipeline { int stage; };
struct ds4_metal_model_buffer { uint64_t inner; };
typedef ds4_metal_rms_args ds4_gpu_rms_norm_args;
typedef ds4_metal_mm_args ds4_gpu_mul_mm_args;

/* Original C shader-packet initializers, used only as the independent oracle. */
static ds4_gpu_mul_mm_args ds4_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

typedef struct {
    ds4_metal_hc_launch spec;
    _Alignas(8) unsigned char args[144];
} record;
static record seen[2];
static ds4_metal_hc_pipeline scale_pipeline = {DS4_METAL_HC_SCALE_STAGE};
static ds4_metal_hc_pipeline mm_pipeline = {DS4_METAL_HC_PROJECT_STAGE};
static ds4_metal_model_buffer weights = {4096 + 37};
static ds4_gpu_tensor out = {1ull << 44}, scratch = {1ull << 44}, input = {1ull << 44};
static char model, command;
static uint32_t flags, facts, count, norm_calls, mm_calls;
static uint32_t wrap_calls, weight_releases, acquire_calls, releases, finishes;
static uint32_t want_width, want_outputs, want_rows;
static int mm_available = 1, scale_available = 1, norm_ok = 1;
static int weights_ok = 1, acquire_ok = 1, finish_ok = 1, fail_encode;
static int weights_live, command_live, borrowed;

int ds4_gpu_init(void) { return 1; }
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
int ds4_zig_metal_tensor_has_buffer(const ds4_gpu_tensor *t) { return t != NULL; }
int ds4_zig_metal_env_present(const char *name) { return getenv(name) != NULL; }
int ds4_zig_metal_error(const char *message) {
    assert(strstr(message, "ds4: Metal HC RMS scale projection ") == message);
    return 0;
}
uint32_t ds4_zig_metal_hc_facts(void) {
    return (flags ? DS4_METAL_HC_M3 : 0) |
        ((facts & 1) ? DS4_METAL_HC_TEST : 0) |
        ((facts & 2) ? DS4_METAL_HC_SCALE_PRESENT : 0);
}
const ds4_metal_hc_pipeline *ds4_zig_metal_hc_mm_pipeline(int bc_in, int bc_out) {
    assert(bc_in == (want_width % 32 != 0));
    assert(bc_out == (want_outputs % 64 != 0 || want_rows % 32 != 0));
    return mm_available ? &mm_pipeline : NULL;
}
const ds4_metal_hc_pipeline *ds4_zig_metal_hc_scale_pipeline(void) {
    return scale_available ? &scale_pipeline : NULL;
}
ds4_metal_model_buffer *ds4_metal_retain_model_range(const void *map, uint64_t size,
        uint64_t offset, uint64_t bytes, uint64_t *inner) {
    assert(map == &model && size == 1ull << 44 && offset == 37);
    assert(bytes == (uint64_t)want_width * 2 * want_outputs && !weights_live);
    wrap_calls++;
    if (!weights_ok) return NULL;
    weights_live = 1;
    *inner = weights.inner;
    return &weights;
}
void ds4_metal_release_model_buffer(ds4_metal_model_buffer *p) {
    assert(p == &weights && weights_live && !command_live);
    weight_releases++;
    weights_live = 0;
}
void *ds4_zig_metal_command_acquire(int *owned) {
    assert(weights_live && !command_live);
    acquire_calls++;
    if (!acquire_ok) return NULL;
    command_live = 1;
    *owned = !borrowed;
    return &command;
}
int ds4_zig_metal_hc_encode(void *cb, const ds4_metal_hc_launch *launch) {
    assert(cb == &command && command_live && weights_live && count < 2);
    assert(launch->stage == count);
    assert(launch->pipeline == (count ? &mm_pipeline : &scale_pipeline));
    assert(launch->args_bytes <= sizeof(seen[count].args));
    seen[count].spec = *launch;
    memcpy(seen[count].args, launch->args, launch->args_bytes);
    count++;
    return fail_encode != (int)count;
}
int ds4_zig_metal_command_release(void *cb, int owned, int ok, const char *label) {
    assert(cb == &command && command_live && weights_live && owned == !borrowed);
    assert(!strcmp(label, "HC RMS scale F16 projection"));
    releases++;
    command_live = 0;
    if (ok && owned) finishes++;
    return ok && finish_ok;
}
int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *x,
        uint32_t width, uint32_t rows, float eps) {
    assert(dst == &scratch && x == &input && width == want_width && rows == want_rows && eps == 1e-6f);
    norm_calls++;
    return norm_ok;
}
int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *dst, const void *map, uint64_t size,
        uint64_t offset, uint64_t width, uint64_t outputs, const ds4_gpu_tensor *x, uint64_t rows) {
    assert(dst == &out && map == &model && size == 1ull << 44 && offset == 37);
    assert(width == want_width && outputs == want_outputs && x == &scratch && rows == want_rows);
    mm_calls++;
    return 1;
}
static void reset(void) {
    assert(!weights_live && !command_live);
    count = norm_calls = mm_calls = 0;
    wrap_calls = weight_releases = acquire_calls = releases = finishes = 0;
}
static int call(void) {
    return ds4_gpu_hc_rms_scale_project_f16_tensor(&out, &scratch, &model,
        1ull << 44, 37, want_width, want_outputs, &input, want_rows, 1e-6f);
}
static int mode(void) {
    if (want_rows <= 8 || (want_width != 16384 && want_width != 20480 && want_width != 28672) || want_outputs != 24) return 0;
    if (want_width == 20480 && want_rows < 512) return 0;
    if ((want_width != 20480 && !flags && !(facts & 1)) || getenv("DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ")) return 0;
    return (facts & 2) != 0;
}
static void check_fused(void) {
    assert(count == 2 && !norm_calls && !mm_calls);
    ds4_gpu_rms_norm_args a = ds4_gpu_make_rms_norm_args(want_width, want_rows, 1e-6f);
    ds4_gpu_mul_mm_args b = ds4_gpu_make_mm_args(want_width, want_outputs, want_rows, (uint64_t)want_width * 2), got;
    assert(seen[0].spec.args_bytes == sizeof(a) && !memcmp(seen[0].args, &a, sizeof(a)));
    assert(seen[1].spec.args_bytes == sizeof(b));
    memcpy(&got, seen[1].args, sizeof(got));
#define EQ(field) assert(got.field == b.field)
    EQ(ne00); EQ(ne02); EQ(nb01); EQ(nb02); EQ(nb03); EQ(ne12); EQ(nb10); EQ(nb11);
    EQ(nb12); EQ(nb13); EQ(ne0); EQ(ne1); EQ(r2); EQ(r3);
#undef EQ
    assert(seen[0].spec.groups_x == want_rows && seen[0].spec.groups_y == 1);
    assert(seen[0].spec.threads == 1024 && seen[0].spec.shared_bytes == 128);
    assert(seen[0].spec.x == &input && seen[0].spec.scales == &scratch);
    assert(!seen[0].spec.out && !seen[0].spec.weights && !seen[0].spec.weight_offset);
    assert(seen[1].spec.groups_x == ((uint64_t)want_rows + 31) / 32);
    assert(seen[1].spec.groups_y == ((uint64_t)want_outputs + 63) / 64);
    assert(seen[1].spec.threads == 128 && seen[1].spec.shared_bytes == 8192);
    assert(seen[1].spec.x == &input && seen[1].spec.out == &out && seen[1].spec.scales == &scratch);
    assert(seen[1].spec.weights == &weights && seen[1].spec.weight_offset == weights.inner);
    assert(wrap_calls == 1 && weight_releases == 1 && acquire_calls == 1 && releases == 1);
    assert(finishes == !borrowed && !weights_live && !command_live);
}
int main(void) {
    const uint32_t widths[] = {512, 16384, 20480, 28672};
    const uint32_t outputs[] = {24, 32}, rows[] = {1, 8, 9, 31, 32, 128, 511, 512, 8192};
    uint32_t cases = 0, fused = 0;
    for (uint32_t disabled = 0; disabled < 2; disabled++) {
        if (disabled) setenv("DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ", "0", 1);
        else unsetenv("DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ");
        for (uint32_t wi = 0; wi < 4; wi++) for (uint32_t oi = 0; oi < 2; oi++)
        for (uint32_t ri = 0; ri < 9; ri++) for (facts = 0; facts < 4; facts++)
        for (uint32_t device = 0; device < 2; device++) {
            flags = device;
            want_width = widths[wi]; want_outputs = outputs[oi]; want_rows = rows[ri];
            reset();
            assert(call());
            if (mode()) { check_fused(); fused++; }
            else assert(!count && norm_calls == 1 && mm_calls == 1 && !wrap_calls);
            cases++;
        }
    }
    unsetenv("DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ");
    facts = 2; flags = 0; want_width = 20480; want_outputs = 24; want_rows = 8192;
    reset(); mm_available = 0;
    assert(call() && !count && norm_calls == 1 && mm_calls == 1 && !wrap_calls);
    mm_available = 1;
    reset(); scale_available = 0;
    assert(!call() && !count && !norm_calls && !mm_calls && !wrap_calls);
    scale_available = 1;
    reset(); weights_ok = 0;
    assert(!call() && wrap_calls == 1 && !weight_releases && !acquire_calls);
    weights_ok = 1;
    reset(); acquire_ok = 0;
    assert(!call() && weight_releases == 1 && acquire_calls == 1 && !releases);
    acquire_ok = 1;
    for (borrowed = 0; borrowed < 2; borrowed++) for (fail_encode = 0; fail_encode <= 2; fail_encode++) {
        reset();
        assert(call() == !fail_encode);
        assert(count == (fail_encode == 1 ? 1u : 2u));
        assert(weight_releases == 1 && releases == 1 && finishes == (!borrowed && !fail_encode));
    }
    borrowed = fail_encode = 0;
    reset(); finish_ok = 0;
    assert(!call() && weight_releases == 1 && releases == 1);
    finish_ok = 1;
    reset(); input.bytes = 1;
    assert(!call() && !count && !norm_calls && !wrap_calls);
    input.bytes = 1ull << 44;
    reset(); scratch.bytes = 1;
    assert(!call() && !count && !norm_calls && !wrap_calls);
    scratch.bytes = 1ull << 44;
    reset();
    assert(!ds4_gpu_hc_rms_scale_project_f16_tensor(&out, &scratch, &model, 1, 37,
        20480, 24, &input, 8192, 1e-6f) && !wrap_calls);
    reset(); want_rows = 1; norm_ok = 0;
    assert(!call() && norm_calls == 1 && !mm_calls && !count);
    norm_ok = 1;
    reset();
    assert(!ds4_gpu_hc_rms_scale_project_f16_tensor(&out, &scratch, &model, 1ull << 44,
        37, 20480, 24, &input, 0x80000000u, 1e-6f) && !wrap_calls);
    printf("Native HC projection: PASS %u cases (%u fused), fallback and ownership failures\n", cases, fused);
}
