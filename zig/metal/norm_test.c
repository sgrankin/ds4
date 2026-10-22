/* Production native-controller CPU oracle. Fake tensors allocate no GPU/model
 * memory; packet fields and retained-buffer/command sequencing are inspected. */
#include "bridge.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct ds4_gpu_tensor { uint64_t bytes; int buffer; };
struct ds4_metal_model_buffer { uint64_t inner; int live; };
static ds4_gpu_tensor x={UINT64_MAX,1}, y={UINT64_MAX,1}, out={UINT64_MAX,1}, aux={UINT64_MAX,1};
static ds4_metal_model_buffer weights[2];
static ds4_metal_norm_launch seen;
static char model, command;
static uint32_t expected_mode, expected_width, expected_kv_width, expected_rows;
static const ds4_gpu_tensor *expected_x, *expected_y;
static ds4_gpu_tensor *expected_out, *expected_aux;
static unsigned wrapped,released,acquired,encoded,finished,commands_released,cases;
static int init_ok=1,wrap_failure,acquire_ok=1,encode_ok=1,finish_ok=1,borrowed,command_live;

int ds4_gpu_init(void) { return init_ok; }
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t?t->bytes:0; }
int ds4_zig_metal_tensor_has_buffer(const ds4_gpu_tensor *t) { return t&&t->buffer; }
int ds4_zig_metal_error(const char *message) { assert(strstr(message,"ds4: Metal ")==message);return 0; }
ds4_metal_model_buffer *ds4_metal_retain_model_range(const void *map,uint64_t size,
        uint64_t offset,uint64_t bytes,uint64_t *inner) {
    assert(map==&model&&size==UINT64_MAX&&offset==37u+wrapped*101u);
    assert(bytes==(uint64_t)(wrapped?expected_kv_width:expected_width)*4);
    assert(wrapped<2);
    unsigned index=wrapped++;
    if ((int)wrapped==wrap_failure) return NULL;
    weights[index].inner=4096+offset;
    weights[index].live=1;
    *inner=weights[index].inner;
    return &weights[index];
}
void ds4_metal_release_model_buffer(ds4_metal_model_buffer *w) {
    assert(!command_live&&w->live);
    w->live=0;
    released++;
}
void *ds4_zig_metal_command_acquire(int *owned) {
    acquired++;
    assert(!command_live);
    if (!acquire_ok) return NULL;
    command_live=1;
    *owned=!borrowed;
    return &command;
}
int ds4_zig_metal_norm_encode(void *cb,const ds4_metal_norm_launch *launch) {
    assert(cb==&command&&command_live);
    seen=*launch;
    encoded++;
    assert(launch->mode==expected_mode);
    assert(launch->x==expected_x&&launch->y==expected_y);
    assert(launch->out==expected_out&&launch->aux_out==expected_aux);
    if (expected_mode==DS4_METAL_NORM_PLAIN) assert(!launch->weights0&&!launch->weights1);
    else {
        assert(launch->weights0==&weights[0]&&weights[0].live);
        assert(launch->weight_offset0==weights[0].inner);
    }
    if (expected_mode==DS4_METAL_NORM_QKV) {
        assert(launch->weights1==&weights[1]&&weights[1].live);
        assert(launch->weight_offset1==weights[1].inner);
    } else assert(!launch->weights1&&launch->weight_offset1==0);
    return encode_ok;
}
int ds4_zig_metal_command_release(void *cb,int owned,int ok,const char *label) {
    assert(cb==&command&&command_live&&owned==!borrowed);
    const char *expected=expected_mode==DS4_METAL_NORM_PLAIN?"plain RMS norm":
        expected_mode==DS4_METAL_NORM_ADD?"add+RMS norm":
        expected_mode==DS4_METAL_NORM_QKV?"fused q/kv RMS norm":"weighted RMS norm";
    assert(!strcmp(label,expected));
    command_live=0;
    commands_released++;
    if (expected_mode!=DS4_METAL_NORM_PLAIN) assert(weights[0].live);
    if (expected_mode==DS4_METAL_NORM_QKV) assert(weights[1].live);
    if (owned&&ok) finished++;
    return ok&&finish_ok;
}
static void reset(void) {
    assert(!command_live&&!weights[0].live&&!weights[1].live);
    wrapped=released=acquired=encoded=finished=commands_released=0;
}
static ds4_metal_rms_args expected_rms(uint32_t width,uint32_t rows) {
    uint64_t row=(uint64_t)width*4,bytes=row*rows;
    return (ds4_metal_rms_args){
        .ne00=(int32_t)width,.ne00_t=(int32_t)(width/4),.nb1=row,.nb2=bytes,.nb3=bytes,.eps=1e-6f,
        .nef1={(int32_t)rows,1,1},.nef2={1,1,1},.nef3={1,1,1},
        .nbf1={row,row,row},.nbf2={bytes,row,row},.nbf3={bytes,row,row},
    };
}
static uint64_t expected_threads(uint32_t width) {
    uint64_t n=32,width4=width/4;
    while(n<width4&&n<1024)n*=2;
    if(n>width4)n=width4;
    return n?n:1;
}
static int call(int alias) {
    expected_x=&x;
    expected_y=NULL;
    expected_out=alias?&x:&out;
    expected_aux=NULL;
    cases++;
    switch(expected_mode) {
        case DS4_METAL_NORM_PLAIN:
            return ds4_gpu_rms_norm_plain_rows_tensor(expected_out,&x,expected_width,expected_rows,1e-6f);
        case DS4_METAL_NORM_WEIGHTED:
            return ds4_gpu_rms_norm_weight_rows_tensor(expected_out,&x,&model,UINT64_MAX,37,expected_width,expected_rows,1e-6f);
        case DS4_METAL_NORM_WEIGHTED_BF16:
            return ds4_gpu_dsv41_rms_norm_weight_bf16_rows(expected_out,&x,&model,UINT64_MAX,37,expected_width,expected_rows,1e-6f);
        case DS4_METAL_NORM_ADD:
            expected_y=&y;expected_aux=&aux;
            return ds4_gpu_add_rms_norm_weight_tensor(expected_out,&aux,&x,&y,&model,UINT64_MAX,37,expected_width,1e-6f);
        case DS4_METAL_NORM_QKV:
            expected_y=&y;expected_aux=alias?&y:&aux;
            return ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(expected_out,&x,&model,UINT64_MAX,37,expected_width,
                expected_aux,&y,138,expected_kv_width,expected_rows,1e-6f);
    }
    assert(0);return 0;
}
static void check_packet(void) {
    unsigned expected_weights=expected_mode==DS4_METAL_NORM_PLAIN?0:expected_mode==DS4_METAL_NORM_QKV?2:1;
    assert(encoded==1&&acquired==1&&commands_released==1&&finished==(unsigned)!borrowed);
    assert(wrapped==expected_weights&&released==expected_weights);
    assert(seen.rows==expected_rows&&seen.tasks==(expected_mode==DS4_METAL_NORM_QKV?2u:1u));
    assert(seen.threads==expected_threads(expected_width)&&seen.shared_bytes==128);
    if(expected_mode==DS4_METAL_NORM_QKV) {
        ds4_metal_qkv_rms_args expected={
            .q_n=(int32_t)expected_width,.q_n4=(int32_t)(expected_width/4),
            .kv_n=(int32_t)expected_kv_width,.kv_n4=(int32_t)(expected_kv_width/4),
            .q_row_stride=(uint64_t)expected_width*4,.kv_row_stride=(uint64_t)expected_kv_width*4,.eps=1e-6f,
        };
        assert(!memcmp(&seen.args.qkv,&expected,sizeof(expected)));
    } else {
        ds4_metal_rms_args expected=expected_rms(expected_width,expected_rows);
        assert(!memcmp(&seen.args.rms,&expected,sizeof(expected)));
    }
}
int main(void) {
    const uint32_t widths[]={4,16,128,512,1280,5120,20480},rows[]={1,7,32,513,8192};
    for(borrowed=0;borrowed<2;borrowed++)for(unsigned wi=0;wi<7;wi++)for(unsigned ri=0;ri<5;ri++)
    for(expected_mode=0;expected_mode<=DS4_METAL_NORM_QKV;expected_mode++)for(int alias=0;alias<2;alias++) {
        if(expected_mode==DS4_METAL_NORM_ADD&&rows[ri]!=1)continue;
        expected_width=widths[wi];expected_kv_width=wi?widths[wi-1]:4;expected_rows=rows[ri];
        reset();assert(call(alias));check_packet();
    }
    borrowed=0;expected_width=5120;expected_kv_width=512;expected_rows=32;expected_mode=DS4_METAL_NORM_QKV;
    reset();wrap_failure=2;assert(!call(0)&&wrapped==2&&released==1&&!acquired);wrap_failure=0;
    reset();acquire_ok=0;assert(!call(0)&&released==2&&!commands_released);acquire_ok=1;
    for(borrowed=0;borrowed<2;borrowed++) {
        reset();encode_ok=0;assert(!call(0)&&released==2&&commands_released==1&&!finished);encode_ok=1;
    }
    borrowed=0;
    reset();finish_ok=0;assert(!call(0)&&released==2&&commands_released==1);finish_ok=1;
    reset();wrap_failure=1;assert(!call(0)&&wrapped==1&&!released&&!acquired);wrap_failure=0;
    reset();x.bytes=1;assert(!call(0)&&!wrapped&&!acquired);x.bytes=UINT64_MAX;
    reset();x.buffer=0;assert(!call(0)&&!wrapped&&!acquired);x.buffer=1;
    reset();expected_width=0;assert(!call(0)&&!wrapped&&!acquired);expected_width=5120;
    reset();assert(!ds4_gpu_rms_norm_weight_rows_tensor(&out,&x,&model,1,37,5120,32,1e-6f)&&!wrapped&&!acquired);
    reset();init_ok=0;assert(!call(0)&&!wrapped&&!acquired);init_ok=1;
    /* Scalar public wrappers use exactly one row. */
    expected_rows=1;expected_mode=DS4_METAL_NORM_PLAIN;
    reset();expected_x=&x;expected_y=NULL;expected_out=&out;expected_aux=NULL;
    assert(ds4_gpu_rms_norm_plain_tensor(&out,&x,expected_width,1e-6f));check_packet();
    expected_mode=DS4_METAL_NORM_WEIGHTED;
    reset();assert(ds4_gpu_rms_norm_weight_tensor(&out,&x,&model,UINT64_MAX,37,expected_width,1e-6f));check_packet();
    puts("native RMS packets, shapes, aliases and retained resource failures: PASS");
    printf("%u ordinary controller cases\n",cases);
}
