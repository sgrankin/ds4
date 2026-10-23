/* Actual native controllers, fake resource bridge: no GPU/model allocation. */
#include "bridge.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
struct ds4_gpu_tensor { uint64_t bytes, offset; };
static ds4_gpu_tensor x={UINT64_MAX,0}, y={UINT64_MAX,16}, q={UINT64_MAX,32}, k={UINT64_MAX,48}, out={UINT64_MAX,64};
static char token;
static unsigned acquired,encoded,released,finished,copied,cases;
static int init_ok=1,encode_ok=1,acquire_ok=1,finish_ok=1,copy_fail,borrowed,live,disable;
static ds4_metal_quant_launch quant;
static ds4_metal_engram_launch engram;
static ds4_metal_pool_launch pool;
static uint64_t copy_offset[2],copy_bytes[2];
int ds4_gpu_init(void) { return init_ok; }
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t?t->bytes:0; }
uint64_t ds4_zig_metal_tensor_offset(const ds4_gpu_tensor *t) { return t?t->offset:0; }
int ds4_zig_metal_env_present(const char *name) {
    assert(!strcmp(name,"DS4_METAL_DISABLE_V41_LINEAR_BF16"));return disable;
}
void *ds4_zig_metal_command_acquire(int *owned) {
    acquired++;assert(!live);if(!acquire_ok)return NULL;
    live=1;*owned=!borrowed;return &token;
}
int ds4_zig_metal_command_release(void *command,int owned,int ok,const char *label) {
    assert(command==&token&&live&&owned==!borrowed&&strstr(label,"V4.1 ")==label);
    live=0;released++;if(owned&&ok)finished++;return ok&&finish_ok;
}
int ds4_zig_metal_quant_encode(void *command,const ds4_metal_quant_launch *launch) {
    assert(command==&token&&live);quant=*launch;encoded++;return encode_ok;
}
int ds4_zig_metal_engram_encode(void *command,const ds4_metal_engram_launch *launch) {
    assert(command==&token&&live);engram=*launch;encoded++;return encode_ok;
}
int ds4_zig_metal_pool_encode(void *command,const ds4_metal_pool_launch *launch) {
    assert(command==&token&&live);pool=*launch;encoded++;return encode_ok;
}
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst,uint64_t dst_offset,const ds4_gpu_tensor *src,uint64_t src_offset,uint64_t bytes) {
    assert(!live&&dst_offset==0&&copied<2);
    assert(dst==(copied?&k:&q)&&src==(copied?&y:&x));
    copy_offset[copied]=src_offset;copy_bytes[copied]=bytes;copied++;
    return (int)copied!=copy_fail;
}
static void reset(void) {
    assert(!live);acquired=encoded=released=finished=copied=0;
    memset(&quant,0,sizeof quant);memset(&engram,0,sizeof engram);memset(&pool,0,sizeof pool);
}
static void command_ok(void) {
    assert(acquired==1&&encoded==1&&released==1&&finished==(unsigned)!borrowed);
}
int main(void) {
    const uint32_t widths[]={1,16,32,128,512,5120},rows[]={1,7,8192};
    for(borrowed=0;borrowed<2;borrowed++)for(disable=0;disable<2;disable++)
    for(unsigned alignment=0;alignment<2;alignment++)for(unsigned wi=0;wi<6;wi++)
    for(unsigned ri=0;ri<3;ri++)for(unsigned format=0;format<=DS4_V41_FP4_E4M3;format++) {
        uint32_t w=widths[wi],r=rows[ri],block=format==DS4_V41_FP4_E4M3?16:32;
        if(format!=DS4_V41_BF16&&w%block)continue;
        reset();x.offset=alignment?4:0;
        assert(ds4_gpu_dsv41_quantize(&x,w,r,(ds4_v41_activation_format)format));command_ok();
        unsigned linear=format==DS4_V41_BF16&&!disable&&!alignment;
        ds4_metal_quant_args args={w,r,format};
        assert(!memcmp(&quant.args,&args,sizeof args)&&quant.count==(uint64_t)w*r);
        assert(quant.x==&x&&quant.linear==linear&&quant.threads==(linear?256u:32u));
        assert(quant.groups_x==(linear?(quant.count+1023)/1024:((uint64_t)w+block-1)/block));
        assert(quant.groups_y==(linear?1u:r));cases++;
    }
    borrowed=0;disable=0;x.offset=0;
    reset();assert(ds4_gpu_dsv41_quantize(&x,1,UINT32_MAX,DS4_V41_BF16));
    assert(quant.linear&&quant.count==UINT32_MAX&&sizeof quant.count==8);
    reset();assert(ds4_gpu_dsv41_quantize(&x,2,UINT32_C(2147483648),DS4_V41_BF16));
    assert(!quant.linear&&quant.count==UINT64_C(4294967296));
    for(borrowed=0;borrowed<2;borrowed++)for(unsigned masked=0;masked<2;masked++) {
        reset();assert(ds4_gpu_dsv41_engram_add(&x,&y,&q,&k,masked?&out:NULL,5120,7,1e-6f));command_ok();
        ds4_metal_engram_args args={5120,7,1e-6f,masked};
        assert(!memcmp(&engram.args,&args,sizeof args));
        assert(engram.residual==&x&&engram.kv==&y&&engram.q_weight==&q&&engram.k_weight==&k);
        assert(engram.mask==(masked?&out:&x)&&engram.groups_x==7&&engram.groups_y==4&&engram.threads==32);cases++;
    }
    for(borrowed=0;borrowed<2;borrowed++)for(uint32_t start=0;start<4;start++)
    for(uint32_t r=1;r<=8;r++) {
        reset();uint32_t pairs=(r+(start&1))/2,last_even=(start+r-1)&~1u;
        assert(ds4_gpu_dsv41_pool2(pairs?&out:NULL,&x,&y,&q,&k,512,r,start));
        if(pairs) {
            command_ok();ds4_metal_pool_args args={512,pairs,start&1};
            assert(!memcmp(&pool.args,&args,sizeof args));
            assert(pool.out==&out&&pool.kv==&x&&pool.scores==&y&&pool.previous_kv==&q&&pool.previous_scores==&k);
            assert(pool.grid_x==512&&pool.grid_y==pairs&&pool.threads==256);
        } else assert(!acquired&&!encoded&&!released);
        if(last_even>=start) {
            assert(copied==2&&copy_offset[0]==(uint64_t)(last_even-start)*512*4&&copy_offset[1]==copy_offset[0]);
            assert(copy_bytes[0]==512*4&&copy_bytes[1]==512*4);
        } else assert(!copied);
        cases++;
    }
    borrowed=0;
    reset();encode_ok=0;assert(!ds4_gpu_dsv41_pool2(&out,&x,&y,&q,&k,512,8,0));
    assert(released==1&&!finished&&!copied);encode_ok=1;
    reset();finish_ok=0;assert(!ds4_gpu_dsv41_pool2(&out,&x,&y,&q,&k,512,8,0)&&!copied);finish_ok=1;
    reset();copy_fail=1;assert(!ds4_gpu_dsv41_pool2(NULL,&x,&y,&q,&k,512,1,0)&&copied==1);copy_fail=0;
    reset();acquire_ok=0;assert(!ds4_gpu_dsv41_quantize(&x,512,7,DS4_V41_BF16)&&!released);acquire_ok=1;
    reset();init_ok=0;assert(!ds4_gpu_dsv41_quantize(&x,512,7,DS4_V41_BF16)&&!acquired);init_ok=1;
    reset();assert(!ds4_gpu_dsv41_engram_add(&x,&y,&q,&k,NULL,512,7,NAN)&&!acquired);
    reset();assert(!ds4_gpu_dsv41_pool2(&out,&x,&y,&q,&k,512,2,UINT32_MAX)&&!acquired);
    reset();assert(!ds4_gpu_dsv41_quantize(&x,17,1,DS4_V41_FP4_E4M3)&&!acquired);
    reset();x.bytes=1;assert(!ds4_gpu_dsv41_quantize(&x,512,7,DS4_V41_BF16)&&!acquired);x.bytes=UINT64_MAX;
    printf("PASS: %u activation packets, alignment/env choices, masks, pool carries and command ownership cases\n",cases);
}
