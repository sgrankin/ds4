#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while (0)
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int check_qkv_norm(void *model,size_t model_bytes) {
    float max_abs=0, max_scaled=0;
    const unsigned widths[]={128,1280,5120}, rows[]={1,7,32};
    for(unsigned wi=0;wi<3;wi++)for(unsigned ri=0;ri<3;ri++)for(unsigned alias=0;alias<2;alias++) {
        unsigned qn=widths[wi],kn=wi?widths[wi-1]:512,r=rows[ri];
        size_t qb=(size_t)qn*r*4,kb=(size_t)kn*r*4;
        uint64_t qo=64,ko=(uint64_t)(qn+32)*4;
        CHECK(ko+(uint64_t)kn*4<=model_bytes);
        float *q=malloc(qb),*kv=malloc(kb);CHECK(q&&kv);
        for(size_t i=0;i<qb/4;i++)q[i]=((int)(i*31%1009)-500)/127.0f;
        for(size_t i=0;i<kb/4;i++)kv[i]=((int)(i*17%1013)-500)/127.0f;
        ds4_gpu_tensor *qi=ds4_gpu_tensor_alloc(qb),*ki=ds4_gpu_tensor_alloc(kb);
        ds4_gpu_tensor *qr=ds4_gpu_tensor_alloc(qb),*kr=ds4_gpu_tensor_alloc(kb);
        ds4_gpu_tensor *qbase=ds4_gpu_tensor_alloc(qb+32),*kbase=ds4_gpu_tensor_alloc(kb+64);
        CHECK(qi&&ki&&qr&&kr&&qbase&&kbase);
        ds4_gpu_tensor *qa=ds4_gpu_tensor_view(qbase,16,qb),*ka=ds4_gpu_tensor_view(kbase,32,kb);CHECK(qa&&ka);
        CHECK(ds4_gpu_tensor_fill_f32(qbase,12345,qb/4+8)&&ds4_gpu_tensor_fill_f32(kbase,12345,kb/4+16));
        CHECK(ds4_gpu_tensor_write(qi,0,q,qb)&&ds4_gpu_tensor_write(ki,0,kv,kb));
        CHECK(ds4_gpu_tensor_write(qa,0,q,qb)&&ds4_gpu_tensor_write(ka,0,kv,kb));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(qr,qi,model,model_bytes,qo,qn,r,1e-6f));
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(kr,ki,model,model_bytes,ko,kn,r,1e-6f));
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(qa,alias?qa:qi,model,model_bytes,qo,qn,
            ka,alias?ka:ki,ko,kn,r,1e-6f));
        CHECK(ds4_gpu_end_commands());
        /* Fused Q/K uses one threadgroup width for both reductions. Separate
         * calls can choose different trees, so compare ordinary f32 error;
         * the existing weighted/BF16 oracle below remains byte-exact. */
        const float *reference[]={ds4_gpu_tensor_contents(qr),ds4_gpu_tensor_contents(kr)};
        const float *actual[]={ds4_gpu_tensor_contents(qa),ds4_gpu_tensor_contents(ka)};
        const size_t counts[]={qb/4,kb/4};
        for(unsigned part=0;part<2;part++)for(size_t i=0;i<counts[part];i++) {
            float error=fabsf(reference[part][i]-actual[part][i]);
            float scaled=error/(1+fabsf(reference[part][i]));
            CHECK(scaled<=2e-6f);
            max_abs=fmaxf(max_abs,error);max_scaled=fmaxf(max_scaled,scaled);
        }
        const float *qguard=ds4_gpu_tensor_contents(qbase),*kguard=ds4_gpu_tensor_contents(kbase);
        for(unsigned i=0;i<4;i++)CHECK(qguard[i]==12345&&qguard[qb/4+4+i]==12345);
        for(unsigned i=0;i<8;i++)CHECK(kguard[i]==12345&&kguard[kb/4+8+i]==12345);
        ds4_gpu_tensor_free(qa);ds4_gpu_tensor_free(ka);
        ds4_gpu_tensor_free(qi);ds4_gpu_tensor_free(ki);ds4_gpu_tensor_free(qr);ds4_gpu_tensor_free(kr);
        ds4_gpu_tensor_free(qbase);ds4_gpu_tensor_free(kbase);free(q);free(kv);
    }
    printf("PASS: 18 QKV fused/separate norm shape, alias and offset cases max_abs=%.9g max_scaled=%.9g\n",max_abs,max_scaled);
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init());
    const size_t page=(size_t)sysconf(_SC_PAGESIZE), model_bytes=((20480*4+page-1)/page)*page;
    float *weights=NULL; CHECK(!posix_memalign((void **)&weights,page,model_bytes));
    for (unsigned i=0;i<20480;i++) weights[i]=((int)(i*17%257)-128)/64.0f;
    CHECK(ds4_gpu_set_model_map(weights,model_bytes));
    const unsigned widths[]={128,512,1280,5120,20480}, rows[]={1,7,32,513};
    for (unsigned wi=0;wi<5;wi++) for (unsigned ri=0;ri<4;ri++) for (unsigned alias=0;alias<2;alias++) {
        const unsigned n=widths[wi], r=rows[ri]; const size_t count=(size_t)n*r, bytes=count*4;
        float *input=malloc(bytes), *ref=malloc(bytes), *actual=malloc(bytes); CHECK(input&&ref&&actual);
        for (size_t i=0;i<count;i++) input[i]=((int)((i*2654435761u)%65537)-32768)/8192.0f;
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(bytes), *a=ds4_gpu_tensor_alloc(bytes+32), *b=ds4_gpu_tensor_alloc(bytes+32);
        CHECK(x&&a&&b);
        ds4_gpu_tensor *av=ds4_gpu_tensor_view(a,16,bytes), *bv=ds4_gpu_tensor_view(b,16,bytes); CHECK(av&&bv);
        CHECK(ds4_gpu_tensor_fill_f32(a,12345,count+8)&&ds4_gpu_tensor_fill_f32(b,12345,count+8));
        CHECK(ds4_gpu_tensor_write(x,0,input,bytes)&&ds4_gpu_tensor_write(av,0,input,bytes)&&ds4_gpu_tensor_write(bv,0,input,bytes));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(av,alias?av:x,weights,model_bytes,0,n,r,1e-6f));
        CHECK(ds4_gpu_dsv41_quantize(av,n,r,DS4_V41_BF16));
        CHECK(ds4_gpu_dsv41_rms_norm_weight_bf16_rows(bv,alias?bv:x,weights,model_bytes,0,n,r,1e-6f));
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(av,0,ref,bytes)&&ds4_gpu_tensor_read(bv,0,actual,bytes));
        CHECK(!memcmp(ref,actual,bytes));
        const float *guard=ds4_gpu_tensor_contents(b); CHECK(guard);
        for (unsigned i=0;i<4;i++) CHECK(guard[i]==12345&&guard[count+4+i]==12345);
        if (n==5120&&r==1&&!alias) {
            for (unsigned pass=0;pass<4;pass++) {
                double begin=seconds(); CHECK(ds4_gpu_begin_commands());
                for (unsigned i=0;i<2000;i++) {
                    if (pass==1||pass==2) CHECK(ds4_gpu_dsv41_rms_norm_weight_bf16_rows(bv,x,weights,model_bytes,0,n,r,1e-6f));
                    else { CHECK(ds4_gpu_rms_norm_weight_rows_tensor(av,x,weights,model_bytes,0,n,r,1e-6f)); CHECK(ds4_gpu_dsv41_quantize(av,n,r,DS4_V41_BF16)); }
                }
                CHECK(ds4_gpu_end_commands());
                printf("norm pass=%u fused=%u 2000 calls=%.3f ms\n",pass,pass==1||pass==2,(seconds()-begin)*1000);
            }
        }
        ds4_gpu_tensor_free(av);ds4_gpu_tensor_free(bv);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(a);ds4_gpu_tensor_free(b);
        free(input);free(ref);free(actual);
    }
    CHECK(!check_qkv_norm(weights,model_bytes));
    ds4_gpu_cleanup(); free(weights);
    puts("PASS: 40 weighted norm/BF16 shape and alias cases, exact bytes and guards"); return 0;
}
