#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while (0)
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
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
    ds4_gpu_cleanup(); free(weights);
    puts("PASS: 40 weighted norm/BF16 shape and alias cases, exact bytes and guards"); return 0;
}
