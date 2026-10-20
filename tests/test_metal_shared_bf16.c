/* Fused shared-expert projections must preserve all three BF16 boundaries. */
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
    const size_t page=(size_t)sysconf(_SC_PAGESIZE);
    const size_t bank=(((size_t)5120/32*34*2304+page-1)/page)*page, bytes=bank*2;
    unsigned char *w=NULL; CHECK(!posix_memalign((void **)&w,page,bytes)); memset(w,0,bytes);
    for (unsigned b=0;b<2;b++) for (size_t i=0;i<(size_t)5120/32*2304;i++) {
        uint16_t scale=0x1800+(i%7)*128; memcpy(w+b*bank+i*34,&scale,2);
        for (unsigned k=0;k<32;k++) w[b*bank+i*34+2+k]=(unsigned char)(i*17+k*31+b*53);
    }
    CHECK(ds4_gpu_set_model_map(w,bytes));
    const unsigned widths[]={128,5120}, rows[]={1,6,32,128}, outputs[]={32,2304};
    unsigned shapes=0;
    for (unsigned wi=0;wi<2;wi++) for (unsigned ri=0;ri<4;ri++) for (unsigned oi=0;oi<2;oi++) {
        unsigned n=widths[wi],r=rows[ri],o=outputs[oi]; size_t count=(size_t)r*o, size=count*4;
        float *input=malloc((size_t)n*r*4); CHECK(input);
        for (size_t i=0;i<(size_t)n*r;i++) input[i]=((int)((i*2654435761u)%65537)-32768)/32768.0f;
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((size_t)n*r*4), *storage[6], *v[6]; CHECK(x);
        for (unsigned i=0;i<6;i++) {
            storage[i]=ds4_gpu_tensor_alloc(size+32); CHECK(storage[i]);
            CHECK(ds4_gpu_tensor_fill_f32(storage[i],12345,count+8));
            v[i]=ds4_gpu_tensor_view(storage[i],16,size); CHECK(v[i]);
        }
        CHECK(ds4_gpu_tensor_write(x,0,input,(size_t)n*r*4));
        for (unsigned ci=0;ci<2;ci++) {
            float clamp=ci?7.0f:0.0f;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(v[0],w,bytes,0,n,o,x,r));
            CHECK(ds4_gpu_dsv41_quantize(v[0],o,r,DS4_V41_BF16));
            CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(v[1],w,bytes,bank,n,o,x,r));
            CHECK(ds4_gpu_dsv41_quantize(v[1],o,r,DS4_V41_BF16));
            CHECK(ds4_gpu_swiglu_tensor(v[2],v[0],v[1],count,clamp,1));
            CHECK(ds4_gpu_dsv41_quantize(v[2],o,r,DS4_V41_BF16));
            CHECK(ds4_gpu_dsv41_shared_gate_up_bf16_rows(v[3],v[4],v[5],w,bytes,0,bank,n,o,x,r,clamp));
            CHECK(ds4_gpu_end_commands());
            for (unsigned i=0;i<3;i++) CHECK(!memcmp(ds4_gpu_tensor_contents(v[i]),ds4_gpu_tensor_contents(v[i+3]),size));
            for (unsigned i=3;i<6;i++) {
                const float *p=ds4_gpu_tensor_contents(storage[i]);
                for (unsigned k=0;k<4;k++) CHECK(p[k]==12345 && p[count+4+k]==12345);
            }
            shapes++;
        }
        if (n==5120 && o==2304 && (r==1||r==32)) for (unsigned pass=0;pass<4;pass++) {
            double begin=seconds(); CHECK(ds4_gpu_begin_commands());
            for (unsigned rep=0;rep<200;rep++) {
                if (pass==1||pass==2) CHECK(ds4_gpu_dsv41_shared_gate_up_bf16_rows(v[3],v[4],v[5],w,bytes,0,bank,n,o,x,r,7));
                else {
                    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(v[0],w,bytes,0,n,o,x,r));
                    CHECK(ds4_gpu_dsv41_quantize(v[0],o,r,DS4_V41_BF16));
                    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(v[1],w,bytes,bank,n,o,x,r));
                    CHECK(ds4_gpu_dsv41_quantize(v[1],o,r,DS4_V41_BF16));
                    CHECK(ds4_gpu_swiglu_tensor(v[2],v[0],v[1],count,7,1));
                    CHECK(ds4_gpu_dsv41_quantize(v[2],o,r,DS4_V41_BF16));
                }
            }
            CHECK(ds4_gpu_end_commands());
            printf("shared rows=%u pass=%u fused=%u 200 calls=%.3f ms\n",r,pass,pass==1||pass==2,(seconds()-begin)*1000);
        }
        for (unsigned i=0;i<6;i++) { ds4_gpu_tensor_free(v[i]); ds4_gpu_tensor_free(storage[i]); }
        ds4_gpu_tensor_free(x); free(input);
    }
    ds4_gpu_cleanup(); free(w); printf("PASS: %u shared-expert BF16 shapes, all intermediates and guards exact\n",shapes); return 0;
}
