/* Exhaustive independent conversion reference plus cached-row timing. */
#include "../ds4_engram.c"
#include <assert.h>
#include <stdio.h>
#include <time.h>

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    engram_decode_init();
    for (unsigned scale = 0; scale < 256; scale++) {
        for (unsigned code = 0; code < 256; code++) {
            int exp = (code >> 3) & 15, mantissa = code & 7;
            double value = exp ? ldexp(1.0 + mantissa / 8.0, exp - 7) : mantissa / 512.0;
            if (code & 128) value = -value;
            float expected = (float)ldexp(value, (int)scale - 127);
            uint32_t bits; memcpy(&bits, &expected, 4);
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&expected, &bits, 4);
            bool valid = (code & 127) != 127 && scale != 255 && isfinite(expected);
            float actual = engram_decode_lut[scale][code];
            assert(isfinite(actual) == valid);
            if (valid) assert(memcmp(&expected, &actual, 4) == 0);
        }
    }
    char path[] = "/tmp/ds4-engram-lut-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];
    for (unsigned i=0; i<256; i++) raw[i]=(i==127 || i==255) ? 0 : i;
    for (unsigned i=256; i<sizeof(raw); i++) raw[i]=127;
    assert(write(fd,raw,sizeof(raw))==sizeof(raw)); close(fd);
    ds4_engram_table t; assert(ds4_engram_table_open(&t,path,0,1));
    uint32_t rows[24]={0}; float reference[24*256], actual[24*256];
    unsetenv("DS4_ENGRAM_DECODE_LUT");
    assert(ds4_engram_read(&t,rows,24,reference));
    for (unsigned pass=0; pass<4; pass++) {
        bool lookup=pass==1 || pass==2;
        if (lookup) setenv("DS4_ENGRAM_DECODE_LUT","1",1);
        else unsetenv("DS4_ENGRAM_DECODE_LUT");
        double begin=seconds();
        for (unsigned i=0; i<2000; i++) assert(ds4_engram_read(&t,rows,24,actual));
        printf("engram pass=%u lookup=%u 2000x24 rows=%.3f ms\n",pass,lookup,(seconds()-begin)*1000);
        assert(!memcmp(reference,actual,sizeof(actual)));
    }
    unsetenv("DS4_ENGRAM_DECODE_LUT");
    ds4_engram_table_close(&t); unlink(path);
    puts("PASS: all 65536 FP8/scale pairs and cached-row bytes");
    return 0;
}
