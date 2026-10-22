#include "ds4_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_jpeg(const char *path, uint32_t width, uint32_t height,
                      const char *expected_fp) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        return 0;
    }
    char got[65] = {0};
    int ok = hex_fingerprint(image.fingerprint, got, sizeof(got)) &&
             image.width == width && image.height == height &&
             strcmp(got, expected_fp) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u fp=%s, expected %ux%u fp=%s\n",
                path, image.width, image.height, got,
                width, height, expected_fp);
    }
    ds4_image_free(&image);
    return ok;
}

/* Optional inputs let the paired C/native loop inspect actual oriented pixels,
 * rather than qualifying a decoder using only a matching dimensions summary. */
static int dump_image(const char *path) {
    ds4_image image = {0};
    char error[160] = {0}, fingerprint[65] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "%s: %s\n", path, error);
        return 0;
    }
    hex_fingerprint(image.fingerprint, fingerprint, sizeof(fingerprint));
    printf("{\"width\":%u,\"height\":%u,\"fingerprint\":\"%s\",\"rgb\":\"",
           image.width, image.height, fingerprint);
    const size_t bytes = (size_t)image.width * image.height * 3;
    for (size_t i = 0; i < bytes; i++) printf("%02x", image.rgb[i]);
    puts("\"}");
    ds4_image_free(&image);
    return image.width == 0 && image.height == 0 && image.rgb == NULL;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--bench")) {
        unsigned long repeats = strtoul(argv[2], NULL, 10);
        if (!repeats || repeats > 10000) return 2;
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (unsigned long i = 0; i < repeats; i++) {
            ds4_image image = {0};
            char error[160] = {0};
            if (!ds4_image_decode_file(&image, argv[3], error, sizeof(error))) {
                fprintf(stderr, "%s\n", error);
                return 1;
            }
            ds4_image_free(&image);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double seconds = (double)(end.tv_sec - begin.tv_sec) +
                         (double)(end.tv_nsec - begin.tv_nsec) / 1e9;
        printf("{\"repeats\":%lu,\"seconds\":%.9f}\n", repeats, seconds);
        return 0;
    }
    if (argc > 1) {
        for (int i = 1; i < argc; i++) if (!dump_image(argv[i])) return 1;
        return 0;
    }
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    return 0;
}
