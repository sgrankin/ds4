#!/bin/sh
set -eu
WORK=/Volumes/Models/ds4/.workspaces/idea-resident-functional-runtime-native
OUT="$WORK/speed-bench/ideas/artifacts/resident-functional-native-direct-v1"
AUX=/Volumes/Models/ds4/speed-bench/ideas/artifacts/suffix-diagnostic-native-wrap-build-20261002
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk
for unit in ds4_help ds4_prompt_prefix linenoise ds4_gpu_args; do
    /usr/bin/cc -isysroot "$SDK" -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -I "$WORK" -c "$WORK/$unit.c" -o "$OUT/$unit.o"
done
/usr/bin/cc -isysroot "$SDK" -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -I "$WORK" -c /Volumes/Models/ds4/speed-bench/experiments/resident-functional-hour2-20261003/ds4_cli_metrics.c -o "$OUT/ds4_cli.o"
/usr/bin/cc -isysroot "$SDK" -O3 -g -mcpu=native "$OUT/ds4_cli.o" "$OUT/ds4_help.o" "$OUT/ds4_prompt_prefix.o" "$OUT/linenoise.o" "$OUT/ds4_gpu_args.o" "$OUT/ds4.o" "$OUT/functional.o" "$OUT/runtime.o" "$OUT/qualification.o" "$AUX/ds4_image.o" "$AUX/ds4_distributed.o" "$AUX/ds4_tp.o" "$AUX/ds4_ssd.o" "$AUX/ds4_metal.o" "$AUX/ds4_layer_pack.o" "$AUX/ds4_engram.o" -lm -pthread -framework Foundation -framework Metal -o "$OUT/ds4-functional"
