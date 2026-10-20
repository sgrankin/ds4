#!/bin/sh
set -eu
HERE=/Volumes/Models/ds4/speed-bench/experiments/resident-functional-hour2-20261003
WORK=/Volumes/Models/ds4/.workspaces/idea-resident-functional-runtime-native
OUT="$WORK/speed-bench/ideas/artifacts/resident-functional-native-direct-v1"
AUX=/Volumes/Models/ds4/speed-bench/ideas/artifacts/suffix-diagnostic-native-wrap-build-20261002
FRONT=/Volumes/Models/ds4/.workspaces/idea-resident-native-weights-next/speed-bench/ideas/artifacts/resident-native-weights-next-build-plan-v3/source/quality_probe.c
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk
for band in 0.025 0.05 0.10; do
    /usr/bin/cc -isysroot "$SDK" -O3 -fno-fast-math -ffp-contract=off -g -mcpu=native -Wall -Wextra -Werror -std=c11 -DDS4_FM_ROUTER_BAND="$band" -I "$WORK" -c "$HERE/ds4_resident_functional_band.c" -o "$OUT/functional-band$band.o"
    /usr/bin/cc -isysroot "$SDK" -O3 -g -mcpu=native -Wall -Wextra -std=c11 -I "$WORK" "$FRONT" "$OUT/ds4.o" "$OUT/functional-band$band.o" "$OUT/runtime.o" "$OUT/qualification.o" "$AUX/ds4_image.o" "$AUX/ds4_distributed.o" "$AUX/ds4_tp.o" "$AUX/ds4_ssd.o" "$AUX/ds4_metal.o" "$AUX/ds4_layer_pack.o" "$AUX/ds4_engram.o" -lm -pthread -framework Foundation -framework Metal -o "$OUT/quality-functional-band$band"
    /usr/bin/cc -isysroot "$SDK" -O3 -g -mcpu=native "$OUT/ds4_cli.o" "$OUT/ds4_help.o" "$OUT/ds4_prompt_prefix.o" "$OUT/linenoise.o" "$OUT/ds4_gpu_args.o" "$OUT/ds4.o" "$OUT/functional-band$band.o" "$OUT/runtime.o" "$OUT/qualification.o" "$AUX/ds4_image.o" "$AUX/ds4_distributed.o" "$AUX/ds4_tp.o" "$AUX/ds4_ssd.o" "$AUX/ds4_metal.o" "$AUX/ds4_layer_pack.o" "$AUX/ds4_engram.o" -lm -pthread -framework Foundation -framework Metal -o "$OUT/ds4-functional-band$band"
done
