#!/bin/sh
set -eu
runtime=${1:-.}
output=${2:-speed-bench/ideas/quality-probe}
policy_flag=${3:-}
source=${4:-speed-bench/ideas/quality_probe.c}
core_object=${5:-"$runtime/ds4.o"}
cc -O3 -Wall -Wextra -std=c11 $policy_flag -I "$runtime" \
  "$source" \
  "$core_object" "$runtime/ds4_image.o" "$runtime/ds4_distributed.o" \
  "$runtime/ds4_tp.o" "$runtime/ds4_ssd.o" "$runtime/ds4_metal.o" \
  "$runtime/ds4_layer_pack.o" "$runtime/ds4_engram.o" \
  -lm -pthread -framework Foundation -framework Metal -o "$output"
