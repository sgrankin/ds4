# Overnight decode experiments

The authorized decode session continues until 06:55 Eastern. Its first
completed comparison tested gathering selected packed FP4 main KV rows directly
to half scratch. This can reduce intermediate traffic and fuse staging without
changing stored KV or checkpoints. Five alternating pairs each at 8K and 32K
showed neutral to slightly favorable latency overall, with phase variation in
both directions. All 768 tested fixed-history logit rows and continuation
hashes matched; these checks do not establish practical task quality.

The resident-fast and persistent-fast presets adopt the existing path, with an
explicit `DS4_METAL_V41_PACKED_F16_GATHER=0` override. This FP4-owner result is
separate from earlier F16-storage qualification. The
[result report](../../../speed-bench/experiments/2026-11-04-decode-pass/f16-gather.md)
contains the setup, samples and limitations. Further session results remain
in progress.
