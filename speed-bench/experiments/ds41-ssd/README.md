# DS4.1 SSD experiments on Apple M5 Max, 128 GiB

Base: bd66c402. Model: gguf/DeepSeek-V4.1-Flash-Q2.gguf. SSD streaming is required.

## 01: baseline and vision-loaded text-only benchmark

Added `ds4-bench --vision FILE` to isolate encoder loading from agent prompt changes.
`make -j2 ds4-bench` passed. Initial 2048-token raw ds4.c prefix, ctx=2053,
4 decode tokens, default streaming cache: 130.59 tps without vision, 130.95
with vision. These single runs do not reproduce the reported agent slowdown.
Logs and CSVs are adjacent. GPU busy profile is cumulative and sampled, not
a precise per-stage breakdown. Baseline prefill wall time is 15.68 seconds;
GPU busy reaches approximately 6.1 seconds near prefill completion.

Command: `DS4_METAL_GRAPH_PREFILL_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming
--prompt-file ds4.c --ctx-start 2048 --ctx-max 2048 --gen-tokens 4`
Add `--vision gguf/DeepSeek-V4.1-Flash-Vision.gguf` for loaded vision.

`run_abba.py` compares controls and candidates in fresh processes, retains
commands and profiles, and requires identical final full-vocabulary logit
values (9 significant digit float serialization). Copy the baseline ds4.c to
/tmp/ds41-prefill-input.c before changing source so token inputs remain fixed.
Results with full logits belong in /tmp; retain summaries here.

## 02: one prefetch reader versus default eight (no default change)

`python3 speed-bench/experiments/ds41-ssd/run_abba.py /tmp/ds41-threads1
--candidate-env DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS=1`

ABBA, fresh processes, 2048 tokens, 8 teacher-forced decode tokens.
Control: 131.08, 134.16 tps. Candidate: 135.14, 136.48 tps (about 2.4%
faster by aggregate time). All full-vocabulary final logits match exactly.
Steady decode: 17.13/17.32 control, 16.82/17.34 candidate.
The improvement is too small relative to run variation to change a shared
backend default. Retain this as a documented negative/inconclusive experiment.
