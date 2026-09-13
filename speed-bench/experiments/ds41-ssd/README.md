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
