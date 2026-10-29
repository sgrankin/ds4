# ROCm qualification reference

ROCm source and historical test plans are retained; the current Makefile does not expose the old ROCm targets. The procedures below come from QA snapshot `70537d44` (2026-10-28). Select a matching historical revision and verify its build and fixtures before use. They are not evidence of a current hardware pass and cannot substitute for Mac Metal or CUDA qualification.

## Host and memory safety

On Strix Halo, the historical build was `make clean && make strix-halo`, with no compiler warnings. Run `make test-linux-memory test-rocm-memory` while idle. Linux admission must subtract `CmaFree` from `MemAvailable` even if `CmaTotal` is zero and reject oversized pinned allocations before the driver. An independent process-group watchdog must monitor `max(0, MemAvailable - CmaFree)` and stop below 3 GiB usable. Keep logs outside `/tmp`, avoid builds and primitive benchmarks alongside even a paused resident model, and inspect kernel logs for OOM and GPU errors; a watchdog termination is not a pass.

## Primitive checks

After attention changes, `make test-glm53-kda-rocm test-glm-attention-rocm` and a repeat with `DS4_ROCM_GLM_SELECTED_ATTN_HEAD_TILE=1` cover padded, out-of-range and empty FP16/FP32 selections, continued causal masking, indexer normalization and split attention. After routed MoE or MXFP4 changes, `make test-mxfp4-rocm` must show zero failures in both `mid` and `out` for 1, 3, 32, 128 and 512 tokens and report PASS. The primitive benchmark is not whole-model timing.

## Model integration and continuation

For the full, non-Flash GLM 5.3 Q2 model, use bounded SSD streaming. The historical smoke command was:

```sh
./ds4 --rocm -m gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf --ssd-streaming --ctx 4096 --nothink --tokens 4 -p 'Reply with exactly: OK'
```

Startup must select a safe cache budget without an override; compact indexed prefill and decode must finish. Repeat with an excessive byte hint and one-expert target: reduce the former using available non-CMA memory and use per-layer fallback for the latter. After each run, confirm SSH is responsive and check that there was no OOM, GPU reset or reboot. A `--mtp-timing --temp 0` run must complete a verification cycle without `glm mtp step failed`. Separately, for GLM 5.3 Flash Q2, test initial and continued prefill across 2051 and 4096, including non-multiple-of-four frontiers, against scalar attention and matching long continuation fixtures. Short 100-case scores cannot establish long-context quality. Record cache split, context, lowest usable RAM and corruption markers. Default prequantized Q8 decode and `DS4_ROCM_Q8_PREQUANT_DECODE=0` diagnostic must both pass quality; `--quality` stays full FP32 activation.

## Historical reference

The old Strix Q2 reference with Promessi Sposi, 8192 allocated context and 32 generated tokens was about 77 t/s initial 4096 prefill, 68 t/s 2048 append and 11.7 t/s decode at 6144; the source called for three-run medians, and did not identify a separate measurement revision. The long rendered Z.AI reference was NLL 0.661721610, 6/8 first-token matches. Original Flash and GLM 5.2 numbers describe retired models, not current gates. See [GLM qualification](glm.md) for vision and cross-backend checks.
