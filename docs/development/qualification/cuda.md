# CUDA qualification reference

CUDA source and historical test plans are retained, but the current Makefile does not expose the old CUDA build targets. The procedures below describe the 2026-10-28 QA snapshot at `70537d44`; use a matching historical revision and verify its build and fixtures before attempting them. They do not establish a current release pass. Current Mac Metal qualification is separate. DeepSeek V4.1 CUDA network TP and DSpark are implemented in retained source; implementation is not physical qualification.

## Hardware and service prerequisites

On a DGX Spark, check for existing inference services before use. Record their process, model, ports and launch command before any authorized interruption; confirm workers have exited, and restore and verify service health afterward. Do not use high-performance Hugging Face Xet while a model is resident. The eight-GPU host requires explicit permission for the current QA pass. The historical build commands were `make clean && make cuda-spark` and `make cuda-regression`; require warning-free Spark and eight-GPU builds where those targets exist.

## Primitive checks

Run applicable historical tests under Compute Sanitizer as well as normally. Pool initialization, graph replay and cleanup must restore prior release thresholds and trim unused allocations. Grouped Q8 must match scalar rows exactly for ragged shapes, compact slices and 8/16/32-row boundaries with untouched tails. Token-tile and indexed attention must agree with CPU references through ring wraps, compression boundaries, masked rows, peaky queries and changed graph-replay inputs; use memcheck, racecheck and synccheck. After projection, router-stream or scratch changes test delayed side-stream input, graph replay with changed data, invalid ranges, undersized scratch and output guards. Check full-model logits as well as primitive outputs.

DSpark proposal tests cover ties, negative scores, vocabulary tails, invalid previous tokens, all supported Markov ranks and canaries against CPU reference. Batched compressor state must equal sequential updates across compression boundaries; cache copies must match ordered copies byte for byte, including snapshot, restore, accepted prefixes, overlap fallback and unaligned spans. `make test-cuda-dspark-moe CUDA_ARCH=sm_121` and its sanitizer check cover 1–8 rows, overlapping expert selections, invalid selections and multilaunch Q2 assignments; distinct rows must equal separate calls. These arithmetic fixtures do not replace V4.1 supported-model testing.

For MXFP4, the historical `make test-mxfp4-cuda CUDA_ARCH=sm_121` and `make test-mmvq-cuda CUDA_ARCH=sm_121` cover dense and routed matrix/vector paths, fused gate/up and down, odd rows, K=256/768, bounds and canaries. Test staged output exactly, including misaligned rows and fallback, and retain Q8 activation accuracy bounds. Compare whole-model continuation and prefill/decode timings, not only primitive speed.

## V4.1 single Spark integration and streaming

The 341 GiB Q2 file included 189 GiB disk-only Engram; do not load it resident on a 128 GB Spark. Test one model process per host with SSD streaming, automatic and explicit expert budgets. Resident mode must reject oversized weights before warming or allocation. The historical `tests/test_deepseek41_cuda`, `tests/test_cuda_ssd_cache`, `tests/test_cuda_q8_rows` and `tests/test_deepseek41_prefill --cuda MODEL speed-bench/promessi_sposi.txt` exercised routing ties, 384 experts, absolute positions, masked IDs, cache eviction and prefetch, cancellation, changed budgets, buffered I/O and continued prefill. Read-ahead must protect active inputs and pending slots, publish only complete reads and evict unused read-ahead before demand-hot experts. Unused prefetched entries are not demand hits, and prefill must not accumulate a second unbounded expert cache. Check full saved state and logits after restore, callbacks and cancellation. Cover 2–8 row and 256-token prefill dispatch boundaries, 8,192-row projections and long deferred-state boundaries through 60K. Under driver 580.173.02 the host backtrace collector could crash; `--show-backtrace device --report-api-errors no --error-exitcode 99` retained device checking.

### Continuation and physical qualification

Score the matching V4.1 short and long manifests including 8/16/32K sparse frontiers and continued prefill. The snapshot recorded general100 NLL 0.363135483 (2696/2994 API tokens), boundary17 0.405201234 (126/141), long9 0.558581488 (491/576). Run real agent editing, server tool loops, streaming cancellation and slot reuse. Test native 2–8-row SSD batches with isolated full-logit, cache and snapshot comparisons; five prefill plus three decode rows is native, whereas 128 prefill plus three decode rows uses ordered fallback. Monitor memory and swap; measure actual appends and subsequent decode with read-ahead and medium sweep both enabled and disabled.

### Historical workload reference

The historical run used a 64 GiB cache hint, 64K context, Promessi Sposi and 32 teacher-forced tokens. Fresh prefill was 84.90 t/s at 2K, 116.34 at 4K, 228.44 at 8K and 384.45 at 32K; an 8K-to-12K append was 111.15 t/s. These are single runs, not comparative medians. The same historical setup used a 64 GiB hint resolved to 56.88 GiB dynamic cache plus 7.12 GiB prefill reserve. Its 3,241-token append after 32K was 87.77 t/s:

```sh
./ds4-bench --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming \
  --ssd-streaming-cache-experts 64GB --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 --ctx-max 36009 --step-incr 3241 --ctx-alloc 65536 \
  --gen-tokens 32 --teacher-forced-decode --csv .workspaces/cuda-append.csv
```

This is a historical command requiring the matching CUDA build. Record the resolved budget, since memory pressure can reduce the hint. Also run at least 256 following decode tokens and compare total latency with read-ahead and medium sweep disabled: the append can displace decode-hot experts. Eight sessions at 1K were about 11 aggregate t/s; do not report this as per-client throughput. The original Flash/Spark and eight-L40S results belong to retired models and do not set a V4.1 floor. See [distributed](distributed.md) for network TP.
