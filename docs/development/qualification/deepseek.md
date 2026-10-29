# DeepSeek qualification on Metal

Use a checkpoint-matched V4.1 Flash model and support files. V4.1 is a different
architecture from retired V4 Flash; do not refresh old vectors through an API
alias now serving V4.1. Run model jobs serially under the [pressure guard](../benchmarking.md#guarded-execution),
with admitted memory and context. The [shared checks](native-and-protocol.md)
cover storage, protocol and performance. All commands below start at the repo root.

## Artifact, quality and basic inference

Audit the release GGUF and native Engram rows against pinned source with
`python3 gguf-tools/deepseek41_validate_gguf.py --payload`; for calibrated weights use
the actual imatrix and record expert coverage. Q4 requires `--quant q4` in both
converter and validator, and conversion from source safetensors, not Q2.
Inventory validation is not quality validation. Check downloads and Q4 split
assembly, interrupted transfer, space and checksums before opening the result.

```sh
make test-engram test-deepseek41-gguf test-quality-api test-frontends test-download-model
make quality-score
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ctx 32768 --nothink -p 'Explain C pointers in one paragraph.'
./ds4 --nothink --temp 0 --dump-logprobs .workspaces/ds4-logprobs.json --logprobs-top-k 20 -p 'Explain C pointers.'
```

Also exercise thinking and max-thinking, coding, long recall and realistic
continuations. Use the matching `deepseek-v4.1-flash-20260910` fixture and its
100-case `20260910-general` companion under `gguf-tools/quality-testing/`;
score long and extended manifests for sparse boundaries and 8/16/32K and
64/96K prefixes. Compare teacher-forced probabilities and API top-token
agreement as well as practical tasks. For boundary changes cover 127/128/129,
511/512/513, 1023/1024/1025 and 16383/16384/16385 tokens, including continued
prefill. Keep calibration coverage and the fallback for unobserved experts.

## State, admission and prefill

Build `./tools/zig-build graph-test` and run
`zig-out/bin/test-deepseek41-graph MODEL --session-fixture` and `--partitions`
under Metal validation. Check continuation logits, compression carry, sliding
window wrap, cancellation and malformed snapshots. Use `--long-sessions` with
the fixture's `prompts/case_016.txt` for restored continuation through 16,385
tokens; its two sessions share a 64 GiB expert cache. Zero-weight fixtures do
not establish quality. Never run full main-model residency on a 128 GiB Mac;
Engram stays disk-only. Check `--memory-plan` with Q2 and Q4, then verify actual
host residency separately; simulated admission is not physical qualification.
`--warm-weights` must warm only a TP shard and skip a full-file pass in SSD mode.

### Prefill parity

After scheduling changes, use `--prefill-parity speed-bench/promessi_sposi.txt`
under validation to compare full logits and saved cache spans with token-major
execution, including continued prefill, restoration and cancellation. A partial
layer stack must not become a snapshot. For resident encoder changes compare
`--encoder-parity`, `--encoder-long-parity` and `--encoder-cancel` against
`DS4_METAL_DISABLE_V41_ENCODER_RESIDENCY=1`; verify cache budget restoration,
no pinned encoder pages after cancellation, and two-layer fallback when it
cannot fit. Measure initial and continued prefill, loading and first decode,
not just prefill throughput.

### Batch isolation

The former session-batch and MoE fixture sources do not have current Zig build
targets. Restore production-linked fixtures before running these objectives;
do not use stale C-era binaries as release evidence. For session batches, check reversed row order, at least six decode steps, mixed
prefill/decode, invalid batches and subsequent continuation. Compare complete
logits, not just argmax. Isolation from companion rows and scalar equivalence
are separate checks. Q4 native batching has a historical 0.0002 full-logit
bound against scalar controls; isolation and cache state remain exact. Require
`native_ds41=1` and nonzero `prefill_rows` for admitted mixed native shapes;
image-bearing sessions and two-row TP can take ordered fallback. Test 2, 4 and
8 sessions and real concurrent server cancellation and prefix reuse. Report
model-step latency and aggregate tokens/s without validation. For MoE coverage include IQ2 eight-expert and Q4/MXFP4 six-expert routes,
cold and hot evictions, physical buffer release and table admission. Compare
Q2 and Q4 decode batches with independent single-row calls across ownership
partitions, including untouched output tails.

### Physical TP

Physical TP requires authorized hardware and separate fixtures; these
objectives are not established by the current native build alone. Use the [distributed guide](../../backends/distributed.md) and
an authorized pair. Verify rank memory admission, matching prefill chunks and
actual RDMA device/GID, not merely IP connectivity. Exercise TCP and RDMA,
identity mismatch, worker disconnect and paused-peer timeout, session
invalidation and recovery; preserve both logs. Do not bypass memory guards.
Score full vocabulary across ranks and investigate persistent differences.

## DSpark

Use the checkpoint-matched `DeepSeek-V4.1-Flash-DSpark.gguf`, not a V4.0
drafter. Its approximately 8.04 GiB resident support model needs headroom along
with target and context. For conversion run
`python3 tests/test_deepseek41_dspark_conversion.py` and verify every tensor
against pinned source using `python3 gguf-tools/deepseek41_dspark.py --hf SOURCE --source-revision REVISION --out DSPARK --verify`.
Preserve source values and MXFP4 experts.

```sh
./tools/zig-build dspark-native-oracle
zig-out/bin/dspark-native-oracle --candidate-mask
zig-out/bin/dspark-native-oracle TARGET DSPARK DUMP_DIRECTORY
python3 tests/deepseek41_dspark_reference.py SOURCE DUMP_DIRECTORY
zig-out/bin/dspark-native-oracle TARGET DSPARK --runtime
```

The independent oracle checks captured target features, all three stages,
Markov logits and confidence. Runtime checks must cover changed rejected future
tokens, retained prefixes zero through six, stop rewinds, continuation, save
and load, and both sampling modes. Missing draft features on older snapshots
must be warmed rather than reused stale. With a matched drafter, compare 400 and
800 token temperature-zero continuations to `--dspark-strict`; record first
difference, acceptance, direct commits, fallbacks and speed. Floating-point
ordering means byte identity is not required; verifier errors, invalid text and
material task regressions require investigation. Exercise real code and prose,
tool boundaries, client stops, exact sampling and cancellation near capacity.
Working speculation does not itself establish a speedup.

The former `score_official_verify` has no current Make or Zig build target.
Restore a production-linked verifier scorer to compare scalar and `--verify-batch 6`
on `gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/manifest-sensitivity.tsv`
at context 34816, and its `manifest-long.tsv` with `--continued-prefill 32`.
Use the same target, drafter, context and `--ssd-streaming
--ssd-streaming-cache-experts 40gb` settings for each pair. Report paired
teacher-forced NLL and API top-token agreement, and compare the complete
vocabulary. Repeat on physical TP with matched workers if TP is claimed.
Sampled text equality and greedy prefix length do not replace this check.

Physical TP requires
matching drafter options on both ranks and separate runtime and mismatch checks.

## Retained PRO path

PRO is experimental. On an admitted PRO-capable host, test a short Q2 prompt,
template, thinking behavior and endpoint aliases, and compare SSD and resident
continuations where each fits. The [scorer documentation](../../../gguf-tools/quality-testing/README.md)
explains checkpoint-specific fixtures: the June preview uses `data/pro/manifest.tsv`
and the August 0813 release uses `data/pro-0813/manifest.tsv`. Do not compare
a checkpoint against the other fixture as a quantization result. Run Q4
distributed only on suitably admitted high-memory hosts. If PRO cannot be
exercised, build the binaries and review changes to model shapes, tensor lookup,
routed expert mapping, templates and KV payload compatibility, then explicitly
mark runtime and quality scope unvalidated.

## Vision and agent

Audit the separate V4.1 encoder against pinned source. Compare encoder outputs
with independent official BF16 results for OCR, layout, diagrams and screenshots;
check finite logits, changed image embeddings, replay and restored logits in
resident, SSD and physical TP where claimed. Exercise server vision cache and
agent image tools, including changed or reordered images. A plausible description
alone is insufficient. With `tests/test_agent_vision.py`, independently check
two-image code edits and prefix reuse, including a long context with enough room
for image tokens and tool results. Test thinking levels and changes in a live
conversation: the old prefix must be invalidated without losing messages.

## Historical reference values

These stored scores are historical fixture references, not bit-identical
cross-hardware gates. Pair cases on the same prefill schedule; hardware and
routing can change rounding. The original record did not pin a revision for
these values, so re-establish a baseline before a current release comparison.

| Model and execution | Fixture | Mean NLL | API top-token agreement |
| --- | --- | ---: | ---: |
| Q2, M5 SSD | General 100 | 0.364576009 | 2697/2994 |
| Q2, M3 Ultra resident | General 100 | 0.365680596 | 2705/2994 |
| Q2, Metal TP | General 100 | 0.361471738 | 2699/2994 |
| Q4, M5 SSD | General 100 | 0.246523256 | 2896/2994 |
| Q4, M3 Ultra resident | General 100 | 0.252409161 | 2896/2994 |
| Q4, M5 SSD | Long 9 | 0.335651410 | 542/576 |
| Q4, M3 Ultra resident | Long 9 | 0.328760094 | 538/576 |
| Q2, M3 Ultra resident | Extended 6 | 0.558586149 | 327/384 |
| Q2, Metal TP | Extended 6 | 0.560655370 | 328/384 |
| Q4, M3 Ultra resident | Extended 6 | 0.366465846 | 360/384 |

A historical Q4 resident M3 Ultra 512 GiB sweep, 36,864 allocation, 128 output
tokens, excluded loading and reported medians of three complete sweeps after warmup. At 4K, 8K, 16K
and 32K frontiers its prefill was 341.75, 339.10, 641.49 and 715.96 tokens/s;
decode was 18.69, 18.67, 18.32 and 18.11 tokens/s. Later frontiers were appends.
Use `./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf --prompt-file speed-bench/promessi_sposi.txt --ctx-start 4096 --ctx-max 32768 --step-mul 2 --ctx-alloc 36864 --gen-tokens 128 --show-output --csv RESULT.csv` to reproduce the workload, but use the current five-sample methodology for a comparison.
