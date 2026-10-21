# F16 storage for already-quantized compressed KV

**Measured main-cache allocation halves with identical outputs on three fixed
fixtures and an exact odd-frontier canonical round trip. No speed gain is
established. A default broad GPU-suite failure also reproduces on the parent.** Item48
stores only the main compressed-KV tensors in F16 instead of F32, enabled by
`DS4_V41_COMPRESSED_KV_F16=1`. It preserves the existing quantizer and uses
existing typed attention with a selected-row gather that expands to F32.
This halves the allocated main-cache bytes. It is not native bitpacked FP4
storage, a new quantizer or a weight conversion. Presets remain unchanged.

## Algorithm and representability assumption

The ordinary main-cache publication path already applies the FP4/E4M3 quantizer
and its final BF16 boundary. Its bounded output alphabet has nonzero magnitudes
from `2^-10` through 2688 and BF16 precision, all exactly representable in F16.
The experiment changes the container for those existing values:

```text
publish: existing quantizer -> existing BF16 values -> store as F16
scalar decode: gather selected F16 rows -> exact F32 expansion -> attention
batch attention: use existing typed compressed-cache reader
canonical save/load: expand/convert across the ordinary F32 wire representation
```

Index caches, raw sliding-window rings and unfinished compression-pair state
remain F32. The attention algorithm, positions, routing and weights are unchanged by this
storage choice; typed readers may still change arithmetic rounding. The representability argument applies to the existing bounded
quantized alphabet; it is not a guarantee for arbitrary unquantized F32 caches.
The focused GPU check passed exact comparisons for the quantizer, offset
storage conversion and sparse gather. It did not test standalone attention.
Typed attention can choose differently rounded arithmetic, so storage
representability alone does not prove full-model logit identity. The actual
fixture/round-trip qualifications below test that separately.

## Runtime and persistence boundaries

This local implementation is an Apple Metal opt-in. Set it at process start
and retain the setting for the lifetime of its sessions; allocation planning
also reads the process setting. Other backends keep their current layout.
Canonical snapshots retain the existing F32 format and header: saving expands
main-cache values, and loading converts into the receiving session's layout.
This is intended to preserve existing canonical prefixes and their disk bytes;
this feature does not halve canonical SSD-cache files or snapshot staging RAM.

Original-backed suffix capture/relocation currently rejects packed sessions.
This is unfinished typed-copy and rotary-repair work, not a fundamental
incompatibility with the independent BF16 pre-RoPE archive. The current graft
copies compressed rows using four-byte offsets, and its rotary repair operates
on F32 views; those operations cannot be used directly on two-byte F16 storage.

Supporting relocation would require typed compressed-row copies and bounded
unpack/rotate/requantize/repack work for the rotary tails. It would not require
a permanent full-cache F32 mirror. Until that work is implemented and measured,
canonical SSD prefixes remain usable with suffix reuse disabled; the combined
packed-layout/suffix mode is unsupported.

Use the server directly with suffix reuse disabled while qualifying the option:

```sh
DS4_V41_COMPRESSED_KV_F16=1 DS4_SERVER_CONTEXT_SUFFIX_REUSE=0 \
  ./ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 64GB \
  --ctx 262144 --prefill-chunk 8192 --mtp-draft 1
```

This is a launch example, not a measured throughput recommendation. Neither
`context-fast` nor `persistent-fast` is changed to select this layout.

## Measurement method and capacity scope

The fixed model experiment keeps context capacity 262,144, live prompts 8,192,
expert cache 64 GiB, native routing and decoder replay off. The bracket compares
F32/F16/F16/F32 with the same changed binary. The observed native allocated
main-cache capacities are 1,342,185,472 bytes for F32 and 671,092,736 bytes for
F16, confirmed by the allocation logs.
Index capacity 335,546,368 bytes and raw rings 10,485,760 bytes are unchanged.
These are **allocated tensor capacities**, not process RSS or physical memory
pressure measurements. Expert-cache capacity stays fixed 64 GiB; saved KV bytes
are not automatically assigned to additional expert slots. There is no 2×
whole-process/session-capacity or serving-speed claim. At context 8,192 the
same main-cache formula saves 20.004 MiB. At 262,144 the main+index+raw tensor
capacities total 1,610.010→970.006 MiB (39.75% less), excluding other buffers.

The narrow [cross-layout probe](../speed-bench/kv_pack_roundtrip_bench.c) opens
one engine, prefills its own harmless 8,192-token history and evaluates one known
teacher token before snapshot A. That odd 8,193-token frontier exercises a live
raw ring and pending compression state. It then:

1. Saves canonical A in native F32, evaluates the next common teacher token and
   retains those freshly computed logits.
2. Creates an F16 session, loads A and saves B; compares every canonical byte.
   Evaluates the same next teacher token and compares finite logits bytewise.
3. Creates a fresh F32 session, loads B and saves C; compares B/C bytes, then
   evaluates that same fresh token and compares its logits with the reference.

Only two in-memory snapshots coexist; no model-sized artifacts are written.
Loaded logits are not counted as fresh quality evidence. The probe's test-only
layout changes occur after freeing each previous session. Run it under the
existing pressure guard after building, with an existing fresh output directory:

```sh
make -j2 speed-bench/kv_pack_roundtrip_bench
mkdir -p .workspaces/item48-roundtrip
python3 speed-bench/ideas/pressure_run.py .workspaces/item48-roundtrip-guard -- \
  speed-bench/kv_pack_roundtrip_bench \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --out .workspaces/item48-roundtrip
```

## Results and credit

The 0,1,1,0 bracket measured native practical sums 68.955283/70.964805 s and
candidate sums 68.997398/69.567117 s. Candidate mean 69.282258 s versus native
mean 69.960044 s is 0.969% faster than the mean but 0.474% slower than the faster
control, with 2.914% native bracket drift. **Latency is unresolved/neutral.**
All four arms' 22 common teacher rows were byte-identical (ΔNLL=KL=0), and all 12
independent first answers stopped and passed. These are three fixed 8,192-token
fixtures at large allocated capacity, not a full 262k-context quality test.

The odd 8,193-token canonical check passed: A/B/C were 63,479,352 bytes each,
A/B and B/C were byte-exact, and all 129,280 freshly computed logits after the
next common token at 8,194 were finite and byte-identical across both layout
transitions. This is a canonical in-memory compatibility result, not an SSD
cache lifecycle, HTTP, coding or agent-workflow qualification.

Focused quantizer/storage/gather GPU tests and model-free server/agent checks
passed. The default full Metal suite stops at a quantization assertion before the new
packed case. The extracted `af29fced` parent with the same flags, support
objects and shaders fails identically, so it is a confirmed baseline failure;
the default broad suite is not passing. The diagnostic
`DS4_METAL_MATH_SAFE=1 ./tests/test_deepseek41_metal` passed the full suite,
including the packed case. Defaults and timed model runs retain fast math;
neither the runtime nor oracle was changed to resolve the baseline failure.
[Experiment report](../speed-bench/experiments/packed-kv-hour48-20261020/README.md)
retains the first results, repeat ranges and failure logs. Presets stay unchanged.

This builds on [antirez/ds4](https://github.com/antirez/ds4)'s quantizer and typed
Metal attention/cache readers. The broader cache-layout investigation followed
[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)
and the [native scale pilot](07-native-scale-packing.md). This F16 container is a
local storage adaptation, separate from DeepInfer's distributed cache formats
and the native MXFP4 expert/Engram dictionary experiments.
