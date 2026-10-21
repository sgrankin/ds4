# F16 storage for already-quantized compressed KV

**Measured main-cache allocation halves with identical outputs on three fixed
fixtures and an exact odd-frontier canonical round trip. No speed gain is
established. A default broad GPU-suite failure also reproduces on the parent.** Item48
stores only the main compressed-KV tensors in F16 instead of F32, enabled by
`DS4_V41_COMPRESSED_KV_F16=1`. It preserves the existing quantizer and uses
existing typed attention with a selected-row gather that stays in F16.
This halves the allocated main-cache bytes. It is not native bitpacked FP4
storage, a new quantizer or a weight conversion. Presets remain unchanged.

## Algorithm and representability assumption

The ordinary main-cache publication path already applies the FP4/E4M3 quantizer
and its final BF16 boundary. Its bounded output alphabet has nonzero magnitudes
from `2^-10` through 2688 and BF16 precision, all exactly representable in F16.
The experiment changes the container for those existing values:

```text
publish: existing quantizer -> existing BF16 values -> store as F16
scalar decode: gather selected F16 rows -> F16 attention staging
batch attention: use existing typed compressed-cache reader
canonical save/load: expand/convert across the ordinary F32 wire representation
```

Item53 adds a packed scalar reader, enabled automatically with F16 storage.
Gather copies the selected F16 words
directly into scratch and supplies F16 to attention, avoiding the expansion
and subsequent repacking. Existing gathered staging fuses the raw-ring and
compressed-row copies where supported. The current DS4.1 path retains its
existing reduction, BF16 boundary and separate inverse-RoPE steps.

Two clean primitive runs measured a 1–16% reduction for gather plus attention,
depending on shape and repeat, with exact outputs. Direct task timings remained within control
drift; a same-frontier continuing-decode measurement was effectively neutral
(20.7582 versus 20.7405 tokens/s). All 273 paired frontier rows matched exactly.
This does not establish a full-model gain. The reader is nevertheless the
default within the opt-in F16 mode: it removes duplicate conversions at
unchanged measured outputs and neutral overall performance. Set
`DS4_METAL_V41_PACKED_F16_GATHER=0` to restore the old expanded reader for
comparison. No extra switch is needed for ordinary use:

```sh
DS4_V41_COMPRESSED_KV_F16=1 \
  ./ds4-local persistent-fast \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

[Item53's method and phase measurements](../speed-bench/experiments/f16-gather-hour53-20261021/README.md)
also show that 94% of Item51's apparent F16 edit penalty was in prefix rebuilding,
not the measured graft-plus-fresh-tail phase. That gap does not establish
intrinsically slower F16 arithmetic.

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

Item48 deliberately rejected suffix capture/relocation for packed sessions;
that historical measurement covered storage and canonical restoration only.
Item51 implements the missing typed graft path. Captured state retains the
shared canonical F32 wire representation, and the immutable pre-RoPE archive
retains its independent BF16 rotary originals. F32 and F16 receiving sessions
can therefore consume the same source layout without changing the disk format.

For packed main-cache grafts, a bounded **256 KiB F32 tile** receives canonical
source rows, repairs their rotary tails with the existing operations, then
stores the rows as F16. The remaining lanes are preserved; raw/index caches and
pending pairs keep their existing F32 paths. There is no permanent expanded
main-cache mirror. Even-aligned left, right and same-position reuse use the
same eligibility checks. Right shifts and recapture of approximate suffixes
require immutable originals; without originals the legacy inverse/forward
repair supports left shifts only. Zero displacement skips rotary repair.

The [Item51 report](../speed-bench/experiments/f16-suffix-hour51-20261021/README.md)
retains the cross-layout, relocation and recapture qualification below. The
low-level matrix passed. All seven HTTP steps and six actual file reads also
passed, including native SSD restoration and two approximate suffix grafts
after restart.
To opt into the combined agent configuration:

```sh
DS4_V41_COMPRESSED_KV_F16=1 ./ds4-local persistent-fast \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

For the separate large-capacity, native-routing storage experiment from Item48:

```sh
DS4_V41_COMPRESSED_KV_F16=1 DS4_SERVER_CONTEXT_SUFFIX_REUSE=0 \
  ./ds4-server --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 64GB \
  --ctx 262144 --prefill-chunk 8192 --mtp-draft 1
```

This is a launch example, not a measured throughput recommendation. Neither
`context-fast` nor `persistent-fast` is changed to select this layout.

## Item51 suffix compatibility and edit cost

All six low-level arms passed. Five original-backed arms covered F32/F32,
F16/F16, both cross-layout directions and a closing F32 repeat. Each restored
byte-identical canonical prefix state and BF16 originals, then completed
right66/left128/right514/zero moves with 513 reused and 128 fresh tokens. All 20
approximate recaptures retained originals; the primary source/tape hashes stayed
immutable.

Each arm computed 16 native and 16 grafted teacher-logit rows. Across layouts,
the native rows matched their F32 native counterparts bytewise, and the grafted
rows matched their F32 grafted counterparts bytewise: **no additional F16 ΔNLL,
KL or argmax change** on these positions. Each combined logit file was
16,547,872 bytes with SHA256
`23b1961a2ac2d14c1610ee256da80ae07476deb2dee716931b1f24309812020b`.
The existing stale-suffix approximation still differs from fresh native
recomputation: mean ΔNLL +0.00004128895 nats/token, mean KL 0.00001934012 and
zero argmax flips over 16 rows, identically in all five layouts.

The separate no-original F16 left128 arm passed four teacher rows, with mean
ΔNLL +0.00002345393/KL 0.00001457319 versus its fresh native computation and no
argmax flips. There was no F32 legacy control, so that arm does not establish
cross-layout logit equality for legacy repair.

| Four-move edit-time sum | Native recomputation | Suffix reuse |
|---|---:|---:|
| F32 opening / closing |190.043 / 190.346 s|142.541 / 144.386 s|
| Full F16 |196.497 s|152.864 s|

The single full-F16 arm was 3.31% slower for native edits and 6.55% slower for
reused edits than the corresponding F32 bracket means. These sums exclude
source setup, prefix compatibility checks, recapture and teacher continuation.
The benefit is allocated storage: main KV at context 8,192 falls
40.008→20.004 MiB, while expert slots stay fixed and disk bytes do not halve.
This screen shows an edit-cost tradeoff, not a new speed gain. Mixed-layout
timings are retained descriptively in the
[summary](../speed-bench/experiments/f16-suffix-hour51-20261021/summary.json).
The test uses fixed teacher continuations, not independent greedy answers or a
whole-agent quality suite. Keep this option explicit; presets remain unchanged.

## Item48 measurement method and capacity scope

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

## Item48 results and credit

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
