# Item51: F16 compressed KV with suffix reuse

2026-10-21; one-hour budget 01:53–02:53 UTC. completed02:34 UTC, within budget. **All six low-level arms and all seven HTTP
steps pass. Original-backed layout comparisons are byte-exact; SSD native and
approximate restoration work after restart.**
The [Item48 report](../packed-kv-hour48-20261020/README.md) remains the historical
storage-only qualification, where packed sessions rejected suffix reuse.

## Implementation and bounded method

Suffix sources retain shared canonical F32 state and independent BF16 pre-RoPE
rotary originals. A packed receiving session loads main-cache source rows into
a 256 KiB F32 staging tile, applies existing original-based or legacy rotary
repair, then stores F16. Raw/index/pending state remains F32. No full expanded
main-cache mirror or new disk format is added. Even-aligned left/right/zero
moves keep existing eligibility and fresh-tail checks; right shifts and
approximate recapture require immutable originals. Legacy no-original repair
is qualified separately on a left shift.

Reuse the existing [hour39 probe](../recipe-rightshift-hour39-20261008/repeat.c),
extended with `--source-f16 0|1`, `--destination-f16 0|1`, optional fourth
zero-displacement move, and `--moves 1 --prerope 0` for legacy left128. Default
three-move CLI behavior is retained. Each layout option defaults to the startup
environment. Source and receiving sessions are sequential on one engine.

The matrix covers F32/F32, F16/F16, F32/F16 and F16/F32. A source at the odd
1,537-token frontier has 513 reusable tokens. Before edits, the receiving
session restores its exact prefix and recaptures byte-identical canonical state
and originals. It then compares fresh native recomputation with suffix reuse
for right66, left128, right514 and zero shift, computing 128 fresh tokens after
each graft. Pending-pair, ring-wrap and source/tape hash checks are retained.
Every original-backed reused endpoint is recaptured after the fresh tail to
check its approximate tag and originals, then freed; the original immutable
source remains the source for all measured moves.

The probe records four fixed-teacher rows per move, ΔNLL/KL, argmax changes and
native/reuse logits. It does not independently generate a greedy answer or
perform an agent/tool workflow. Zero shift still follows an edited prefix and
does not imply equality with fresh recomputation; only rotary repair is skipped.
Edit timing excludes source setup, compatibility checks, recapture and teacher
continuation. Run one model job at a time under the existing pressure guard.

## Low-level results and limits

Source base `2436f414f083b7b04a9b0dedec4788e6f6bd62d0`; commands, binary
hashes and successful server/agent model-free checks are in
[evidence/checks/checks.json](evidence/checks/checks.json). The verified 1017
IQ2_XXS/Q2_K model runs with native routing, context8192/chunk512, expert
cache64GiB, selected tile64/limit1024, replay0. Source setup, prefix
compatibility, approximate recapture and teacher time are outside edit timing.

All six arms pass. The five original-backed arms have identical full logit
files (32 fresh rows,16,547,872 bytes each), SHA256
`23b1961a2ac2d14c1610ee256da80ae07476deb2dee716931b1f24309812020b`.
F16 adds ΔNLL=KL=0 and no argmax changes against the corresponding F32 native
AND grafted histories. Prefix restoration and recapture preserve canonical
source bytes and original tape. All20 original-backed grafts recapture as
approximate with originals; every source/tape hash stays unchanged. The
unsupported no-original right shift rejects before mutation in each arm.
Zero shift reports no rotary repairs. All moves copy three pending pairs.

The already-approximate graft's own difference from native recomputation is
mean ΔNLL +0.00004128895 nats/token, mean KL0.00001934012, max KL0.00010298367
on16 teacher rows; all five layouts share those values, with no argmax flips.
The legacy F16 left-only branch passes four rows, mean ΔNLL +0.00002345393 and
KL0.00001457319, no flips. There is no F32 legacy control; do not assign that
entire legacy approximation error to packing or claim its layout equivalence.
These are four-token access-code teachers, not independent greedy answers or
broad coding qualification. [Metrics and hashes](summary.json),
[per-row data](evidence/moves/02-f16/rows.csv),
[first traces](evidence/moves/02-f16/stderr.log).

| Source→receiving layout | Native edit sum | Grafted edit sum |
|---|---:|---:|
| F32→F32 before |190.043s|142.541s|
| F16→F16 |196.497s|152.864s|
| F32→F16 |197.626s|148.394s|
| F16→F32 |192.067s|148.103s|
| F32→F32 after |190.346s|144.386s|

The full F16 arm is3.31% slower for native and6.55% slower for grafted edits
than the two F32 control means. Control drift is+0.160%/+1.295%. Mixed-layout
runs also vary despite identical logits; this is one ordered technical matrix,
not a full serving ABBA. At this capacity main allocation is41,951,232→20,975,616
bytes (40.008→20.004MiB); the main cache halves, not the entire runtime. Keep
F16 optional: memory falls, but this small edit workload has an observed cost.
Saved bytes are not automatically reassigned to expert slots, and disk bytes
remain F32. Item48 retains the separate large-capacity allocation results.

All1,969 pressure samples were normal; peak wired+compressed99.170GiB and max
swap growth22.0MiB. One heavy job ran at a time. Raw logits/binaries remain
under `.workspaces/`; published evidence keeps small CSVs, hashes and traces.
The analyzer uses only standard Python.

## Commands

```sh
mkdir -p .workspaces
make -j2 ds4-server ds4-agent
clang -O3 -g -mcpu=native -Wall -Wextra -std=c99 -I. \
  speed-bench/experiments/recipe-rightshift-hour39-20261008/repeat.c \
  ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o \
  ds4_layer_pack.o ds4_engram.o -lm -pthread \
  -framework Foundation -framework Metal -o .workspaces/item51-relocate
python3 speed-bench/ideas/pressure_run.py .workspaces/item51-moves-guard -- \
  python3 speed-bench/experiments/f16-suffix-hour51-20261021/run.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --binary .workspaces/item51-relocate --output .workspaces/item51-moves
python3 speed-bench/experiments/f16-suffix-hour51-20261021/analyze.py \
  .workspaces/item51-moves --output .workspaces/item51-summary.json
```

## Application restart qualification

The existing Item12 launcher accepts an explicit model, F16 and one native
prefix probe after restart. Two histories retain unchanged inventory tool
results, remove obsolete prose, archive facts in real JSON files and require
fresh reads after each edit. Replay and resident substitution are off. This
checks native prefix restoration and subsequent approximate SSD suffix use,
including recaptured sources after restart. All7 first steps pass, with6 real
file reads (two inventory reads, four archive reads). Both first edits graft
native SSD sources into F16 (1242/1244 reused tokens). After restart the
unchanged native probe restores1368 tokens, HTTP200 in0.584s; its one-token
output is deliberately capped and is not a task grade. Both subsequent edits
graft832 tokens from original-backed approximate SSD sources, then return the
newly read RIVER-74/CEDAR-92 facts and correct416/202 inventory totals. The
entire latest User message remains fresh. This demonstrates one recaptured
approximate-source continuation across restart, not an unbounded trajectory.

[Requests, first outputs and reads](evidence/http/http),
[native prefix proof](evidence/http/native-prefix-probe-result.json),
[restart traces](evidence/http/after-server.stderr.log). The edited workflows
take28.458/28.354s before restart and30.591/32.136s after. These are functional
F16 integration observations with no paired F32 application control; they do
not establish a whole-agent speed effect. Saved state remains canonical F32
plus BF16 originals; disk quota4096MiB/hot0, two resident sessions. Both servers
exit normally; after restart the cache retains30 endpoints/1647 chunks using
462,726,941 bytes with255 disk chunk reads and no evictions. All231 HTTP
pressure samples were normal, peak wired+compressed95.772GiB, maximum swap
growth0.94MiB.

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/item51-http-guard -- \
  python3 speed-bench/experiments/server-workflow-hour12-20261004/launch.py \
  --mode reuse --compressed-kv-f16 --native-prefix-probe --fresh 64 \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item51-http --cache .workspaces/item51-cache \
  --shared .workspaces/item51-shared
```

## Decision

The option remains
`DS4_V41_COMPRESSED_KV_F16=1`; existing presets do not enable it automatically.
This compatibility follow-up does not establish a new serving-speed gain or
extend Item48's fixed-fixture quality result to arbitrary contexts.

[Portable method and enabling example](../../../recipes/13-packed-compressed-kv.md).
