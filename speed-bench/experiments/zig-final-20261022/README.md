# Handwritten Zig restart: final qualification

Final implementation snapshot: `9649f2a5`, based on frozen C main `ec5424de`.
This is a selective migration of 27 handwritten implementation files, not a
complete engine rewrite. The [module guide](../../../zig/README.md) lists the
boundaries, individual experiments, retained costs and substantial remaining
C/Objective-C scope. Reports named `hourN` identify sequential items, each with
a one-hour ceiling; they do not mean 27 elapsed hours.

## Combined checks

- All **78 fixture invocations** and four builds pass. Existing C/native
  fixtures run in Debug and ReleaseFast; five additional native bridge/vocabulary
  fixtures run in both modes. These link the selected production implementations.
- The actual **Debug F32 router GPU test passes 90 cases**, under the existing
  pressure guard. Both recorded samples are normal, with unchanged swap.
  The return status and command are retained here; its case output was displayed
  in the execution transcript, rather than captured by the guard.
- The default `tools/zig-build` installation passes all five program help
  checks and both preset binary-selection checks. All five installed executable
  hashes match the final ReleaseFast fixture-build installation.
- `zig fmt --check build.zig zig` passes. Independent source reviews found no
  additional ownership, ABI or format regression in the recent boundaries.

[Fixture results](fixtures-results.json), [fixture binaries](fixtures-binaries.json),
[Debug GPU command](debug-router-command.json), [guard result](debug-router-result.json),
[pressure samples](debug-router-pressure.jsonl), [installation checks](install-results.json),
[installed binaries](install-binaries.json), [toolchain](toolchain.json).
Raw fixture/help output remains under `.workspaces/zig-final-20261022/`.

The CPU sweep complements these actual model/GPU workflows:

| Workflow | Result and scope |
| --- | --- |
| SSD/KV interchange and persistent context | Both C/native restart directions retrieve archived facts through actual tools; three context edits reuse SSD/pre-RoPE suffix state and answer correctly. [Combined serving snapshot](../zig-combined-20261022/README.md) |
| Sampling | Twelve model runs retain paired continuations across exposed sampling/penalty modes; the actual numerical fixture covers top-k. [Sampling](../zig-sampling-hour22-20261022/README.md) |
| Resident routing | All six coding implementations and substitution counts match across C/native/C; native whole-task time is 1.60% above the bracket mean. [Resident policy](../zig-resident-route-hour23-20261022/README.md) |
| Agent files | Actual read/edit/write tools produce identical continuations and correct bytes, metadata and attributes. [File operations](../zig-agent-file-hour24-20261022/README.md) |
| Model input | All 532,480 decoded vocabulary entries and 96 token vectors match across three families; actual DeepSeek/Qwen model-opening paths pass. [GGUF input](../zig-gguf-input-hour25-20261022/README.md) |
| F32 Metal controller | Both implementations pass the real router/model-view fixtures. Four saved model-frontier logit vectors match; steady decode is effectively unchanged. [F32 controller](../zig-metal-f32-hour26-20261022/README.md) |
| Responses persistence | Both C-writer/native-reader and native-writer/C-reader survive actual server restarts, GET/DELETE records and reconstruct a child with `previous_response_id`; all four model calls answer correctly. [Raw storage](../zig-response-store-hour27-20261022/README.md) |

These records qualify their own snapshots and workflows. They do not establish
all possible model histories or a whole-runtime speedup. Known small image
rounding differences and the C/native shared default-fast-math FP4 oracle
failure remain documented in the guide. No shader/tolerance change hides them.

## Final persistent-agent integration

The final C and native installations also run the existing six-turn agent loop
with fresh per-arm caches: 32 GB expert cache, context 8192, chunk 512, greedy
sampling, pre-RoPE suffix reuse and fresh tail 128. The checkpoint cache is
SSD-only (`hot_mb=0`, 512 MB SSD). This keeps the original workflow settings;
the other model/GPU screens above used their separately recorded 48 GB setting.

Both arms complete six turns and three external context edits. Each of the
three graded queries actually reads the archive again and answers exactly
`{"active_total":416,"archived_code":"RIVER-73"}`. Each arm makes three
SSD-backed pre-RoPE grafts; two occur after an earlier approximate state.
C reuses 2064/1539/1118 tokens, native 2065/1542/1121. Geometry comes from the
actual trace, without token-parity retries. As in the original loop, this does
not independently prove that an original pre-RoPE master was never modified.

All 437 pressure samples are normal; swap remains 0.012695 GiB and peak wired
memory is 42.703 GiB. The ordered arms take about 225/228 seconds. This is a
functional integration check, with different per-arm paths and opaque turn IDs;
those two times do not isolate language/runtime speed.

[C result](agent-c-result.json), [native result](agent-native-result.json),
[C execution/settings](agent-c-execution.json), [native execution/settings](agent-native-execution.json),
[outer commands](agent-run-results.json), [guard result](agent-guard-result.json),
[pressure summary](agent-pressure-summary.json), [repeat wrapper](agent_workflow.py).
Full transcripts, edited contexts, cache files and pressure samples remain under
`.workspaces/zig-final-20261022/runtime-agent*`.

## Build comparison

Both builds use the same final source, Apple toolchain, warm caches and `-j12`,
building all five programs in isolated source copies. One content-forced full
rebuild and two no-op/edit points per implementation:

| Operation | Make C | Native Zig |
| --- | ---: | ---: |
| Full rebuild | 9.184 s | 9.557 s |
| No-op, mean | 0.0127 s | 0.3997 s |
| F32 controller edit, mean | 3.2460 s | 0.5005 s |

The full native point costs 4.1% more. The selected controller edit takes 84.6%
less time because it recompiles the small native module instead of the large
Objective-C unit. This is not a blanket incremental-build gain: earlier cache
and CLI edits through Zig cost more than their small C counterparts. No-op
overhead remains visible. These few local samples are observations, not a
confidence interval or a result for every development machine.

Edits append comments, preserving runtime semantics. Make's 1.05-second
timestamp barriers are outside every timer. Successful compiler steps verify
that an edit did work; the native edit compiles one Zig object and no C units.
Both source copies are restored byte-for-byte against their original manifest.
[Commands, individual points and source hashes](build-results.json).

The first attempt is retained in [build-first-attempt.json](build-first-attempt.json).
It stopped because the measurement script mistook verbose Zig compiler
invocations for actual compilation on a cache hit. The corrected loop reads
Zig's standard `--summary all --color off` success/cached statuses, preserving
verbose invocations separately. A [second stopped attempt](build-second-attempt.json)
encountered the summary's collapsed `reused dependencies` spelling. Its failed
no-op row is retained and excluded. After accepting that standard spelling, the
remaining two native no-op/edit points were appended using the same runner
functions and existing source/cache; completed full/Make points were not
repeated. Both stops were measurement failures, with no runtime source change.
The literal recovery loop remains in the useful workspace as `finish.py`.

## Repeat the checks

Run from the repository root with the toolchain above. Choose a fresh output
directory each time; the loops leave commands, output and results there. Do not
run model or compiler jobs alongside the timed build comparison.

```sh
python3 -B speed-bench/experiments/zig-final-20261022/fixtures.py \
  --out .workspaces/zig-final-rerun

python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD -- \
  .workspaces/zig-final-rerun/native-debug/bin/test-metal-dense-mpp --router

# Final selected C/native programs, six turns and three context edits per arm.
python3 -B speed-bench/ideas/pressure_run.py NEW_AGENT_GUARD -- \
  python3 -B speed-bench/experiments/zig-final-20261022/agent_workflow.py \
  --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --binaries .workspaces/zig-final-rerun --out .workspaces/zig-final-agent-rerun

python3 -B speed-bench/experiments/zig-final-20261022/measure_builds.py \
  --out .workspaces/zig-build-rerun

tools/zig-build
./ds4-local --bin-dir "$PWD/zig-out/bin" persistent-fast --model MODEL
```

The original Make build and `tools/zig-build -Dnative=false` remain available.
Native modules do not automatically enable approximate runtime features.
