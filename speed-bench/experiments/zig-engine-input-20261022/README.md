# Native engine input: mapping, configuration and weight binding

The first engine rewrite milestone passed the existing negative-file checks,
three-family mapping/lifecycle comparison in ReleaseFast and Debug, and a short
native Metal generation check. Work started at 12:18 UTC on 2026-10-22, after the
[modular migration qualification](../zig-final-20261022/README.md). This is an
ownership and compatibility result; it does not establish a startup or inference
speed gain.

## Scope

The active native path now uses four handwritten modules:

- [model.zig](../../../src/engine/model.zig) owns the file descriptors, mapping,
  metadata and tensor directory. `init` uses `errdefer` for partial construction;
  `deinit` closes the owner, and `take` transfers the actual shared C record
  without allocating another owner. Engram and Qwen n-gram tails remain outside
  the resident weight mapping. The Qwen disk descriptor is checked against the
  same file before transfer.
- [config.zig](../../../src/engine/config.zig) constructs a validated DS4.1
  `ModelConfig` by value, including its real shape and layer policies. Its tiny
  `1e-20` RMS epsilon and source/consumer layer roles are documented explicitly.
- [weights.zig](../../../src/engine/weights.zig) binds and validates the DS4.1
  pointer table with an explicit configuration. Tensor descriptors borrow the
  `Model` lifetime; weight binding owns neither payload bytes nor mappings.
- [input.zig](../../../src/engine/input.zig) is the narrow compatibility
  facade. The actual model, configuration and weight records are shared headers,
  rather than parallel Zig layouts.

GLM and Qwen configuration validation and weight binding still use C. Session,
graph and execution code also remain C. The compatibility adapter still copies
configuration into the engine's existing global settings; this milestone does
not yet remove that global coupling. `-Dnative=false` retains the C input path.

A source review found ordinary named helpers, scoped cleanup and explicit
ownership in these modules. Comments explain borrowed strings, tail unmapping,
shared versus private mapping, the tiny epsilon, and the distinction between
cache-source and consumer layers. No semantic admission change was found in the
validated DS4.1 binder.

## Qualification

[ReleaseFast results](evidence/releasefast/results.json) and
[Debug results](evidence/debug/results.json) each contain ten successful fixture
processes: the existing sparse/invalid GGUF fixture and Qwen n-gram fixture for
both implementations, followed by C/native lifecycle checks on three real files.
Each real-file process opens, configures, binds and closes twice: first a private
mapping, then a shared mapping. Across the two campaigns, all **24 real
open/configure/bind/close cycles** passed.

The lifecycle fixture compares all shape scalars and compression ratios, a digest
of every named tensor descriptor and offset, and a digest of every populated
weight-table slot's descriptor index. It also checks that close resets the record,
closes both descriptors where applicable, and unmaps the retained range. C/native
and private/shared identities matched for all three files:

| Model | File bytes | Mapped bytes | Tensors | Bound slots |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix | 365,713,686,528 | 162,955,640,832 | 1,046 | 1,044 |
| GLM-5.3-Flash-Q2 | 96,505,816,384 | 96,505,816,384 | 1,412 | 1,412 |
| Qwen3.8-Flash-Next-Q4 | 177,280,286,720 | 74,879,795,200 | 1,256 | 1,256 |

DeepSeek's Engram tail and Qwen's n-gram tail were outside the mapping; Qwen also
retained its separate disk-read descriptor. The GLM file had no disk-only tail.
No tensor payload warming or generation occurs in the lifecycle driver. Timings
are retained as observations from an ordered screen, with warm-cache effects;
they are not a matched performance experiment.

The [native GPU check](evidence/gpu/result.json) used the 1017 Q2 model, Metal,
48 GB expert cache, context 1,024, prefill chunk 512, MTP width 1, seed 73,
temperature zero and a 16-token limit. It returned exactly `ORCHID42` in 6.906 s
for the whole process. This one short answer qualifies the input seam through
actual execution; it is not a coding, long-context or throughput measurement.

After restoring the unused C `DS4_SHAPE_FLASH41` reference constant for
model-free fixtures, the [final ReleaseFast rerun](evidence/releasefast-final/results.json)
also passed all ten processes and all twelve real lifecycle cycles with the
same identities. Native production configuration was unchanged. Its six
recorded source hashes match the final source capture. The production-linked
session-state fixture also passed (`session state tests: ok`); its
[build record](evidence/native-session-build.json) completed successfully.

All four [pressure guards](evidence/pressure-summary.json) completed with child
status zero, normal pressure in every sample and no swap growth. C/native
production build stderr was empty. The retained [build commands](evidence/build-results.json)
include successful final native and Debug fixture builds; their 1.078 s and
2.397 s elapsed times are cached builds, not clean-build comparisons.

## Reproduction

Run from the repository root, with the three model filenames shown above under
`gguf/`. Build the original C and selected native paths, then run the existing
fixtures serially under the existing pressure guard:

```sh
tools/zig-build -Dnative=false --prefix .workspaces/zig-engine-input-c \
  install gguf-model-test qwen-ngram-test
tools/zig-build --prefix .workspaces/zig-engine-input-native \
  install gguf-model-test qwen-ngram-test
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-engine-input-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-input-20261022/model-qualify.py \
    --c-bin-dir .workspaces/zig-engine-input-c/bin \
    --native-bin-dir .workspaces/zig-engine-input-native/bin
```

The copied [driver](model-qualify.py) defaults to the fresh directory
`.workspaces/zig-engine-input-rerun`; choose `--out` for another run. Existing
output directories fail visibly. Add `-Doptimize=Debug` to native fixture builds
and use a separate prefix/output directory to repeat the Debug screen. The
production engine remains release-optimized by the build graph.

The committed driver records the actual new engine/header paths. The original
local driver used stale path names for those new files; the recorded RF/Debug
results preserve its original source subset and exact binary hashes.
[Post-qualification hashes](evidence/source-sha256.json) add the actual new paths
separately. The RF/Debug source predates the C-only reference-constant
restoration; the final RF evidence records that updated source. These new path
hashes are not presented as hashes taken retroactively during the original run. Raw fixture outputs, command lines, hashes and guard records are
retained under `evidence/`; no model payloads are copied.
