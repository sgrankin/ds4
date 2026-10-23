# Selected-engine checkpoint: build and runtime measurements

The selected engine package passed the final C/native six-turn task pair,
ordinary frontier comparison, CPU smoke checks and source-restoration checks.
All five default installed applications match the qualified native binaries.
This checkpoint includes Resource 72; the broader rewrite goal remains active
for the residual C/Objective-C owners listed below.

The measured source is checkpoint `3979cbb1`, against main
`ec5424de5562269cbe6952cdad7408989df293de`. The
[317-file frozen manifest](evidence/source-freeze.json) records the actual
source hashes and confirms that source stayed unchanged through qualification.
[summary.json](summary.json) collects the results; commands and raw records are
retained under [evidence](evidence/).

## Final agent workflow

Both matching applications passed six turns, three external context edits and
three fresh archive reads. Each follow-up returned active total 416 and archived
code `RIVER-73`. Both runs observed three SSD-backed pre-RoPE suffix grafts,
two after a prior approximate state; the third selected source was itself tagged
approximate. The results explicitly make no immutable-original-source claim.

| Ordered task arm | Whole workflow | Result |
| --- | ---: | --- |
| C control | 235.391 s | All task and experiment checks pass |
| Selected native package | 233.770 s | All task and experiment checks pass |

The [pair comparison](evidence/agent-comparison.json) finds byte-identical
stdout, the same three graft geometries after excluding trace timings, the
same fixed test time and equal initial visible context after removing display
IDs. Both bootstrap prefills used 2,372 tokens. These are practical
archival/retrieval checks, not full-logit or complete input-token parity.
The native arm was 0.69% shorter in this single ordered pair; this does not
establish a speed gain.

[Control](evidence/agent-control/result.json) and
[native](evidence/agent-native/result.json) evidence retain each fixture,
execution settings, raw output and four gzip traces. The artifact indexes
([C](evidence/agent-control/artifacts.json),
[native](evidence/agent-native/artifacts.json)) retain hashes and original paths
for the context text files left in the workspace. After C completed, its shared
workspace/cache was moved to
`.workspaces/zig-final-agent-20261022/control/shared`, allowing native to use a
fresh directory at the same original shared path. This happened after execution
and does not change the recorded commands or observed C facts.

Both pressure guards completed with child return code zero. All 227 C samples
and 226 native samples were normal, with zero swap growth. The existing
[agent driver](../zig-combined-20261022/run_agent.py) is unchanged; exact commands
are in the two guard records and [campaign records](evidence/campaigns/).

## Ordinary runtime bracket

[ordinary.py](ordinary.py) ran C/native/native/C with 512 prefill tokens,
64 forced teacher steps, context 1,024, chunk 512, a 32 GB SSD expert-cache
target, F32 compressed KV, replay/resident routing off and no DSpark model.
All four executions completed the requested counts and produced byte-identical
complete prefill-frontier vectors. The source hashes before and after match.
The frontier is dumped before decode, so this does not compare all 64
teacher-step outputs.

| Arm | Whole process | Prefill tokens/s | Steady decode tokens/s |
| --- | ---: | ---: | ---: |
| C opening | 22.398 s | 35.99 | 17.63 |
| Native first | 23.967 s | 32.76 | 17.92 |
| Native second | 20.944 s | 33.03 | 17.94 |
| C closing | 21.220 s | 32.96 | 17.90 |

Mean whole-process time was 21.809 s C and 22.456 s native, a 2.97% increase.
The C prefill rate drifted down 8.42% across the bracket. These observations are
near-neutral given this drift and do not demonstrate a performance win or prove
non-regression. [Exact metrics, commands and binary/source hashes](evidence/ordinary/results.json),
[raw CSV/logs](evidence/ordinary/) and a
[full frontier gzip](evidence/ordinary/frontier.logits.json.gz) are retained.
All four original frontier files share SHA-256
`93d6454045b400ebb4dc6dad93cbe67ede9430be620510708a2b606cdd32c26c`;
[their index](evidence/ordinary/frontier-files.json) records the workspace paths.
The guard completed successfully: all 86 samples were normal, with zero swap
growth.

## Build cost

[measure_builds.py](measure_builds.py) reuses the existing serial five-program
comparison: Apple Make C versus the selected handwritten native package at
12 jobs. Source-only copies in a fresh ignored workspace warm both builds,
then measure one content-forced full rebuild, two no-ops and two single
Metal-controller edits per variant. The Make edit is `ds4_metal.m`; the native
edit is `zig/metal/matmul_f32.zig`. Temporary comments are restored in `finally`;
the Make 3.81 timestamp barrier is outside timed intervals. No binaries, tests
or models execute during this comparison.

| Measured build operation | Make C | Selected native |
| --- | ---: | ---: |
| Warm content-forced full rebuild, one point | 9.731 s | 8.929 s |
| No-op, mean of two | 0.013 s | 0.524 s |
| Single F32 controller edit, mean of two | 3.481 s | 0.618 s |

The full native point compiled 15 C/Objective-C units and 37 Zig objects,
with two objects cached/reused. Native no-ops compiled nothing, with 39
cached/reused objects; each controller edit compiled one Zig object and reused
38. The selected source list has 72 Zig files, including imported owners and
facades; it is not a count of standalone compiler objects. Warm full rebuild
and controller-edit observations favor the selected package here; no-op
overhead remains roughly half a second. One full point is not a statistical
build-speed conclusion or a cold toolchain-cache measurement.

All commands returned zero and all original source bytes were restored.
[The full build record](evidence/build/build-performance.json) retains measured
and setup/restoration runs, per-step compile summaries and restoration hashes.
[Source hashes and raw logs](evidence/build/) accompany it; larger logs are
gzip-compressed. There were no failed runs in this final campaign.

## Matching installation and CPU smoke checks

Both source variants installed all five applications and built the existing
server/agent fixtures under
`.workspaces/zig-final-all-20261022/{c,native}/bin`. C uses `-Dnative=false`;
the selected package uses `-Dnative=true`.

| Final installation/check | Whole process | Result |
| --- | ---: | --- |
| C install plus server/agent fixture builds | 3.488 s | Exit 0 |
| Native install plus server/agent fixture builds | 0.567 s | Exit 0 |
| C `ds4_test --server` | 0.140 s | Exit 0 |
| C `ds4_agent_test` | 4.996 s | Exit 0 |
| Native `ds4_test --server` | 0.137 s | Exit 0 |
| Native `ds4_agent_test` | 4.993 s | Exit 0 |

[Exact commands and raw output](evidence/install/) retain these successful
checks. The install times reuse build caches and are setup observations, not
full-rebuild measurements. The final default installation completed in 0.554 s;
[all five installed binary hashes](evidence/installed.json) match the qualified
native prefix exactly. Earlier module-specific GPU, payload, suffix, HTTP and
agent qualifications remain in the [milestone guide](../../../src/README.md).

## Reproduction

Run the scripts serially after functionality qualification, with no competing
compiler or model jobs. Use fresh paths. The build script executes no programs;
the runtime and agent scripts use the existing pressure guard.

```sh
python3 -B speed-bench/experiments/zig-engine-final-20261022/measure_builds.py \
  --out .workspaces/zig-final-rerun/builds
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-final-rerun/ordinary-guard -- \
  python3 -B speed-bench/experiments/zig-engine-final-20261022/ordinary.py \
    --control-bin-dir .workspaces/zig-final-rerun/builds/make-source \
    --native-bin-dir .workspaces/zig-final-rerun/builds/native/bin \
    --out .workspaces/zig-final-rerun/ordinary
```

The model is `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
The runtime script records full commands, environment, binary/source hashes,
CSV counts and frontier identity. Finite successful execution and requested
counts are checked; disagreements are recorded rather than rounded or silently
rejected. Resource ownership/ARC files are explicitly in the frozen source
list. The actual executions used `.workspaces/zig-final-all-build-20261022`
and `.workspaces/zig-final-ordinary-20261022`; raw caches and large source trees
remain there, outside the report.

## Source inventory and retained C scope

At this checkpoint the explicit selected list contains 71 owner/facade Zig
files with 20,767 physical lines plus a 60-line binding adapter: 72 source
entries and 20,827 physical lines in total, including comments and blank lines.
A bounded final peer review found no concrete blockers; selected Zig/build
formatter checks and the final diff check passed. Most public records come
from actual C/SDK headers; the earlier checkpoint-cache `Blob`/`Stats` manual
extern-ABI exception is documented in the [guide](../../../src/README.md).

The selected package owns active V4.1 model/configuration/weights,
graph/resources, scalar/batch attention, execution/MoE/prefill/streaming,
payload/suffix and speculative orchestration, plus the qualified host/front-end
modules in the guide. C retains public engine/session facades and some complete
diagnostic services, other model-family compute paths, server
accept/materialization/durable-publish policy and agent worker/UI transactions.
Most Objective-C kernel-dispatch support and all existing Metal shaders remain.
These remaining owners are subsequent work; this report qualifies the selected
checkpoint and does not close the whole-rewrite goal.
