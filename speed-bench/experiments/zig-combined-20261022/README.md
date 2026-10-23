# Combined handwritten Zig qualification

The frozen code snapshot is `b8c71f98` (2026-10-22), before the subsequent numeric
sampler experiment. Main remains `ec5424de5562269cbe6952cdad7408989df293de`.
All five Metal programs build with selected native modules or C references.
The retained source is handwritten, uses real project headers, and keeps SDK
binding generation in compiler caches. [Current module guide](../../../src/README.md).

## Actual workflows

All runs are sequential under the existing pressure guard. The model is
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`; prefill chunks are512 rows.
Server/vision runs use48GiB expert cache; the existing agent runner uses32GiB.
No runtime approximation defaults were changed by selecting Zig.

| Qualification | Outcome |
| --- | --- |
| C publishes, native server restarts | Restored8622 tokens; HTTP cached-token count agrees; actual tool discovery, note read and exact answer pass. |
| Native publishes, C server restarts | Restored8623 tokens; the same accounting/retrieval checks pass. |
| Repeated live-context edits | Six turns, three edits, three fresh archive reads; every answer is `{"active_total":416,"archived_code":"RIVER-73"}`. |
| SSD/pre-RoPE suffix reuse | Three grafts reuse2068/1543/1126 tokens; two follow a previously approximate context. All report `source_ssd=1`, `pre_rope=1` and latest-user-fresh. |
| Image-to-answer and recall | Both C/native encode a112×84 red PNG, answer RED, then recall RED. Complete stdout matches. |

Each server pair uses distinct processes, a byte-identical first request and
fresh disk-only application cache (`hot=0`). The two directions use separate
fixtures, hence their one-token path-length difference. All three agent answers
require a new actual archive read; facts are checked in storage after context
removal. The225.23-second agent run is a qualification, not a matched task-speed
benchmark. Repeated reuse does not independently prove master-cache immutability.
The19–21second C/native vision runs are also not alternating speed controls.

The pressure guard completes normally, with kernel pressure level1 throughout
and maximum swap0.0127GiB. Large RAM use is permitted by the guard; these settings
are measurement controls rather than a new global memory cap.

## Build and CPU checks

All58 consolidated fixture invocations pass. The sweep covers selected storage,
server, agent, tokenizer, GGUF, Engram, image and Metal-controller CPU objects in
C/native × Debug/ReleaseFast where supported. It follows the individual actual
GPU and model qualifications; CPU stubs do not replace those GPU results.
Shell syntax and actual selected agent/server help checks pass for the new
`ds4-local --bin-dir DIR` selector. Original preset names/defaults remain.

All-five builds use Apple Clang for residual C/ObjC and Zig0.17 for native modules.
The following are warm-toolchain source-edit measurements, not installation or
cold filesystem benchmarks:

| Build | Full source rebuild | No-op | Small cache edit | CLI edit |
| --- | ---: | ---: | ---: | ---: |
| Make C | 9.205s | 0.0125s | 0.218s | 0.273s |
| Zig C | 9.480s | 0.156s | 0.288s | 0.350s |
| Zig native | See bracket below | 0.348s | 0.567s | 0.583s |

The first native forced rebuild takes12.547s. Follow-up actual-compilation
profiles give9.498s(j12),9.304s(j8),9.627s(j18),9.883s(j12). All native objects
actually compile; the remaining90k-line C engine TU is the roughly9-second
critical path. Keep the existing simple j12 default: one scheduling sweep does
not justify another build policy. The first slower result is retained. Native
small edits cost a few tenths of a second more; no blanket faster-build claim.

The first Make timing loop is **invalid for edit comparisons**: system Make3.81
misses changes made within the same whole second. The corrected loop waits
outside measured intervals and verifies actual compiler commands:22 for each
full rebuild, one for each small edit, zero for no-ops. Both corrected full
samples are9.199/9.211s. All isolated source bytes are restored. No harness
framework or special cache implementation was added to fix this measurement.

## Decisions and reproduction

Retain the selected modules. The earlier activation comparison has identical
full-frontier logits and21.785 mean steady decode tokens/s in both arms.
Component tokenizer/chat/decode gains and the rare image-rounding difference
are recorded in their own reports, not promoted into a whole-model speed claim.
The unchanged default-fast-math FP4 oracle failure in both C/native is retained;
existing strict-math full GPU suites pass. A server JSON survey retains C:
standard-scanner surrogate/nonfinite/nesting behavior differs from protected
API behavior, and adapting it would expand this migration.

`tools/zig-build` builds selected native programs; `-Dnative=false` builds their
C reference. [Fixture loop](fixtures.py), [runtime loop](runtime.py),
[SSD restart driver](serve.py), [persistent-agent driver](run_agent.py) and
[vision check](vision.py) are retained here. They use explicit fresh workspace
paths and existing fixtures, not another parser or grading framework. From a
fresh checkout, first run the fixture loop to create C/native installs, then:

```sh
mkdir -p .workspaces/zig-combined-20261022
python3 speed-bench/experiments/zig-combined-20261022/fixtures.py
python3 speed-bench/ideas/pressure_run.py .workspaces/zig-combined-20261022/runtime-guard -- \
  python3 speed-bench/experiments/zig-combined-20261022/runtime.py
```

Use a new result-directory name when repeating an already populated experiment.
[Recorded commands, binary identities, all outcomes and timing samples](results.json).
Full logs, contexts, traces and actual cache files remain in
`.workspaces/zig-combined-20261022/`. Build-loop sources are retained for inspection;
its original Make edit results must be replaced by the separately corrected
Make loop as described above.
