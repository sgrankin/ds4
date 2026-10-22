# Handwritten benchmark frontend

Completed 2026-10-22 04:46 UTC, inside the one-hour ceiling. Base `895f4b52`.
Native `ds4-bench` now uses `zig/bench.zig` and bindings generated from the
small real-header boundary `zig/bench_api.h`. C remains the selectable reference.
The build linker accepts a prepared frontend object, so selection stays outside
the linker rather than accumulating optional arguments.

Options, borrowed argument strings, token/logit slices and cleanup are explicit.
The original libc numeric grammar and CSV/JSON precision are preserved. The
current Apple options, TP/distributed public parsers, teacher-forced decode,
speculative control flow, snapshot/replay rules and timing boundaries remain.
No engine layout or translated frontend is imported. CUDA-only profiler calls
and inactive ROCm aliases are outside this Metal build, as in the C baseline.

All five C/native ReleaseFast programs and the native Debug benchmark build.
Thirteen help/early-option cases match C/native RF/native Debug byte-for-byte
in status/stdout/stderr (39 runs). Initial compiler errors caught an ignored
C return value and an inferred unsigned count passed to `%d`; both received
small explicit discard/type fixes, with original failures retained. Debug
applies to the native frontend and host modules, while the engine stays fast.

Six serial real-1017 runs compare the preceding native-host binary with C
frontend against the new native frontend, using a 48 GiB expert cache, chunk
512 and context allocation 256. Profiles: 64/128 frontiers with eight forced
tokens and snapshots; the same with snapshot-disabled replay; 32/64/128
multiplicative frontiers with two greedy tokens. **Every corresponding full
frontier JSON/logit vector and decoded output matches** (seven frontier pairs).
CSV prefill increments and decode counts match; the first snapshot is nonzero,
whereas forced replay reports zero snapshot bytes. Snapshot save/restore and
replay remain outside both timed regions.

Process C→Zig seconds: snapshot 12.005→10.862, replay 12.186→11.963,
multiplicative 9.936→9.571. These short ordered pairs include startup and do not
establish an inference speedup. Actual distributed/speculative deployment was
source-reviewed, not executed in this item.

Decision: **retain**. `tools/zig-build ds4-bench` builds the native frontend;
`-Dnative=false` selects C. [Commands, hashes and results](results.json),
[model loop](qualify.py). Full build attempts/stdout/guard logs remain in
`.workspaces/zig-bench-frontend-20261022/`.
