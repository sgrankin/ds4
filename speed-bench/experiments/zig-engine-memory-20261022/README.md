# Item77 — Engine memory admission and workspace growth

Eleven actual Engine accounting fields move once into Engine.memory; all C
families use that same record. The117-line handwritten owner controls DS41
host/recommended-limit admission, expert-cache fitting and demand workspace
growth/rollback. A borrowed immutable resource view supplies whole authoritative
model/session facts. Native Session now contains Memory directly: the opaque
Engine pointer and C workspace-grow controller disappear from this path.
Streaming static-byte accounting reuses the actual merged-span owner while
excluding every routed expert class, preserving the guard's original policy.

Integration17:43–17:49UTC starts atdda2c842. The record-only C controller passes
first. C control, native ReleaseFast/Debug and full-C fallback all build cleanly,
pass the existing model-free state fixture and three actual graph fixture modes
(twelve runs): simulated128/256/512GiB/overflow/TP/SSD-fit admission, multi-session
allocation accounting, and scalar Session frontiers/save-load/cancel/continue.
The scalar flag preserves this fixture's established cancellation assumption;
default chunk growth is checked separately, with no assertion weakened.

The fixed real-model8192-prefix/8-teacher C/native pair explicitly grows2048→4096
rows before prefill. All129280 full prefill-boundary logits match exactly; this
is not a decode-step quality claim. Active source hashes match before/after.
Whole processes24.823/24.237s are one ordered functional pair, not a latency
comparison. All five pressure guards stay normal with unchanged swap. No new
runtime knob, test assertion, shader, compiler failure or tolerance change.
Generic startup, Session resource creation and persistence remain later owners.

```
tools/zig-build ds4-bench graph-test session-state-test -Dnative=true \
  --prefix .workspaces/memory-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/memory-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-engine-memory-20261022/qualify.py \
  --prefix .workspaces/memory-repeat/native --out .workspaces/memory-repeat/cases
```

Use -Dnative=false for full C and -Doptimize=Debug for native runtime checking.
qualify.py reuses the existing fixtures/clean environment helper. growth.py is
the preceding Session model loop restricted to this one change/case; the published ROOT alone is adjusted for this folder depth. The exact executed
workspace copy is retained as evidence/workspace-growth.py. Results and
raw logs/settings/hashes are retained; large frontier JSON is gzip-compressed,
with no binaries/model/cache stores checked in.
