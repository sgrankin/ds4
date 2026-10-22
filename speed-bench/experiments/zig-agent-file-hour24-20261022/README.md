# Native agent file transactions and edit anchors

Baseline `f77b4abc`; source preparation and qualification completed within a
one-hour ceiling on 2026-10-22. Five existing primitive operations move into
287 handwritten Zig lines: regular-file opening, capped file reading, atomic
replacement, unique search and ordinary/anchored edit-span selection. No agent
worker or model records cross the interface. The real header owns the 16 MiB
cap and libc ownership contract; generated SDK declarations provide FILE/stat.

Scoped cleanup replaces the transaction's manual exit/cleanup paths. Standard
bounded byte search replaces the local substring loop. Comments preserve
nonblocking regular-file admission, failed-read output timing, overlapping
anchors, suffix-only tail uniqueness, CR/LF trimming, symlink and version checks,
hard-link refusal, metadata ordering, fsync/close-before-publication and new-file
umask/no-overwrite behavior. The original C/Linux implementation remains.

## Qualification

All five programs build in C/native ReleaseFast without warnings. The unchanged
production-linked agent fixture passes in C/native × Debug/ReleaseFast: four
runs. It covers failed partial writes preserving the destination, stale expected
bytes, symlink/mode/owner/xattr preservation, hard-link refusal, temporary cleanup,
nonblocking file admission, streaming reads and anchored edit boundaries.

```sh
tools/zig-build -Dnative=true --prefix .workspaces/file-native install agent-test
.workspaces/file-native/bin/ds4_agent_test
```

The real-model pair uses the preceding otherwise-native agent as the C file-body
control. Both agents perform exactly read → edit → write → DONE, with identical
generated text. The inventory changes only count17→18; the receipt is exactly
eight bytes ORCHID42. Mode0751, uid/gid and a real extended attribute survive.
No fallback shell, retry, tool error or cap occurs. Prompt-file reading also
passes through the selected production reader. Settings: model1017, 48 GiB expert
cache, context8192, chunk512, MTP1, greedy/no-thinking, route/prefill/suffix off.

The two cold whole-process samples take76.327/78.773 seconds. They qualify the
actual caller; one ordered pair does not establish a performance change. All
pressure samples are normal, with no swap growth.

The initial setup attempt stopped before opening a model because this macOS
Python lacks os.setxattr. Its guard result and workspace remain. The corrected
loop uses the standard xattr utility and bytes.fromhex; no engine change was
needed. Reproduce with fresh guard/output paths and both executable paths:

```sh
python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD -- \
  python3 -B speed-bench/experiments/zig-agent-file-hour24-20261022/qualify.py \
  --c-agent CONTROL_AGENT --native-agent NATIVE_AGENT --output NEW_RESULTS
```

Decision: **retain**. [Results, commands and hashes](results.json),
[actual caller](qualify.py). Full logs, files and traces are under
`.workspaces/zig-agent-file-20261022/`. No tool behavior or preset changes.
