# Complete handwritten CLI frontend

Completed 2026-10-22 04:53 UTC, inside the one-hour ceiling. Base `622db937`.
Native `ds4` now uses readable `zig/cli.zig`, shared typed options/progress
helpers and `cli_diagnostics.zig`. Generated declarations stay behind the small
real-header `cli_api.h`; no translated frontend, SDK dump or private engine
record is imported. All76 active Apple flag spellings remain. C is available
under `-Dnative=false`.

One-shot versus REPL decode lifetimes, transcript/EOS updates, session
replacement, signal installation/rollback, callback borrowing and diagnostic
session cleanup are explicit. Defers release owners; SDK signal types remain
actual imported declarations. Complex REPL branches were expanded into ordinary
braced blocks. Numerical diagnostic/printf contracts and libc argument grammar
remain as before. Distributed/speculative public APIs remain present.

All five C/native RF programs and the actual native CLI Debug frontend build.
Fourteen no-model help/early-option cases match status/stdout/stderr byte-for-byte
across C/native RF/native Debug (42 runs). Initial compiler findings were a
shadowed name, signed remainder syntax and an inferred unsigned C-vararg count;
small explicit fixes suffice. A demonstrated missing-option diagnostic mismatch
was corrected without a parser abstraction. Initial failures remain in workspace.

The guarded actual comparison changes only the frontend, using the preceding
native-core C frontend as baseline, 48 GiB expert cache, chunk512, ctx1024,
seed1 and no thinking. All28 serial checks pass:

- Raw/chat token dumps for real DeepSeek1017, GLM5.3 and Qwen3.8 match full output.
- Both ordinary greedy and frontend-owned sampled generation match output.
- Full logits JSON,128-way logprob JSON, live/fresh consistency diagnostics and
  fixed-prefix perplexity output match exactly.
- Piped REPL changes thinking/session capacity, recalls PINE-251 and submits a
  text file via `/read`, with identical output and successful READ42 response.
- SIGINT during prefill emits no canceled answer; the next turn succeeds with
  RESUMED42. Both implementations show input lengths **142 then17**, verifying
  that the canceled transcript was rolled back before reuse.

An additional actual native Debug sampled run matches RF output. REPL fixture
history writes are denied only for the user's `.ds4_history`, preserving ordinary
CLI behavior without adding test commands to personal history. The first
qualification attempt stopped before GPU use because the test treated the mixed
JSON-array/pretty-table token dump as one JSON document; it now uses the standard
JSON parser on the first line and compares full output too. Original evidence
is retained, not mistaken for an engine failure.

The short runs establish functional parity, not a speedup: startup dominates
some process differences. Image input, distributed workers and speculative
multi-token execution are source-reviewed here, not claimed runtime-qualified
by these text checks. They remain available, with image execution planned in
the combined qualification.

Decision: **retain**. `tools/zig-build ds4` builds the native frontend;
`-Dnative=false` selects C. [Commands/hashes/results](results.json), [loop](qualify.py),
[Debug run](debug.py). Complete logs, initial failures and pressure samples live
under `.workspaces/zig-cli-frontend-20261022/`.
