# Bounded GGUF cursor inside the C engine

Budget: 2026-10-22 03:48–04:48 UTC. Base is the KV persistence commit
`576edd70` on `zig-modular`. Decision at 03:56 UTC: **retain**.

`zig/gguf.zig` replaces six bounded read/string/value-skipping operations,
previously embedded in `ds4.c`. It is 107 handwritten lines. `ds4_gguf.h` holds
the original stack-owned cursor, borrowed string, metadata tags and six function
declarations. Standard generated bindings supply those types; the implementation
imports no SDK or libc declarations. Model/tensor tables, mapped-file ownership,
metadata lookup and all graph operations stay C.

Two local `DS4_NATIVE_GGUF` guards retain the original C implementation for the
reference build. Call sites use the six prefixed names. Independent source review
confirmed that undoing those name/visibility changes, guards and moved type/tag
definitions restores the previous engine token stream. Make's engine dependencies
include the new private header; ordinary Make continues using the C bodies.

The port preserves overflow-safe bounds, unchanged destination/position on a
failed primitive, borrowed arbitrary string bytes, consumed string-length prefixes
on a truncated body, first-error text/offset, later successful reads after an
error, scalar-array multiplication checks, the existing depth limit and empty
unknown-type arrays. It adds no UTF-8 checking, rollback or new format rules.

## Evidence

- All five C/native production programs build.
- The small production-linked cursor fixture passes C/native × Debug/ReleaseFast.
- Existing `tests/test_deepseek41_gguf.c` sparse extent and invalid-layout checks
  also pass in all four configurations. Its textual engine include receives the
  native guard, so the native case resolves the actual Zig functions.
- Real 1017 IQ2 model tokenization matches the pre-extraction C binary byte for
  byte, including the multilingual prompt, in C/native/native/C order.

All model-opening checks run serially under the existing pressure guard. The
[simple recorded loop](qualify.py) reuses the existing sparse fixture and CLI;
[raw results](results.json) contain commands, timings and output hashes. The two
native tokenizer runs take 9.48/9.33 ms; the C runs take 14.69/8.38 ms. These tiny
samples qualify output and do not establish a performance improvement. Sparse
checks take about 5 ms each. Initial all-program builds take 11.38/10.99 seconds,
including fresh build setup.

The broader converter-contract manifest suite was **not run**: its original HF
config/tokenizer and source-shard headers were not found in retained evidence.
No candidate-derived inputs were substituted for that independent check.

Build with `tools/zig-build gguf-test gguf-model-test`; `-Dnative=false` selects
C, and `-Doptimize=Debug` selects checked host code. Run `test-gguf-cursor`
directly. Run `test-deepseek41-gguf` and real model diagnostics under
`speed-bench/ideas/pressure_run.py`, one process at a time. Complete commands,
source/binary hashes and logs are in `.workspaces/zig-gguf-20261022/`.
