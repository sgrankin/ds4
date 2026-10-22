# Native typed GGUF input and tensor directories

Baseline `39db30dc`, completed inside a one-hour ceiling on 2026-10-22.
The existing native GGUF module grows from140 to307 lines, adding eight typed
metadata lookups/getters, tensor-byte calculation and complete tensor-directory
parsing. The actual model/tensor/array records move once into the real private
header. Strings and payloads remain borrowed from the mapping.

The original C format table supplies the block constants through its lookup;
there is no second handwritten table. C retains allocation, count admission,
fatal diagnostics, unsupported-format warnings and all mapping/file lifecycle,
including disk-only Engram/Qwen tails. Native code handles the bounded typed
reads, conversion rules, shape products and relative/absolute offsets.

Comments and focused cases preserve first duplicate-key lookup, token-ID
nonnegative/INT_MAX bounds, partial array/tensor outputs and error timing.
A zero-count tensor array avoids narrowing its NULL pointer. The existing C
block-rounding and alignment expressions wrap near UINT64_MAX; this migration
explicitly preserves that policy rather than claiming an admission fix.

## Qualification

All five programs build in C/native ReleaseFast without warnings. The existing
cursor fixture passes C/native × Debug/ReleaseFast with focused real-production
cases for typed conversion, partial fields, q8 block size/offsets and overflow.
Existing DeepSeek sparse/model binding and Qwen disk-only mapping tests also
pass. Qwen mapping runs cover C/native × Debug/ReleaseFast under the guard;
file lifecycle itself remains C.

Across the three real DeepSeek, GLM and Qwen models, all532,480 decoded token
entries match byte-for-byte per implementation, including NUL and invalid-UTF8
fragments. All96 raw/rendered tokenizer vectors match. Actual DS4.1 binding
passes both arms; a native GPU continuation returns ORCHID42. Generic token
modes exercise GLM/Qwen; the ordinary sole-model fixture is DS4.1-specific and
was not incorrectly applied to those families. No startup-speed claim is made.

```sh
tools/zig-build -Dnative=true --prefix .workspaces/zig-gguf-input-20261022/native \
  install gguf-test gguf-model-test qwen-ngram-test
.workspaces/zig-gguf-input-20261022/native/bin/test-gguf-cursor
# Build the C reference under the matching /c prefix first, then:
python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD -- \
  python3 -B speed-bench/experiments/zig-gguf-input-hour25-20261022/qualify.py
```

The recorded loop uses fresh result paths under the workspace. All pressure
samples remain normal, with no swap growth. Source review caught a C-pointer
allowzero propagation issue before compilation; the actual frozen builds/tests
passed on the first attempt.

Decision: **retain**. [Results, commands and hashes](results.json),
[real model loop](qualify.py), [Qwen mapping loop](qwen.py). Raw outputs and guard
records remain under `.workspaces/zig-gguf-input-20261022/`.
