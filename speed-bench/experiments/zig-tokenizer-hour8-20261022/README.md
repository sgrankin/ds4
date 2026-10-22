# Native borrowed maps and span-based BPE

Completed 2026-10-22 04:28 UTC, within the one-hour ceiling. Base is metadata
commit `91dbe48b` on `zig-modular`.

`zig/tokenizer.zig` is a handwritten 151-line implementation. Standard
`StringHashMap` owns map storage and borrows immutable GGUF keys; the C vocabulary
contains two opaque handles. Each BPE piece owns one encoded buffer, a standard
span vector and reusable rank-key scratch. A merge extends a span, without
allocating/copied symbol payloads. The left-to-right greedy scan remains simple.
The C reference retains its original FNV probing and copied-symbol algorithm
behind the same private header. Maps and BPE are selected together.

Preserved contracts: duplicate keys overwrite values; misses leave outputs
unchanged; arbitrary byte/empty keys; GPT-2 byte encoding; leftmost rank ties;
INT32_MAX never merges; encoded-byte fallback omits missing bytes; output tokens
append. C wrappers retain allocation-guard admission and fatal OOM behavior,
and output growth still calls the actual C token-vector helper. Model loading,
Unicode/pre-tokenization rules, chat formatting and special IDs remain C.

All five programs build in C/native configurations. The actual production
map/piece fixture passes C/native × Debug/ReleaseFast: **four successful runs**.
It checks borrowed storage, embedded NUL/empty keys, duplicates, append semantics,
merge/tie/sentinel behavior and UTF-8/fallback. No generated implementation or
SDK dump is checked in. A final source-only unused annotation suppresses the old
C hash helper warning; its unchanged body and before/after hashes are recorded.
Measured binaries retain their original hashes; the next full build covers this
nonbehavioral annotation.

The existing GGUF fixture gains a small tokenizer mode that opens the real
model/vocabulary once and times warm repeated tokenization. C/Zig/Zig/C runs
use six fixed inputs, both raw and rendered-context paths, with 20 repetitions.
**All 48 full token-ID comparisons are identical**, including 23,000-token tool
output, persistent-agent context and multilingual text. The output vector is
warmed; per-piece allocations remain timed. Below are mean milliseconds per
call; savings compare elapsed time, not throughput percentages.

| Input | Mode | C ms | Zig ms | Time saved |
| --- | --- | ---: | ---: | ---: |
| ordinary-code | raw | 0.2919 | 0.1724 | +41.0% |
| ordinary-code | chat | 0.5175 | 0.4119 | +20.4% |
| persistent-context | raw | 1.5855 | 0.9917 | +37.5% |
| persistent-context | chat | 2.8906 | 2.5807 | +10.7% |
| tool-output | raw | 3.4898 | 2.1176 | +39.3% |
| tool-output | chat | 6.6939 | 5.9228 | +11.5% |
| long-piece | raw | 8.3459 | 8.1417 | +2.4% |
| long-piece | chat | 8.3803 | 8.1598 | +2.6% |
| multilingual | raw | 0.2758 | 0.1586 | +42.5% |
| multilingual | chat | 0.4148 | 0.3298 | +20.5% |
| rendered-turns | raw | 0.0120 | 0.0073 | +38.7% |
| rendered-turns | chat | 0.0109 | 0.0110 | -1.4% |

This is a tokenizer gain, not a whole-inference speedup. The tiny rendered-turn
case is effectively unchanged/slightly slower; the deliberately long piece gains
only 2–3% because full greedy rescans remain. Ordinary raw inputs save 37–42%;
rendered ordinary/persistent/tool inputs save 11–20%. Model-open time is about
0.65/1.09 ms (C/Zig); vocabulary loading is about 4.58/3.23 ms, recorded separately
from warm timings. These are short local samples, not cross-machine guarantees.

A final actual Metal prefix-file generation returns exactly **ORCHID42** in both
implementations. All model-opening jobs execute serially under the pressure
guard. [Warm loop](qualify.py), [warm results](results.json), [generation loop](cli.py)
and [generation results](cli-results.json) retain commands and hashes. Fixed
inputs are included in `corpus/`; original cache paths in the manifest identify
the measured files. Full build/stdout/source/pressure artifacts remain under
`.workspaces/zig-tokenizer-20261022/`.

Decision: **retain**. Build with `tools/zig-build tokenizer-test tokenizer-model-test`;
run `test-tokenizer-piece` directly. Under the guard, use
`test-deepseek41-gguf --tokenizer-bench MODEL REPEATS FILE...` for warm comparisons.
`-Dnative=false` keeps the C reference; `-Doptimize=Debug` checks native helpers.
