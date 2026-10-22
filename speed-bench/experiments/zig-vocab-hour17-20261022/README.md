# Native vocabulary ownership and token-byte decoding

Completed 2026-10-22 05:09 UTC, inside the one-hour ceiling. Base `8cefb346`.
`zig/vocab.zig` owns token-array/map construction, model-family marker defaults,
release and token-to-byte decoding. C supplies only real GGUF cursors/counts,
explicit family and optional GLM metadata IDs; it retains model mapping/fatal
error policy. The existing small vocabulary header gains the actual input/API
records. Chat and vocabulary share one generated header module.

Tokens and map keys borrow the model mapping. A failed build releases partial
owners and clears output; release never frees borrowed bytes. Marker defaults,
GLM id0 and missing-required-token diagnostics remain explicit. Token decoding
uses the standard UTF8 decoder with the original masked-byte fallback for
malformed legacy spellings, a compile-time GPT2 inverse and literal fullwidth-bar
pass-through. Results remain NUL-terminated libc-owned byte buffers, including
embedded NULs and fragments that are not valid UTF8. Existing allocation
admission stages/labels remain at the engine boundary.

All five C/native programs and existing loader/CLI fixtures build. Model-free
session invariants pass C/native × Debug/RF. A small native production-linked
fixture checks borrowed ownership, duplicate mappings, marker defaults, embedded
NUL/permissive byte handling, invalid IDs and partial truncated-merge cleanup
followed by rebuild; it passes Debug/RF. It qualifies the actual native lifecycle,
not a second copied C implementation. One intentional test string concatenation
was parenthesized for clean final fixture logs; original warning retained.

The existing readonly GGUF fixture gains a24-line public token-text mode. It
loads each real vocabulary and calls ds4_token_text for every ID, plus both
out-of-range ends, asserting termination and emitting standard JSONL byte arrays.
C/native complete output matches for **all532,480 vocabulary entries**:

| Model | Entries |
| --- | ---: |
| DeepSeek1017 | 129280 |
| GLM5.3 | 154880 |
| Qwen3.8 | 248320 |

Actual raw/rendered tokenizer vectors also match across eight fixed inputs for
all three models (48 native vectors), exercising borrowed maps after loading.
All model openings run serially under the existing guard, without GPU or weight
binding. This item makes no isolated vocabulary/inference speed claim; the C
comparison also includes preceding native tokenizer improvements.

Decision: **retain**. `tools/zig-build vocab-test session-state-test
tokenizer-model-test`; the first two are model-free, the last opens real models.
[Commands/hashes/results](results.json), [all-entry and tokenizer loop](qualify.py).
Full byte outputs, initial logs and pressure records remain in
`.workspaces/zig-vocab-20261022/`.
