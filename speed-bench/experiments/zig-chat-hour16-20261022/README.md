# Native chat formatting and control scanning

Completed 2026-10-22 05:04 UTC, inside the one-hour ceiling. Base `af3515cd`.
`zig/chat.zig` owns full/incremental chat policy, effort prefixes, rendered
control scanning and payload-close escaping for DeepSeek4/4.1, GLM and Qwen.
A small real private vocabulary record moves unchanged to `ds4_vocab.h`; family
selection is explicit at the C adapter. Model/vocabulary loading, engine graph,
multimodal embedding ownership and public engine APIs remain C.

The same ordered25 markers are preserved. The native scanner uses the standard
findAnyPos for the three actual marker-leading bytes, skipping ordinary text
before checking spellings in original order. Plain spans borrow input bytes
and call the selected prelexer directly, removing throwaway NUL copies. This
is a concrete algorithm simplification, not translated strncmp/cast machinery.
The existing allocation-admission and fatal OOM boundaries remain.

Role differences stay explicit: DeepSeek assistant bodies use plain BPE;
GLM/Qwen rendered assistants interpret their explicit controls. User/tool
literal bytes and only exact closing-wrapper/escape collisions retain the
original treatment. Later DeepSeek41 system markers and Qwen C multimodal
open/close callers are preserved. Two independent source reviews found no
concrete policy/lifetime mismatch.

All five C/native programs and model-opening fixtures build without warnings.
The existing model-free session-state/multimodal fixture passes C/native ×
Debug/RF (four runs), with all assertions unchanged. Its synthetic vocabulary
setup now uses the real public opaque-map API instead of obsolete private
pointer/helper assumptions.

Actual C/Zig/Zig/C readonly vocabulary loops for DeepSeek1017/GLM5.3/Qwen3.8
retain all96 native raw/rendered token vectors across eight inputs; closing
controls also match. Eighteen full-prompt CLI runs compare three effort modes
for all three families with literal controls/Unicode; complete output matches.
Both actual V41 CLI prefix-transition/image-offset fixtures also pass.

Warm rendered-chat milliseconds per call, means of two samples (20repeats):

| Family | Input | C renderer | Zig renderer | Time saved |
| --- | --- | ---: | ---: | ---: |
| deepseek | ordinary-code | 0.3599 | 0.1651 | 54.1% |
| deepseek | persistent-context | 2.1855 | 0.9575 | 56.2% |
| deepseek | tool-output | 5.1954 | 2.1144 | 59.3% |
| glm | ordinary-code | 0.6017 | 0.2417 | 59.8% |
| glm | persistent-context | 3.1645 | 1.4655 | 53.7% |
| glm | tool-output | 7.4242 | 2.8962 | 61.0% |
| qwen | ordinary-code | 0.4432 | 0.2733 | 38.3% |
| qwen | persistent-context | 2.6526 | 1.5335 | 42.2% |
| qwen | tool-output | 6.4659 | 3.8165 | 41.0% |

Multilingual inputs save30–40% in this screen; raw BPE is not changed by this
item. This is warm CPU formatting/tokenization throughput, not model token/s
or a whole-task speed claim. Fixed inputs, commands, outputs and all individual
timings remain available rather than replacing them with only best cases.

Decision: **retain**. `tools/zig-build session-state-test tokenizer-model-test
cli-chat-test`; the first fixture is model-free, the other two open models and
run serially under the guard. [Commands/hashes/results](results.json),
[warm loop](qualify.py), [prompt/transition loop](qualify-cli.py), [corpus](corpus/).
Full logs/binaries/pressure records remain in `.workspaces/zig-chat-20261022/`.
