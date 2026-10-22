# Complete native text splitting

Completed 2026-10-22 04:49 UTC, inside the one-hour ceiling. Base `74793699`.
Handwritten `zig/pretokenizer.zig` implements the ordered JoyAI, GLM4 and Qwen35
rules with slices and explicit decoded-character records. The tiny shared
header uses the original generated Unicode tables; vocabulary/maps remain
opaque. Existing C family selection, allocation admission and fatal OOM policy
stay at the engine boundary. C retains its original algorithms behind the same
explicit family/maps/raw-length API. No engine or vocabulary layout is imported.

Meaningful original rule comments are retained: number widths, punctuation
plus trailing newlines, leading spaces, contractions, combining marks and the
existing permissive malformed-UTF8 stepping. This port does not silently
replace the model's tokenizer with a validating Unicode decoder.

Both C/native build all five programs plus the existing warm tokenizer mode.
The production-linked maps/BPE/lexer fixture passes C/native × Debug/RF (four
runs), making family boundaries and one malformed continuation observable via
synthetic merge ranks. All builds/fixtures pass on the first attempt.

The existing readonly loader/vocabulary fixture runs C/Zig/Zig/C for each actual
local DeepSeek1017, GLM5.3 and Qwen3.8 model. Eight fixed inputs cover code,
persistent context, tool output, long pieces, multilingual text and all three
rendered turn spellings. Both raw text and rendered-special-token scanning run
20 warm repeats. **All 96 native output-token vectors match their C references**;
closing C controls also match. The fixture opens no GPU and binds no weights;
these are actual model-vocabulary comparisons, not inference qualification.

Mean warm ordinary-code milliseconds per call (two samples per implementation):

| Family | Mode | C lexer | Zig lexer |
| --- | --- | ---: | ---: |
| deepseek | raw | 0.1626 | 0.1648 |
| deepseek | chat | 0.3777 | 0.3516 |
| glm | raw | 0.1970 | 0.1939 |
| glm | chat | 0.5607 | 0.4610 |
| qwen | raw | 0.2579 | 0.2771 |
| qwen | chat | 0.4783 | 0.4920 |

Microtimings are mixed: Qwen code/tool paths are roughly5–7% slower, around
0.02ms on code; GLM rendered code is faster in this short sample. Other cases
also vary, including a15% DeepSeek multilingual rendered gap of0.046ms. No
whole-model speed claim is made, and this does not erase the preceding span-BPE
savings (both arms already contain that improvement). The language port is
retained for its explicit, readable ownership/rules and matched token IDs.
Further tokenizer optimization should be its own measured change.

Decision: **retain**. `tools/zig-build pretokenizer-test tokenizer-model-test`;
run the tiny fixture directly. Run model-opening fixtures one at a time under
the existing guard. [Commands/hashes/results](results.json), [loop](qualify.py),
[fixed corpus](corpus/). Full logs/guard records stay under
`.workspaces/zig-pretokenizer-20261022/`.
