# Native image preprocessing and layout

Completed 2026-10-22 05:17 UTC, inside the one-hour ceiling. Base `c87d79b0`.
The remaining twelve image preprocessing/layout exports use handwritten Zig,
real public image records, explicit dimensions and owned slices. The existing
C codecs and reference preprocessing remain selectable. Comments preserve
bicubic support truncation, double accumulators, integer pixel rounding,
patch/temporal order, DeepSeek padding and rounding-mode-sensitive `lrint`.

All five C/native programs build. Existing decode and vision image fixtures
pass C/native in ReleaseFast, and vision fixtures also pass both Debug arms.
A libc-only shared library links the same selected production image objects;
the numeric comparison does not copy the implementation or load the GPU.

The comparison covers four model families and eleven geometries, from 1×1
through 2048×2048 and extreme 4096×64/64×4096 strips. All geometry fields,
patch counts and ownership-reset checks pass. DeepSeek1017 geometry also
matches the pure layout functions extracted from DeepSeek's Apache2-licensed
[DeepInfer image processor](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/ve/processor/image_processor.py),
with current local architecture parameters; layout token types and permutation
match exactly. The source hash and parameters are recorded in results.json.

ReleaseFast patch buffers match byte for byte in **43 of 44** cases. The
DeepSeek1017 64×4096 strip differs in **2 of 914340** values by one rounded
pixel: maximum normalized error 0.00784314, RMS 0.00001160. Eight additional
Debug comparisons retain identical geometry, with normalization differences
of at most 0.000000239. These are measured approximation costs, not rejection
criteria. Actual image-to-answer inference is part of combined qualification.

One-pass CPU timings show the native preprocessing around 3–15% slower on
these inputs (roughly hundredths of a millisecond to four milliseconds).
This is not an alternating latency measurement and does not establish a
whole-task regression. Dynamic symbol inspection confirms sqrt/floor/round
already compile inline; replacing their spelling would not remove external
calls. The earlier decode speedup and GPU vision costs are separate.

Decision: **retain**, with the small measured pixel difference and preprocessing
cost explicit. No new resize algorithm, numerical tolerance or experimental
runtime default was introduced. `tools/zig-build image-test vision-image-test
image-library` builds the production-linked fixtures. [Commands, hashes and
all 96 numeric cases](results.json); [comparison loop](qualify.py). Full float
buffers, build logs and pressure records remain under
`.workspaces/zig-image-preprocess-20261022/`.
