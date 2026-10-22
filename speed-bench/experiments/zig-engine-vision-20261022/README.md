# Native DeepSeek vision encoder

Item 63 replaces the complete DeepSeek vision controller with the reviewed
native owner. The one-hour window starts at 14:55 UTC on 2026-10-22 and ends at
15:55 UTC. The C/native ReleaseFast and native Debug encoder arms passed,
followed by the actual native image/read and contextual follow-up check.
Qualification completed at about 14:59 UTC. The complete embeddings match
byte for byte; the ordered times do not establish a speed gain.

The controller owns the thirteen-buffer image workspace, shape checks,
32-block sequencing, BF16 rounding, residual swaps, 3x3 alignment and output
readback. A separate public entry avoids pulling unrelated engine diagnostic
services into image-only callers. Eight small Objective-C leaves retain ARC,
pipeline lookup and shader encoding. Existing shaders, the sidecar binder,
image decode/preprocessing and prompt-image assembly stay unchanged.

## Bounded qualification

[qualify.py](qualify.py) reuses the actual
[`--vision-encoder` fixture](../../../tests/test_deepseek41_graph.c). Each arm
encodes the same image twice, requires finite and byte-repeatable output, and
writes the complete F32 embedding. The serial C/native ReleaseFast pair and
native Debug arm compare every embedding value, geometry and exact
bytes. Numeric differences are reported with maximum absolute drift and the
number of changed values; successful finite execution and the fixture's own
repeatability invariant are required. Ordered process times are retained as
diagnostics, not a latency ablation.

The available sidecar is
`gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf`, previously
[verified](../1017-upgrade-20261020/record.json) at 970,555,552 bytes and SHA-256
`861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf`.
It exercises the 5,120-wide 1017 projection. The image is the existing 112x84
red PNG at `.workspaces/zig-combined-20261022/vision/red.png`; an identical
[small copy](red.png) is retained here for reproduction. There is no local
legacy 4,096-wide DeepSeek sidecar, so that branch is not qualified here.

Build `graph-test` before enabling the new owner and retain that binary as the
C-controller control, with the otherwise-qualified native engine and
speculative-cycle implementation. The retained control was built at `1e27a3f2` in
8.313 seconds at `.workspaces/zig-engine-20261022/vision-control/bin`; this is a
cached development build, not a clean-build measurement. The initial candidate
format check found a reserved local identifier, `resume`; it was renamed
`resume_commands` and the source formatted before builds. After source freeze,
the small loop runs under the existing pressure guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-vision-model-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-vision-20261022/qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/vision-control/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/vision-native/bin \
    --debug-bin-dir .workspaces/zig-engine-20261022/vision-debug/bin \
    --image speed-bench/experiments/zig-engine-vision-20261022/red.png
```

The default fresh output is `.workspaces/zig-engine-vision-rerun`; commands,
binary/source hashes, raw stdout/stderr and complete F32 embeddings remain
there. The source list names the integrated vision files explicitly and does
not collect unqualified drafts by glob.

Optional [cli.py](cli.py) reuses the prior actual red-image REPL screen with an
explicit `--bin-dir`. It reads the image and asks a contextual follow-up,
requiring two `RED` answers and the real image-encoding diagnostic. It uses a
32 GB expert-cache target, context 4,096, chunk 512, MTP width 1 and temperature
zero. The sandbox only prevents writes to the user's history file.

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-vision-cli-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-vision-20261022/cli.py \
    --bin-dir .workspaces/zig-engine-20261022/vision-native/bin \
    --image speed-bench/experiments/zig-engine-vision-20261022/red.png
```

Its default fresh output is `.workspaces/zig-engine-vision-cli-rerun`. This is a
single functional integration screen, not a coding-quality or performance
claim. All GPU/model execution and serial builds belong to root.


## Results

The [recorded build commands](evidence/build/) retain the first reserved-word
format failure. Renaming `resume` to `resume_commands` and formatting resolved
it before compilation. The native ReleaseFast install plus graph fixture built
successfully in 5.373 s; the native Debug graph fixture built in 1.404 s. Both
had empty stdout/stderr. Together with the 8.313 s C-controller build, these
are cached development-build observations rather than clean-build comparisons.

All three [encoder arms](evidence/model/results.json) returned zero and encoded
the image twice. Every output was finite and internally byte-repeatable. The
reported grid was 15x12, with 180 rows of width 5,120: 921,600 F32 values, or
3,686,400 bytes. Both native outputs match the retained C-controller output
exactly, with maximum absolute difference zero and no changed values.

| Encoder controller | Whole fixture process | Full embedding |
| --- | ---: | --- |
| C controller, otherwise native engine | 3.435 s | Reference |
| Native ReleaseFast | 3.256 s | Byte-identical |
| Native Debug | 3.217 s | Byte-identical |

The common SHA-256 is
`35f1d31866e16697ee6894af87efb2a1f3fad62a93063996d2b500e805ae6b92`.
The full-array comparison and hashes are retained as small evidence. The
standard-gzip reference array remains in the ignored workspace at
`.workspaces/zig-engine-20261022/report-embedding.f32.gz`; it is deliberately
excluded from version control. Original command paths, binary hashes,
geometry, stdout/stderr and comparison results remain in this report. All
three raw arrays also remain under the recorded workspace run paths. These three ordered
runs cover functionality and build-mode compatibility, not a latency ablation.

The [actual native CLI](evidence/cli/results.json) returned zero in 19.661 s,
emitted the real image-encoding diagnostic and answered `RED` to both the image
and its contextual follow-up. Its [raw output](evidence/cli/stdout.log),
[diagnostics](evidence/cli/stderr.log) and [scripted input](evidence/cli/stdin.txt)
are retained. This checks one simple integrated image workflow, not broader
visual reasoning or coding quality.

Both [pressure guards](evidence/guards/) completed with child return code zero.
The [29 sampled observations](evidence/pressure-summary.json) all had normal
pressure, with zero swap growth. Peak wired/compressed memory was
4.623/9.044 GiB for the encoder campaign and 38.096/9.152 GiB for the CLI
campaign. The [ten selected source files](evidence/source-verification.json)
still match the hashes captured by the measured encoder loop. The compact
[summary](summary.json) links the results without adding further gates.

The qualified boundary is the complete native controller with unchanged Metal
shaders and existing leaf encoders on the actual 1017 BF16 sidecar. Legacy
4,096-wide sidecars and other images remain untested by this item.
