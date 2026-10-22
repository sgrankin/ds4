# Native image decoding, orientation and SHA identity

Completed 2026-10-22 05:02 UTC, inside the one-hour ceiling. Base `9120ac51`.
`zig/image.zig` replaces the complete decode-memory/file/free lifecycle with
borrowed slices, checked TIFF offsets, standard integer readers and standard
SHA256. Existing Iris JPEG/PNG codec implementations and their progressive-JPEG
fixes remain C. Numerical resizing/patch/layout remains C in this item.

The real codec records come from declarations in `image_api.h`; no copied
vendor structs or translated image controller is imported. Successful RGB
allocations transfer to the public libc owner. Missing/path/I/O failure keeps
the original partial-output contract; decoding clears the output record. Pixel
identity keeps its exact existing format: SHA256(native-u32 width,height,
orientedRGB), rather than encoded file or EXIF bytes.

All five C/native programs and C/native Debug/RF image fixtures build. The two
existing progressive-JPEG fixed-fingerprint checks pass in all four arms.
The existing C fixture gains a small optional JSON pixel dump and repeated
actual decode mode, so the comparison calls selected production objects.

A standard-library Python loop creates RGB, grayscale and RGBA PNGs plus all
eight EXIF orientations in both TIFF byte orders, retaining the two existing
JPEGs and a512×512RGB PNG. **All88 full RGB/fingerprint vectors match** across
C/native × Debug/RF (22images each). An independent standard SHA digest and
matrix transpose/rotation expressions verify identity/orientation; expected
PNG pixels verify grayscale expansion and alpha handling. Free resets the
public ownership record. No malformed-file testing framework is introduced.

Warm C/Zig/Zig/C,64 actual512×512 PNG decode/free calls per sample:

| Implementation | Seconds |
| --- | ---: |
| C before | 0.270917 |
| Zig | 0.180459 |
| Zig | 0.181972 |
| C after | 0.272616 |

Native mean time is **33.32% lower**, with0.63% control drift. This measures the
whole small CPU decode/orientation/fingerprint package, including warm file I/O;
it does not isolate SHA alone or claim a model/vision-inference speedup.
The codec, bytes and output identity are unchanged.

Decision: **retain**. `tools/zig-build image-test`; run test-image-decode for the
original regressions, with file arguments for pixel JSON or `--bench 64 FILE`
for warm decode. [Commands/hashes/results](results.json),
[loop](qualify.py), [generated fixtures](fixtures/). Full logs/binaries remain
in `.workspaces/zig-image-decode-20261022/`.
