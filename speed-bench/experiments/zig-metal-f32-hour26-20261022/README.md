# Native F32 Metal matmul controller

Baseline `61032796`, qualified inside a one-hour ceiling on 2026-10-22.
172 handwritten Zig lines replace F32 matmul validation, packet construction
and plain/short/MV_EXT/router-MM dispatch policy. The two actual 112-byte MV
packets move once into the shared project header; existing MM88, retained model
buffers and command tokens are reused. The small Objective-C bridge queries
cached pipelines and encodes fixed weights/x/out bindings. Shaders and F16
controller logic do not change.

Comments preserve uint32-to-int32 packet bit patterns, unsigned weight-product
wrapping, activation bounds, single-row M5/NR0 override, router quality/env
thresholds, T3 exactness and generic multi-row policy. Retained weights survive
owned command completion; failure does not commit an owned command or end a
borrowed batch. No generic resource/dispatch framework is introduced.

## Qualification

All five C/native ReleaseFast programs and existing GPU fixtures build without
warnings. The actual router test passes 90 cases per implementation across 15
row counts 1–8192 and six repeated router/MM mode settings. Each runs real GPU
work, output guards and a CPU-double oracle. The C/native error tables match.
The existing full SSD expert fixture also passes both arms, including tiny F32
matvec/model-view offsets and mapping lifetime. The required invocation is
`test-metal-dense-mpp --router`: its no-argument path does not exercise F32.
Likewise the SSD table-admission shortcut misses its F32 checks.

The ordinary model screen runs control/native/native/control with a 512-token
prompt and 128 fixed decode steps, 48 GiB expert cache, context allocation 1024,
chunk 512. Model controls use the preceding otherwise-native bench, isolating
this controller. All four saved 129,280-entry logit vectors at the 512-token
frontier match exactly. This does not independently compare every decode-step
logit vector; the fixed continuation controls the measured history.

Steady decode rates are 21.98/22.08/21.81/22.00 tokens/s: means 21.990 versus 21.945,
about 0.2% lower for native. Whole-process times are 21.908/23.086/23.185/22.959 s;
native is 3.13% above the control mean, with 4.80% whole-control drift. No model
speedup is established, and the observed costs remain recorded. Pressure stays
normal with no swap growth.

```sh
# The model control must be the preceding native build, not the C GPU control.
# If that earlier installation is absent, build it in a separate workspace:
jj workspace add -r 61032796 .workspaces/zig-f32-control
(
  cd .workspaces/zig-f32-control
  tools/zig-build --prefix ../zig-gguf-input-20261022/native ds4-bench
)

tools/zig-build --prefix .workspaces/zig-metal-f32-20261022/native \
  install dense-mpp-test ssd-experts-test
# Build /c with -Dnative=false, then run GPU/model jobs serially:
python3 -B speed-bench/ideas/pressure_run.py NEW_GUARD -- \
  python3 -B speed-bench/experiments/zig-metal-f32-hour26-20261022/qualify.py
```

Decision: **retain** the typed controller and measured neutral steady decode.
[Commands, hashes and results](results.json), [existing-fixture/model loop](qualify.py).
The prompt is retained beside the loop. Its base/result constants select the
workspace paths; select fresh paths for another run. Raw logs, vectors and guard
records remain under
`.workspaces/zig-metal-f32-20261022/`. Runtime defaults are unchanged.
