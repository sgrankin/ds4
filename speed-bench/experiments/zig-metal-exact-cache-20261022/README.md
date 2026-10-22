# Item78 — Exact Metal view cache and lifetime policy

The123-line handwritten owner replaces NSString/NSMutableDictionary cache keys
with a standard value-keyed hash map: model address/size plus page offset/bytes.
It owns range/page admission, configured limit, saturating accounting and
completion-boundary eviction. Six real actions share one20-line export facade.
Two23-line-total ARC leaves create/release retained buffers; cached references
belong to the map, owned views transfer to ARC, and transient views also belong
to the command collection. Existing command completion still decides when
retirement is safe. The original C fallback and lifetime enum stay intact.

Integration17:49–17:54UTC starts at7224c6b9. The same production-linked lifetime
hook first passes the C control, then native ReleaseFast/Debug and full-C
fallback. All24 final GPU cases pass: repeated/page-normalized identity,
distinct owned/transient objects, actual command-array membership, three
nonzero-offset GPU reads, insertion without premature eviction, completion-bound
eviction under1MiB and unlimited caching, each with ordinary and unretained
command buffers; plus each selection's existing SSD and complete safe-math
numerical fixtures. No shader/tolerance or assertion weakened. Native builds
pass on their first compilation with no warnings.

The first C trial supplied unretained=0, but that existing knob is presence-only;
all four short cases therefore used unretained mode. Its artifacts are retained
separately and excluded from the24-case count. Corrected qualification omits
the variable for ordinary commands and reruns the C control before activation.
The hook begins the command scope before creating transient views, so its
membership assertions cover the real lifetime rather than an earlier scope.

A frozen-source real1017 GGUF C/native pair returns identical MAPLE output.
Whole processes5.805/4.867s are one ordered warm-cache functional pair, with no
latency claim. All guards stay normal with unchanged swap. Exact settings,
commands, hashes, wire/fixture outputs and original trial are retained; no
binaries/model/cache stores are checked in.

```
tools/zig-build command-memory-test metal-test ssd-experts-test ds4 \
  -Dnative=true --prefix .workspaces/exact-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/exact-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-metal-exact-cache-20261022/qualify.py \
  --prefix .workspaces/exact-repeat/native --out .workspaces/exact-repeat/cases
```

Use -Dnative=false for full C and -Doptimize=Debug for native checks. Reuse the
published clean_exec helper from the Session report. model.py retains the exact
workspace prefixes of the real-model pair. Other Metal dispatch and broader
Engine/Session lifecycles remain queued owners.
