# Item76 — Native Metal model-view policy

The138-line handwritten owner handles page-aligned overlapping view ranges,
coverage/lookup, bounded registry order and map removal/clear. One actual shared
registry owns the original strong MTLBuffer fields. Zig changes metadata; four
small ARC/driver leaves query device limits, create, assign/move and retire real
buffers. Partial-success accounting, cumulative wrap statistics, map identity,
maxBufferLength and original knob acceptance remain unchanged. The C policy
fallback uses the same actual record, with no Engine/Session mirror.

Integration17:40–17:43UTC starts at389ea7d4. C policy control, native ReleaseFast,
native Debug and full-C fallback all build cleanly and pass the same three
production GPU fixtures (twelve runs). The SSD normal mode checks4100 target
map replacements, disjoint spans and retained auxiliary-map GPU reads; using
--table-admission alone would not qualify those actions. Router and complete
numerical tests pass; the full numeric diagnostic uses the previously required
DS4_METAL_MATH_SAFE=1 for all selections. No shader or tolerance changed.

A fixed real1017 IQ2_XXS/Q2_K GGUF C/native generation pair also passes, returning
byte-identical `MAPLE\n` through the full model mapping. Its sources match before
and after. Whole processes5.460/2.728s are an ordered warm-cache functional pair,
not a latency result. Every guard stays at normal pressure with fixed swap.
No new test assertions or compiler/runtime failures were introduced.

```
tools/zig-build ds4 ssd-experts-test dense-mpp-test metal-test \
  -Dnative=true --prefix .workspaces/views-repeat/native
python3 -B speed-bench/ideas/pressure_run.py .workspaces/views-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-metal-model-views-20261022/qualify.py \
  --prefix .workspaces/views-repeat/native --out .workspaces/views-repeat/gpu
```

Use -Dnative=false for full C and -Doptimize=Debug for native checks. Published
qualify.py uses cwd-relative existing clean_exec.py. model.py retains the exact
workspace prefixes from the actual pair. Results/evidence preserve commands,
settings, hashes and outputs; no binaries/models or backing stores are checked
in. Exact-range cache/lifetime and other Metal dispatch remain subsequent owners.
