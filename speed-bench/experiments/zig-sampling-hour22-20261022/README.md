# Native numeric sampler

Completed 2026-10-22 inside the05:33–06:33UTC one-hour ceiling. Code baseline
`b8c71f98`; the C-sampler model control is that snapshot's otherwise-native CLI.
Seven primitive APIs move argmax/exclusion, xorshift draws, filtering,
materialized probabilities and speculative residual sampling into500 lines
of handwritten Zig. The small real header exposes arrays/counts/knobs/RNG and
optional scratch, never engine/session layouts. One C xmalloc bridge preserves
allocation admission and fatal OOM policy. Existing C implementations remain.

Session and stop-token policy, overthinking marker configuration/thresholds,
penalty application and caller-owned scratch stay C. Native comments preserve
the finite-1e30 sentinel, first-index ties/NaN anchors, zero-seed substitution,
wrapping multiplication, RNG draw consumption, heap/fallback proof and filter
order. Strict float regions retain classification and scalar accumulation.
Typed standard sorting replaces generic C qsort without changing its total
candidate ordering. The two initial obsolete array-repeat spellings were
replaced with Zig0.17 typed@splat before compilation; first formatter log retained.

All five C/native programs build warning-free. Unchanged tests/test_sampling.c
links the selected actual sampler and actual engine test TU, not copied bodies.
It passes C/native × Debug/ReleaseFast, covering256 seeds per ordinary case,
ties/nonfinite/cutoff values, null scratch, penalty/thinking thresholds and
speculative correction/distributions. Primary test math matches existing Make:
optimized engine, finite-aware reference test source. Both actual baselines pass;
no alternate diagnostic math mode or production-flag change was introduced.

## Alternating CPU measurement

The existing fixture samples a fixed163840-entry synthetic vocabulary100 times
per measurement. C/native/native/C runs all pass, with identical checksums and
RNG states. Reference-loop timings stay stable.

| Selected sampler | C mean | Native mean | Less CPU time |
| --- | ---: | ---: | ---: |
| Default min-p | 70.139ms | 29.6285ms | 57.76% |
| Top-p plus min-p | 170.4035ms | 97.382ms | 42.85% |

Savings against the faster C controls are57.51%/42.24%; C bracket drift is
1.16%/2.13%. These are CPU sampler costs, not model decode-throughput gains.
At100 samples, default cost drops from0.701 to0.296ms per draw in this fixture.

## Actual model callers

Twelve guarded model runs compare the preceding C-sampler native CLI with the
new native-sampler CLI. Greedy, nucleus, default min-p, top-p/min-p, full softmax
and enabled overthinking-penalty cases all return successful nonempty output.
Every paired continuation is byte-identical; greedy returns ORCHID42 and both
penalty runs report the enabled C policy. This exercises actual normal sampling
and its existing session/penalty integration. It is a fixed-seed caller check,
not a broad quality suite or an alternating model-speed benchmark.

The first driver used a nonexistent --top-k CLI flag and stopped immediately
with usage status2; those results are retained. The corrected driver uses the
actual CLI's exposed controls. Top-k remains covered by the existing public-API
fixture; no new CLI option was added just for qualification. Output differences
would be recorded for inspection rather than automatically rejected.

Decision: **retain**. `tools/zig-build sampling-test` builds the unchanged
production-linked fixture; run bin/test-sampling from the selected prefix.
[Exact commands, hashes, CPU measurements and all real outcomes](results.json),
[build/CPU loop](build.py), [actual model loop](qualify.py). Full first logs,
continuations and pressure records remain under `.workspaces/zig-sampling-20261022/`.
