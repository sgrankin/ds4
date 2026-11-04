# Preserving contribution rounding in indexed prefill attention

Later [combined qualification](../2026-11-04-combined-prefill/README.md) adopted
the original shortcut in the named fast presets, with explicit numerical and
practical limitations. The experiment below records its earlier decision.

**Remove the rounded-contribution prototype.** It increased fixture error and
did not retain the isolated operation benefit of skipping unit rescaling.
The one-hour window is November 4, 2026, 12:09–13:09 Eastern. The baseline is
`1312c914`; the measured prototype revision is `46c7b454`.

## Question and mechanism

The [unit-rescale experiment](../2026-11-04-attention-unit-rescale/README.md)
skipped `exp(M - M)` and multiplication of the accumulated output by one
when an attention row did not change the finite running maximum. It reduced
isolated operation time by about 9–10% and full-model inference total by about
1.1–1.5% on its measured workloads. Small kernel differences propagated to
model logits, so that version remains opt-in and is not enabled by the fast
presets.

The new candidate tests a possible explanation: removing the identity multiply
may change floating-point contraction and the rounding of the weighted KV
contribution. It explicitly rounds that contribution before using
`fma(output, 1, contribution)`, with local `#pragma clang fp contract(off)`
and `reassociate(off)`. This is a hypothesis about contraction, not evidence
of the instructions emitted for the baseline. The fixture and operation checks
test the numerical effect and whether any speed benefit survives.

The prototype `kernel_dsv4_indexed_mixed_attention_heads16_rb4_unit_scale_rounded`
specialization is selected by `DS4_METAL_V41_ATTN_UNIT_SCALE=2`; `1` selects
the prior approximation and `0` selects the original path. Four-row staging
must be enabled with `DS4_METAL_V41_PREFILL_ATTN_RB4=1`. The change is limited to
eligible indexed prefill attention and changes no packet or storage format.
The prototype was removed after measurement; the earlier mode 1 remains opt-in.

## Measurement setup

On an Apple M5 Max with 128 GiB RAM, we compared mode 2 with the expanded
attention reference and mode 0 using the existing six-shape fixture, then ran
five alternating operation pairs at 512, 2048 and 8192 rows. Full-model timing,
quality and a long-context coding task were conditional on promising operation
results and were not run. Local raw artifacts are in
`.workspaces/hillclimb-20261104/06-attention-contribution-rounding/`.

## Fixture results

All six packed controls and all three fallback cases matched exactly. Mode 2
changed more active-path elements and had higher RMS error than the prior
mode 1 approximation on each active fixture input:

| Rows / heads | Changed elements | Maximum absolute error | RMS error |
| --- | ---: | ---: | ---: |
| 2 / 64 | 45,130 / 65,536 | 1.335144e-5 | 1.32256e-6 |
| 33 / 32 | 375,633 / 540,672 | 1.144409e-5 | 1.25328e-6 |
| 3 / 64, short-window tail | 18,811 / 98,304 | 2.9802e-8 | 3.20145e-9 |

The prior report contains the mode 1 values. These results do not establish
full-model quality or identify the baseline's compiled instructions.

## 2048-row operation

Five alternating pairs used the warmed 64-head fixture with four calls per
sample. The timed scope includes sorting, encoding and GPU completion, but
excludes model execution, expert loading and allocation. In the mode 2
comparison, the control averaged 31.92515 ± 1.06997 ms per call and the candidate 31.88940 ±
0.07237 ms, a 0.112% reduction between means. Four of five candidate samples
were slower than their paired baseline. The first baseline sample was
33.83550 ms; the other four ranged from 31.38125 to 31.55725 ms. All samples
are retained, and variation is sample standard deviation.

A separate five-pair mode 1 comparison on the same binary averaged 31.46375 ±
0.11993 ms for its baseline and 28.57200 ± 0.12126 ms for the candidate,
a 9.191% reduction; all five pairs favored mode 1. These are separate paired
comparisons against the original path, not an alternating mode 1 versus mode 2
comparison.

| Pair | First | Mode 2 control / candidate (ms/call) | Mode 1 control / candidate (ms/call) |
| --- | --- | --- | --- |
| 1 | Control | 33.83550 / 31.84750 | 31.61350 / 28.76575 |
| 2 | Candidate | 31.38125 / 31.93575 | 31.47650 / 28.47850 |
| 3 | Control | 31.43475 / 31.99400 | 31.29275 / 28.61675 |
| 4 | Candidate | 31.55725 / 31.82600 | 31.52125 / 28.50275 |
| 5 | Control | 31.41700 / 31.84375 | 31.41475 / 28.49625 |

The local CSV files retain the samples. The pressure guard reported no
interruption; its initial reported swap was approximately 24.97 GB.

## 512- and 8192-row operations

The same procedure gave 8.28165 ± 0.12836 ms control and 8.40785 ±
0.08140 ms candidate at 512 rows, a 1.524% increase between means. At 8192
rows it gave 124.68350 ± 1.07000 ms and 126.21370 ± 0.03899 ms, a 1.227%
increase. Four of five candidate samples were slower in each comparison. The
first 8192-row control, 126.59350 ms, remains included. Variation is sample
standard deviation.

| Pair | First | 512 control / candidate (ms/call) | 8192 control / candidate (ms/call) |
| --- | --- | --- | --- |
| 1 | Control | 8.50700 / 8.47525 | 126.59350 / 126.18825 |
| 2 | Candidate | 8.22175 / 8.51325 | 124.12100 / 126.16400 |
| 3 | Control | 8.22075 / 8.34800 | 124.22200 / 126.21350 |
| 4 | Candidate | 8.26375 / 8.37250 | 124.17125 / 126.24350 |
| 5 | Control | 8.19500 / 8.33025 | 124.30975 / 126.25925 |

## Decision

Remove mode 2: it did not achieve the intended fixture-rounding improvement
and offered no operation advantage on the tested shapes. This decision does
not rest solely on a small, noisy slowdown, and the fixture errors do not
establish worse practical task quality. No full-model timing, quality or
practical task was run for mode 2. The original shortcut remains opt-in; a
subsequent combined qualification will assess the package and preset choice.

## Reproduction

The measured source is preserved at `46c7b454`. Build that revision with
`./tools/zig-build -Doptimize=ReleaseFast ds4-bench metal-test -j4`. The
current fixture was restored along with the runtime and no longer accepts the
prototype's mode argument. At the prototype revision, run these under the
pressure guard, using separate guard directories for each invocation:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/rounding-fixture-guard \
  -- ./zig-out/bin/test-deepseek41-metal --fp4-attention
python3 speed-bench/ideas/pressure_run.py .workspaces/rounding-op-guard \
  -- ./zig-out/bin/test-deepseek41-metal --fp4-attention-bench 2048 2
```

Repeat the operation command with 512 and 8192 rows, and with mode `1` for
the separate prior-shortcut comparison. The local CSV files and guard records
retain the original samples and conditions.
