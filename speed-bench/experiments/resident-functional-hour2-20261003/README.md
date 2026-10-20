# Resident experts: second hour

Measured 2026-10-03, 12:29–13:17 UTC. All behavior is in private opt-in
prototypes; MAIN runtime defaults were not changed.

The strongest simple affinity policy reduced decode time by 16–19.5% and
logical expert demand reads by 17.2% on 24 fixed-input tasks. Mean delta-NLL
was -0.01260 nats/token and KL was 0.02056. Functional similarity did not
establish an advantage over this simpler selector. These are measured
tradeoffs, not an accuracy guarantee or a default-promotion decision.

## Measurements

Each task scores 32 fixed, independently written continuation tokens. Means
are token weighted. Native controls before, after and late reproduced all
768 vectors exactly. Their decode times were 43.45, 43.94 and 43.82 seconds.
Prefill improved later in the run, so the later control is included and
prefill savings are not attributed entirely to routing.

| Policy, 24 cases | Delta-NLL | KL | Decode seconds | Logical expert reads, GB |
| --- | ---: | ---: | ---: | ---: |
| Native before | 0 | 0 | 43.45 | 700.61 |
| Affinity, gap .10, one replacement | -.00880 | .00698 | 42.37 | 674.13 |
| Functional, one replacement | -.00342 | .00900 | 42.51 | 664.84 |
| Functional with router band .025 | -.00153 | .01008 | 40.98 | 663.70 |
| Affinity, three replacements, mass .20 | -.01260 | .02056 | 36.70 | 580.26 |
| Affinity, three replacements, mass .25 | -.01260 | .02056 | 35.26 | 580.26 |
| Native late | 0 | 0 | 43.82 | 700.61 |

The two three-replacement policies made identical routing decisions and
produced identical quality scores; increasing mass .20 to .25 was not binding
on these cases. Different timings for identical decisions illustrate run
variation. The worst case mean delta-NLL at this point was +.06217. The
profiles were enabled in every arm. Logical reads include demand prefill and
decode reads, not independently observed physical SSD bytes.

The four-case screen varied affinity gap, replacement count and functional
router band. Pure functional selection had +.02590 delta-NLL there; band
.025 reduced it to +.00118, but that improvement did not generalize to the
24-case mean. All measured points remain in the CSV and plots. No candidate
was discarded for disagreeing with native vectors.

In total: 23 fixed-input configurations, 8,064 scored rows, plus 20 generated
tasks across five policies. See [results.csv](results.csv),
[per_case.csv](per_case.csv), [decode tradeoff plot](pareto_decode.png) and
[combined prefill/decode plot](pareto.png). Grey control ranges show measured
variation, not confidence intervals.

## Generated answers

Four independently graded tasks used temperature zero and a 256-token cap.
All five policies completed two of four tasks; the other two reached the
cap, including native. Native, one-replacement affinity, pure functional
and band .025 had two correct first JSON answers. Three-replacement affinity
had three correct first answers, but still only two completed tasks. Its
decode time fell from 47.88 to 40.82 seconds (14.7%) and whole-task runtime
from 82.66 to 75.40 seconds (8.8%). Different generated histories make these
task-time comparisons, not paired distribution comparisons.

Native produced a wrong endian sum and a wrong weighted median, then began
correcting both in prose before hitting the cap. The stronger affinity
policy gave the correct median in its first JSON object. Baseline failures
and caps are retained in [generation/results.csv](generation/results.csv).
Four tasks are too few to infer an accuracy bound.

An initial generation pilot used the default temperature-zero engine path,
which bypasses session hooks and our timing footer. It was deliberately
stopped. The corrected campaign sets existing DS4_CLI_FORCE_SESSION=1 and
checks actual functional activation and the generation footer. Pilot files
are retained separately as generation-engine-path, not counted above.

## Implementation and reproduction

Use [run.py](run.py) under the existing MAIN pressure_run.py guard. The private
functional runtime source is revision 0b5b5ae4528f361c44a8089d6f210c0d15dbbdd5
in .workspaces/idea-resident-functional-runtime-native. The existing completed
functional map is reused, without a model rescan. Its 16 representative
normalized inputs and output projections rank eligible resident experts.
The six native GPU routing coefficients are retained; only expert IDs change.

The strongest affinity configuration uses gap .90, rank 64, maximum individual
mass .25, maximum three substitutions, total substituted mass .50, replace
policy and native weights. Context is 8192, expert cache 64GB, chunk 512,
selected tile 64 and limit 1024. All per-arm environments and argv are saved
in execution.json files.

The band experiment is a small compile-time restriction: functional ranking
chooses only among residents close to the best eligible router score. See
[ds4_resident_functional_band.c](ds4_resident_functional_band.c) and
[build_bands.sh](build_bands.sh). The CLI timing change is in
[ds4_cli_metrics.c](ds4_cli_metrics.c), built by [build_cli.sh](build_cli.sh).
Host fixtures passed optimized and ASAN/UBSAN checks; builds had no warnings.

Across 1,930 pressure samples, pressure stayed normal, swap did not grow and
peak wired plus compressed memory was 79.08 GiB. Only the deliberate pilot
interrupt stopped a guard. Raw vectors, compiled binaries and map artifacts
are retained locally and excluded from the commit.

Next: measure the original-expert arrival policy against this fixed affinity
policy, including its staged reads and the earlier mandatory pool join.
