# First-attempt closing-token pilot

The opt-in ADD2 closing-token boost produced no useful completion improvement
on the four frozen counting cases. All original first attempts are retained:
each arm has two completed correct answers and two capped INVALID answers.
The original pressure guard and producers returned 3; analysis integrity passes
without rewriting those exit codes or statuses.

| Arm | Completed correct | Capped INVALID | Generated tokens | Changed cases |
| --- | ---: | ---: | ---: | ---: |
| Native before | 2 | 2 | 5985 | 0 |
| ADD2 after 1024 thinking tokens | 2 | 2 | 5984 | 1 |
| Native after | 2 | 2 | 5985 | 0 |

Both native arms exactly match the retained original effort25 reference across
all four token/RNG/piece sequences, status/final fields and raw output bytes.
The circular case receives zero active interventions and matches exactly.
Only the derangements answer trajectory changes, from 1680 to 1679 generated
tokens. Both capped cases retain their native token/RNG/piece/status sequences
despite 1024 active boost steps each. Total active steps are 2701.

The sampled active-step mean KL values are 1.7427e-10 for derangements,
1.3550e-12 for lattice and 2.2153e-12 for subset. These are same-prefix
full-vocabulary **pre-filter** raw-to-policy distributions at actual FP32
temperature 0.6000000238418579 and the actual rounded FP32 ADD offset to
ID128822. They are not post-top-p probabilities, corpus loss or cross-prefix
native/candidate KL. No speed, quantization-causality or general accuracy
conclusion follows from this four-case diagnostic pilot.

The original d02 analyzer failed closed before reading results because its
mocked release schema differed from ROOT's producer. That refusal is retained.
Corrective source bb9336921b50081d1f642a2e644297481aa71fad changes only the
release gate to the real typed ROOT schema. Sixteen model-free CPU tests pass;
checkout independently reviewed the original analyzer and corrective gate.
The executed corrected analyzer is pinned in the evidence archive.

ROOT release metadata records COMPLETE/childrc3, a reacquired MAIN lock, no
live model processes, 887 normal pressure samples, zero bad pressure samples,
zero swap growth and peak wired plus compressed memory 80.1580 GiB. No model,
tokenizer or GGUF metadata was opened by this analysis.

`closing-first-attempt-evidence` contains the compact report, complete compressed
100-file completed-run archive, original source/large-artifact path/hash pins,
CPU preparation and fixtures, the preserved refusal and independent-review
record. Original raw traces and first-attempt outputs remain at their pinned
paths. Every archive member and all 194 frozen source assets were verified;
no completed study, runtime, policy or prompt was changed.
