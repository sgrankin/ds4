# Independent native-effort counting study

This six-problem study compares native effort levels 0, 25 and 75 on the same
frozen question strings. It evaluates requested reasoning effort with this Q2
runtime; it does not establish quantization causality. Different effort
prefixes and generated histories make cross-prefix KL/NLL unsuitable here.
No runtime, routing, logit policy, word penalty or closing boost is changed.

| Case | Independent gold methods | Exact answer |
| --- | --- | ---: |
| Exactly three fixed points among nine labeled positions | Enumeration / derangement recurrence | 22260 |
| Twelve-bead necklaces with four black beads | Canonical rotation orbits / Burnside fixed shifts | 43 |
| Onto functions from seven labels to four labels | Enumeration / inclusion-exclusion | 8400 |
| Domino tilings of a 3-by-8 board | First-empty-cell enumeration / closed recurrence | 153 |
| Five nondecreasing positive parts of 20, maximum 7 | Tuple enumeration / bounded partition recurrence | 32 |
| Connected labeled six-vertex graphs with seven edges | Edge-set connectivity enumeration / component recurrence | 5700 |

The questions are absent from the four counting calibration cases. Their
inputs, golds and methods are frozen in `input-freeze-v1.json` before any
model attempt on these questions. Every case has two exact integer methods
that agree; no model or tokenizer was consulted to choose or verify them.

Balanced process order is **0 → 25 → 75 → 75 → 25 → 0**. Each process uses
the same fixed six-case order, same native core objects and same helper. Each
case resets the sampler RNG to 20261002 and synchronizes an independent full
chat prompt. The existing native `session_sync` behavior refills the backend
when the prior case is not a prefix. Expert-cache warmth can carry between
cases; identical ordering and a fresh process for each block are maintained.

All blocks use Tfloat 0.6000000238418579, top-p float0.949999988079071, min-p0,
cap2048, SSD expert cache64 GiB, context8192, prefill chunk512, two CPU threads,
power100 and one draft token with ordinary scalar sampling. Word penalty is
zero, and every token checks that the native policy is inactive. No extra RNG
draw, model forward or full-vocabulary logit copy is used for diagnostics.

Level0 uses the actual native no-thinking protocol: no effort system text and
the assistant's supplied `</think>` close. Its final-answer region begins at
byte offset0. Levels25/75 use their native effort text and `<think>` prefix;
their final region starts after the last generated exact closing piece.
Prompt IDs, every sampled token including terminal stop, token piece bytes,
RNG before/after, thinking state and raw outputs are retained. The literal
strict-final-v2 C scorer is reused; an independent typed Python analyzer
reconstructs raw/final bytes and checks complete integer answers.

First-attempt outcomes are primary. All wrong, capped and INVALID outputs
remain unchanged, with original producer rc3 for any INVALID answer. Completed
correct requires a parsed correct integer, native stop and no cap; capped
correct is separate. Two repeated blocks per effort must match all six prompt
ID sequences, full token/RNG/piece sequences, raw outputs and result fields
exactly, excluding measured host times. A failed repeat or incomplete block
is a control failure, never an invitation to rerun or choose a better attempt.

There are 36 first attempts. The absolute generation cap is 73,728 non-stop
tokens plus up to 36 terminal stops. At 12–20 tokens/s this permits roughly
60–100 minutes of decode; budget **90–110 minutes** including six model starts
and prefill. Timings are diagnostic and do not support a speed claim. The
MAIN pressure guard must serialize the entire prepared study under a separate
explicit ROOT grant, after the growth and suffix follow-ups. No model opens
occur during preparation. CPU preparation pauses during ROOT timing freezes.

The helper-only link uses the eight pinned native effort25 core/auxiliary
objects from `idea-quant-effort`; no core or Metal rebuild is needed. Its
build proof records the unchanged object hashes, literal scorer source and
model-free self-test. The launch authority, required inventory, pressure
framework and exact parent identity will be frozen separately before grant.
