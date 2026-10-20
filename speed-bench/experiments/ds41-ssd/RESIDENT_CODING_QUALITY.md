# Resident routing: actual generated coding screen

Native and resident routing each generated a correct fix for the original
agent-session fixture's first reconciliation task within a 275-second wall
budget. External checks passed all five original unit tests, 10,000 exact-cent
conversions, duplicate/refund/void/zero-account semantics and generator input.
The CSV and original test file remained byte-identical; original test ASTs
were unchanged. Only ledger/money.py and ledger/report.py changed in each arm.

Both agent turns timed out before their final waiting marker. Generated-code
correctness passed; completed_model_turn and complete_harness_success are
false for both. The later since/filter and CLI prompts were not evaluated.
No completed-turn timing comparison or speed benefit can be inferred here.

Settings: DeepSeek-V4.1-Flash-Q2.gguf, SSD cache64GiB, context8192,
prefill chunk512, temperature0, seed1234, per-generation cap4096. Baseline
GAP=0; candidate GAP=.05 MAX_MASS=.12 MAX_SUBS=1. Frozen root agent SHA256:
8b4b611d3964e48f07c6afa167e60a274b8c86af2b5bc4c5484389e8550772f5.
Metal shaders and the original buggy fixture were frozen for both arms.

The candidate command/provenance confirms an explicit DS4.1 path, SSD mode,
normal scalar agent generation, and no oracle, async, prefetch, quality,
vision, TP or speculative flags incompatible with the routing path. The
positive GAP path rejects incompatible graph modes during decode setup;
actual model generation and tools proceeded. Its atexit counter report was
not emitted after SIGTERM, so this coding run has no measured substitution
count. Earlier complete replay/teacher-quality experiments supply those
counts and their own separate performance/quality evidence.

Both jobs used the MAIN ROOT pressure guard and shared model-job lock.
All pressure samples had level1, zero adverse samples, and max swap0.03186GiB.
The guard cleaned each timed-out model job. No further model work followed.

Persistent main-root evidence directory:
`speed-bench/ideas/artifacts/resident-coding-pair-20261002T1705/`

Compact files: manifest.json, coding-quality-summary.json,
pressure-summary.json, and for baseline-ds41 / resident-ds41-full275:
command.json, provenance.json, generated-code-checks.json,
generated-files-sha256.json, project-final/ledger/money.py and report.py.
Candidate eligibility.json records eligibility and its counter limitation.
Raw stdout/trace logs, shaders/binary, and original/final projects are retained.
The root artifact postcheck_generated.py is the independent CPU helper.

Excluded attempts: initial baseline/resident tried ds4flash.gguf, whose existing
symlink resolves to Qwen3.8; startup rejected SSD streaming before inference.
The symlink was left unchanged. The resident-ds41 attempt inherited an earlier
short deadline and is excluded; resident-ds41-full275 restarted from the
ORIGINAL unchanged fixture with a full275-second budget, never baseline fixes.

This evidence-only child is based on86722bec (resident replay runner), without
optional parser hardening or weak-miss drop prototype ancestry.
