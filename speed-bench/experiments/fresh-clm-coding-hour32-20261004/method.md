# Fresh first-attempt CLM coding tasks

Two unused event-processing tasks use the frozen Item17 fixture format: task_order, tests_per_task161 and tasks with key/prompt/system/config; each generated file must contain solve(payload, config). The project-visible config contains the actual policy and one sample input, never its gold. Use the existing Item27 real READ → remove two obsolete note spans → first WRITE → completion flow, with all live task requirements/config result retained. This folder prepares only fixtures/grades; ROOT owns any later caller/launcher adaptation and model launch.

`dedupe` tracks the last ACCEPTED timestamp for each exact string key. A rejected duplicate never refreshes expiry; elapsed time >= ttl accepts, so the equality boundary expires. ttl0 accepts all events. Return accepted and rejected input-index lists. Empty keys, equal timestamps, Unicode distinctions, interleaved keys, exact boundaries and repeated rejected events are covered.

`bucket` starts full at time0. Every event refills integer tokens up to capacity, advances the previous timestamp even when rejected, then accepts cost <= available tokens and spends only on acceptance. Oversized costs spend nothing; zero cost accepts at capacity0. Return accepted/rejected indices, remaining balance after every event and final tokens. Empty payloads, equal timestamps, no refill, zero capacity/rate/cost, saturation, oversized rejection and large integer arithmetic are covered.

Each private cases.json table is literal161 cases: one visible sample,16 fixed edges and144 deterministic cases using seeds20261004321/322. Expected answers were authored before any model. Four fixed hand answers and all322 gold rows were checked by a separate per-key clock for dedupe and one-time-unit refill for bucket; check-results.json records GOLD_SANITY_PASS. These are small fixture computations, not model output tests. Reference implementations remain private and comply with the existing allowed Python language.

grade.py borrows the existing language/JSON/nonmutation checker. One simple161-case loop reports passed/failed counts and the first failing case/reason, preserving partial success rather than stopping at a wrong answer. Syntax or missing solve refuses execution; runtime errors and input mutation are visible failures. The first implementation is never repaired/retried. Full161 success remains a distinct task grade, separate from response completion, caps and actual read/write contract.

Fixture-authoring command already completed once:

```sh
python3 -B speed-bench/experiments/fresh-clm-coding-hour32-20261004/practical/prepare_fixtures.py
```

Preparation and this small computation count toward Item32's hour. No model, tokenizer, native build, runtime/caller/launcher edit or jj command was executed by the fixture author. ROOT may choose a client example instead; these fixtures remain reusable.

ROOT copied the reviewed Item27 caller and Item28 launcher with only fixture paths, dedupe/bucket names, cache-key label, and profile selection changed. READ high/unrestricted, WRITE high/budget128 and FINAL high/budget1, global2048 cap, temperature0 and the same actual read/edit/write requirements stay fixed. Planned order is native-before (suffix0/routing0), context-fast (suffix64/routing0), persistent-fast (suffix64/strong3), native-after. canonical1/PAD1 are fixed; every arm warms in its own new cache/store under its real forward profile and advertises the same fresh32-visible paths. Labels select explicit equivalent settings; this is not a ds4-local wrapper execution test.

Before any model, ROOT ran both reference files through the actual grader; each passed161/161. The frozen JSON hashes and command results are recorded in build.json/check-results.json. Later code is measured once; no prompt, fixture or gold will change in response to generated code. Partial check counts, completion, caps and actual tool-contract failures remain separate outcomes.
