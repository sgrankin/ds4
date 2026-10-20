# Optional parity padding

Reuse item23's one-arm launcher and unchanged item19 four-session driver. Run serial `existing-before`, `padded`, `existing-after` with `DS4_SERVER_CONTEXT_SUFFIX_PAD=0/1/0`. Canonical materialization is0 throughout. Other settings remain suffix64/hot0/SSD4096 MiB, Q2/cache64/ctx8192/chunk512/batch2/GAP0, full-input/storefalse, stable tools/instructions, temperature0 and effort none. Every arm has a fresh cache/store and the same advertised paths; the same arm profile survives its server restart.

ROOT owns implementing and source-checking the proposed padding policy before launch. Candidate1 considers a newline after the first USER boundary only when the padded candidate gives a longer supported even-aligned suffix than the ordinary candidate. This does not implement odd cache relocation. Record which candidate was selected, padding and source/destination/reused/fresh lengths from actual server logs; do not assume PAD1 is always used. The newline changes the model input, so outputs, warm/prefix cost and later histories may change. This is an opt-in representation/eligibility approximation rather than native parity or a bug fix.

The unchanged groups are warm A/B, warm C/D, edited A/B, and edited C/D after restart: eight reply grades and eight required fresh reads. Preserve strict READY failures, wrong fact JSON, caps and exact fallbacks. Compare warm cost, edited group makespan, whole launcher wall and native-bracket spread alongside actual graft geometry and source/cache path. Count group log slices once. No retries, no parity tuning of the fixed fixtures, and no paired KL/NLL from divergent generated histories.

Example ROOT invocation, repeated with fresh `padded` and `existing-after` namespaces:

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/parity-suffix-hour24-20261004/guard-existing-before -- python3 -B speed-bench/experiments/parity-suffix-hour24-20261004/launch.py --arm existing-before --output speed-bench/experiments/parity-suffix-hour24-20261004/existing-before --cache .workspaces/server-agent-day-20261004/parity24-cache-before --store .workspaces/server-agent-day-20261004/parity24-store-before --shared .workspaces/server-agent-day-20261004/concurrent-visible
```

Phase execution and launcher-summary JSON record requested PAD/canonical values; `http/{before,after}` preserves the unchanged driver records. ROOT owns builds, model jobs and pressure qualification. Only source preparation and syntax checks have been performed by the author. `source-note.md` is owned separately by checkout and is not edited here.

The primary padded arm completed at18:47:20UTC: all8 steps/reads passed, with edited group times64.848/64.562 seconds versus opening71.692/72.369. Logs show A/C pad1 and527 reused, B/D pad0 and527 reused. This justifies the bounded canonical-padded follow-up before the closing control: keep PAD1 and change only canonical0→1, then restore canonical0/PAD0. Record it separately from the primary padding comparison. The runtime and fixed fixtures are unchanged; all work remains within the19:25 deadline.
