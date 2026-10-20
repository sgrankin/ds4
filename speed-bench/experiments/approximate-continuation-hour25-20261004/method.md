# Approximate live continuation

Use unchanged item19 `run.py` through the item24 one-arm launcher. Three serial arms are `before`, `fixed`, `after`. `--server PATH` is required: ROOT explicitly selects the saved old binary (runtime00da420b, SHA256 begins dd601278) for both brackets and the newly qualified binary for fixed. Both phases of an arm, including the restart, use that same selected path; phase execution and summary JSON record it. No implicit fallback to MAIN's current binary is allowed.

All arms request canonical1/PAD1, suffix64, hot0/SSD4096 MiB, GAP0, selected64/limit1024, Q2/cache64/ctx8192/chunk512/batch2, stable full-input tools/instructions, temperature0 and effort none. Each arm gets fresh cache/store paths and uses the same advertised A–D files. The existing four groups and eight graded replies/fresh-read checks remain unchanged: warm A/B, warm C/D, edited A/B, restart, edited C/D.

ROOT's proposed runtime difference is one continuation condition: permit the Responses visible-history path on a valid approximate live graph with the same identity hint, while retaining canonical rendering for native warm histories. Only the visible-prefix tier is re-enabled: call-ID-only and memory-text tiers stay bypassed for canonical full-input clients. It retains the approximate state's old conditioning; it does not make it a canonical exact-prefix cache or qualify arbitrary history/branch reuse. ROOT owns implementing, checking and compiling that condition. This launcher does not change the driver, core or request semantics.

Report warm/edited group makespans, queue-inclusive step durations, before/after tool-request prefill/cache counters, actual graft geometry/source, format and fact grades, caps and guard results. Preserve all eight steps and all generated text, including nondeterministic warm output or caps. A bracket mean is not a fair causal latency comparison if warm/edited histories materially diverge; describe those differences rather than silently remove failures or tune the input. Generated divergent histories have no paired KL/NLL or hidden-state parity claim.

Example ROOT command using its explicitly saved baseline binary; repeat with the new binary for fixed and saved binary again for after, all in fresh namespaces:

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/approximate-continuation-hour25-20261004/guard-before -- python3 -B speed-bench/experiments/approximate-continuation-hour25-20261004/launch.py --arm before --server /ABSOLUTE/PATH/TO/SAVED/OLD/ds4-server --output speed-bench/experiments/approximate-continuation-hour25-20261004/before --cache .workspaces/server-agent-day-20261004/continuation25-cache-before --store .workspaces/server-agent-day-20261004/continuation25-store-before --shared .workspaces/server-agent-day-20261004/concurrent-visible
```

The binary path placeholder is intentionally not guessed; ROOT must provide the actual retained artifact. The launcher records the actual command/path and requested flags. Preparation counts toward item25; no builds, model launches or runtime edits were performed by the author.
