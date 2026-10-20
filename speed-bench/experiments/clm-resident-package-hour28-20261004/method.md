# Resident routing on actual suffix CLM coding

Use the repaired Item27 caller unchanged: the frozen Item17 ledger/LRU contracts and first-write161 graders, actual config READ, removal of exactly two obsolete note spans, then a fresh WRITE user after the real function_call_output. Full input, stable instructions/tools, store:false, temperature0, READ high/0, WRITE high/128, FINAL high/1 and global output cap2048 stay fixed. The absolute first-code grader path repair is retained. The keys and advertised confirmation-visible paths match Item27.

Run `native-before` → `subs3` → `native-after`, each with a fresh cache/store. All three arms enable suffix64 with canonical1/PAD1, hot0/SSD4096, Q2/cache64GB, ctx8192/chunk512/two slots, selected tile64/limit1024. Here native means native routing; it does not disable suffix reuse. Brackets explicitly use GAP0/native_weights0. The candidate uses GAP.90, mass.25, rank64, max_substitutions3, total_mass.50, replace and native_weights1. Other inherited DS4/PT settings are cleared. Only resident routing changes.

Each arm completes ledger, restarts the same server binary, then completes LRU. The same route profile is used before and after restart; both tasks warm their actual native sources under that arm's policy. Do not import GAP0 endpoints into the candidate. The client is serial despite two configured slots. LRU's READ happens after restart, so this does not claim a pending LRU checkpoint survives restart.

Keep wrong first code, grades, caps, errors and exact fallbacks. Report each task's READ/edit/WRITE/FINAL wall, first-code161 result and actual final reply separately from startup/restart/whole launcher wall. Preserve observed graft geometry/fresh span/PAD/source and existing route/cache counters from server logs. No new telemetry is needed. The candidate may degrade code quality; no retry or repair is allowed. Different generated histories do not support paired NLL/KL or an exact hidden-state/forgetting claim.

Example ROOT command; repeat with fresh output/cache/store paths for subs3 and native-after:

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/clm-resident-package-hour28-20261004/guard-native-before -- python3 -B speed-bench/experiments/clm-resident-package-hour28-20261004/launch.py --arm native-before --output speed-bench/experiments/clm-resident-package-hour28-20261004/native-before --cache .workspaces/server-agent-day-20261004/coding28-cache-native-before --store .workspaces/server-agent-day-20261004/coding28-store-native-before --visible .workspaces/server-agent-day-20261004/confirmation-visible --timeout 900
```

ROOT owns builds, launches and the pressure guard. Only authored source and AST checks were performed during preparation; they count toward the experiment budget.

At20:13 UTC, the completed strong point left enough budget for a bounded
`subs1` point before the closing native control. It changes only the maximum
substitutions3→1 under the same strong settings and warms its own new cache.
This supplies a lighter routing tradeoff rather than an assumed improvement.

The first strong LRU grade failed because the restricted execution scope
omitted Python's builtin `iter`, used by `next(iter(cache))`. The task allows
builtins and its language validator accepts that code. Add only that missing
builtin to the existing scope; retain original failed grades/child3 and regrade
the saved first codes in every already-completed arm. Gold fixtures,161 checks,
actual code and HTTP generations are unchanged; no model retry or repair.
