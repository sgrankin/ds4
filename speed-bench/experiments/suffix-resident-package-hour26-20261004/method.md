# Suffix and resident package

Use item19's unchanged four-session driver directly. All arms keep suffix reuse enabled, tail64, hot0/SSD4096 MiB, Q2/cache64/ctx8192/chunk512/batch2, stable full-input tools/instructions, temperature0 and effort none. Following Items24/25, ROOT selects explicit `--canonical 1 --pad 1` in every arm. The narrow approximate visible-continuation runtime also remains fixed. Record the actual chosen geometry/padding and continuations from logs.

Run serial `native-before`, `subs3`, `native-after` with fresh cache/store namespaces and the same advertised shared paths. “Native” here means routing GAP0/native weights0; suffix remains approximate in both brackets. `subs3` adds GAP.90, mass.25, rank64, maxsubs3, totalmass.50, replace/nativeweights1. Source warmup and replay both use the arm's fixed profile, including the server restart. Never import a GAP0 cache into the resident arm. Existing server/core permits this combination, but permission to configure it is not qualification of its quality or performance.

Preserve warm A/B, warm C/D, edited A/B and edited C/D after restart; eight independently graded replies and eight fresh reads per arm. Report warm cost, edited group makespan, whole launcher wall, route counters, observed suffix source geometry/cache path, strict correctness, caps and fallbacks. Count group log slices once. Wrong code/total/format remains an outcome; no repair or retries. Changes to warm generated text may affect later histories, so savings are whole-package evidence rather than the sum of earlier isolated gains. No paired NLL/KL is claimed from these divergent histories.

Example ROOT command, repeated with fresh `subs3` and `native-after` output/cache/store namespaces:

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/suffix-resident-package-hour26-20261004/guard-native-before -- python3 -B speed-bench/experiments/suffix-resident-package-hour26-20261004/launch.py --arm native-before --canonical 1 --pad 1 --output speed-bench/experiments/suffix-resident-package-hour26-20261004/native-before --cache .workspaces/server-agent-day-20261004/package26-cache-before --store .workspaces/server-agent-day-20261004/package26-store-before --shared .workspaces/server-agent-day-20261004/concurrent-visible
```

ROOT owns runtime qualification and all launches. Only this launcher/method were prepared; no model or build was run by the author. The directory is the preparation label; ROOT may schedule it under a later item number.
