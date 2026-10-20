# Host hot-cache curve

One change from item19 reuse64: host hot-cache capacity 0, 128 or 512 MiB. Reuse the exact item19 driver directly, without copying or changing its fixtures, prompt paths, tool protocol or grading. Native item19 is prior context; the proposed curve compares reuse64 at the three capacities rather than attributing native/reuse differences to RAM caching.

Each arm gets a fresh cache/store directory and runs four distinct full-input identities, two simultaneous clients on two resident server slots, SSD limit 4096 MiB, Q2/cache64/ctx8192/chunk512/GAP0 and suffix fresh tail 64. Warm A/B, warm C/D, edited A/B, then server restart and changed edited C/D. Advertised A–D paths stay identical between arms. The existing driver resets only its known active/archive files and preserves eight reply grades, eight actual fresh-read checks, all raw requests/responses, group makespans and queue-inclusive step durations. Strict format errors, wrong answers, caps and fallback histories remain measured; no retries or parity tuning.

RAM hot entries disappear at restart; persisted source manifests remain on SSD. Thus compare pre-restart edited A/B and restarted C/D separately. This is not a cold physical-SSD benchmark: the OS page cache may also serve file reads. Cache capacity does not prove hot residency or actual hit counts. Report hot/disk chunk counters and owned bytes only if recorded from the implementation; ROOT will decide a small capacity screen from actual payload sizes before launching the curve. No extra cache-stat instrumentation is part of this launcher.

Example single-arm ROOT command, repeated serially with fresh `hot128`/`hot512` output/cache/store namespaces and the same `--shared` path:

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/hot-cache-hour21-20261004/guard-hot0 -- python3 -B speed-bench/experiments/hot-cache-hour21-20261004/launcher.py --hot-mb 0 --output speed-bench/experiments/hot-cache-hour21-20261004/hot0 --cache .workspaces/server-agent-day-20261004/hot21-cache0 --store .workspaces/server-agent-day-20261004/hot21-store0 --shared .workspaces/server-agent-day-20261004/concurrent-visible
```

The launcher records the chosen capacity in phase execution files and `launcher-summary.json`; the existing item19 driver produces `http/{before,after}/result.json` and CSVs unchanged. Startup, client, shutdown, restart and whole-arm wall are separate fields. Existing item19 reuse64 took about five minutes per arm; three serial arms would take roughly fifteen minutes. Preparation and any authorized capacity screen count toward item21's hour. The author has not launched a model or cache screen.
