# Resident routing with bounded thinking

Use the unchanged item20 natural runner directly, frozen hour17 ledger/LRU task files and first-write graders. Stable task instructions and tool definitions remain identical across requests. READ is high effort/budget0; WRITE high/budget128; FINAL high/budget1. Stored `previous_response_id` carries native private reasoning. Each task retains the shared 2048 output allowance, first implementation, 161-check grade, actual final text, caps, errors and per-role cached input/usage. No repairs or retries.

The only varied runtime feature is resident expert routing:

| Arm | Gap | Maximum substitutions | Other settings |
|---|---:|---:|---|
| native-before | 0 | 0, routing disabled | native routing |
| subs3 | .90 | 3 | mass .25, rank64, total mass .50, replace, native weights |
| subs1 | .90 | 1 | same requested settings; only max substitutions changes |
| native-after | 0 | 0, routing disabled | native routing |

The native arms disable routing through GAP0 rather than supply the unsupported `MAX_SUBS=0`. For subs1 the implementation's single-replacement mass gate applies; its aggregate mass option is only consumed for multiple substitutions. Do not interpret a maximum as a guaranteed replacement count.

Every arm has a fresh cache/store, the same confirmation-visible paths, server runtime supplied by ROOT, Q2/cache64/ctx8192/chunk512/batch1, selected64/limit1024 and temperature0. Suffix reuse is explicitly off in every arm; this does not qualify resident and suffix approximations together. Existing server stderr retains resident-route substitutions/skipped loads/bytes/native-weight counters and cache diagnostics when emitted. Skipped route bytes are logical avoided expert loads, not measured physical SSD bytes; absent counters remain absent. No new telemetry is added by the launcher.

Compare completed whole-task wall and READ/WRITE/FINAL costs against native bracket mean/spread, alongside first-write correctness and completion. Approximate routing may produce wrong code, alter thinking length or cap; all such outcomes remain visible and code grade stays separate. This two-task generated-workflow screen has no paired NLL/KL or broad accuracy guarantee. Earlier reasoning experiments motivated the fixed budget choices; this screen measures the package against the same reasoning settings with routing disabled.

Proposed ROOT launch (not executed by the author):

```sh
python3 -B speed-bench/ideas/pressure_run.py speed-bench/experiments/resident-combined-hour22-20261004/guard -- python3 -B speed-bench/experiments/resident-combined-hour22-20261004/launch.py --output speed-bench/experiments/resident-combined-hour22-20261004/http --max-wall-seconds 1500
```

The four arms are serial; expected duration is about 8–10 minutes. ROOT owns builds, model execution, pressure checks and results. Preparation counts toward item22's hour.
