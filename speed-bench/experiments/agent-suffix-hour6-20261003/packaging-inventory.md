# Packaging the measured options

Use one entry point with explicit named presets. Environment variables can
remain internal implementation details; users should not need to reconstruct
the tested combination by hand. This is the inventory and proposed next item,
not a configuration framework. Update: `ds4-local` now implements context-fast
and resident-fast as a small shell wrapper.

| Option | Current main agent status | Packaging action |
|---|---|---|
| Selected prefill tile 64 / limit 1024 | Available through existing environment settings | Include the measured settings. |
| Idle context edits and suffix reuse | Available in main source and rebuilt agent, with `--live-context-file` | Offer `context-fast`: reuse 1, fresh minimum 128, four frontiers, 512 MiB staging, resident GAP 0. |
| Strong resident affinity with native weights | Measured ID-only coefficient-preserving path forwarded into main and rebuilt | Available as resident-fast; defaults remain opt-in. |
| WRITE-only reasoning cutoff 256 | Successful hook exists in the private practical helper; `PT_WRITE_REASONING_BUDGET` is ignored by main | Integrate the small hook after the actual READ→WRITE synchronization before offering `tools-fast`. |
| Marker penalties / no-thinking | Available today | Keep explicit options; they did not produce the measured coding winner. |
| Asynchronous original arrivals | Private prototype; age 8 timing-neutral, age 1 mixed quality/cost | Keep outside the first package. |

Start `context-fast` from the measured context 8192, expert cache 64 GB and
prefill chunk 512, with explicit caller overrides. It accelerates accepted
external edits; the caller still decides what to archive and where. The task
screen used no-thinking; this does not justify silently disabling reasoning
for arbitrary persistent-agent work.

Current suffix reuse requires resident GAP zero, so `context-fast` and the
future `tools-fast` must be separate modes until that interaction is tested.
An environment file alone cannot supply missing runtime code. Reuse the
current argument/environment parsers and keep the implementation small.

The wrapper and native-weight core path are built; shell syntax/help checks
and the existing agent unit suite pass. [Integration records](../preset-integration-20261003/checks.json).
There is no new model performance qualification. The private WRITE cutoff has
not been forwarded, so the combined tools-fast preset is still pending.

User correction: packaging known values is a small shell task, not an hour-long
experiment. Durable suffix checkpoints and pre-RoPE relocation measurements
receive separate budgets in the sequence.

Keep the two headline results separate: **24.65%** on the real agent's long
repeated-context workflow, and **23.90%** on two private code-helper tasks.
Neither is a measurement of the combined package.
