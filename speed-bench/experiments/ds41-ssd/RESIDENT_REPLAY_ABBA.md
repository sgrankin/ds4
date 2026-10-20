# Approximate resident replay timing

The standalone `speed-bench/agent-session/resident_replay_abba.py` runner copies
one executable, fixed teacher-forced replay and Metal sources, then runs ABBA
or BAAB in separate processes. Arm A sets resident routing GAP to zero; arm B
defaults to GAP=.05, MAX_MASS=.12, MAX_SUBS=1 using the frozen v4 binary, whose
one-substitution behavior matches v3. Any DS4 parameter can be overridden with
`--candidate-env NAME=VALUE`; `--base-env` applies common parameters to both arms.

```sh
python3 /tmp/ds4-resident-run/speed-bench/agent-session/resident_replay_abba.py \
  /tmp/resident-replay-ABBA \
  --model /Volumes/Models/ds4/ds4flash.gguf \
  --replay /Volumes/Models/ds4/speed-bench/agent-session/session-v1.txt
```

The two-substitution exploratory arm overrides MAX_SUBS=2, MAX_MASS=.14 and
MAX_TOTAL_MASS=.24. Use a fresh output path for every trial. Cache defaults to
64 GiB and prefill chunk to 512; pass explicit values to reproduce a different
configuration. The runner performs no sampling or tool work. Phase timings
exclude startup, final state writes and hash I/O; process wall time includes
initialization and replay output writes.

`manifest.json` records executable/replay/shader SHA-256 hashes, exact DS4
environments and model path/size/mtime. It avoids reading the entire model to
hash it before a timing trial. `summary.json` records every phase, output logits
and snapshot hashes, cumulative cache/pending/timing profiles and resident-route
counters. Cache `miss_pread` values are reported in rounded GiB by the runtime
and represent requested read traffic, not proven physical device traffic.
`comparison.json` records group means and candidate percent changes, including
decode and appended-prefill-plus-decode time. Full logs and timings are retained;
large state/logit outputs are removed after hashing unless `--keep-state` is set.

Hashes are diagnostic: approximate arms may differ. The runner validates equal
teacher-forced phase boundaries but does not assert cross-arm state equality.
Quality remains a separate teacher-forced/held-out validation. It does not
convert faster replay into a task-success or quality-neutral speed claim.

CPU validation uses a fake subprocess replay to exercise ABBA and BAAB, accepts
different approximate state/logit hashes, checks inherited route-environment
isolation, parses cache/route profiles and verifies aggregate timing arithmetic.
The validation performs no GPU work.
