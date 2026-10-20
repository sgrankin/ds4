# Original pre-RoPE tails and repeated relocation

Item 8: 2026-10-03 23:43:58 to 2026-10-04 00:43:58 UTC, including implementation,
builds, runs and interpretation. Runtime revision
`52278b7b50111dde99cd888e55f288feacc9e5a3`; see `revision.json` for build details.

The opt-in Metal V4.1 path archives normalized BF16 rotary tails before RoPE and
cache quantization. Every relocation restores those original tails, applies
RoPE at the destination once and quantizes only the rotary lanes. Canonical
exact-prefix restore imports the original tape alongside its state. Private
approximate checkpoints can retain and supply originals for later moves; they
remain excluded from exact prefix lookup/export. Pair boundaries currently
require even left shifts. Unsupported coverage/geometry falls back to rebuilding.

## Eight moves from one immutable source

Q2 Flash, context 8192, prefill chunk 512, 64 GiB expert cache; other
approximations disabled. Source span 2048:4097 (2049 reused tokens), 128 fresh
old-context tokens, even shifts 2/66/128/256/514/768/1024/1280. An odd endpoint
exercises three unfinished compression pairs. Each native/reuse arm includes
its destination prefix evaluation and four teacher tokens. Source canonical
and original-tape hashes stayed identical across every arm.

| Measurement | Result |
| --- | ---: |
| Native / reuse total | 819.10 / 342.74 seconds |
| Edit latency saving | 58.16% |
| Mean delta-NLL | -0.00001536 nats/token |
| Mean / maximum KL | 0.00001351 / 0.00006633 |
| Greedy-token changes | 0 / 32 |
| Original tape, checkpoint / live ctx8192 | 3.15 / 5.67 MiB |

These likelihood numbers cover four fixed continuation tokens per move. They
measure this changed-register fixture, not a general task-accuracy guarantee.
Numerical relocation stability does not remove old-prefix conditioning.

Command (wrapped by the existing pressure guard):

```sh
speed-bench/experiments/pre-rope-hour8-20261003/suffix-repeat-bench \
  --model gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 8192 --chunk 512 \
  --cache-gb 64 --prerope 1 --moves 8 \
  --out speed-bench/experiments/pre-rope-hour8-20261003/core-prerope8
```

Exact argv/settings are retained in `guard-prerope8/command.json`; rows and
source identities are in `core-prerope8/rows.csv` and `moves.csv`. Pressure:
1142 samples, no bad samples, no swap growth; peak wired plus compressed RAM
85.60 GiB on the 128 GiB host. The existing pressure guard remained enabled.

## Ordinary agent edits

Six turns and three idle removals of old audit prose, preserving the complete
active inventory tool result. Each new question must freshly retrieve the
archived fact from an actual JSON file and calculate an independent expected
answer. The latest question stays fresh. Full process time includes initial
prefill, captures, edits, reconstruction, tools and generated answers.

Both arms use the frozen item8 binary. The copied system-prefix image included
a different live-context path: the initial native arm rebuilt 1987 system
tokens (56.10 seconds), while the suffix arm then hit the updated cache.
Its initial full-process saving therefore includes this warmup mismatch; a
matching warmed native control is being measured. The edited-turn timings
remain separately reported. The first native grading pass selected a tool-result User
marker rather than the query; the saved transcript proves all three fresh tool
reads. Its `result.json` was regraded after the one-line fix; original guard
status and stdout are preserved. No model execution or timing changed.

Commands are in each `execution.json` / guard `command.json`; useful transcripts
and traces remain in `agent-native/` and `agent-suffix/`. Results follow after
both arms complete. Existing agent suite and focused suffix mock passed.

| Agent arm | Full process seconds | Three edit-to-answer seconds | Correct answers + fresh reads |
| --- | ---: | ---: | --- |
| Native | 347.45 | 201.40 | 3/3 |
| Pre-RoPE | 146.88 | 58.32 | 3/3 |

The initial, warmup-confounded full process time fell **57.73%**; the three edited turns fell **71.04%**.
All three grafts activated, two from an already approximate live graph.
The selected original sources differ across edits; the separate core screen
proves eight moves from one unchanged original. This is one ordered pair on a
small retrieval/arithmetic fixture, with no reversed-order timing confidence
interval or generated-history KL claim. The first screen ended at 00:26:37 UTC; the corrected warmed control follows.

### Corrected warmed comparison

The follow-up native arm confirmed a 1987-token system-cache hit, matching the
suffix arm. It passed all three independently graded answers and fresh file
reads. Full process time: **293.93 s native versus 146.88 s reuse**,
**50.03% saving**. The three edited turns totalled 201.53 versus
58.32 seconds, **71.06% saving**. Use this pair for the full-process
comparison; the first cold-system arm remains visible above. This is a single
ordered fixture pair, not a general speed or quality guarantee.

Item8 complete at 00:37:34 UTC, within its original deadline.
