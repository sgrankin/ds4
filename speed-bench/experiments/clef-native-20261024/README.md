# Native Clef qualification

Native Zig/Metal text and media serving is implemented for Clef-Flash/full/Omni,
through both decision APIs in the existing server. Qualification includes2,000
Flash text requests,100 each for full/Omni, all three BLINK28 samples, controlled
end-to-end media checks and mixed-model serving. Actual independent DS41 engines
and the existing four model families pass the recorded ownership/regression
gates. The sections below retain each experiment's boundaries and limitations.

- [Setup, APIs and limits](../../../docs/CLEF.md)
- [GPU head: chosen default, numerical controls and timings](HEAD.md)
- [Vision tower comparisons](vision.md), [CPU media/audio](MEDIA.md),
  [actual media tasks](MEDIA-TASKS.md)
- [Mixed Flint/Flash serving](mixed-serving.md)
- [Omni sparse tiles and long-input row-bound repair](OMNI.md)
- [Broader text/BLINK input staging](STAGING.md)
- [Completed task scores and calibration](BROAD_QUALITY.md)
- [Actual engine ownership and lifetime](OWNERSHIP.md),
  [existing-model numerical regression](ENGINE-REGRESSION.md)
- [Cooperative scheduling tradeoff](SCHEDULING.md)

## Artifact and oracle checkpoint — 2026-10-25

Started from plan revision `6262ae0ee7ea`, with main at `ce3afb7999b7`.
This first checkpoint adds downloads, reference outputs and a small public
quality sample. It does not by itself qualify native serving or all modalities.

`nu download-model.nu clef-flash-int4`, `clef-int4` and `clef-omni-int4`
now download complete pinned model directories into `gguf/`. The aliases are
`clef-flash`, `clef` and `clef-omni`. Every file receives size/SHA-256 verification.
Per-model identity tables prevent repeated names such as `config.json` from
colliding with Flint or one another. The `ds4flash.gguf` symlink is unchanged.
The offline downloader suite passes **22 tests**, including all three bundles,
repeat downloads and a corrupted scoped metadata file.

| Checkpoint | Revision | Downloaded size (decimal GB) |
| --- | --- | ---: |
| [Flash INT4](https://huggingface.co/mlx-community/clef-flash-4bit) | `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1` | 6.194 |
| [Full INT4](https://huggingface.co/mlx-community/clef-4bit) | `b973b6f405d02f8e9b83208c7e82ddf43a742eb6` | 16.311 |
| [Omni INT4](https://huggingface.co/mlx-community/clef-omni-4bit) | `040a6e49d2e5d2f4f3ba68e70a6a0654833e23e1` | 19.773 |

All three were downloaded and verified locally. These are disk sizes, not peak
runtime memory. Backbones are affine INT4; joint heads retain their published
BF16 weights. Media towers and Omni router precision follow each checkpoint.

## Reference numerical and warm timing baseline

The offline [oracle](oracle.py) uses the checkpoint's pinned `clef_mlx.py`,
MLX 0.32.3, mlx-lm 0.32.0 and a **FP32 joint head**. This head dtype matches the
initial native CPU implementation; it differs from the reference CLI's BF16
head default. Production serving does not import Python or MLX.

Hardware: M5 Max, 128GB. Run under the existing pressure guard, after the other
model job released the GPU. Four fresh independent records; eight timed warm
forwards per record. Numbers include backbone plus head, and exclude encoding,
loading and HTTP. No prompt/cache reuse was requested.

| Record | Full input tokens | Questions | Reference median |
| --- | ---: | ---: | ---: |
| Duplicate payment routing | 169 | 1 | 65.23 ms |
| Structured incident, predicate + severity | 266 | 2 | 99.69 ms |
| French proposition | 157 | 1 | 65.94 ms |
| Structured state, fire predicate + action | 238 | 2 | 85.54 ms |

The expected decisions are billing; urgent=true/severity2; planet=true; and
fire=false/ordinary monitoring. All agree. This is a tiny numerical/smoke corpus,
not a broad accuracy benchmark. The guard reports normal pressure throughout
and zero swap growth; child exit 0.

Reproduce after installing the reference-only environment:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/clef-reference/guard -- .workspaces/clef-artifacts-20261024/venv/bin/python speed-bench/experiments/clef-native-20261024/oracle.py gguf/clef-flash-int4 speed-bench/experiments/clef-native-20261024/requests.jsonl .workspaces/clef-reference/output --repeats 8
```

Raw output remains in `.workspaces/clef-artifacts-20261024/reference/`: each case
has its record, exact IDs/spans, full FP32 final hidden rows, dequantized lexical
rows and joint-head logits/probabilities. An additional raw tokens 1..16 forward
provides a small backbone-only oracle. `tests/fixtures/clef-record.json` and
`clef-omni-record.json` contain ten tokenizer/span cases each, generated from
AST-extracted official record functions and each model's local tokenizer.
Official encoder revisions are Flash `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`
and Omni `df2d7c9505e62ba8a527b147513a4fa66a77dc5d`. This avoids importing
PyTorch merely to prepare text records. Fixtures cover normalization, control
spellings, structured values, option ordering and all three decision primitives.

## Decision Index smoke sample

Use [Decision Index](https://github.com/apolinario/decision-index) revision
`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` and its HTTP engine/scorers.
The prepared sample has **100 requests: 25 each** from API-Bank, BANKING77,
CLINC150+OOS and When2Call. It uses the kit's case-preserving stratified sampler,
seed 20260919 and pinned exclusions under edition 0.2.1. Rows, source downloads
and labels remain local under `.workspaces/clef-artifacts-20261024/decision-index/`.
Do not publish these source datasets or compare a four-task sample index with
an official full-suite number.

Rebuild with the upstream CLI:

```nu
let kit = '.workspaces/clef-references/decision-index'
$env.PYTHONPATH = $kit
let python = '.workspaces/clef-artifacts-20261024/venv/bin/python'
^$python -m decision_index suite rebuild --edition 0.2.1 --work .workspaces/clef-artifacts-20261024/decision-index --only 3 4 5 62
```

The kit's partial rebuild deliberately cannot pass the full frozen-suite hash.
Its base rows are retained in `release-v1-rebuilt/selected-rows.jsonl.gz`; added
When2Call rows are in `release-v2-rebuilt/added-rows.jsonl.gz`. Stage those into
a local partial Suite, copy `hub/excluded-questions.json`, and call
`decision_index suite sample --edition 0.2.1 --dir PARTIAL --n 100 --seed 20260919`.
The sample's base and added rows are also staged as `smoke-suite` for native scoring.
CRUXEval was omitted from this first sample: its grouped normalizer unexpectedly
requires GSM8K data too. That recorded failed rebuild did not justify changing
the evaluator or broadening setup before a useful four-task run.

Once a native server is running, use:

```nu
nu speed-bench/experiments/clef-native-20261024/evaluate.nu http://127.0.0.1:19285 clef-flash .workspaces/clef-native-quality/flash
```

The driver invokes the upstream runner and task scorers. Inspect per-task
accuracy/F1, coverage/errors and latency in `benchmark-summary.json`; no
full-index comparison is meaningful for this subset.
Broader edition 0.2.1 and current public 0.3 evaluations remain later checkpoints.

## First native Flash HTTP checkpoint

The native candidate serves both decision protocols at
`http://127.0.0.1:19285`, launched as:

```nu
.workspaces/clef-native-20261024/build/bin/ds4-server --model gguf/clef-flash-int4 --port 19285 --ctx 16384
```

The pressure-guarded server binary SHA-256 was
`f4f05a4f5dcd7346497d2fe8050b2f4a720b50afd07fdf1c65f7f550f3f66efe`.
This is the initial candidate in the working tree; subsequent commits may contain
further optimization. The server stayed loaded while the following client jobs
ran sequentially; no other model job/build ran during the final warm screen.

The 100-request smoke completes **100/100 HTTP requests**, without errors or
unsupported cases. This is completion coverage, not 100% answer accuracy.
Upstream per-task scorers give:

| Task | Cases | Metric | Native Flash score | Median HTTP latency |
| --- | ---: | --- | ---: | ---: |
| API-Bank | 25 | Accuracy | 88.0% | 2921 ms |
| BANKING77 | 25 | Macro-F1 | 90.91% | 726 ms |
| CLINC150+OOS | 25 | Macro-F1 | 80.88% | 1403 ms |
| When2Call | 25 | Accuracy | 64.0% | 288 ms |

Overall HTTP p50 is 1,032 ms, p95 is 3,008 ms; the run took 135.5 seconds. These are
unequal workloads: the first API-Bank request has 6,475 actual input tokens and
53 options, whereas the short reference corpus has 157–266 tokens. This does
not establish an across-model or across-hosted-provider latency ranking.
Small per-task sample scores are not the published full-suite metrics.

All 100 fields are categorical. Native unrounded distributions yield mean
NLL 0.56563 nats, multiclass Brier 0.22677 (sum of squared option errors per field)
and 10-bin equal-width ECE 0.12107. These are absolute smoke-set measurements,
not delta-NLL or well-powered calibration conclusions. The scorer uses raw
maximum option probability, independently of the wire confidence transformation.

The upstream full-index aggregator crashes when tasks are missing after writing
a valid task summary. [score_subset.py](score_subset.py) therefore invokes the
same upstream task scorers directly and adds these elementary distribution
statistics; it deliberately computes no headline index. No evaluator or source
dataset was patched. The maintained [driver](evaluate.nu) uses this task-only
path. The scorer can re-read saved responses without more server work; the
runner supports resuming an incomplete run.

Official OpenAI Python SDK **3.37.0** successfully calls local
`client.decisions.create`: predicates, duplicate and absent names, typed bool
versus string choice values, usage and parsed response objects pass. The client
uses a local-only base URL and dummy local key; no OpenAI service was contacted.
The broader Nu HTTP fixture also covers discovery, both APIs, practical answers,
ordered/typed values and explicit model/type/context errors.

Finally, [warm.nu](warm.nu) sends one untimed then eight timed fresh HTTP
requests per short record against the loaded server:

| Tokens | Reference backbone+head | Native HTTP wall | Native / reference difference |
| ---: | ---: | ---: | ---: |
| 169 | 65.23 ms | 77.88 ms | +19.4% |
| 266 | 99.69 ms | 126.28 ms | +26.7% |
| 157 | 65.94 ms | 78.23 ms | +18.6% |
| 238 | 85.54 ms | 107.92 ms | +26.2% |

These are intentionally different timing boundaries: native includes record
preparation and HTTP; the reference excludes both. Profile matched engine-only
components before attributing the difference to kernels. All six top answers
match the reference; the largest absolute probability difference in these HTTP
responses is 0.000583. No speed advantage is claimed from this first screen.

```nu
nu speed-bench/experiments/clef-native-20261024/warm.nu http://127.0.0.1:19285 speed-bench/experiments/clef-native-20261024/requests.jsonl .workspaces/clef-native-quality/warm.json --repeats 8
```

Raw native responses, SDK output and warm timings stay under
`.workspaces/clef-artifacts-20261024/`; server/guard logs stay under
`.workspaces/clef-native-20261024/http-first/`. This checkpoint qualifies Flash
text only. Full/Omni, media, co-resident engines, caching and throughput tuning
remain separate work.

## Same-weight quality and stage comparison

The same 100 frozen requests were then run through the offline MLX adapter
[mlx_reference.py](mlx_reference.py), using the existing Decision Index runner,
identical INT4 weights and a FP32 head. Both runs have identical input-token
counts and payload hashes. **All 100 top choices match**, hence all four task
scores above are identical between native and reference execution.

- Mean KL(reference || native): **0.00008574**.
- Mean delta-NLL: **+0.0001580 nats per decision** (native 0.5656261,
  reference 0.5654681).
- Largest individual option-probability difference: **0.02219**.

This isolates runtime drift for the chosen quantization. It does not measure
INT4 versus unquantized-model loss. The reference run took 106.5 seconds, versus
135.5 seconds for native HTTP; reference timing includes encoding and inference,
but omits HTTP. All four guards report normal pressure and no swap growth.

A separate synchronized stage screen retains the original combined warm loop
and adds four backbone/head timing passes per case:

| Flash tokens | MLX backbone | MLX FP32 head | MLX combined |
| ---: | ---: | ---: | ---: |
| 169 | 63.20 ms | 2.46 ms | 65.37 ms |
| 266 | 96.91 ms | 2.93 ms | 99.91 ms |
| 157 | 63.90 ms | 2.13 ms | 65.77 ms |
| 238 | 83.07 ms | 2.84 ms | 85.72 ms |

The reference head is cheap on GPU. The initial native CPU head profile takes
12–20 ms; the subsequent [GPU head experiment](HEAD.md) is completed and adopted.
Component measurements use extra synchronizations and need
not sum exactly to the separate combined-loop medians.

Full Clef also has complete four-record and raw 16-token oracles. Its reference
combined medians are 224/331/232/292 ms. The initial native four-record outputs
have all six top answers equal to reference and max probability gap 0.0001224.
Omni reference combined medians are 70.46/86.21/70.61/81.61 ms at
166/262/161/234 tokens, with all six expected decisions correct. Omni uses
MLX-VLM 0.7.6, the checkpoint's tested patch version. The raw 16-token oracle and
full hidden/head tensors were saved for native comparison. That initial reference
checkpoint alone did not qualify native Omni; subsequent numerical/task/kernel
qualification is complete in [OMNI.md](OMNI.md).

Raw paired results and comparison statistics are in
`.workspaces/clef-artifacts-20261024/quality-flash-100-reference/` and
`quality-flash-100-comparison.json`; split/full/Omni oracles are in
`reference-stages/`, `full-reference/` and `omni-reference/` under the same root.
The first warm driver saved timings plus the final response per case. It did
not save every intermediate response, so that screen cannot establish repeated
output invariance. The driver now saves each repeat's answers for the next
qualified screen.
