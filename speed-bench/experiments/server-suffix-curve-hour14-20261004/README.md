# Original-backed suffix tails: item 14

Budget14:07–14:57 UTC,2026-10-04, including10 minutes of prior preparation.
The 64-token minimum is the fastest tested practical point. Keep128 available;
this small screen supplies no evidence that256 buys useful quality. Runtime
behavior remains opt-in. The unchanged server binary is revision3fb450f7;
the core caller/build record and serial loop were committed at37253201.

## Actual file/tool HTTP workflow

Four serial arms used the unchanged item12 workflow, in order64,256,128,native
(the earlier item12 comparison ran native first). Two sessions read actual
archive files, delete completed logs, turn over a two-slot server, restart with
disk-only checkpoints, then read changed archive facts. Every arm passed7/7
steps and fulfilled all6 requested file reads. Three of four edited turns grafted;
the second B edit again fell back because no aligned source was eligible.

| Minimum fresh tail | Ordinary edits, seconds | Changed files after restart | All four edited turns | Saving across edited turns | All7 tasks including cold sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native |106.386|85.286|191.671|—|298.581|
|64|77.036|72.261|149.297|22.11%|257.780|
|128|80.253|71.980|152.233|20.58%|260.435|
|256|82.860|78.444|161.304|15.84%|269.666|

At64, ordinary edits saved27.59%, changed-file edits15.27%, and all tasks
including cold source preparation13.66%. The same A first edit still evaluated
784 fresh tokens at every setting: endpoint availability and the complete fresh
query can outweigh the configured minimum. B's first edit used an additional
eligible endpoint at64, reducing actual fresh tokens to99 versus227 at128.
All grafts loaded from the disk tier; A's second graft loaded an approximate
source after restart. OS page caching is not excluded by a disk-tier load.

These are whole HTTP step times including file handling and context edits.
Server startup/restart and launcher wall are recorded separately. This is one
timing order on simple arithmetic/retrieval tasks, not a balanced general
benchmark. Generated histories differ, so HTTP times have no paired NLL/KL
claim. [Raw task analysis](http-analysis/summary.json),
[CSV](http-analysis/summary.csv), [plot](http-analysis/http-latency.png), and
[HTTP detail](http-results.md) preserve actual reuse, fallback and fresh counts.

## Fixed-context core curve

Three families—ordinary deletion, self-contained favorable suffix, changed
fact—each compare identical final tokens at tails64/128/256 with a freshly
computed reference. Geometry is prefix1024, source edit256→128, unchanged
suffix512, even left shift128. Pre-RoPE BF16 originals are retained. Source
capture happens before generating an answer; the entire incoming query stays
fresh. Every fresh/reuse greedy answer was3141 and stopped normally.

Across the three families,64 saved69.6–70.0% of diagnostic core edit time;
128 saved60.2–60.7%;256 saved41.3–41.5%. Mean paired delta-NLL ranged from
−0.0000309 to+0.0000155 nats/token; mean KL(fresh||reuse) from0.00000589
to0.0000211. Full-vocabulary raw logits were recomputed by the analyzer.
Each continuation is only three tokens including EOS; the corrected fact is
explicitly visible in the freshly evaluated prefix even in the stress family.
These tiny lookup continuations cannot establish broad reasoning quality.

Core edit timing excludes creating/capturing the source (36.4–40.5 seconds),
model startup and scoring/generation. It must not be substituted for the HTTP
workflow savings. [Paired CSV](results.csv), [analysis](analysis.json),
[latency/NLL curve](pareto.png), [KL curve](kl.png), and
[core detail](core-results.md) retain the measured points, including small
non-monotonic likelihood changes rather than treating them as rejection gates.

## Reproduction and pressure

Use the existing `speed-bench/ideas/pressure_run.py` to run `run.py` or
`http_sweep.py`; exact argv/settings are in each guard's `command.json`, each
point's `execution.json`, and `build.json`. Both use64GB expert cache, native
routing weights, resident substitution off, and current Metal sources. HTTP
uses8192 context, two slots,512 prefill chunk, hot state cache0 and SSD1024MiB.
Core uses4096 context. HTTP sampling is temperature0/effortnone; a sent seed
has no effect in this Responses implementation.

Core621 and HTTP1069 pressure samples were all normal(level1), with zero swap
growth. Peak wired+compressor was83.42/85.08GiB respectively. Both guarded
jobs completed with exit0. No unrelated model job ran during measurement.
