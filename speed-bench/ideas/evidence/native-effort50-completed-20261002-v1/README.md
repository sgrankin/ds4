# Completed native effort-50 counting run

Producer and MAIN guard completed with **rc3**: one completed correct answer and
three INVALID answers capped at 2048 tokens with unclosed reasoning and no stop.
The accepted f27 checker validated all four records and token traces without
changing the producer's strict statuses. Total generated tokens are 6336, including
6335 reasoning tokens. The circular10 answer completed in 192 tokens; the other
three each used 2048. All four raw outputs, token JSONL, original records, exact
prompts/options, stop/cap/closure fields and diagnostic sampling records are kept.

| Native effort | Completed correct / 4 | INVALID caps | Tokens | Reasoning tokens |
| --- | --- | --- | --- | --- |
| 25 | 2 | 2 | 5985 | 5982 |
| 50 | 1 | 3 | 6336 | 6335 |
| 75 | 1 | 3 | 6681 | 6680 |

This is a descriptive comparison of the same four frozen problems at one seed.
The thinking prompt changes with effort. It supports no cross-effort KL, equivalent
logit, quantization-causality, generalization or speed claim. The 75 row uses only
the penalty-zero native arm of the completed narrow pair. The 25 archive is linked
in the comparison JSON rather than copied again.

Settings were native-only penalty zero, cache 64 GiB, max output 2048, seed
20261002, temperature 0.6/top-p 0.95/min-p 0, hesitation markers and stride 32.
ROOT session 34982 release time 1790983622.190748 follows guard rc3, with the lock reacquired
and no matching live model processes. All 301 pressure samples were normal;
there were zero adverse samples and zero swap growth.

The original source revision cee2d1426766064ec18740621c34faf006f9919a, actual
helper SHA f0596bf4fb3035871c4bd7b1393161b005377fca2bc0a6e28b868b852cd3f571,
queue launcher, full 98-file preflight identities, frozen queue, fixture, source
and existing-object proof are retained. Preflight hashes are preserved evidence;
the archiver reverified the helper, copied sources and archived files, not every
historical object. No current model stat, tokenizer/model open, model payload hash,
guard acquisition, build or helper execution was performed. No default is promoted.
