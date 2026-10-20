# Completed native effort-25 counting run

The producer and MAIN guard completed with **rc3**: two completed correct answers
and two INVALID capped answers. The latter reached 2048 tokens with reasoning
unclosed and no stop token. Integrity checks passed; this is not an all-correct
run. All four full raw outputs, token JSONL traces, original typed records and
generation options are preserved here without scoring changes.

The run used the first four frozen counting problems, seed 20261002, temperature
0.6/top-p 0.95/min-p 0, cache 64 GiB, hesitation markers, diagnostic stride 32,
native-only penalty zero, and effort 25. It generated 5985 tokens, including 5982
reasoning tokens. ROOT session 39064 released ownership after 294 normal pressure
samples, zero adverse samples and zero swap growth. Its release proof records
the reacquired lock and no matching live model processes.

The comparison uses only the penalty-zero native arm of the previously completed
effort-75 narrow pair: 6681 tokens and one completed correct answer. Native 25
used 10.4176% fewer tokens and completed two correct answers on these four problems.
The effort prefix changes the thinking prompt, so this single-seed comparison
does not establish causality, generalization, cross-effort KL or equivalent raw
logits. Diagnostic CPU work and warm cache reuse prevent a clean speed claim.
Native 50 was still running and no partial outcomes were read for this archive.

CLM's accepted f27 checked analysis and frozen native25 summary were reused;
analysis was not rerun. Producer source, guard, queue launcher, fixture, generation
queue, manifest, existing-object proof and all 98 preflight identity records are
retained. The helper's actual SHA is f0596bf4fb3035871c4bd7b1393161b005377fca2bc0a6e28b868b852cd3f571.
Model identity is copied from the original run manifest; no current model stat,
tokenizer/model open, model payload hash, build or helper execution was performed
by the archiver. Original outputs remain untouched. No default is promoted.
