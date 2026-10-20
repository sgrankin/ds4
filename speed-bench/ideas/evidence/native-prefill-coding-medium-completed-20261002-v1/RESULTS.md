# Medium-append generated coding checks: PASS

Native64, selected64 with tile64/limit1024, and native80 each passed all four unchanged coding tasks and all34 independent gold checks, without repair, extra attempts or cap hits. Native64 emitted574 tokens; selected64 and native80 each emitted569. All completed with EOS.

Each prompt begins with47 native warm tokens, then393–414 appended tokens at nonzero position. The selected64 arm warms under the same native64 defaults; its four full warm-logit vectors are byte-identical to native64. Appended native token IDs match across all arms. Eligibility was validated from actual native counts, prefix position, environment trace and the frozen source branch, not a profiler.

Selected64 and native80 match each other in all four initial post-append full-vocabulary vectors, code bytes and token-decision records excluding timing. Against native64, round_cents and split_escaped code/decisions match, while merge_intervals and topological_order differ and still pass every gold check. These are four bounded tasks; this supports task correctness on them, not broad quality or task throughput. Scorer worker RSS is sampled, with the retained Darwin hard-limit caveat; an observed peak of0 means the short worker exited before a positive sample.

All pressure samples were normal, with no swap growth. The guard completed0, the native helper is absent and MAIN model lock was reacquired/released. Raw initial/warm vectors remain in the persistent ignored artifact directory with manifest hashes. Timings are diagnostic and excluded from speed claims; the separate profile-off ABBA owns the20.23% replay turn-time result.
