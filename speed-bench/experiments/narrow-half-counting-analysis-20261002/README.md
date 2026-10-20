The completed frozen narrow-half λ0 / λ0.5 effort 75 pair contains eight structurally valid producer records. The accepted corrected wrapper `f27b7d5a` reports analysis integrity PASS (analysis exit 0), while the original producer/guard returned 3 because six records are INVALID. The original records, scores, cap flags and raw/token traces remain unchanged.

| Case | Gold | Native status / tokens | λ0.5 status / tokens | Common sampling prefix |
| --- | ---: | --- | --- | ---: |
| circular 10 | 123 | CORRECT / 537 | CORRECT / 406 | 224 |
| restricted_derangements 8 | 10860 | INVALID / 2048 | INVALID / 2048 | 273 |
| lattice 8 x 6 | 2002 | INVALID / 2048 | INVALID / 2048 | 389 |
| subset_mod 7 | 132 | INVALID / 2048 | INVALID / 2048 | 202 |

Each arm has one completed CORRECT answer out of four, zero valid incorrect answers, and three unclosed-reasoning INVALID records at the 2048-token cap. There are no capped-CORRECT records. The solved circular case returns 123 in both arms and uses 537 versus 406 generated tokens (−24.3948%). The other three cases each remain capped at 2048 tokens in both arms.

Totals are 6681→6550 generated tokens (−131, -1.960784%) and 6680→6549 reasoning tokens. The aggregate reduction comes entirely from the single solved case. Strict accuracy does not improve; the cap limits observable reasoning length on the other cases. This is four related counting cases with one fixed seed and does not identify quantization as the cause of overthinking.

The corrected analysis validates typed case/gold/prompt/options, contiguous token/RNG/stop identities, exact raw-byte reconstruction, generated-thinking activity bounds, final-offset/closure semantics and completed guard status. It preserves the unchanged strict-final-v2 producer scores and original aggregation/math. It additionally distinguishes completed_correct from capped_correct. No rescore, repaired answer or cross-effort KL comparison was performed. Native effort 25/50 remain unrun in this archive.

All original results, marker mapping, eight raw-text traces and eight complete token JSONL traces are copied byte for byte into `producer-output/` (under 4 MiB total). Their original persistent artifact paths, byte sizes and SHA identities are in `manifest.json`. Full guard command/result/pressure telemetry, ROOT release proof, frozen producer manifest, launcher/preflight/core proofs, frozen queue and freshly checked source/shader identities are retained. No GGUF metadata or payload was accessed during analysis or archival.

Guard pressure: 656 samples, all kernel level 1, no adverse samples or swap growth; swap stayed 0.017763672 GiB. Maximum simultaneous wired+compressor was 79.5915 GiB. ROOT session 25806/guard 31276 returned 3 and ROOT independently reacquired/released the MAIN lock with no owned helper remaining at 22:46:10 UTC before later separately granted runs.

Recorded decode totals are 319.117103 s native and 314.897155 s candidate. They include generation diagnostics and ordered/warm-cache reuse; they remain exploratory and support no profile-off speed claim. Detailed per-case token diagnostics/counterfactuals in `checked-summary.json` retain their original scope: sampled baseline thinking prefixes and ideal pre-top-p policy, without extrapolation to unobserved steps.
