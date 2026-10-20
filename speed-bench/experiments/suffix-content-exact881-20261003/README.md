# Exact suffix content replay on the fixed881 fixture

The full-content replay and clipped dependency repair matched the native-before and native-after controls: all 32 full-vocabulary FP32 rows and all 20 serialized state payloads passed an independent byte audit. This includes source reconstruction, edited prefix2096, content frontier2977, prompt3388 and final3392 histories. Four common ordinary teacher IDs and the terminal vector remained exact. The clipped diagnostic state differs only in the required DSSD magic tag on three payloads; its conservative public export/import guard remains active.

| Content path | Layer output rows | Warm KV rows | Content-phase returned pread bytes | Content time, diagnostic |
| --- | ---: | ---: | ---: | ---: |
| Native before | 35,240 reference | 0 | 52,642,897,920 | 19.792 s |
| Full-content replay | 35,240 | 0 | 43,037,982,720 | 18.355 s |
| Clipped repair | 32,104 | 832 | 38,947,184,640 | 17.334 s |
| Native after | 35,240 reference | 0 | 141,575,454,720 | 20.878 s |

Clipping removes 8.899% of content-layer output rows. Relative to native-before, which shares the exact starting logical cache counters, it returns 26.016% fewer pread bytes during content replay. These are native logical IO counters, not a physical SSD throughput measurement. The native-after arm intentionally reconstructs its prefix from zero and has a different starting cache; its IO does not provide a matched performance bracket. Timing is diagnostic and other small CPU evidence work overlapped. A separately frozen balanced all-bootstrap repeat is required for a latency claim.

Independent FP64 likelihood recomputation agrees with the native CSV within 3.198e-14 nats. Complete raw vectors are byte-identical, so ΔNLL is 0 on each of the four teacher tokens; KL, TV and argmax flips are 0 on all five teacher/terminal vectors. This establishes numerical parity on this fixture, not general correctness across prompts or layouts.

This is exact reconstruction after an 11-token prefix edit, with 881 content tokens and 409 fresh tokens. It recomputes the necessary lower-layer content and skips upper-layer output rows whose dependency cone no longer reaches the final cache. It does not establish exact reuse of an old prefix-conditioned suffix across arbitrary edits. Earlier direct suffix reuse on this same fact fixture showed substantial likelihood drift; the separate CLM isolation prototype changes the serving policy to remove mutable-prefix dependencies.

The selected path used 13×64+49 causal scalar publication tiles. Native carry 354,287,616 bytes already existed; it is not incremental memory claimed by this experiment. The clipped known-owned buffers plus explicit reserves peaked at 42,295,861 bytes, below the 128 MiB admission bound. Existing model/cache, driver, global RSS and native carry are outside that bound. System pressure stayed normal through 428 samples, with peak wired-plus-compressed 84.688 GiB and measured swap growth 0.003789063 GiB.

All four first native calls returned 0; no retries, alternate geometry, threshold fitting, sampler, code grader or post-result repair was used. Runtime defaults and all 100 MAIN runtime/shader authority files remain unchanged. Complete original vectors/state files remain protected in the persistent materialization workspace and will be indexed in its result archive.
