# Balanced exact suffix repair: fixed881

A quiet native–repair–repair–native run passed complete raw-vector, state-body and logical cache/IO controls. The existing instrumented content clock averaged **12.146% lower** for clipped dependency repair; whole-helper time averaged **5.195% lower**. Both native content clocks are above both repair clocks. This is one fixed workload, two repeats per path, with every arm using the same source bootstrap.

| First-attempt arm | Content clock | Whole helper | Content returned pread bytes |
| --- | ---: | ---: | ---: |
| Native before | 19.464450 s | 71.399343 s | 52,642,897,920 |
| Clipped repair 1 | 17.388412 s | 66.310624 s | 38,947,184,640 |
| Clipped repair 2 | 16.773738 s | 65.162073 s | 38,947,184,640 |
| Native after | 19.420699 s | 67.276962 s | 52,642,897,920 |

The independent ROOT audit checked all32 full-vocabulary FP32 rows and all20 serialized state payloads, exact histories and finite logits. The six required DSSD magic tags on the two repair arms were normalized; complete remaining state bodies matched. Each of four teacher-token ΔNLL values is0; KL/TV/argmax flips are0 on all five teacher/terminal vectors. Independent FP64 NLL recomputation differs from the native CSV by at most3.198e-14 nats, reported without fitting.

All four arms matched the seven logical counters at engine-ready, source reconstruction and pre-content. Both repeated native arms and both repeated repair arms also matched these counters at all five measured stages. The runtime does not expose a complete cache-table fingerprint. Physical process-read bytes may differ and were not an equality gate. Repair returns **26.016% fewer logical pread bytes** and computes8.899% fewer content-layer output rows, with832 lower dependency KV rows warmed.

The content clock includes its existing checkpoint, budget and stdout work. Whole-helper time also includes source reconstruction and serialization. Other collaborating CPU/source/read work was paused throughout the model run. ABBA reduces order bias; two repetitions on one fixture do not establish a general latency result or physical SSD throughput. No retry, tuning, changed binary, sampler, grader or default promotion occurred.

Geometry is unchanged: old source2988, edited prefix2096, suffix881, fresh continuation409, four common CODE3141 teachers, ctx8192/chunk256/cache64GiB, causal13x64+49 content publication. This repairs the necessary lower content and skips upper output rows with no remaining dependency. It is **not an exact stale-prefix suffix-cache graft**. Direct reuse and the separate CLM isolation policy have distinct quality evidence.

The model job completed in271.776s; ROOT released the MAIN lock with263 normal pressure samples,0 bad samples,0 measured swap growth and peak wired-plus-compressed80.299GiB. All100 MAIN runtime/shader authority files remained byte-identical. The complete original vectors/states remain in the persistent balanced workspace and are referenced by the independent audit.
