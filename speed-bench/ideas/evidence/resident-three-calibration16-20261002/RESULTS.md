# One, two and three resident substitutions

All three native controls matched all512 full-vocabulary reference rows and full SHA1993. The two-substitution arm repeated all prior serialized quality fields and routing counters; it now has its own retained full raw export, without claiming retrospective byte proof for the older scalar-only run.

| Maximum substitutions | Delta-NLL nats/token | Mean KL | Argmax changes | Worst prompt delta-NLL | Maximum row KL | Substitutions / own original misses |
|---|---:|---:|---:|---:|---:|---:|
|1|+.0210325|.0126620|13|+.136065|1.11871|3521 /11777|
|2|+.0138603|.0126514|20|+.113244|.61608|6464 /12718|
|3|+.0059292|.0166944|18|+.090175|1.41472|8299 /13433|

Three substitutions improve mean target NLL and worst-prompt NLL on this calibration, while average/full-row maximum KL increase. Its prompt-bootstrap95% NLL interval spans−.015014 to+.026892. This mixed result justifies generated-task and clean-latency tests, without establishing superior general quality or promotion. The policy keeps three strongest originals protected and caps total original replaced mixture mass at.40; one/two retain their existing protections and.35 budget. All use gap.30, per-original mass.20 and rank48.

Miss populations differ with policy/cache history. Route counters report tentatively skipped original requests, not net demand-read reduction. Exact demand-byte I/O is unavailable from this frozen helper, so no physical SSD or net-I/O claim is made. Timings are instrumented and CPU preparation ran concurrently; no speed claim follows from them. Full raw outputs remain persistent with hashes in the compact summary. Pressure was normal throughout with no swap growth. The confirmation16 split remains untouched.
