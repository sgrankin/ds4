# Resident expert functional signatures: first complete native scan

The first native construction run scanned all **15,360 experts** (40 layers ×384 experts), with16 captured inputs per layer and16 signed projection features per output. It completed all **245,760 expert/input outputs** and wrote7,864,320 bytes of little-endian FP16 signatures. An independent ROOT audit checked all3,932,160 halfwords, exact completion counters and successful source/scanner/constructor/admission/GPU cleanup.

This gives the resident-routing experiment real expert-output data. It does not yet establish that a substitute preserves model quality. Map publication, native runtime integration and fixed quality evaluation are separate steps.

| Actual first attempt | Result |
| --- | ---: |
| Native guarded command | exit0,83.939s diagnostic wall |
| Expert pread calls | 46,080 |
| Logical requested and returned expert bytes | 152,882,380,800 |
| Checked pair/down/completion counts, each | 245,760 |
| Projected output rows | 245,760 |
| Known application page/stack charge peak | 31,735,808 bytes |
| Guard pressure samples | 81 normal,0 bad |
| Measured swap growth | 0 GiB |
| Peak wired plus compressed memory | 13.144 GiB |

The constructor uses bounded expert slabs and no normal inference engine, tokenizer or64GiB expert cache. Its64MiB allowance covers declared application page/stack charges, not globalRSS, OS or Metal allocations. Other small CPU qualification jobs overlapped this diagnostic run;83.939s is not a quiet inference benchmark or physical SSD throughput claim.

Descriptive output checks found zero nonfinite halfwords, zero projected rows with zero norm, zero complete duplicate expert signatures and1,408 subnormal halfwords. Projected norm squared spans0.00341449 to11115.17 across layers and probes. These observations did not select or tune a quality gate. Exact raw SHA: `e3d03efbf384ec16c6a54d36624efcbee37ee2912031297e32967c9fcf9efcad`.

Inputs came from independently audited immutable capture026. The metadata-only admission packet was separately published and independently byte-audited before this scan. No retry, map publication, policy evaluation or default promotion occurred. Two ROOT reporting failures were retained: an incorrect pressure-field name and an access-time-sensitive file identity comparison; corrected artifact checks ran after native ownership release without repeating the scan.

All100 MAIN runtime/shader authority files remained byte-identical. Full original raw signatures, pressure logs, completed native footer, build/admission provenance and retained reporting failures live in the persistent constructor workspace.
