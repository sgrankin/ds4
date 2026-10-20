# Endpoint-aligned suffix follow-up CPU preparation

Private frontend and runner only, in a new child of frozen41c9e357. The original helper/runner, runtime core, auxiliary objects,26 shaders and first odd65 plan are unchanged. No model/tokenizer/GGUF metadata/engine/GPU work was performed.

Three separately guarded cases: zero identity P2017/delta0/C65/tail1/chunk512; inverse parity P1408/delta65/C65/tail1/chunk512; physical wrap P1951/delta65/C129/tail1/chunk65. Each uses ctx4096, cache64GiB, tape1024 and16 teacher tokens. Five arms per case: untouched native-before, experimental native-off, tape-native, suffix-reuse, untouched native-after. No automatic inter-case chain.

Both native source and edited-fresh paths stop at their suffix endpoints before the fresh tail. On zero shift, the frontend first checks source/fresh canonical bytes and initial logits exactly. Approximate zero reuse must match all17 continuation vectors and all diagnostic graph bytes after normalizing only the validated DSSD→DSV4 magic. The runner independently checks these retained raw/state files; small KL does not substitute for identity.

The128MiB unique app-owned staging contract is retained. The extra edited-endpoint token vector is charged; the new streamed state comparator uses32KiB stack storage within the existing64KiB overhead reserve. No third full snapshot or accumulated GPU-vector matrix was added. The base engine/cache/graph/backend metadata remain guard-monitored separately.

CPU validation: both helper arithmetic selftests, actual C state-comparator fixtures across32KiB boundaries, and14 Python driver tests passed. Fixtures cover all three geometry counts, false identity claims, raw extent/CSV failures, exact guard parent and bounded absolute ps, execute-mode drift despite unchanged hashes, and preservation of the first rc3 helper exit without another arm. No failed compile/test attempts occurred. Build logs/proof preserve the earlier12-test stage and final14-test stage.

The benchmark remains diagnostic: one related literary trajectory per geometry, approximate old-conditioned inputs, no task accuracy/forgetting/default promotion claim. Configured chunk65 changes native prefill as well as replay, and therefore gets its own full native controls.

All model jobs remain unrun and need explicit ROOT grants. Preparation uses copy2 and verifies probe executable permissions; the first-pilot V2 execute-bit failure is preserved independently by ROOT.
