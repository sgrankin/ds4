# Prepared v2 native-fast64 versus three-slot resident ABBA

Preparation only. No model launch has been granted for v2. All CPU agents must
explicitly pause before timing. MAIN runtime and guard sources are untouched.

Use only `speed-bench/ideas/artifacts/resident-three-fast64-abba-plan-20261002-v2`.
Manifest SHA256: `13adc6b7ce0202caa4f2d25bd5e275b889020ff6275bd5b8d1c427d54f77a317`.
Prepared-command SHA256: `fece863fbe1b73caccc2c012fd117cbd1809298848c4421dd7a02adf9840d4cc`.

V1 and its failure remain intact. Its MAIN pressure guard completed with child
return code1 when the exact parent argv check rejected macOS's framework Python
path, before model metadata, helper launch, or creation of a run directory.
The launcher shim and framework executable are different files. V2 separately
pins the observed framework interpreter path and SHA256, requires `samefile`
against that path, and checks its hash. Every argument after argv0 must match
the prepared guard command exactly. The guard source hash and held MAIN lock
are still required before model metadata/helper execution and before each arm.
Interpreter and other asset hashes are also checked after each arm.

All inherited environment names beginning with `DS4` are removed before the
fixed configuration and shader paths are installed. Binary, shaders, session,
prior controls, object proof and arm policies are byte-identical to v1.
Runtime `b9408c3b8e3ec5b22ad7efff3e365fd81818c317` remains opt-in. Both arms use cache64,
context8192, chunk512, TILE64/LIMIT1024, configured6136 slots. Native uses GAP0;
candidate uses gap.30/mass.20/total.40/rank48/MAX_SUBS3/apply/replace. Fresh-process
order remains ABBA once, without diagnostic/IO/route profiling.

Eight CPU tests passed: twelve toy ABBA trials, alias/suffix/hash/lock rejection
checks, and an actual tiny child under the unchanged MAIN pressure guard. That
child only checked parent identity and lock; it did not invoke the timing
wrapper run, replay helper, tokenizer, model metadata or GPU. The first suite
was blocked by sandbox access to a pressure sysctl and is retained separately;
the same suite passed with the CPU-only sandbox override. `cpu-proof.json`
records both logs and the preserved v1 failure. The previous model-free helper
arity and build evidence remain in v1; no helper build or execution was repeated.

Both native repeats must retain the prior tile64/limit1024 seven-vector SHA
`f91a39503f60233d633b08586229fb1b2e18958e60632efaec9d742ab977e652` and final-state SHA
`f71f06ea4d72e4329eb7c67a4ba462b284d21352756f8ec50022ec8c9276f2b0`.
Candidate repeat hashes must match each other. Raw vectors and state remain
retained, and each launch uses an exclusive fresh run/pressure directory.
Estimated time remains330–450 seconds. This fixed replay does not promote a
default or establish generated-task quality.
