# Independent item25 qualification audit

The narrow visible-only change works on this four-session file workflow. Each
arm passed all eight steps and performed all eight required fresh file reads.
The fixed arm retained four approximate post-READ checkpoints instead of
performing the two additional B/D grafts seen in both old controls. This audit
read the completed small results, group/server logs, execution records, guards
and source; it ran no model, compiler, grader or test.

## Source and binding

`slot_probe_reuse_locked` keeps its preceding approximate-state profile gate:
the request must have the same nonempty context-cache profile, which includes
the client `prompt_cache_key`. A valid checkpoint, matching live token count,
image identity and the **entire remembered visible transcript** remain required.
Only the Responses-visible branch now permits
`!canonical_suffix || ds4_session_is_approximate(slot->session)`.

The shared `canonical_suffix` value remains unchanged. Canonical full-input
requests still bypass call-ID-only, memory-text and legacy disk-text splices;
changing an earlier public message therefore cannot be hidden by a matching
call ID. Exact token reuse remains available. Native source warming, default
canonical0 behavior and previous-response/conversation/incremental handling
retain their existing conditions. The earlier broad-condition hazard and
corrected source are recorded in [source-audit.md](source-audit.md).

`build_live_prompt_suffix` retains the actual approximate checkpoint IDs and
appends only the new visible suffix. It preserves the deliberately inserted
newline without adding that hidden pad to the public history. This is continued
approximate conditioning, not restoration of a native-exact graph.

## Observed mechanism and tasks

All three arms have four initial SSD grafts with `reused=1034`, `fresh=95`,
`graft_fresh=31`, `source_approximate=0`. A/C use unpadded plans; B/D use
`prefix_pad=1`. Each old arm then regrafts B/D at the tool-result-to-final step:
`reused=1034`, `fresh=214`, `graft_fresh=150`, still from native SSD sources.
Thus each old control has six grafts.

The fixed arm has four grafts total and four subsequent
`RESPPROTO match=visible-prefix ... cached=1530 ... approximate=1` records.
These are A/B before restart and C/D after restart; all keep the full live
post-READ checkpoint. Their final requests stage from token1530 rather than
1034. Source endpoints remain immutable; this change affects continuation
selection, not the graft algorithm or source contents.

The four independent edited gold pairs are A `(313, BEACON-46)`,
B `(231, ORBIT-57)`, C `(267, CEDAR-70)` and D `(197, DELTA-83)`.
Totals are respectively `23*14-9`, `31*8-17`, `19*15-18`, `27*8-19`.
Each client actually read its own advertised `active.json` during warmup and
its own `archive.json` when edited. C/D read newly changed archive codes after
server restart. All four warm answers are exactly READY. All16 HTTP responses
per arm are completed with no incomplete/cap records. The old before/after
brackets have byte-identical final texts and identical tool arguments/file
contents for all eight steps; the fixed arm may differ in JSON whitespace.

## Timing and pressure

Concurrent request latencies are not summed: the edited measurement is the
sum of the two joined group makespans. Launcher wall includes cold source
warmup, startup, shutdown and restart.

| Arm | Warm groups, s | Edited groups, s | Whole launcher, s | Steps / reads | Grafts / approximate visible continuations |
| --- | ---: | ---: | ---: | --- | --- |
| Old before |181.293259|147.420960|334.893003|8/8 / 8|6 / 0|
| Fixed |186.559008|102.474748|295.812091|8/8 / 8|4 / 4|
| Old after |186.357652|147.061732|340.360119|8/8 / 8|6 / 0|

Against the old bracket mean, edited makespan falls30.404% and whole launcher
wall12.385%. Against the faster old control these are30.319% and11.670%.
This is one old/fixed/old sequence on an existing workload family. It is not
a generated-history paired-NLL measurement or a general latency guarantee.

All guards completed with child0: before323, fixed286, after329 normal samples;
every sample has level1, no reason and zero consecutive bad samples. Swap is
flat at0.012998GiB in all three arms, so growth is zero. Peak wired+compressor
is84.360992/84.357559/85.141815GiB respectively.

## Execution and log attribution

All executions use the same Q2 model path, ctx8192, chunk512, two slots,
expert cache64GB, tile64/limit1024, GAP0, fresh64, canonical1/pad1,
hot0/SSD4096 and the same advertised `concurrent-visible/{A-D}` paths. Each
arm has a separate cold cache/store and restarts against its own persisted
cache. See [build.json](build.json), each arm's `*-execution.json`,
`http/{before,after}/result.json`, group logs and `guard-*/{result,pressure}`.
Both ordinary server binary hashes freshly match `build.json`: old saved
`dd601278…` (revision00da420b), fixed `3164ce82…` (revision91c9cf21).
The GGUF was neither opened nor statted by this audit.

Empty per-step `suffix_log_lines` are intentional, not a parser failure.
The unchanged concurrent driver passes `server_log=None` to each individual
turn, then reads the server log once after joining the pair. Its group arrays
contain the actual graft lines and it explicitly reports “Concurrent logs
attributed to groups only.” Count group or full-server logs once; no driver
fix is needed.
