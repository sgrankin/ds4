# Item46: incremental tool-schema announcements

2026-10-20; ABBA completed 22:16:52 UTC. Source base 2639960c; same changed server
binary in all four arms, flag order 0/1/1/0. All four arms and eight case workflows
passed on first outputs; all servers exited 0. Full settings and binary SHA256
are in [executions](evidence/abba/02-late/execution.json) and [summary](summary.json).

| Case | Native workflow mean | Flag 1 mean | Workflow saving | R2+R3 saving | Native workflow bracket drift |
|---|---:|---:|---:|---:|---:|
| Ordinary full-input |29.470 s|31.878 s|−8.17%|−20.48%|+1.78%|
| Favorable previous-response history |178.838 s|96.045 s|46.30%|66.34%|−0.094%|

Workflow includes three HTTP requests and the actual client file read; server
startup is excluded. R2+R3 excludes discovery and local tool time. Against the
faster native control, favorable workflow saving is 46.27% (R2+R3: 66.27%);
ordinary workflow regresses 9.13%. Raw arm times and the exact formula are in
[summary.json](summary.json): `saving = 1 - candidate_mean / native_mean`.

The ordinary case already hits native tool-output-ID continuation. The flag
adds 73 input tokens and costs time. Favorable begins with 4,487 tokens and
returns 13 schemas in namespace `archive__`; parent replay forces comparison of
the rendered prefix. Baseline rewrites its earlier schema preamble and reuses
zero tokens on R2/R3. Incremental announcements preserve 4,541 and 6,961 cached
tokens while appending schemas at discovery time. This explains the measured
benefit; it is not a GPU-kernel, resident-routing or suffix-cache speedup.

## Retained evidence

* [ABBA result](evidence/abba/result.json): first answers, actual-file-read proof,
  hashes, namespace/path grades, per-request usage and timing. Each arm directory
  retains exact execution, result and server stderr; each case retains all three
  request/response/wire JSONs. No retry or best-output selection.
* [Synthetic fixtures](evidence/abba/fixtures/favorable/fixture.json): ordinary
  and favorable note contents are retained beside their fixture definitions.
* [ABBA pressure summary](evidence/abba-guard/pressure-summary.json) and
  [guard result](evidence/abba-guard/result.json).
* [Baseline compatibility first outputs](evidence/baseline-qualification/result.json):
  separate pre-change ordinary qualification, not part of the paired ABBA.
* [Build logs](evidence/checks/build.log), [follow-up build](evidence/checks/build-2.log)
  and [server checks](evidence/checks/server-tests.log). The initial source build
  history is retained; no binaries, model payloads or production context are copied.

## Scope and replication

Read the [portable recipe](../../../recipes/11-incremental-tool-schemas.md) for
algorithm, opt-in launch and reproduction command. The original driver is
[run.py](run.py). Both cases used temperature 0/no thinking, native routing,
suffix reuse off and application SSD KV cache off. Ordinary precedes favorable
in each server, so shared prefix state may be warm; usage is retained rather
than assuming cold starts. These are one fixture per case and a single ABBA,
not whole-agent or general quality evidence. Native-versus-incremental rendering
can change output probabilities; equal graded answers do not prove equality.

Credit the [official DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)
for the incremental schema direction. Feature remains opt-in:
`DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS=1`.

Post-measure review fixed current-request metadata precedence over historical
same-name tool definitions. The fix only reparses metadata; it changes no rendered
prompt in the measured unique-name fixtures. A focused regression and the existing
server suite passed after rebuilding ([build](evidence/checks/build-3.log),
[checks](evidence/checks/server-tests-3.log)); ABBA timings remain from the recorded
pre-fix binary, with no benchmark retry.
