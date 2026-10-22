# Native V4.1 canonical payloads

Item 59 began at 13:32 UTC on 2026-10-22 with a one-hour budget ending at
14:32 UTC. All eight ReleaseFast arms and the additional native Debug F16 check
passed. The change moves the numerical DSV4 v2 payload owner into native Zig,
using the actual graph and validated configuration. Public session policy, approximate
state exclusions, HTTP/agent envelopes and private replay wrappers remain with
their callers. No disk format change is proposed by this fixture.

## Reuse and the missing boundary

Existing `tests/test_deepseek41_graph.c --session-fixture` covers positions
1, 2, 3, 127, 128 and 129, in-memory snapshots, nonzero cache-span roundtrips and
a fresh teacher token after restore. `--long-sessions` extends that to
511/512/513, 1023/1024/1025 and 16383/16384/16385, with a fixed 64 GiB cache and
a long prompt. Those are useful existing checks, but neither exchanges actual
canonical files between two binaries.

The new [public-API helper](payload_roundtrip.c) keeps the short frontier set
and adds that exchange. It includes only `ds4.h`, links the actual selected
production objects, and uses one session at context 256, chunk 256, MTP width 1
and a 32 GB expert-cache target. It has no private Session/graph layout, copied
serializer, model-text parser or GPU fault stub.

At each frontier the helper:

1. Independently computes the token prefix and saves canonical DSV4 v2 bytes.
2. Evaluates common fresh teacher token `987`, checks all full-vocabulary logits
   are finite, and saves them as the computed continuation reference.
3. If given a peer directory, compares the peer's canonical bytes and computed
   continuation against its own independent values. Same-storage exchange is
   strict; cross-storage differences are recorded separately.
4. Loads the peer (or its own) payload, resaves it, then evaluates `987` again
   and compares freshly computed full logits. Same-storage bytes/logits must
   match exactly. Cross-storage conversion records equality and maximum
   absolute logit drift without treating intentional rounding as failure. Loaded saved logits
   alone are not continuation evidence.

The odd frontiers retain pending paired compression; 127/128/129 cover the raw
ring boundary. F32 and F16 use the same canonical F32 wire format, while
conversion into F16 runtime storage may round values. Wire compatibility and
cross-storage numerical equivalence are measured separately. Raw payloads
and logit arrays are useful local outputs and should stay under `.workspaces/`,
rather than being copied into the publication evidence.

## Three concrete failure checks

At the final frontier only, the helper checks:

- A bad magic byte rejects the payload while preserving the entire previous
  canonical payload, position and checkpoint-valid flag.
- An out-of-vocabulary first token does the same, before any graph mutation.
- A normal file declares the full logical payload length but physically ends
  just after logits. Header/token validation succeeds, then the cache read
  fails after restore has begun mutating logits. Public checkpoint validity and
  payload size must become invalid/zero; canonical save must reject it. A good
  reload followed by fresh computation must recover exactly.

These are bounded admission/ownership checks, not an exhaustive malformed-file
suite. Diagnostics need to identify a failure; their exact platform wording is
not compared.

## Serial exchange

[qualify.py](qualify.py) first runs two strict same-storage cycles using a
frozen C-payload control and selected native-payload candidate, with the rest of
the engine held fixed: C/F32 → native/F32 → C/F32, then C/F16 → native/F16 →
C/F16. Every exported canonical file and freshly computed continuation must
match that storage mode's independent control at its frontier.

Two additional native imports measure C/F32 → native/F16 and C/F16 → native/F32
conversion. Their execution, finite outputs and failure/recovery invariants
must succeed. Wire equality, own/source fresh-logit equality and maximum
absolute drift are recorded; legitimate conversion differences do not fail the
canonical compatibility campaign. The helper uses production conversion code
and does not implement another half converter. `--source-f16` identifies the
source layout; it defaults to the destination layout.

This is two bounded exchange loops and two conversion observations, not a
matrix of unrelated malformed inputs. The helper binary is `payload-roundtrip`;
root owns actual build integration and guarded model execution.

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-payload-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-engine-payload-20261022/qualify.py \
    --control-bin-dir .workspaces/zig-engine-20261022/execution-native/bin \
    --native-bin-dir .workspaces/zig-engine-20261022/payload-native/bin
```

The driver defaults to fresh `.workspaces/zig-engine-payload-rerun`, records
commands, binary hashes, JSONL results and artifact hashes, and fails visibly
if outputs already exist. This is a canonical-state compatibility experiment,
not a latency or broad task-quality benchmark. The completed results follow below.


## Results

The [frozen control helper build](evidence/build/payload-control-build.json) took
2.356 s. The first native build passed in 10.516 s with one unused C-reference
helper warning; a narrow unused attribute retained that fixture helper. The
[clean-warning native rebuild](evidence/build/payload-native-build2.json) passed
in 16.172 s, and the [Debug build](evidence/build/payload-debug-build.json) passed
in 2.857 s. These cached development builds are not clean-build benchmarks.
The control uses the previous execution-stage native engine with its C payload
implementation; the candidate adds the native payload owner. Commands and binary
hashes are preserved with each run.

All [eight ReleaseFast arms](evidence/model/results.json) passed:

| Check | Arms / frontiers | Outcome |
| --- | --- | --- |
| F32 C → native → C | 3 / 18 | Canonical files and freshly computed full logits identical |
| F16 C → native → C | 3 / 18 | Canonical files and freshly computed full logits identical |
| C F32 → native F16 | 1 / 6 | Wire, resave and fresh logits identical; maximum absolute drift 0 |
| C F16 → native F32 | 1 / 6 | Wire, resave and fresh logits identical; maximum absolute drift 0 |
| Native Debug F16 import | 1 / 6 | Same checks passed against the F16 control |

Each fresh continuation evaluated token `987` through the restored cache and
compared all 129,280 logits. The cross-storage results are observations for this
fixture, not a proof that arbitrary F32 cache values convert losslessly to F16.
Canonical payload sizes were 613,944; 693,820; 790,592; 11,742,768; 11,822,644;
and 11,837,496 bytes at positions 1, 2, 3, 127, 128 and 129, respectively, in both
storage modes. Raw payloads and full logit arrays remain in the ignored workspace;
the publication retains their hashes and the small per-frontier records.

Every one of the nine runs passed the bad-header and bad-token preservation
checks, the post-mutation short-read invalidation, and successful reload followed
by identical fresh logits. The [Debug records](evidence/debug/results.jsonl) and
[summary calculations](summary.json) preserve the actual outcomes.

Both [pressure guards](evidence/pressure-summary.json) completed with child
status zero and normal pressure throughout. The explicit
[33-file source snapshot](evidence/active-source-sha256.json) matches the active
sources at publication; unintegrated HTTP, vision and other draft owners are
excluded. This is canonical-state compatibility evidence at the six specified
frontiers, not a throughput, whole-agent or broad task-quality claim.
