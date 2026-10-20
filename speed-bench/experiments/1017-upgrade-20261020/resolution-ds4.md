# ds4.c foundation conflict resolution

Resolved the16 jj conflict hunks in the earliest compact foundation, old local `7cc542d0` / base `0aaea5a2`, onto upstream `5379c1c9`. Only `ds4.c` was edited in tracked runtime sources. No jj commands, builds, tests, model jobs or new bounded-replay option were run/added. Original conflicted source and the three pinned source versions are retained in `resolution-ds4/` for read-only review.

Semantic choices:

- **1–4: graph ownership/workspace.** Retained upstream DSpark forward declarations and graph owner, plus our explicit/demand-sized workspace helpers and pre-RoPE archive. Actual graph teardown now frees upstream DSpark and Engram reader owners before the existing local workspace/predict/oracle/tape teardown. Workspace-only growth does not free these graph-lifetime owners. Reset clears both DSpark and pre-RoPE coverage.
- **5,8: norm fusion.** Kept upstream CUDA pointwise fusion and our opt-in Apple BF16 norm helper in their mutually exclusive platform branches; ordinary fallback remains.
- **6–7: shared expert and routed SSD work.** Kept new upstream CUDA shared-work balance helpers and ROCm selected-load startup, alongside local Metal fused shared projection, resident routing, exact-route oracle and early/async/predict load features. The non-Apple `shared_here` remains mutable because upstream balanced ownership can enable it after queuing shared work.
- **9–11: batch MoE/TP/verification.** Local bool/coordinator flow and resident-prefill/overlap/exact tiles stay. Upstream balance ownership/mask conditions and explicit `DS4_TP_GATE_FFN` arguments stay. The routed resident flag is **`force_resident && !g->verify_prefix`**, preserving both selected-small nonresident operation and the upstream requirement not to force residency during prefix verification.
- **12–13: scalar failure cleanup.** Kept local gate-predict setup but adopted upstream's `bool ok` entry flow so a failed tensor write/begin/embed does not return before waiting on the active Engram reader. Predict setup is armed only if initial GPU setup succeeded. Reader completion and local deferred route validation both run before scope exit.
- **14: capture plus trace.** DSpark target-feature capture remains before batch before-attention, then the existing local norm/split trace hooks run.
- **15–16: sampler dispatch.** Retained the local overthinking-penalty scalar-evaluation fallback **before** upstream's new V4.1 DSpark speculative dispatch, for greedy and stochastic entrypoints. Other upstream dispatch/settings remain.

Source-only checks completed:

- All16 conflict markers removed;587 preprocessor conditional blocks are balanced.
- `git diff --no-ext-diff --no-textconv --check -- ds4.c` passed.
- Compared functions against the pinned upstream source: `ds41_attention_candidates` and `ds41_index_batch` are identical. `ds41_attention_batch`, scalar/batch KV publication, and `ds41_decoder_prepare` differ only by the retained pre-RoPE archive capture calls. Thus the upstream compressed-attention causal indexing/masking paths were not replaced by the old implementation.
- Checked all `ds41_prefill_limit`, graph-memory/allocation, MoE-batch and TP sum call sites for the combined local signatures/new gate arguments.
- Detailed function-scoped upstream comparisons, including the scalar MoE merge, are in `resolution-ds4/*.diff`.

This is ready for ROOT's independent source review, snapshot/descendant propagation and focused compilation. No compile/runtime parity is claimed by the source checks.
