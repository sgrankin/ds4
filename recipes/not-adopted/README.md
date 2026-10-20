# Tested ideas kept out of defaults

Keep useful failures beside the recipes so another harness can avoid repeating
the same attribution mistakes. These decisions apply to the measured settings,
not every possible version of the idea.

| Idea | Why it was not adopted | Detail |
|---|---|---|
| Segmented LRU expert eviction | A.73% replay gain became 6.45% fresh/1.77% restored actual-agent regressions. Runtime support was removed; the offline simulator is not a deployable flag. | [Cache-policy chapter](../06-cache-policy-and-dead-ends.md#route-frequency-aging) |
| Two-layer prefetch lookahead |87–94 tokens/s versus 128–130 controls despite exact logits; reduced explicit wait did not mean lower completion time. | [Other scheduling directions](../06-cache-policy-and-dead-ends.md#other-tested-directions-not-promoted) |
| Automatic demand-workspace default | Useful small-session gains, but growth regressed 25–63% in the combination and 30% with workspace alone. Cause unresolved; opt-in remains. | [Workspace resizing](../06-cache-policy-and-dead-ends.md#workspace-demand-and-safe-cache-trimming) |
| Hesitation-marker logit penalties | Easy traces sometimes did not change; harder guarded cases lost a correct final to a cap. All tested Item4 penalties failed intervals. Use phase-specific budget evidence separately. | [Reasoning controls](../05-reasoning-controls.md#tried-hesitation-word-logit-penalties) |
| Streaming displaced ideal experts back into resident substitutions | After removing an avoidable join, age 8 was timing-neutral with extra logical reads; age 1 remained slower. Private prototype, not the combined preset. | [Resident routing](../01-resident-routing.md#what-did-not-earn-its-complexity) |
| More host hot-cache RAM as an automatic speed win | Hot512 served all pre-restart chunks from RAM but did not establish faster task completion against drift-matched hot 0 behavior. | [State cache](../03-state-cache.md#evidence-and-limits) |
| Native MXS3 scale packing as a serving feature | Python and C passed byte checks on six native expert matrices and saved 3.670404%; C expands to ordinary native MXFP4 blocks, so no expert RAM benefit. A packed cache/kernel path and whole-serving measurement remain absent. Sampled 1017 routed Q2 scale fields are ineligible; eight sampled Engram windows remain eligible. | [Native scale pilot](../07-native-scale-packing.md) |
| Approximate decoder bounded replay | Promising official-source idea, but no local feature or quality/speed result adopted yet. Keep separate from exact suffix/state restore. | [Official inference lessons](../08-official-inference-lessons.md#candidate-approximate-decoder-bounded-replay) |
| Selected-cache IQ2 register epilogue | Fixed-teacher ABBA averaged 19.530 versus 19.525 tokens/s, no useful gain. SSD-address cases passed; full SSD-experts tests failed the same retained-RSS check on baseline and candidate. Default shader unchanged. | [Kernel experiment](../10-q2-kernel-epilogue.md) |

The measured mixed-prefill-quantum curve has a 128/2048 tradeoff; 512 was
dominated on this fixture. See the [scheduling experiment](../08-official-inference-lessons.md)
for measurements and their scope.

Historical rejection of a changed-logit scheduling patch meant it failed its
then lossless contract. The project subsequently allowed explicit approximations
with measured task quality; numerical disagreement alone is not a universal
reason to reject a new experiment.
