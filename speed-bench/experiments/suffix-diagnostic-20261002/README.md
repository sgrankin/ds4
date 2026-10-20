# Private suffix diagnostic serialization

Based on integrated source `80cd7047`. Diagnostic snapshots now use DSSD magic, retaining the native 13-u32 header layout and every subsequent byte. The canonical payload loader and exact suffix-source parser already require canonical magic; they reject these inspection bytes. Canonical export defaults and formats are unchanged.

The model-free test uses the real opaque-session implementation: a valid synthetic minimal DS41 canonical source is accepted by the actual parser, the identical bytes carrying diagnostic magic are rejected, and actual public payload/snapshot loaders reject the diagnostic on an engine-less session before any GPU call. Snapshot comparisons assert canonical/diagnostic tags and compare every other byte exactly, including version, checkpoint, logits, and tensor body. Both rebuilt oracle arithmetic/ownership controls pass. No model was opened; no native oracle run is claimed.

The original frozen5c/7008 core and helper binaries remain untouched. New frozen binaries and manifests reside under `speed-bench/ideas/artifacts/suffix-diagnostic-build-20261002`; native validation requires a separate root grant.

The later integrated native physical-wrap run passed diagnostic import/source
rejection, identity, prefix ownership, branch, cancellation and export controls.
Its full-vocabulary logits exactly match the previous frozen5c delta66 run.
See the [native result](../suffix-diagnostic-native-wrap-20261002/NATIVE_RESULTS.md).
The earlier helper's canonical pre-cancel comparison was corrected in a separate
child before that run; its original frozen binary remains unchanged.
