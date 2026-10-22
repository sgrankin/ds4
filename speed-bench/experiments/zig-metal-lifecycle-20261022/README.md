# Closing native Metal lifecycle owners

Model publication/residency/warmup, GPU bootstrap/retirement sequencing and optional selected-route record/replay/hotlist transactions now use native owners. Existing ModelViews and RoutedSingle sole entries compose the new pure owners; Bootstrap adds one entry over the same generated Metal binding. Actual state moves once into shared records. Literal pipeline catalogues, actual Objective-C resources and fixed ARC operations remain driver leaves.

Independent source reviews passed before integration. C fallback and native ReleaseFast/Debug builds pass all98 fixture invocations across four configurations. The C sampling fixture exposed12 stale test-only Engine member references, now adjusted to the existing actual Core record. Native integration needed a standard string header and a retained legacy GLM hotlist adapter. First failures are retained; no numerical assertion or runtime tolerance changed.

All45 final C/native/Debug GPU invocations pass: full DeepSeek fixture, shared BF16, dense MPP, parallel/deferred scheduling, local TP gates, exact-view lifetime and all three SSD expert formats. The exact-view eviction fixture initially ran without its documented1MiB setting and failed in original C; supplying its required setting passes both eviction and unlimited modes in all three builds. The original failed invocation remains visible. Full kernel fixtures retain the previously required math-safe setting; no shader change was introduced.

Six additional real SSD-fixture invocations record selected IDs/hotlists, replay the C binary tape and merge hotlists. C/native/Debug produce identical binary tape and hotlist bytes in both stages, preserving the independently discovered selected-cache hotness side effect. These are local fixtures, not two-host TP qualification.

The complete matched model/performance campaign follows separately. [Commands, results and first failures](evidence/). Useful binaries remain under `.workspaces/zig-qualified-20261022/`.
