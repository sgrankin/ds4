# Models Larger Than RAM

[Serving](../serving/overview.md)

Use resident inference when the model and runtime fit: it is faster.
SSD streaming keeps a bounded cache of routed experts and reads missing
experts from the GGUF. It trades speed for capacity; it does not remove the
memory needed for other weights, activations, scratch, and the context.

Metal supports streaming for DeepSeek and GLM. CUDA and ROCm retain streaming paths in source; they are not newly qualified by this manual. Do not infer support for every model
and tensor layout from the existence of the flag.

## Start with the automatic budget

```sh
./ds4 -m ds4flash.gguf --ssd-streaming
```

The startup report shows the effective cache, resident layers where applicable,
and memory requirements. Prefer the automatic budget for a first run. Use a
local SSD and leave memory for the OS and other applications.

Examples:

```sh
# V4.1 Flash Q2 on a 128 GB Mac.
./download-model.nu ds41f-q2
./ds4 --ssd-streaming --ctx 32768 --nothink

# GLM 5.3 Flash Q4 on a 128 GB Mac.
./download-model.nu glm53-q4
./ds4 --ssd-streaming --ctx 4096

# PRO Q2 on a 128 GB Mac: usable for inspection, but slow.
./download-model.nu pro-q2-imatrix
./ds4 --ssd-streaming --ctx 32768 --nothink
```

Generation is usually more sensitive to cache misses than prefill. A large
model that starts successfully can still be too slow for interactive work.
Use a short generation before committing to a long task.

## Adjusting the cache

To leave more room for context or other sessions:

```sh
./ds4 --ssd-streaming --ssd-streaming-cache-experts 32GB
```

A byte budget is a target, not a guaranteed allocation. DwarfStar reserves
routed-prefill headroom and fits the cache to the remaining model, graph,
context, and backend budget. The effective value may be smaller than requested.
Non-routed weights and KV state are additional to that expert-cache budget.

A plain number, such as `--ssd-streaming-cache-experts 4000`, requests dynamic
expert slots rather than a byte budget. It is also subject to memory limits.

By default, GLM spends the cache budget on selected experts across all layers.
`--ssd-streaming-full-layers N` reserves full routed prefix layers instead.
Metal also keeps the non-routed weights resident when it can; they are needed
by every token, and paging them back in delays generation after a prompt.
An oversized expert cache can displace those weights and slow decoding.
More cache helps only while the rest of the working set still fits.

More context and more server sessions consume more memory. Metal GLM budgeting
includes planned and already resident sessions; a request can still be refused
when its mandatory allocations cannot fit. Disabling the memory guard is not
a remedy for insufficient RAM.

Leave expert preloading enabled for normal use. `--ssd-streaming-cold` and
`--ssd-streaming-preload-experts N` are mainly useful for controlled measurements.
See [benchmarking](../development/benchmarking.md) and the [release QA guide](../development/qualification/README.md).

## How Metal stores cached experts

The default Metal streaming cache groups expert slots into large buffers called
slabs. Each slot holds an expert's gate, up and down weights; those three views
share one Metal buffer and use different offsets. The
[storage policy](../../src/metal/expert_storage.zig) targets 4,096 MiB per slab,
rounded down to whole slots. `DS4_METAL_STREAMING_EXPERT_SLAB_MB` changes that
allocation size independently of the requested expert-cache capacity. The last
slab can be smaller. There is a 256-slab limit, so arbitrarily small slabs are
not a way to preserve a large cache.

The CPU locks populated slots with `mlock`. GPU resource declarations have a
different granularity: selected-address prefill tells Metal which whole slab
buffers its indirect expert addresses will read. Locking a slot against host
paging does not establish the cost of making that buffer available to the GPU.
This distinction matters when profiling a wait before expert kernels begin.
The [slab-size comparison](../../speed-bench/experiments/2026-11-03-expert-slabs/README.md)
tested 4,096 and 512 MiB at unchanged logical capacity. Smaller slabs shortened
the first decode step but increased time in the remaining generation window;
they did not reliably improve the complete measured workload or solve the slow
append. The default remains 4,096 MiB. This is a measured latency tradeoff, not
evidence that every operation benefits from smaller resources.

An experimental `DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY=1` registers the
actual slab buffers in an engine-owned Metal residency set attached to the
runtime command queue. Each new slab is committed to that set before its slots
are published. The association includes the whole set in the queue's command
buffers; it does not guarantee permanent residency under pressure.
Existing resource declarations and expert-cache pins still govern accesses,
and the set covers unused slots within those allocations too. The cache detaches
and clears the set before trimming, unlocking slots or responding
to an `mlock` failure, then disables it until a full cache clear. Unsupported
systems or set-creation failures retain ordinary resource declarations.

Setting the same flag to `2` instead associates that set with each command
buffer that encodes cached expert kernels or blits that populate the cache.
Pure router or shared-work buffers can omit it; a buffer containing both expert
work and a later router remains covered. The association applies to the whole
command buffer, not an individual encoder. This mode changes neither command
boundaries nor the set's allocations.
It requires the macOS 15 command-buffer residency API and falls back to ordinary
declarations if that API is unavailable.
Seed buffers receive the association after their destination slabs are
registered and before their existing blit encoder opens. Snapshot synchronization
buffers still omit it; queue mode also associates the slab set with those
buffers. These scope differences matter when locating initialization costs.

Both modes preserve the existing cache pins and resource declarations. For
unretained command buffers, mode `2` keeps an extra reference to the set through
completion. That reference preserves the set object, not a snapshot of its
allocations, and does not replace the existing ownership of in-flight buffers.
The [first command-buffer comparison](../../speed-bench/experiments/2026-11-05-scoped-residency/README.md)
covered only compute reads. It restored a large fresh-start gap, which motivated
the [seed-write extension](../../speed-bench/experiments/2026-11-05-seed-residency/README.md).
The extended mode confirms association on seed buffers and no longer shows
that large gap in the tested comparison, with matching checked outputs. Five
pairs per context still show mixed phase costs without establishing a
whole-workload gain.
Keep mode `2` as unselected groundwork; the experimental latency package retains
queue mode `1`. These observations do not establish a driver-level cause.

Command scope also limits how this policy interacts with shared-expert overlap.
The next router can follow previous-layer expert work in the same buffer, so
its event still belongs to an associated buffer. The
[overlap comparison](../../speed-bench/experiments/2026-11-05-scoped-overlap/README.md)
found large positive event-adjacent gaps in those mixed buffers and slower
decode with the large cache. A smaller cache retained the same associations
without comparable large gaps and benefited from overlap in warm decode;
cache contents and misses differed too. Narrowing association alone therefore
does not justify restoring overlap globally. The selected experimental package
continues to disable it.

This option remains off in ordinary runtime behavior and fast presets. Its cost
depends on the refresh schedule. With the selected-ID shared-event overlap
enabled, the [first comparison](../../speed-bench/experiments/2026-11-05-streaming-slab-residency/README.md)
shortened the fresh first step but slowed complete decode. A
[standing request without queue attachment](../../speed-bench/experiments/2026-11-05-standing-slab-residency/README.md)
avoided the new refresh waits but retained the original first-step gaps; that
policy is no longer the implementation behind this flag.

The [combined scheduling comparison](../../speed-bench/experiments/2026-11-05-slab-event-interaction/README.md)
pairs queue residency with `DS4_METAL_V41_ROUTE_SHARED_OVERLAP=0` in the
experimental GPU-routing package. CPU refreshes then use full waits and the
existing shared-down/HC-expansion fusion, while ordinary GPU-routing steps keep
their existing kernels. Five pairs at each frontier show faster fresh
128-token decode and total inference, but lower throughput after the first
token. Warm decode was mixed at 8K and slower in four of five pairs at 32K.
This is a latency tradeoff, not a general steady-throughput gain. All 512
fixed-history logit rows matched.
The driver cause of the event-adjacent waits remains unproven. Co-resident helper
performance and older Apple hardware have not been qualified.

With shared-event overlap disabled, the refresh path still submits shared
gate/up work before the routed handler prepares its cache entries and GPU
addresses. Shared GPU execution can overlap that CPU preparation even when the
selected experts are cached. The
[submission-removal experiment](../../speed-bench/experiments/2026-11-05-refresh-shared-flush/README.md)
reduced refresh command buffers without establishing a whole-workload gain;
the separate submission remains. Putting both sets of work in one batch does
not itself enable parallel FFN: the ordinary streaming parallel path also
requires its resident-only selection and cache-pin proof. A separate
[cached-refresh prototype](../../speed-bench/experiments/2026-11-05-cached-refresh-parallel/README.md)
adds a proof for the exact six CPU-selected entries and retains serial miss
fallback. It makes the concurrent schedule possible, but has not earned
selection in the latency package: shorter sampled FFN work coincided with larger
GPU inter-buffer gaps before that work starts. The [model backend manual](../models/deepseek/apple-silicon.md#resident-expert-substitution)
owns its admission conditions.

## M5 Max measurements

On a 128 GB M5 Max, with automatic cache sizing and no speculative decoding:

| Model | Initial prefill | Continued prefill | Generation after each | Runs |
| --- | ---: | ---: | ---: | --- |
| GLM 5.3 Flash Q4_K, 177.77 GiB | 121 t/s | 104 t/s | 11.9 / 14.9 t/s | Three-run median |

September 6, 2026. GLM used a 2K prompt and a 1K append, generating 128 tokens
at each frontier. Generation includes
the first-token wait. Cache reuse depends on the prompt, so these are
workload references, not a speed
guarantee for every model larger than RAM.

Full GLM 5.3 IQ2_XXS (196.58 GiB) also benefits from caching short tool-result
prompts. With an 8K context and a 61.35 GiB effective expert cache, a 16-token
append fell from 30.8 to 2.9 seconds; generation afterward rose from 4.09 to
5.11 t/s (three-run medians, 64 generated tokens). All compared logits and
generated text were identical. This is a short-append improvement, not an
equivalent gain in initial prefill or every generation workload.
