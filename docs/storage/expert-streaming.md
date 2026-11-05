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
actual slab buffers in an engine-owned Metal residency set. It issues one
standing residency request after the first allocation is committed; later slab
additions commit into the same set. Metal may defer the request under pressure.
Existing resource declarations and expert-cache pins still govern accesses,
and the set covers unused slots within those allocations too. The cache ends
the request and clears the set before trimming, unlocking slots or responding
to an `mlock` failure, then disables it until a full cache clear. Unsupported
systems or set-creation failures retain ordinary resource declarations.

This option remains off in presets and ordinary runtime behavior. An earlier
[queue-attached residency experiment](../../speed-bench/experiments/2026-11-05-streaming-slab-residency/README.md)
included the whole slab set in each command buffer on the shared runtime queue. It
shortened the first scalar evaluation after fresh prefill but made complete
128-token decode slower in its five-pair comparisons at 8K and 32K. The profile
shows a useful change in
waiting: large gaps between command buffers shrink, while new gaps appear
near shared-event boundaries within refresh buffers. That supports further
investigation of residency policy; it does not establish a driver root cause
or justify enabling that queue-attached form. The
[standing-request follow-up](../../speed-bench/experiments/2026-11-05-standing-slab-residency/README.md)
found no added refresh waits in its separate diagnostics, but the original fresh
first-step gaps remained. Five-pair timings at 8K and 32K are mixed and do not establish an
overall gain, so the standing request remains unselected. This preserves the
resource-lifetime implementation for further bounded tests of the queue/event
interaction. Co-resident helper performance and older Apple hardware have not
been qualified.

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
