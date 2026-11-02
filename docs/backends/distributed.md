# Inference across machines

[Documentation index](../README.md)

The supported native build is macOS Metal. Its two-Mac path includes the
upstream functionality through `890898cf`, including the DeepSeek V4.1 execution
described below. The completed port has not been qualified on a physical pair
of Macs. Local transport and kernel fixtures establish narrower properties;
they do not establish RDMA correctness or model speed across two machines.
Use matching revisions and consult the [release QA record](../development/qualification/README.md).
CUDA and ROCm source paths remain, but their historical build targets are absent
from the current Makefile; those sections preserve earlier setups.

There are two modes:

| Mode | Split | Main use |
| --- | --- | --- |
| Tensor parallelism | Routed experts and per-layer work across two Macs or two Sparks | Resident inference with lower per-token work on each GPU |
| Pipeline parallelism | Complete layer ranges across several machines | Fit larger models and overlap long prefills |

These are separate from [CUDA layer placement](cuda.md).
Network protocols have no authentication or encryption. Use trusted machines
and a trusted network; run the same commit on every peer. Model paths and
artifacts must agree. Update all TP peers together when changing versions.

## Tensor parallelism between two Macs

This uses exactly two ranks: a coordinator and one worker. Do not pass `--layers`.
Both GPUs work on the same token and exchange partial results. Expert storage
and work assignment depend on the model and available memory; attention
partitioning is also model-specific. Splitting the work can reduce generation
latency, but the gain depends on the model, link and workload.

Two 128 GB Macs can run GLM 5.3 Flash Q4 or V4.1 Flash Q2, with disk-only
Engram tables for V4.1.
A larger quant may need larger machines even though its tensor layout is supported.

### DeepSeek V4.1: capacity versus balanced work

The capacity-oriented path stores one half of the routed expert bank on each
Mac. Both ranks select the same experts for a token, then each evaluates only
the experts it owns. This halves expert storage per machine, but not necessarily
the token's work: five of six selected experts might belong to one rank, leaving
the other waiting at the next exchange.

When both Macs can hold the full expert bank, scalar decode instead divides
the selected list in half. For V4.1 Flash's six selected experts, rank 0 computes
the first three and rank 1 computes the other three. Each retains the original
routing weights, and the ranks add their partial outputs. This balances the
number of experts without substituting approximate experts or changing router
selection. It trades greater resident memory for less uneven work; floating-point
summation order can differ from the sharded path.

Each rank makes its admission decision before mapping the weights. The estimate
includes the full model, sidecars, session graphs, narrow decode-weight copies,
drafter reserve, prefill headroom and a 2 GiB system reserve. The total must fit
below both Metal's recommended working-set limit and seven eighths of physical
RAM. The peers then negotiate: balanced routing activates only if both offered
full residency. Otherwise both use static expert ownership, even if one rank
already mapped the full bank. `DS4_METAL_DISABLE_V41_TP_BALANCED_ROUTE=1` disables
the full-bank offer. This selection applies to resident inference, not the SSD
expert-streaming path.

The admission policy lives in [memory.zig](../../src/engine/deepseek41/memory.zig),
the optional-mode agreement in [ds4_tp.c](../../src/ds4_tp.c), and the per-token
division in [moe.zig](../../src/engine/deepseek41/moe.zig). Required execution modes
must still match: optional residency can fall back, while incompatible shared
work or speculative-decoding settings fail the handshake.

### Shared-expert work and overlap

Every token also passes through a shared expert, independently of the routed
experts. Computing that whole contribution on both ranks duplicates work.
The default V4.1 scalar path splits a Q8 shared expert along its intermediate
dimension: each rank computes half of the gate/up activations, followed by the
corresponding columns of the down projection. The two down-projection partials
are summed at the FFN exchange. This requires Q8 gate/up/down weights and an
intermediate width divisible by 64. When that split is unavailable or disabled,
the existing ownership path assigns the whole shared expert to one rank,
alternating the owner between layers. Disabling shared ownership restores
replicated shared computation.

On the eligible resident MXFP4-routed/Q8-shared path, the shared work overlaps
the routed expert work through the native concurrent FFN scheduler. The two-Mac
version requires balanced routing and the shared split; the single-Mac version
also uses the scheduler, without a split. The BF16 boundaries remain explicit:
gate/up sums are rounded before SwiGLU, and its intermediate is rounded when
the down projection reads it. Split down projections remain F32 until the two
ranks combine them. This avoids adding an extra rounding to each partial.

These paths are selected in [moe.zig](../../src/engine/deepseek41/moe.zig) and
encoded by [parallel_ffn.zig](../../src/metal/parallel_ffn.zig). Diagnostic controls
are `DS4_METAL_DISABLE_V41_TP_SHARED_SPLIT`,
`DS4_METAL_DISABLE_V41_TP_SHARED_OWNER`,
`DS4_METAL_DISABLE_V41_TP_CONCURRENT_FFN` and
`DS4_METAL_DISABLE_V41_CONCURRENT_FFN`; setting a disable flag restores the
corresponding simpler path. These are scalar-decode choices, not a claim that
every prefill or multi-session batch follows the same schedule.

### Exchanges without a command-buffer break at every gate

A gate is the point where the GPU needs the other rank's partial result.
The default resident scalar path writes partials directly into the shared
transport slab, publishes the payload and arrival flag with coherent stores,
then waits on a CPU-written release word. A service thread observes arrival,
exchanges the partials over the selected transport, and publishes the release
sequence only after the exchange finishes. The GPU can continue in the same
command buffer, avoiding a submit/event round trip at each gate. Bounded
periodic command submission still ensures that queued work reaches the GPU.

The GPU wait has an iteration bound. Exhausting it marks the evaluation as
failed; it is not permission to return a result computed without the peer's
partial. The service releases a waiting GPU even after transport failure so
commands can drain. Session-batched execution uses separate event/sequence
handling, and ordinary scalar decode can fall back to the earlier poll/event
path. `DS4_TP_DISABLE_DECODE_POLL_GATES=1` disables the inline decode path;
`DS4_TP_EVENT_GATES=1` also selects event-based arrival. See the ordering in
[tp_control.zig](../../src/metal/tp_control.zig) and the service thread in
[ds4_metal.m](../../src/metal/ds4_metal.m).

Large prefill exchanges allow more time for the peer to arrive: cold kernels
and host-side Engram reads can delay one rank. The default arrival allowance is
30 seconds. `DS4_TP_BIG_GATE_ARRIVAL_MS` accepts positive values up to
600,000 ms; larger overrides are ignored. The selected allowance is clamped
to at least the normal gate timeout plus two seconds. After the header
handshake, the normal payload timeout resumes. Scalar decode keeps its shorter
timeout, and a closed peer connection still fails immediately.

### Optional residency workaround

`DS4_METAL_TP_DETACH_RESIDENCY_SET=1` retains the committed Metal residency set
but removes its attachment to the command queue. Upstream observed multi-second
submission stalls when a queue repeatedly requested residency for very large
model mappings on M3 Ultra pairs. Detaching avoids that queue-level request;
it does not unload or shrink the model. This remains opt-in because upstream's
validation used a disabled wired-memory collector, and we have not validated
the workaround on a physical pair. Leave the default attachment in place unless
investigating that specific stall. `DS4_TP_STALL_TRACE=200` reports gates with
more than 200 ms of GPU-arrival wait or 50 ms of exchange time, helping distinguish
submission delays from transport delays.

### Link setup

Use a Thunderbolt cable. RDMA requires an active verbs device with an
IPv4-mapped GID; a working ping alone does not establish that.

```sh
rdma_ctl status
ibv_devinfo -v
```

Addresses must be on the cabled member interfaces, not only the Thunderbolt
bridge. For example, after checking which interfaces are active:

```sh
# Machine A, example member interface en1.
sudo ifconfig en1 inet 10.99.0.2/30 alias
# Machine B, example member interface en6.
sudo ifconfig en6 inet 10.99.0.1/30 alias
```

For the large tested shards on otherwise idle 128 GB Macs, the setup raised
the per-boot GPU wired-memory limit on both machines:

```sh
sudo sysctl iogpu.wired_limit_mb=120000
```

This is specific to that memory configuration. It grants a larger GPU budget;
it does not create more RAM. Check model, context, and system headroom before
raising a limit on your machine.

### Start the pair

Download the same model on both machines. For GLM 5.3 Flash Q4:

```sh
./download-model.nu glm53-q4
```

Start the worker first; it retries while the coordinator loads:

```sh
# Machine B.
./ds4 --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma --ctx 8192

# Machine A.
./ds4 --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma --ctx 8192
```

The verbs device and GID are selected automatically. If ambiguous, specify
`--rdma-device` and `--rdma-gid-index` from `ibv_devinfo`. Use `--transport tcp`
on both peers when RDMA is unavailable. Do not probe the waiting coordinator
with `curl` or `nc`: it may treat the connection as a worker handshake.

Keep workers running in a terminal or managed session and retain both logs.
Do not treat repeated handshake or RDMA timeouts as successful QA merely
because a retry works.

The coordinator can be `ds4`, `ds4-agent`, `ds4-server`, or `ds4-bench`;
workers run `ds4`. For vision, pass the same `--vision FILE` to both peers;
for GLM MTP, enable `--mtp` on both; for DeepSeek DSpark, use matching support
models and options. V4.1 supports vision and DSpark in two-rank Metal TP.
The [DSpark QA](../development/qualification/deepseek.md#dspark)
lists the prescribed physical-pair checks; the checklist alone does not
establish completed qualification at this revision.

TP disk-cache restore currently rebuilds the exact saved token prefix on both
ranks rather than restoring the coordinator alone. Expect prefill on restore.
See [speculation](../serving/sampling-and-speculation.md) and [serving](../serving/overview.md).

## Tensor parallelism between two Sparks

V4.1 Flash Q2 text inference supports one GPU per rank with a 50/50 expert
split. Each Spark holds about 81 GiB of main weights, plus context and runtime
buffers. V4.1 Engram tables stay on disk. Do not add `--ssd-streaming` or
`--cuda-tensor-parallel`: those select different memory/execution modes.

Historically, both machines were built with `make cuda-spark` at the same commit;
that target is absent from the current Makefile. Download `ds41f-q2` on both.
RDMA needs the libibverbs development headers at build time,
its runtime library, and an active RoCEv2 link. Check `ibv_devinfo -v` and use
the addresses of the directly connected ports, not the management network.

For example, with the coordinator at `172.31.250.1` on the direct link:

These historical commands name the earlier Q2 file; the current `ds41f-q2`
download selects the 1017 filename. Match the artifact to the chosen revision.

```sh
# Worker.
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 32768 \
  --tensor-parallel --role worker --coordinator 172.31.250.1 9911 \
  --transport rdma

# Coordinator.
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 32768 \
  --tensor-parallel --role coordinator --listen 172.31.250.1 9911 \
  --transport rdma
```

The link address selects the matching verbs device and GID. If selection is
ambiguous, use `--rdma-device` and `--rdma-gid-index`. TCP is also available
with `--transport tcp` on both peers.

The coordinator can be `ds4-agent`, `ds4-server` or `ds4-bench`. Match context
sizes and any `--prefill-chunk` overrides on both sides. For V4.1, five or more
ready sessions use native batched decode; smaller groups run in order.
Use `--batched-session 8` on the server and allow memory for all eight contexts.
Retained CUDA network TP implements matching V4.1 DSpark, but not vision or
other model/quant layouts. Physical-pair qualification must be checked in the
[DSpark QA](../development/qualification/deepseek.md#dspark); a prescribed test is not a completed result.

## Pipeline parallelism

The retained PRO engine supports split Q4 artifacts; see the commands below.

Each process maps only its assigned layers, retaining that slice of the KV
state. Layer ranges are inclusive. `N:output` includes the final layer and
output head. Activations travel from one stage to the next over TCP.

Normally give the output head to the final worker. With several workers,
choose non-overlapping ranges covering the entire model. Workers register
their ranges with the coordinator; intermediate workers forward activations
directly to the next stage.

Long prefill chunks can occupy different stages simultaneously. A single
generation stream cannot use that overlap: each token must finish the route
before the next one is sampled. Use pipeline mode primarily for capacity and
long-prefill throughput, not as a guaranteed decode speedup.

### Full PRO Q4

For two 512 GB Mac Studios, use the split artifacts:

```sh
# Machine A.
./download-model.nu pro-q4-layers00-30
./ds4 -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator --layers 0:30 --listen 10.99.0.2 9911

# Machine B.
./download-model.nu pro-q4-layers31-output
./ds4 -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker --layers 31:output --coordinator 10.99.0.2 9911
```

These downloads do not change `ds4flash.gguf`. Startup is expensive because
each side must make its model slice resident.

### Tuning and recovery

Keep default chunk sizes first. `--dist-prefill-window N` controls the number
of chunks in flight; `--dist-prefill-chunk N` overrides the session-derived
chunk size. `--debug` shows route and per-hop timings.

Activations use 32-bit transport by default. `--dist-activation-bits 16` halves
the payload; `8` is more aggressive. These change numerical precision on the
wire, not weights or KV storage. Validate output when changing them.

A disconnected worker invalidates the route. In-flight work can fail; later
requests need a complete route before proceeding, and the coordinator can
replay the saved token prefix to rebuild worker state. Pipeline snapshots
serialize all layer slices into one payload and redistribute them when loaded.

For protocol details, see [ds4_distributed.c](../../src/ds4_distributed.c)
and [ds4_tp.c](../../src/ds4_tp.c).
