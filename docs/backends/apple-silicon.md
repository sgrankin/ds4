# Apple Silicon and Metal

The supported native build runs on macOS and uses Metal. Shared runtime code owns
the process Metal device and queue, while each model owns its graph and resources.
Native Zig controllers choose buffer offsets, dispatch geometry and operation
order. The Objective-C driver creates Metal devices, queues, libraries and
pipeline states, binds actual `MTLBuffer` objects, submits commands and retains
resources until their work is safe to retire. MSL shaders perform GPU math.
[Source ownership](../../src/README.md) describes the remaining C boundaries.

This separation matters for asynchronous inference: a submitted command can
outlive the host call that assembled it. Resource lifetime follows the in-flight
submission, not merely the scope of the controller that requested it. Model
mapping, residency and warmup policy are controlled by the native lifecycle;
the driver owns the corresponding Apple objects. The resource-lifetime
[qualification](../../speed-bench/experiments/zig-metal-resources-20261022/)
records the tested boundary.

The same build selects hardware-specific paths automatically. Dense and routed
matrix operations, attention, quantization and expert loading use shared host
controllers and shaders; model-specific data layout, kernels and tradeoffs are
explained in the [DeepSeek](../models/deepseek/apple-silicon.md),
[GLM](../models/glm53-flash/apple-silicon.md),
[Qwen](../models/qwen38-flash-next/apple-silicon.md),
[Flint](../models/flint/apple-silicon.md) and
[Clef](../models/clef/apple-silicon.md) backend manuals.

## Capacity and startup

Install Apple's command-line tools and follow [build and test](../development/build-and-test.md).
Model files and runtime allocations both matter; these are starting points, not
fit guarantees for every context or session count:

| Unified memory | Starting configuration |
| --- | --- |
| 64 GB | Qwen3.8 Flash Next Q2, starting at 8K context |
| 96 GB | Qwen Q2; stream larger models from SSD |
| 128 GB | V4.1 Flash Q2 with SSD streaming, or GLM 5.3 Flash Q2 with modest context |
| 256 GB | Resident V4.1 Flash Q2 or GLM 5.3 Flash Q4 |
| 512 GB | Larger models, including PRO Q2 |

For example, `./download-model.nu ds41f-q2` followed by
`./ds4 --ssd-streaming --ctx 32768` starts with automatic expert-cache sizing.
See [artifacts](../models/artifacts.md) for file and disk requirements and
[SSD streaming](../storage/expert-streaming.md) before changing the budget. Two Macs can
use the historical [network TP setup](distributed.md); the exact hardware and
model combinations still require their own qualification. For PRO,
`--power 70` trades throughput for lower sustained GPU load; V4.1 and GLM
require `--power 100`.
