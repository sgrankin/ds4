# Documentation

Start with [architecture and data flow](architecture.md) for component boundaries,
then follow the relevant manual. [Project direction](project.md) records current
goals and capability limits.

| Area | What it explains |
| --- | --- |
| [Serving and clients](serving/README.md) | Requests, stateful Responses, scheduling, the local agent and external clients. |
| [Models](models/README.md) | Per-model reference, runtime and Apple Silicon manuals, plus the artifact catalogue. |
| [Storage and memory](storage/README.md) | Expert weights, inference checkpoints and durable conversation history, with their distinct lifetimes. |
| [Backends](backends/README.md) | Shared Metal execution and retained CUDA, ROCm and multi-machine paths. |
| [Development](development/README.md) | Building, testing, benchmark tools and documentation practice. |
| [Lab journal](lab/README.md) | Why experiments were tried, what happened and the evidence behind decisions. |
| [Design ideas](design/README.md) | Unselected proposals and constraints worth retaining for future work. |

The [DeepInfer comparison](models/deepseek/deepinfer/comparison.md) belongs to the
DeepSeek manuals. [Recipes](../recipes/README.md) explain techniques and their
tradeoffs. Historical result reports retain detailed samples; manuals describe
current understanding.

Follow the [writing guide](writing-guide.md) and [lab methodology](lab/methodology.md)
when updating these documents. Substantial changes receive independent source,
prose, explanatory-flow and reader-effort reviews, followed by an editorial read
of the related area.
