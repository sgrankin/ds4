# Model documentation

Each model has three complementary accounts. Start with its reference manual
for the model's architecture and input/state contract, as established by the
available reference sources. For Qwen and GLM, parts of this account are derived
from the local conversion and inference contract rather than an independent
official model specification. The implementation page
then traces how this runtime loads, executes and saves that model. The Apple
Silicon page explains memory movement, Metal execution and optimization tradeoffs.

The quick starts assume the [native build prerequisites](../../src/README.md#build-and-run)
are installed. Each builds `ds4-server` into `zig-out/bin`; the downloader only
installs model artifacts.

| Model | Reader entry point |
| --- | --- |
| DeepSeek V4.1 Flash | [DeepSeek](deepseek/README.md): routed experts, compressed/indexed attention, Engram and persistent inference state. |
| Flint | [Flint](flint/README.md): a native port of the reference INT4 model and its resident Metal decoder. |
| Qwen 3.8 Flash Next | [Qwen](qwen38-flash-next/README.md): hybrid recurrent/attention execution, routed experts and disk-backed n-gram embeddings. |
| GLM 5.3 Flash | [GLM Flash](glm53-flash/README.md): recurrent KDA plus sparse attention, routed experts and embedded speculation. |
| Clef, Clef-Flash and Clef-Omni | [Clef](clef/README.md): learned bounded decisions, with model-specific media inputs and a joint answer head. |

The [serving architecture](../architecture.md) owns protocol, scheduling and
conversation lifetimes shared by these models. Model pages describe their own
state and eligibility rather than implying every backend supports the same
cache, batching or media features.

The [artifact catalogue](artifacts.md) retains download targets and packaged
checkpoint sizes. DeepSeek V4 Pro and full GLM 5.3 retain legacy C engines;
their manuals are outside this native-model batch. Original V4 Flash and
GLM 5.2 are retired.
