# Release qualification

Select checks for the models, features and hardware the release claims; this is
not a requirement to exercise every optional feature. Start with
[the native build and shared protocol checks](native-and-protocol.md), then use
the applicable model pages: [DeepSeek V4.1](deepseek.md),
[GLM 5.3](glm.md), [Qwen](qwen.md), [Flint](../../models/flint/usage.md)
and [Clef](../../models/clef/usage.md). For a release claiming distributed or
other backend support, see [distributed](distributed.md), [CUDA](cuda.md) and
[ROCm](rocm.md). Those pages describe separate qualification, not capabilities
established by the native Metal build.

| Scope | Use | Evidence boundary |
| --- | --- | --- |
| Current native Metal | Run the exposed build, fixtures and applicable model paths. | A model-free fixture does not establish model or hardware quality. |
| Fixture unavailable in the current build | Treat the described case as a test objective and restore a production-linked fixture before relying on it. | An old binary or source file alone does not establish a current pass. |
| Historical reference or physical backend | Compare only matching workloads, and test physical hardware where support is claimed. | Old figures and simulated admission are not current measurements or physical qualification. |

Record each selected check as passed, failed or not run, with reasons for failures
and omissions. Record the revision, hardware, model and checksum, prompt, context, flags,
commands, results and any failures or skipped checks. Check the release artifact
against the selected tree; after a checkout or history change, ensure the binary
was rebuilt rather than relying on restored timestamps. Keep detailed results in
a QA report. Run one model-opening job at a time under the pressure guard, and
monitor pressure and swap. Use the current [measurement methodology](../../lab/methodology.md)
for comparative performance and quality.

Sign off only on the scope actually exercised: ordinary model paths, matching
continuation fixtures, relevant batching and fallback paths, disk KV, streaming
APIs and real agent/tool interruption and recovery. Exercise physical distributed
or non-Metal paths if the release claims them; otherwise state they were not
validated. Explain unexpected slowdowns, intentional quality tradeoffs and every
skip. Original V4 Flash and GLM 5.2 checkpoints and vectors are historical, not
current V4.1 or GLM 5.3 gates.

The former upstream QA host names are historical contact information, not an
authorization to access those systems. In particular, `192.168.60.250` required
specific owner permission for each QA pass. Arrange permission and an appropriate
host before any remote build, process control or test.
