# Design ideas

These documents preserve useful questions and design constraints from project
discussions. They are proposals rather than an execution queue or a description
of implemented capabilities. The [project page](../project.md) owns current
direction; experiments need a separately selected outcome.

- [Model composition](model-composition.md): independent engines, shared
  scheduling and memory, Flint execution lessons and weight-layout options.
- [Voice and decision models](voice-and-decision-models.md): candidate helpers,
  native implementation references, audio protocols and backend task lifetime.
- [Agent services](agent-services.md): memory and thread operations, bounded
  jobs and recoverable background compaction. The generic hosted orchestrator
  was removed; this preserves possible focused capabilities.
