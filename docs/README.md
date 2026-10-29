# Documentation

The technical manuals describe the current architecture and implementation.
The lab journal records why experiments were tried and what their evidence
establishes. Start with the system overview, then follow the relevant model.

- [Architecture and data flow](architecture.md)
- [Model manuals](models/README.md): reference, runtime and Apple Silicon
  accounts for DeepSeek, Flint, Qwen, GLM Flash and Clef, with setup guides.
- [Semantic checkpoint experiments](lab/journal/2026-10-28-deepseek-semantic-checkpoints.md):
  the historical question, comparisons, findings and decision.
- [Documentation guide](writing-guide.md): how we organize, write and review
  technical documents, including explanatory flow and source correctness.
- [Experimental methodology](lab/methodology.md): measurement conventions and
  the preferred lab-entry format.

The [DeepInfer comparison](models/deepseek/deepinfer/comparison.md) belongs
with DeepSeek's implementation. Shared serving, storage and backend topics
will get their own focused accounts as that documentation is synthesized.
The [artifact catalogue](MODELS.md) retains download targets and packaged sizes.

Documents receive independent explanatory-flow, source, prose and reader-effort reviews,
followed by an editorial read of the related area. Existing experiment records
remain evidence; they are not the current technical manuals or a work queue.
