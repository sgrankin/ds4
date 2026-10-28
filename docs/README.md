# Documentation

This tree is being rewritten around the implementation rather than experiment
chronology. The first pilot covers the architecture, lab methodology and a
source-backed DeepSeek example. Existing manuals remain in place during review.

- [Architecture and data flow](architecture.md)
- [DeepSeek on Apple Silicon](models/deepseek/apple-silicon.md): current
  execution, layouts, weight movement and optimization tradeoffs.
- [Semantic checkpoint experiments](lab/journal/2026-10-28-deepseek-semantic-checkpoints.md):
  the historical question, comparisons, findings and decision.
- [Writing guide](writing-guide.md): document responsibilities and explanatory style.
- [Experimental methodology](lab/methodology.md): measurement conventions and
  the preferred lab-entry format.

The intended areas are `serving/`, `storage/`, `backends/` and `models/`.
Each model will have a reference manual, implementation account and relevant
backend documents. DeepSeek's DeepInfer comparison belongs with that model.
Date-prefixed lab narratives belong in `lab/journal/`.

New documents are reviewed for prose, technical correctness and consistency
before replacing the material they synthesize. This pilot does not archive,
rename or delete the existing documentation or experiment folders.
