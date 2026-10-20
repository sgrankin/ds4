# Paired external-edit suffix reuse: first attempts

The reuse arm actually reused881 cached suffix tokens, rebuilt1320 compressed pairs, and kept the latest409-token user span fresh. Its application-owned repair peak was132,716,631 bytes under the160MiB allowance. All six pre-edit controls were byte-identical between reuse and native: opaque old/edited IDs, source text, full canonical state, and129,280 raw logits.

Both task attempts failed. Reuse wrote the obsolete CODE=2718; its first grader rejected the current value and obsolete-value checks. Native attempted a forbidden live-context read, then a list call, and hit512 tokens in its last round without producing a graded file. The first two control turns passed in both arms. These outcomes cannot establish a semantic cache regression, acceptable quality, speed, forgetting, or persistence. No repair or rescore was performed.

The complete copied source, preparation, pressure, refusal, output and analysis inventory is preserved below with its original workspace-relative layout. The13 files over2MiB remain explicit hash/size/mode references in persistent `.workspaces` directories. Generated traces record token pieces; they are not full generated-step logits.

[Paired analysis](speed-bench/experiments/ds41-ssd/agent-suffix-pair-external-evidence/first-paired-A-B-20261003-v1/paired-analysis.json) · [Independent ROOT audit](root-independent-audit.json) · [Import proof](root-import-proof.json).
