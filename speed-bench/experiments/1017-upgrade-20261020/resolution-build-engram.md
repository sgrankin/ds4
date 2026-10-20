# Makefile and Engram rebase resolution

Resolved only these two root files at the earliest compact foundation, onto
upstream `5379c1c9`, preserving local foundation `7cc542d0` features.

- All five Makefile conflicts are additive unions: upstream DSPark/GPU-copy and
  image prerequisites, CUDA/TP/verification targets and cleanup rules remain;
  local state-cache links, store/header dependencies and native/kernel tests remain.
- All three Engram conflicts preserve upstream POSIX random-access advice,
  the default compact BF16 lookup/error sentinel and persistent token-reader API.
  Local Darwin file-cache diagnostic, opt-in float lookup diagnostic and
  `DS4_ENGRAM_SMALL_BATCH_PARALLEL` threshold remain. Default decoding uses
  upstream's BF16 lookup; the older float lookup is used only when requested.
- Source inspection finds no conflict markers. `git diff --check` passed for
  these files. No compiler, tests, model job or jj mutation was run.

Frozen source for ROOT to snapshot:

- `Makefile` SHA256 `1c18c2471a2e9ca2483b461353f6fd7a561996d717b3e48b5c697785c9374c0c`
- `ds4_engram.c` SHA256 `b88907ed18d1a1041c33c79722d0d8cd8aeee493d58abbe5b222e32bd88861b1`
