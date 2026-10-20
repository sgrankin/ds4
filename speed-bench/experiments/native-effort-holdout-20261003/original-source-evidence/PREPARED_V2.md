# Corrected holdout launch freeze

The original V1 plan remains unrun and all 55 original assets still pass its
default model-free preflight. V2 uses the independently reviewable parser-only
source a5b6aa6e0320da7c824b8a9056af33f36c8d2cb3 in a separate workspace.
The existing helper binary, eight native objects, six inputs and all generation
settings are reused unchanged; no build or model metadata access is performed.

V2 default preflight passes all 58 pinned assets. The extra preparation paths
come from reusing the old helper build and adding the separate correction proof
and documentation. The old CPU proof remains labeled as historical V1 evidence.
New source/tests and the correction proof have their own explicit hashes.

Prepared runtime assets are under
`speed-bench/ideas/artifacts/native-effort-holdout-prepared-20261003-v2` in
`idea-native-effort-holdout-signed64`. `prepared-v2-evidence` retains the complete
small command/plan/launcher/frontend/proof/index bytes, while the unchanged
helper binary is retained at its precise pinned path. The command carries the
authoritative plan SHA and invokes only the MAIN pressure guard. The launcher
defaults to CPU preflight; `--execute` still requires a separate ROOT model grant.

The fixed six-case order, first attempts, capped/wrong/INVALID preservation,
balanced effort0/25/75/75/25/0 repeats and conservative 90–110 minute budget
are unchanged. No outcome has been generated or inspected for these inputs.
