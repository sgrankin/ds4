# Borrowed GGUF metadata indexing

Completed 2026-10-22 04:21 UTC, inside the one-hour ceiling. Base is CLI commit
`7596a793` on `zig-modular`.

The existing cursor boundary now also indexes the metadata directory into a
caller-allocated array of borrowed key/type/absolute-value-offset records.
`zig/gguf.zig` contains a 25-line parsing loop plus its contracts. Model/header
validation, count-before-allocation checking, allocation and fatal-error policy
remain in C. The C reference uses the same extracted loop and public record;
there is no duplicated model layout, SDK dump or new allocation abstraction.

Behavior preserved: file-order duplicates; last nonzero UINT32 alignment;
zero/wrong-type alignments ignored; no new power-of-two restriction; independent
alignment peeks; partial fields/record progress and sticky outer errors on
truncation. Values and keys remain borrowed from the original mapping.

All five C/native programs build. The existing cursor fixture, extended with a
mixed table at nonzero offset and a truncated second alignment, passes in
C/native × Debug/ReleaseFast: **four successful runs**. The existing sparse GGUF
model-loader fixture also passes all four configurations under the pressure
guard. Finally, C/Zig/Zig/C real-1017 tokenization returns identical full token
output. Those diagnostics use shared readonly mapping without GPU initialization
or whole-model prefetch. Their 8.7–9.4 ms process times are a correctness screen,
not evidence of a language-port speed difference.

Decision: **retain**. Build fixtures with `tools/zig-build gguf-test gguf-model-test`; run `test-gguf-cursor` directly and the model-opening
`test-deepseek41-gguf` under the existing pressure guard. [Recorded loop](qualify.py)
and [results](results.json) retain real commands/output hashes. Full source/
build/fixture/pressure records are in `.workspaces/zig-gguf-metadata-20261022/`.
