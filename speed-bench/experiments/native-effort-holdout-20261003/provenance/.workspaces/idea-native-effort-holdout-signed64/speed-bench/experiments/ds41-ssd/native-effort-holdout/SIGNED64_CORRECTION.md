# Holdout signed64 scorer correction

The original 5ffaa7eb source and prepared V1 plan remain unrun and unchanged
in the original workspace. Independent review found that Python's `int` digit
limit could abort analysis of an otherwise complete first attempt. A 5000-digit
overflow must be INVALID, while 5000 leading zeroes followed by43 must parse
as43, matching the unchanged literal C `strtoll` scorer.

This child changes only `parsed_integer` in the analyzer. It removes an optional
sign, normalizes leading zeroes and checks signed64 magnitude lexically before
converting at most nineteen digits. Negative minimum, positive maximum,
signed zeroes and arbitrary leading zeroes remain valid. Python's global
conversion limit is unchanged. The wrapper parsing order is unchanged.

Twelve model-free analyzer tests pass, including eighteen direct comparisons
with native `strtoll` plus wrapper variants, signed64 boundaries and long
overflow/leading-zero pieces within a complete six-case first-attempt fixture.
An AST comparison confirms every analyzer function outside `parsed_integer`
is unchanged. The existing helper binary, eight native objects, literal C
scorer, six frozen inputs, arm ordering, generation policy and limits are
unchanged. The original CPU proof remains a historical V1 proof; the separate
V2 correction proof pins the new analyzer and test source. No model, tokenizer,
GGUF metadata, core build or helper build is involved.
