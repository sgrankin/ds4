# Fully fresh prior-secret query fixture

Independent fixture derived from the frozen original redaction arm. The old
register is unchanged; the edited register contains REDACTED. Neutral notes
are unchanged. Prefix instructions and final question ask for exact explicit
evidence, otherwise UNKNOWN. Removed digits occur only in the old register.

Use `--case prior-secret --fresh-tail 128` and require the entire tokenized
question plus assistant prefix to lie after the cache endpoint. The helper
asserts that bound after actual tokenization. Original exact-source greedy
calibration must know the old value; fresh edited greedy must answer UNKNOWN.
A reused old-value answer with valid controls demonstrates stale conditioning
retrieval in this fixture. Probability gains without that answer remain
distributional evidence. Every branch replays the same endpoint logits byte
for byte before scoring or independent greedy generation. Neither output
establishes cache erasure or a general security result.

No native result exists yet. Preserve original semantic fixtures and helpers.
