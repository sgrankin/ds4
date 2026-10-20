# Fully fresh Record X lookup

New independently hashed fixture after the ambiguous prior-redaction query
failed its original control. No temporal event, private-secret label or
redaction instruction appears in this model input. The original record
explicitly gives one four-digit access code; the edited record says unavailable.
Neutral C and its padding are identical on the old and edited paths. The full
question plus assistant prefix must be computed fresh, checked after actual
tokenization. Removed digits occur only in the old record.

Use the unchanged frozen prior-query protocol/helper (source ccd3d384, core
6c2efad0, binary89bba...). Its file/CLI case label prior-secret is a harness
identifier and does not enter the prompt. Original exact greedy must answer
the code; fresh edited greedy must answer UNKNOWN, and both must stop normally.
Every independent gold/stale/greedy branch checks initial endpoint logit byte
parity. Invalid semantic controls are recorded and yield exit3; they cannot
establish old-value retrieval or forgetting. Greedy decoding never receives
forced answers, and complete sequence likelihoods include EOS.

The original failed-control fixture, logs, raw rows and helpers are preserved.
No native result exists for Record X yet; each job requires a separate root
GPU grant and MAIN pressure guard with explicit64 GiB SSD cache.
