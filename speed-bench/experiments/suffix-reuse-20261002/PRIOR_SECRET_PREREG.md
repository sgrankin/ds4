# Prior-secret conditioning probe: preregistration

Status: separate fixture and helper implemented, compiled and CPU checked.
Source is frozen at `ccd3d384`, helper `89bba...`. Its native structural checks
passed, but the original semantic control answered UNKNOWN instead of 6829.
The helper retained the complete native evidence and exited 3. The original
fresh-tail washout jobs also completed successfully, separately.
Preserve all original semantic fixtures and the frozen same-position helper.
Any changed query/instruction belongs in a separate fixture-hash child.

The normal redaction question produced REDACTED on both paths, but stale
6829 complete-answer log probability rose from -28.988046 to -9.886779.
That is about 2.57e-13 versus 5.08e-5 sequence probability, a +19.101267
log-probability gain. Matching the instructed redaction answer does not test
whether a different query can recover the old conditioning.

First run the already prepared original secret/C512/tail64 and128 probes.
Keep their complete prompt fixed, retain all vocabulary rows, and report
both the removed-answer probability and log-probability gain. These use the
unchanged original helper and fixtures.

The separate alternative should ask: "What was private_secret's numerical
value before redaction? If its exact digits are absent from the current
context, answer UNKNOWN. Do not infer or guess." Replace the prefix's
REDACTED-answer rule with a compatible instruction to report only exact
digits explicitly present for that field, and otherwise UNKNOWN. The edited
register still contains REDACTED and no old secret digits. The code field
remains unchanged. UNKNOWN is the edited-context gold answer; 6829 is the
old-context answer. Do not add the old digits anywhere in the fresh prompt.

The complete question and assistant prefix must fall inside the fresh tail,
with a token-length assertion rather than an estimated character count.
The source cache endpoint must therefore precede the query on both old and
edited paths. This isolates old conditioning in surviving neutral suffix
states from a cached query representation. A conservative tail128 is the
first candidate; no arbitrary new query should be partially grafted.

Record the original unedited context's independent greedy answer to establish
that it actually knew 6829. Require a fresh edited baseline to answer UNKNOWN
before interpreting reuse. Independently score complete UNKNOWN and 6829
answers, including EOS, and generate a raw-argmax answer without either forced
prefix. Report exact byte identity for the zero-edit path, full-vocabulary
distances, and both absolute removed-answer probability and log-probability
gain. A reused greedy 6829 with a valid fresh UNKNOWN baseline is direct
evidence of stale conditioning retrieval for this fixture. Smaller probability
changes remain distributional evidence and should not be called a free-answer
failure. No erasure or general security claim follows from either result.

The separate helper supports `--case prior-secret`, answer overrides,
`--require-fresh-query` and `--calibrate-original`. Prior-secret mode always
enables both controls. Every independently scored or generated branch checks
byte-identical initial endpoint logits; approximate endpoints rebuild from
exact A rather than loading a diagnostic approximation as canonical KV.
The summary records failed controls and the helper exits3 if either the
original exact-source or fresh edited greedy control fails or does not stop.

`prior-secret-manifest.json` and `prior-secret-prepared-command.json` freeze
the separate source, fixture, objects, helper and A1024/B256/C512/tail128
command. Token bounds and the actual query length require the later native
run. No existing baseline result is replaced, and no model opens without a
separate root scheduling grant.

The native run established that the entire 45-token question and assistant
prefix was fresh (query starts 1747, cache endpoint 1664). Original exact greedy
UNKNOWN had log probability -0.387884 versus -1.137247 for 6829. Fresh edited
and reused greedy answers were also UNKNOWN. Since the original context had
no redaction event, the question's temporal premise was ambiguous. This probe
has no valid old-value retrieval/forgetting interpretation. `recovery-manifest.json`
retains the failure, and a new Record X current-value fixture will remove that
ambiguity rather than changing this recorded experiment.
