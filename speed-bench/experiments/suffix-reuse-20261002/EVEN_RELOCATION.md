# Even left-shift extension

Status: source `5c8ef1f3` compiled; 117,716 production-layout CPU cases pass.
No native numerical result exists for this source. The validated 128-token
core `6c2efad0` and all previous helpers remain frozen separately.

Only even left shifts with even source/destination boundaries are accepted.
The raw window copy now uses independent physical source and destination
slots. Each segment ends at the next wrap of either side, so a range of at
most128 rows takes at most3 segments. The original bytes come from an
immutable source snapshot. Repair keys at their destination offset using
their old and new absolute positions; do not repair untouched fresh-prefix
rows. Copying only the trailing min(reused,128) rows retains the prefix in a
mixed window. Zero displacement bypasses every repair.

Even displacement preserves all ratio-two pair groups. Existing compressed
and index row shifts, native pair-start RoPE positions, pending unrotated
projection copies and last-three-token Engram reconstruction remain valid.
Odd endpoint pending pairs remain supported; odd displacement, right shifts,
insertion and multiple reused spans remain rejected or unsupported.

`ds4_suffix_layout.h` contains the production admission/segment functions.
`speed-bench/suffix_layout_test.c` calls those functions and compares the
resulting ring against an independent token-at-a-time tagged-state reference.
It checks boundary cases and50,000 seeded random cases, all untouched cells,
both wraps, ratio-two row counts and old/new affine positions. An independent
read-only audit checked252,800 additional layout states. These CPU checks do
not establish native numerical equivalence of shifted keys.

`even-relocation-manifest.json` freezes source, objects, helper and test hashes.
The prepared native commands use A256/newB128 and shifts2/64/130, C64/tail1,
plus C63/tail1 and C63/tail2. The latter distinguishes even from odd cache
endpoints: C includes the fresh tail, so C63/tail1 is even, while C63/tail2 and
C64/tail1 are odd. Every native job requires whole identity graph/logit byte
parity, fresh-prefix cache ownership, cancellation and rejection checks, raw
full-vocabulary metrics and gold/old answer likelihoods. Each job uses the
MAIN pressure guard, explicit64 GiB cache and a separate GPU scheduling grant.
