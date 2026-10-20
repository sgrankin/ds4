# Approximate DS4.1 suffix reuse

Status: the guarded corpus smoke, three same-position semantic probes and one
aligned-deletion probe passed their native oracle checks. Identity edits matched
complete graph and logits bytes exactly. Changed contexts remain approximate;
the secret-redaction probe shows substantial retained stale conditioning.
All operations are explicit API opt-ins. Normal session sync remains unchanged.

Paper transfer: SCR deliberately reuses stale states from surviving suffixes
after an earlier edit. The paper's K=6 and RoPE relocation belong to its own
attention implementation. Its hybrid recurrent-state policy is not a DS4.1
compressor implementation recipe. `paper/AUDIT.md` identifies the precise
primary-source sections and omissions. This experiment begins with K=1.

The caller saves an exact old endpoint A+B+C, restores exact A, and recomputes
B-prime through every DS4.1 layer. The first prototype requires the same token
positions and an even suffix start. It preserves fresh prefix compressed/index
rows, grafts C's old rows for all four owners, and grafts only the relevant
trailing raw-ring rows for all 40 layers. An odd endpoint copies C's unfinished
pair KV/gates. The Engram last-three-token history is reconstructed from the
edited token sequence. Transient selection masks rebuild at their normal owner
layers. No endpoint logits or residuals are copied: at least one fresh tail
token must run before the API returns. Zero relocation preserves the original
quantized RoPE bytes exactly, isolating prefix-conditioned staleness first.

The session is marked approximate. Generic canonical payload/snapshot export
and further exact suffix-cache capture reject that state. Append/eval retains
the mark; full rebuild, invalidation or exact snapshot load clears it. A named
diagnostic snapshot API permits byte comparison without publishing the result
as canonical KV. Invalid candidates leave live state untouched; failures after
copying begins invalidate it before returning. Scope is text-only DS4.1,
single engine/context, non-TP, without images, steering, speculation, imatrix
collection or an outstanding prediction prefetch.

The native oracle must compare an identity edit using identical partitions
(whole diagnostic payload and raw logits byte parity), then a changed block
against matched fresh suffix replay. It reports raw full-vocabulary KL/TV,
teacher-forced delta NLL, argmax flips, subsequent continuation drift, and
elapsed stages. Fresh-tail length is explicit so 1/64/128-token recovery can
be compared. Changed-code and deleted-secret probes should test semantic
leakage independently of generic distribution distances. All native runs use
the main serialized pressure guard and an explicitly granted GPU slot.

A separate prototype can relocate a deletion in multiples of 128 tokens: move
raw-ring positions, compressed/index owner rows and pending state, then apply
the actual DS4.1 inverse-old/forward-new rotary transforms with requantization.
FP4/FP8 cache rounding makes that positional repair approximate too. The first
shifted native probe passed; `RELOCATION.md` records its narrow coverage.

## First native smoke

The frozen oracle ran under the MAIN pressure guard with A=256, B=32,
unchanged=128, fresh tail=1, fixed continuation=4, context=2048, chunk=256,
and explicit 64 GiB SSD cache. It reused 127 tokens and copied 11,225,088 bytes,
including the mixed raw window and all three unfinished pairs. Identity
logits and the complete diagnostic graph payload were byte-identical to exact
replay with the same prefill partitions. Invalid alignment, no fresh tail,
changed suffix tokens, corrupted header and wrong context were rejected
without changing the live exact prefix. Canonical approximate-state export
was rejected; exact load and invalidation cleared the private-state mark.

Fresh edit evaluation took 3966.902 ms; grafted edit evaluation took 791.725 ms
(5.01x, 80.0% lower). This is a single diagnostic comparison, not a balanced
latency benchmark. Mean KL over five vocabulary rows was 0.03950265 and mean
TV was 0.04718605; no argmax changed but no logit row remained byte-identical.
The four teacher-forced targets accumulated delta NLL +1.96735783, dominated
by +1.93376033 at the first target, whose baseline NLL was already 7.87778.
This arbitrary corpus continuation does not establish semantic correctness.

`smoke.log`, `smoke-guard`, `manifest.json` and `prepared-command.json` retain
the exact command, build/source/core hashes, assertions, scores and guard
samples. Full raw reference/candidate vocabulary rows are retained at the
persistent root artifact path recorded in the manifest. The guard completed
normally with no adverse samples or swap growth. No second model run started
automatically; meaningful fact-edit and fresh-tail recovery probes come next.

## Semantic oracle preparation

`speed-bench/suffix_fact_oracle.c` adds separate single-field code replacement
and secret redaction arms. It uses A=1024, B=256, and C=512/1024. Tail=1/64/128
keeps each complete prompt fixed and changes only the source cache endpoint.
The baseline and reuse paths use identical partitions, and each configuration
first requires complete graph and raw-logit byte parity for an identity edit.
Both edited paths score the gold and old answer sequences, including EOS, and
generate an independent answer by argmax of untouched logits. Raw vocabulary
rows on the gold branch, rendered prompts, exact token IDs, per-row metrics,
and greedy token IDs are saved for inspection. A failed fresh greedy answer
is explicitly recorded and cannot establish a semantic success for reuse.

Focused compilation and CPU metric/padding/answer tests passed. The first code
probe completed successfully. `semantic-manifest.json` records source, fixture and
frozen object/helper hashes; `semantic-prepared-commands.json` contains the
twelve exact guarded commands. The first completed job is code/C512/tail1;
all further jobs require a new root scheduling grant. These diagnostic times
are not balanced production latency measurements.

### First fact-edit result

For independent code replacement 2718 to 3141 at A=1024/B=256/C=512,
the API reused 511 suffix tokens, copied 13,764,608 bytes and retained all
three unfinished pairs. The identity edit again matched whole graph and raw
logits bytes exactly, and approximate canonical export was rejected. Both
fresh and reused greedy answers were exactly `3141` and stopped normally.
The complete gold answer (including EOS, three tokens) had log probability
-0.00009641 fresh and -0.00018463 reused. The old answer had log probability
-18.483914 fresh and -17.455711 reused. Reuse therefore increased the old
answer's likelihood somewhat while leaving it far below the edited answer.

Mean full-vocabulary KL was 0.0000133227, TV 0.0000326710, and gold answer
delta NLL +0.0000882217, with zero argmax changes. No changed-context logit
row was byte-identical. Fresh edited-context evaluation took 40,454.067 ms
versus 17,551.076 ms for grafting (2.305x, 56.6% lower) including B-prime
recomputation; setup and answer evaluation were excluded from those times.
This single paired diagnostic establishes a useful same-position fact-edit
case, with further semantic cases and balanced timing still pending.

The MAIN guard completed normally with 195 pressure samples, all at level1,
zero adverse samples and zero swap growth. `semantic-code-c512-t1` preserves
the guard metadata, exact prompts/tokens, row metrics and summary; the log
and raw vocabulary artifact/hash are recorded in `semantic-manifest.json`.
### Redaction and longer suffix

The secret/C512/tail1 probe also returned `REDACTED` in both greedy runs.
However, the complete old answer `6829` gained log probability from -28.988046
fresh to -9.886779 reused: sequence probability rose from 2.57e-13 to 5.08e-5,
a log-probability gain of 19.101267. Greedy redaction does not imply that the
private stale cache forgot the old value. Mean KL on the gold branch was
8.28989e-5 and gold delta NLL was +0.000306869. Fresh versus reused edit
evaluation was 41,186.623 versus 17,599.827 ms (2.340x diagnostic).

For code/C1024/tail1, both runs answered `3141`. Mean KL was 2.18680e-6 and
gold delta NLL was +6.90863e-6. Edit evaluation was 75,507.064 versus
16,841.088 ms (4.484x diagnostic). The old answer's log probability increased
from -19.265330 to -16.224574. These three same-position cases use the original
frozen semantic helper `0fdf...`, not the later relocation implementation.

### First aligned deletion

A256/oldB256/newB128/C64/tail1 relocated 63 surviving suffix tokens left by
128 positions. The refreshed oracle required all 64 remaining fresh-prefix
raw rows per layer, 960 compressed prefix rows and 960 index prefix rows to
match the exact edited-prefix checkpoint byte for byte. Those checks passed,
as did zero-shift identity graph/logit parity, rejection without mutation,
precancel without mutation and postgraft cancellation invalidation. The
relocation repaired 2,520 raw keys, 156 compressed keys and 156 index keys.

Both greedy answers were `3141`. Mean KL was 1.13211e-5, gold delta NLL was
-3.85657e-5, and the old answer's log probability rose from -20.528396 to
-17.261049. Fresh edit evaluation took 4,625.565 ms versus 3,104.604 ms for
reuse (1.490x diagnostic). This covers the mixed raw window and an odd pending
pair endpoint; it does not establish arbitrary relocation or multi-span reuse.

`completed-cases-manifest.json` and the four compact case directories preserve
the commands, exact prompts/tokens, metrics, pressure samples and raw-row
artifact hashes. All four guards completed with level1 samples, no adverse
samples and zero swap growth. The shifted case uses frozen core `6c2efad0`
and ownership driver `a727354a` / helper `d33d...`. No native run starts without
a separate scheduling grant. Secret fresh-tail64/128 washout and a fully fresh
prior-secret query have now completed; the separate recovery manifest records
the results below.

### Fresh-tail recovery

All three secret/C512 full prompts and target token arrays were identical.
The fresh gold-branch full-vocabulary rows were also byte-identical across
tail1/64/128. Both greedy answers were `REDACTED` in every case, and every
identity edit again matched graph and logits bytes exactly.

| Fresh tail | Reused tokens | Old-value probability, reused | Old-value logprob gain | Mean KL | Gold delta NLL | Diagnostic speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 511 | 5.08e-5 | +19.1013 | 8.29e-5 | +3.07e-4 | 2.340x |
| 64 | 448 | 4.97e-12 | +2.9600 | 6.28e-7 | +4.95e-6 | 2.133x |
| 128 | 384 | 9.07e-12 | +3.5617 | 2.07e-6 | +2.02e-5 | 1.962x |

The fresh old-value sequence probability was 2.57e-13 in all three cases.
Recomputing a substantial fresh tail strongly reduced stale conditioning
in this fixture, but recovery was not monotonic. These remain single paired
diagnostic times, not balanced latency results. The original frozen `0fdf...`
helper and fixtures produced all three points.

### Failed prior-query semantic control

The separate `89bba...` probe put its entire 45-token question and assistant
prefix after the cache endpoint. Identity, prefix ownership and initial
endpoint byte equality before independent branches passed. However, the
original exact context answered `UNKNOWN`, rather than the expected `6829`.
Its question asked about the value *before redaction*, while the original
context supplied only a current register and no redaction event. The fresh
edited and reused contexts also answered `UNKNOWN`. The helper retained all
scores and raw rows and exited 3 to mark failed semantic controls. This case
does not establish old-value retrieval or forgetting; the wording repair
belongs to a separate fixture experiment.

`recovery-manifest.json`, the three compact case directories and the batch
ledger preserve this distinction. Their guards completed with level1, no
adverse samples and zero swap growth. All suffix fixtures used native
resident/off forward controls. The cache API itself checks prior suffix-graft
provenance, rather than tracking every possible forward-policy history.
