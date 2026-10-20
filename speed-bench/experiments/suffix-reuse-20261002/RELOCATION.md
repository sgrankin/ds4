# Aligned deletion experiment

Status: isolated prototype compiled; CPU arithmetic/fixture checks and one
guarded native C64 relocation probe passed. Core source is frozen at `6c2efad0`;
the ownership-asserting oracle is frozen at `a727354a`, helper hash `d33d...`.
The original same-position semantic helper remains frozen at its recorded
`0fdf...` artifact; those results do not use this relocation implementation.

This extends K=1 to a left shift in multiples of 128 tokens. The live exact
prefix ends at destination_start. The original exact cache owns a matching
suffix [source_start, source_end). Its destination_end is source_end minus
the displacement. The suffix tokens must match at their new positions,
both starts have even pair alignment, and at least one fresh tail token must
run before returning logits. No arbitrary relocation, insertion or multi-span
reuse is supported. Normal session sync remains exact and unchanged.

The fixed displacement preserves raw-ring phase and every ratio2 pooling
boundary. Only suffix raw rows are copied; fresh prefix rows in a mixed
window remain untouched. Compressed and index rows move from the old owner's
row offset to the new offset, retaining all fresh prefix rows. An odd old
endpoint remains odd and copies each owner's unfinished unrotated KV/gate
projection. Engram history is reconstructed from the new last three token
IDs. The first fresh token rebuilds ordinary index selection masks at their
normal owner layers, including candidate mask generation at layer20.

Copied keys receive native inverse-old then forward-new rotary transforms,
followed by their released cache quantization. Raw windows in layers0–1 use
the plain base10000 frequencies; raw windows in layers2–39 use the compressed
base160000/YaRN frequencies because the original graph chooses frequencies
by the layer's compression ratio, not by raw-versus-compressed storage.
Every compressed/index owner uses the compressed frequencies with absolute
stride2 for the first three owners and stride1 for the last. Raw cache keys
return to FP8_E8M0, compressed KV to FP4_E4M3 and index keys to FP4_E8M0.
Only the final rotary dimensions change; the native kernel rounds rotary
outputs to BF16, so the two passes and requantization introduce additional
error beyond stale prefix conditioning. Zero displacement bypasses all
rotary and quantization repair to retain exact identity bytes.

The separate API `ds4_session_reuse_suffix_relocated` reports both old/new
positions and counts of repaired raw/compressed/index keys. The same-position
API delegates to the shared implementation with equal positions. Approximate
state still cannot be published as canonical token-keyed KV. Invalid candidates
leave the exact prefix unchanged; any failure after the first copy invalidates
the session. The updated oracle tests bad alignment, a right shift, missing
fresh tail, changed suffix tokens, a corrupt source, cancellation before the
first copy, and cancellation during fresh-tail sync after grafting. It requires
whole graph and raw-logit identity for a zero-shift edit before comparing a
real deletion against matched fresh recomputation.

The oracle's `--source-edit` sets the old B length independently of `--edit`.
A256/oldB256/newB128/C64/tail1 is the first completed native smoke. It removes
128 neutral padding tokens and changes one code value while moving the same
question suffix left. The separate gold/old answer likelihoods, independent
greedy answer and raw full-vocabulary rows remain available. Explicit phase
completion lines expose setup and edited-prefix versus suffix timings.
Timing remains diagnostic until balanced measurement is scheduled.

The C64 fixture preserves 64 fresh-prefix raw cells per layer after the final
fresh token, unlike C128 where the final token overwrites the sole remaining
prefix cell. The oracle compared those raw cells for all 40 layers, plus 960
compressed and 960 index prefix rows, against the exact edited-prefix snapshot;
all were byte-identical. Zero-shift identity graph/logit byte equality and all
rejection/cancellation assertions passed. Both edited runs answered `3141`.
Mean KL was 1.13211e-5 and gold delta NLL -3.85657e-5. Single-pair diagnostic
edit times were 4625.565 ms fresh and 3104.604 ms reused (1.490x); the old answer
still gained log probability by 3.26735. See the completed case manifest and
`relocation-code-a256-oldb256-newb128-c64-t1` for compact evidence and raw hashes.

Native runs require the MAIN pressure guard, an explicit root GPU grant and
an explicit 64 GiB SSD cache. The first native guard completed normally with
52 level1 samples, no adverse samples and zero swap growth. Arbitrary even
displacements, insertion, odd pair realignment and multi-span reuse remain
unsupported and untested by this prototype.
