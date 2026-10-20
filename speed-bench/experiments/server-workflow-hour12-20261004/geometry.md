# Item 12: measured suffix geometry

2026-10-04. Read-only inspection of native cached token histories, plus the
actual reuse server logs. No compilation, tokenizer/model opening or inference.

A deletion's user-block parity and the parity of a later reusable suffix can
differ. Full-input replay reconstructs an earlier function call from its public
JSON; here that reconstructed span is one token longer than the sampled call.
Prompt-length subtraction alone therefore gives the wrong graft geometry.

## Direct token-anchor measurements

Read the endpoint manifests under
`.workspaces/server-agent-day-20261004/workflow-native-cache`. Manifest fields
are little endian: token count at byte12, profile length at20, segment count at40;
records begin at96+profile length and contain `(which:u32, bytes:u32,
offset:u64, SHA1:40 ASCII bytes)`. Token IDs occupy state bytes52 through
52+4*token_count. Only overlapping small token chunks were read; their lengths
and SHA1 digests matched their records. Format/read seams are
`ds4_state_cache.c:220` and `ds4_state_cache.c:610`.

For each source endpoint, find its final64 token IDs inside the edited prompt
and extend the identical span backward. Positions are half-open token indices.
The table records the relevant old-history occurrence, excluding a repeated
anchor inside the latest query.

| Edit | Source end | Destination end | Source start | Destination start | Shift | Geometry |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A first, original-user anchor | 1371 | 869 | 807 | 305 | 502 | Even; trim one token to starts808/306 |
| B first, original-user anchor | 1373 | 870 | 808 | 305 | 503 | Odd; incompatible pair alignment |
| A second, prior-query anchor | 1653 | 1151 | 812 | 310 | 502 | Even;841 reusable tokens |
| B second, prior-query anchor | 1654 | 1151 | 813 | 310 | 503 | Odd; incompatible pair alignment |

The full original tool-result endpoints instead match as follows:

| First edit | Source end | Destination end | Shift | Fresh tokens remaining |
| --- | ---: | ---: | ---: | ---: |
| A | 2055 | 1554 | 501, odd | 99 |
| B | 2057 | 1555 | 502, even | 99 |

Both full-endpoint matches leave fewer than the configured128 fresh tokens.
The call span preceding that common tool result measures93 native sampled
tokens versus94 reconstructed tokens in each workflow. This explains the
one-token parity reversal after the call.

## Observed grafts and fallback

Actual records in `reuse/before-server.stderr.log` and
`reuse/after-server.stderr.log`:

| Request | Selected source | Selected destination | Reused | Total fresh | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| A edit1 | 808:1371 | 306:869 | 563 | 784 | Approximate graft, original source |
| B edit1 | 1466:1929 | 964:1427 | 463 | 227 | Approximate graft, original private endpoint |
| A edit2 | 812:1653 | 310:1151 | 841 | 231 | Approximate graft, approximate immutable source |
| B edit2 | — | — | 0 | — | Exact fallback: `no-aligned-source` |

B's source1929 is the private endpoint128 tokens before its2057 prompt end;
it avoids the99-fresh-token problem and selects a suffix whose actual shift is
502, even. Thus the first A/B deletion pair exercises two successful grafts,
although B's original user-block deletion is odd. B edit2 supplies the observed
odd-geometry fallback. A edit2 also demonstrates a graft after restart from an
approximate source. These are measured request geometries, not a general claim
that every odd deletion falls back or that every even deletion is reusable.
