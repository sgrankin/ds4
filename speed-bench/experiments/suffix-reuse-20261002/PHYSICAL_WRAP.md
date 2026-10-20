# Even displacement across both physical ring boundaries

Prepared native case: A256 / old B294 / new B228 / C101 / fresh tail1.
The 66-token left shift reuses 100 tokens from source positions [550,650)
at destination positions [484,584). The helper and core are the frozen
`5c8ef1f3` even-relocation build; no existing helper is replaced.

The raw ring has 128 slots. Its copy must split into three segments:

| Source slot | Destination slot | Rows |
| --- | --- | --- |
| 38 | 100 | 28 |
| 66 | 0 | 62 |
| 0 | 62 | 10 |

This exercises both destination and source physical wraps in one relocation.
After the mandatory fresh token, 27 fresh-prefix raw rows per layer must
remain byte-identical. Before appending the fresh token, 1,210 compressed
and 1,210 index prefix rows must remain byte-identical. Expected repair
counts are 4,000 raw keys, 250 compressed keys, and 250 index keys. The even
cache endpoint has no unfinished pooling pair.

The existing oracle additionally checks zero-displacement whole-graph/logit
byte parity, invalid candidates and pre-copy cancellation without mutation,
post-copy cancellation invalidation, canonical-checkpoint rejection for an
approximate graft, gold-versus-stale answer likelihood, and free answers.
Fresh native evaluation is the quality reference. A passing semantic result
does not establish removal of the old fact from every possible continuation.

The exact command is in `even-relocation-wrap-prepared-command.json`.
Native status: prepared only; an explicit root GPU grant is required.
