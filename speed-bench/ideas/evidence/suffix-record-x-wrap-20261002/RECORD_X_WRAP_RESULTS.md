# Fully fresh retrieval query and a physical wrap

Both guarded native jobs completed with return code 0. These use frozen
helpers, 64 GiB SSD cache, native resident routing off, and fixed fixtures.
The reported edit times are single paired diagnostics, including B′ and the
suffix, excluding common-prefix setup and answer generation.

| Case | Fresh edit | Reuse edit | Mean KL | Gold ΔNLL |
| --- | --- | --- | --- | --- |
| Record X, C512 / fresh128 | 40.606 s | 21.199 s | 0.00339346 | +0.0118461 |
| Left shift66, C101 / fresh1, both physical wraps | 10.188 s | 6.979 s | 0.00000425933 | −0.0000157039 |

Record X calibration is valid: the old exact context answers `6829`, while
the fresh edited and reused contexts both answer `UNKNOWN` and stop. The
entire 36-token question and assistant prefix is freshly evaluated. However,
the stale `6829` sequence logprob increases from −34.1054 to −13.1805
(about 1.54e−15 to 1.89e−6 probability). Even this 128-token fresh tail does
not establish removal of the old fact from the private cache.

The physical-wrap case reuses 100 tokens in three distinct raw copy segments
and repairs 4,000 raw, 250 compressed and 250 index keys. Both branches answer
`3141` and stop. Native byte checks confirm that 27 fresh-prefix raw rows per
layer and 1,210 compressed/index prefix rows survive unchanged. The odd
pooling state is not used in this even-endpoint case.

Both cases pass zero-displacement whole-graph/logit identity, branch-endpoint
byte identity, unchanged invalid candidates and pre-copy cancellation,
post-copy cancellation invalidation, and canonical export blocking.
Pressure stayed at level1 with no bad samples. Record X had 0.000488 GiB
maximum swap growth; the physical-wrap case had none.

Compact command, summaries, raw-row metrics, tokens, rendered prompts and
guard results are in `native-record-x-wrap/`. Full raw-logit files remain in
the persistent root artifacts; `record-x-wrap-manifest.json` records their
paths, lengths and hashes.

The successful original SECRET fresh64/fresh128 washout results and the
failed original-control prior-query run remain independently frozen in
`b7a8612a`, documented by `recovery-manifest.json`. That prior-query failure
is retained as semantic evidence with return code3, not as an infrastructure
failure. No original fixture or helper was replaced by this wording change.
