# Actual native zero-KV persistence: PASS

Frozen private frontend `73414bb7`/binary `5942e2f9…` uses unchanged production agent `7008afd6` and core `5c8ef1f3`. The agent completed all three turns in 210.113 seconds, first answering `CODE=2718` and then `CODE=3141` after editing the earlier record. Genuine approximate reuse skipped556 tokens and evaluated480 tokens fresh; the latest user query began at2585 after the source endpoint2584 and was entirely fresh. All outside/protected edit bytes were preserved, with no rejection.

At the completed-third-turn boundary the private fixture invoked the actual idle-worker save function. It wrote a named3142-token session with **zero KV payload** and printed the approximate-save notice. Saving took0.609ms. The approximate flag remained set, canonical payload bytes stayed0, checkpoint tokens/raw full-vocabulary logits/private diagnostic graph bytes remained **byte-identical**, and exact suffix-source capture remained rejected. The explicit save trace window contains no prefill/sync event. Saving did not force an exact rebuild.

This validates the actual production save function through an inspectable test frontend. It does not add or claim `/save` support in headless stdin. The original production-run raw rc3 persists separately because that earlier runner expected a save it never requested. Optional exact reopen and subsequent query were **not performed** in this grant; their separately prepared fixture remains pending.

MAIN guard completed0:202 normal samples, no adverse sample and no swap growth. See the compact raw results and `native-primary-manifest.json`. Approximate reuse/edit timing24.894s remains an unpaired diagnostic, with no end-to-end speedup claim.
