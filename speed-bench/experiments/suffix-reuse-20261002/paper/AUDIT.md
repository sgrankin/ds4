# SCR primary-paper audit

Source: *Context Language Models*, arXiv:2609.37725v1, user-supplied PDF. PDF page numbers equal printed page numbers. Text was extracted once with Poppler `pdftotext -layout`; no rendering, model execution, or builds occurred. Poppler reported malformed-string/font warnings; all relevant paragraphs were readable.

## Paper algorithm

- Section 4.3, page 6: replacing B in A+B+C with B-prime retains cached states for unchanged C, even though those states depend on the old B. Newly inserted and appended tokens require computation. This is explicitly approximate, and preserving information from removed text is described as a possible benefit.
- Appendix B, pages 14-15: normal SGLang caching remains a token radix tree. SCR compares the current request with that session's previous request, identifies surviving spans beyond the first mismatch, selects the K longest spans, and relocates their cached K/V. K=6 is the main setting.
- Appendix B, page 15: moved keys receive rotary-position correction for their new positions; values are reused. Relocated entries use private slots belonging to the session. They are not published as moved entries in the shared radix tree. Failure to allocate private slots falls back to normal prefill.
- Appendix B, pages 15-16: surviving spans outside the K selections are recomputed. The cap limits accumulated approximation from many simultaneous relocations. A sensitivity experiment covered K in {1,2,3,6,12,64} on 64 BrowseComp-Plus questions; gains mostly saturated around 6, without observed task-accuracy degradation in that small study.
- The paper does not specify a named diff implementation, matching tie-break rules, repeated-token ambiguity resolution, a minimum surviving-span length, exact RoPE equations, or cache-block alignment requirements. Do not attribute those choices to the paper without inspecting its implementation.

## Hybrid-state policy is a second approximation

Appendix B, page 15 describes Qwen3.6-27B, with 48 linear-attention layers and 16 full-attention layers. Linear-attention layers have endpoint recurrent state rather than one cache row per token. SCR retains the state from before the context edit and continues from it; it does not replay inserted B-prime through those layers. B-prime is recomputed in full-attention layers, with its representation influencing later linear layers through their inputs. The paper does not separately describe convolution-buffer treatment. Do not read this as restoring a checkpoint at the textual edit boundary and replaying B-prime through every layer: that would be a different policy.

Appendix B, pages 16-17: SGLang stores recurrent checkpoints at request boundaries. An edit inside such a boundary can force matching back to a much earlier checkpoint. Finer message-boundary checkpoints are proposed to reduce this remaining prefill. This supports exact frontier checkpoints independently of SCR.

## Evidence boundaries

Section 5.3 / Figure 9, page 9: Qwen3.6-27B over 830 BrowseComp-Plus questions reached 60.2% accuracy under both normal and SCR serving. Compute decreased from 10.98 to 7.14 PFLOPs/question, about 35%; this is not a measured wall-clock speedup and not DS4.1 evidence. Figure 12, page 16: 5.3 of the additional 7.8 percentage points of prompt-token reuse came from stripped historical reasoning, leaving 2.5 points from other context edits. A runtime retaining prior reasoning cannot expect that portion without introducing such edits itself.

## Transfer assumptions for DS4.1 (inferences)

DS4.1 source here uses ratio-2 compression before layer 20, ratio-1 after layer 20, raw sliding-window state, four KV-owner layers (2,8,14,20), additional index owners, compressor pending state, Engram history, and model-specific rotary transforms. Ordinary attention K/V relocation alone does not specify how these states move or resume. The Qwen endpoint recurrent-state policy is not an implementation recipe for these DS4.1 states.

Treat any re-labeled or moved stale cache as session-private approximate state, and do not persist it as an exact token-keyed checkpoint. Restoration of the exact prefix and replay of the suffix remain the reference behavior.

## Narrow experiments

1. **Equal-length replacement, one suffix.** Begin with one contiguous B replacement preserving the exact number of token positions, with B ending on a ratio-2 boundary. Keep a long unchanged C spanning the raw-window and Engram history. This removes positional relocation as a variable. Explicitly define which edited B rows are recomputed and which old C/global/index/frontier states survive; copying the entire old endpoint and changing only token labels is a useful stale-state control but does not show that B-prime was ingested.
2. **Aligned deletion, one suffix.** Delete an even-sized block; initially choose a 128-token multiple to retain the ring modulo phase as well as ratio-2 pair alignment. Place both deletion edges at known compressed-row boundaries, retain sufficient C for trailing raw/Engram state, and rerotate moved keys with the actual DS4.1 transform. Move the corresponding index/cache metadata and repair totals/pending frontiers. Even perfect relocation does not make C's states exact after the deletion.
3. **Validation.** Compare no-edit control, cold rebuilt edited prompt, exact-prefix-restored edited replay, and approximate SCR. Record edit latency, full-vocabulary KL/TV and argmax changes, subsequent fixed-token continuation divergence, and answers to deliberately edited facts. Include a contrary replacement fact and a deleted secret to expose old-context leakage. Do not require bit equality for approximate SCR; require it for the no-edit/exact references. Start K=1 and a text-only non-TP, non-vision, non-speculative path before arbitrary multi-span edits.
