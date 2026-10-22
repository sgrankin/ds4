# Item 73 — Native V4.1 Session frontier and synchronization

The native Metal controller now owns V4.1 sync, scalar eval, invalidation,
image identity and scoped encoder residency. The actual 17-field frontier is
embedded once in C Session; the encoder lease is also one shared actual record.
Numerical Executor/Prefill/graph owners remain the existing native implementations.
CPU, other families and distributed transport retain C dispatch. Public
common-prefix/rewrite APIs remain C. This item does not complete the whole rewrite.

The eight-arm campaign passed; all four C/native prefill-frontier pairs have
byte-identical JSON, each containing 129,280 finite logits. F32/F16 ran 512 prefill
and 128 teacher-forced decode tokens; demand-workspace and replay cases ran
8,192 prefill plus eight forced tokens. The saved logits are the **prefill
boundary before decode**, not per-step decode-quality observations. These single
control/native pairs qualify behavior, not a latency improvement.

| Case | Prefill/forced | C controller wall s | Native wall s | Frontier JSON |
| --- | ---: | ---: | ---: | --- |
| f32 | 512/128 | 26.684 | 26.741 | exact |
| f16 | 512/128 | 23.209 | 22.845 | exact |
| growth | 8192/8 | 21.069 | 21.738 | exact |
| replay | 8192/8 | 19.952 | 19.660 | exact |

All use the 1017 IQ2_XXS/Q2_K imatrix GGUF, Metal, SSD streaming, 48GiB requested
expert cache and the identical prompt SHA
`4c5eb5e4d7ba8ef3586f6acad336c1325086749f67bd1ff4fbabe8ead9622952`. Ordinary cases use ctx1024/chunk512; wide cases use
ctx9216 with default chunk for demand growth, or chunk512 for replay256/selected.
Both demand runs logged workspace2048→4096. Both replay runs logged floor7936,
20 selected decoder layers and zero full-map layers. `evidence/model/results.json`
retains exact commands, mode environment, binary hashes, CSV fields and all
34 source hashes before/after; the source sets match.

The control uses the existing selected native modules with the Session
controller still C; the full C fallback was built/tested separately. Source
identity is the recorded file hashes, before the root’s final Item73 commit.

Build/fixture evidence:

- C-controller control, selected native ReleaseFast, selected native Debug, and
  full C fallback production/fixture builds completed. These observed build
  durations were not matched clean-build performance measurements.
- All four modes passed the existing model-free Session state and batch-admission
  fixtures, and their guarded GPU Session-state fixtures.
- C/native/Debug real `--session-fixture` runs passed public prefix/snapshot,
  continued decode, cancellation and bounds with
  `DS4_METAL_DISABLE_V41_LAYER_PREFILL=1`, the explicit scalar fixture setting.
- Record-only control also passed the existing vision-routing fixture.

Final serial default-policy parity results (updated only from completed records):

- `control-prefill`: rc=0, 402.883s; frontiers [257, 513, 1025, 1537, 3585]; V4.1 layer-major cancellation refuses partial state and rebuilds correctly: PASS Guard complete, child rc=0, 388 samples, levels=[1].
- `native-prefill`: rc=0, 407.884s; frontiers [257, 513, 1025, 1537, 3585]; V4.1 layer-major cancellation refuses partial state and rebuilds correctly: PASS Guard complete, child rc=0, 393 samples, levels=[1].
- `control-encoder`: rc=0, 110.392s; V4.1 resident encoder: exact logits/cache/decode and bounded cancellation/release: PASS Guard complete, child rc=0, 106 samples, levels=[1].
- `native-encoder`: rc=0, 110.504s; V4.1 resident encoder: exact logits/cache/decode and bounded cancellation/release: PASS Guard complete, child rc=0, 106 samples, levels=[1].

Both default-prefill runs verified exact logits/cache and three continued decode steps at 257/513/1025/1537/3585, live snapshot publication, partial-state snapshot refusal, cache-budget restoration and clean rebuild after cancellation. Both encoder runs verified residency eligibility/budget gates, all60 locks cleared after partial cancellation, exact logits/cache/next decode, and budget restoration after release.

Failures retained and explained:

- First native compile failed because strict POSIX bindings hide `getpagesize`,
  and the existing encoder fixture lost access to its guarded private lease and
  acquire/release functions. The fix uses `sysconf(_SC_PAGESIZE)` and moves the
  original lease record once into the real graph header. Existing fixture calls
  now delegate the actual native whole-lease actions; assertions were retained.
- The next build passed with eight unused legacy-wrapper warnings; root marked
  those compatibility wrappers unused, then rebuilt cleanly.
- The original default-policy `--session-fixture` failed its expectation that
  cancellation at two work units must publish two scalar tokens. The frozen
  pre-record binary reproduced the same assertion after all 1/2/3/127/128/129
  snapshot frontiers passed: tiny batches now run layer-major, so cancellation
  can precede any completed-token frontier. The explicit scalar arms passed;
  default prefill/encoder parity remains a separate qualification above.

Every heavy run stayed under the existing pressure guard. The eight-arm guard
finished with child rc=0,
176 samples, pressure levels [1],
maximum swap 0.007266GiB. Guard summaries and complete
raw pressure samples are retained; high resident memory is not itself failure.
All final parity guards completed; details appear above.

`evidence/final-source-freeze.json` captures the actual final source files, and
`evidence/final-source/` retains the changed native/shared-record sources.
`evidence/binary-hashes.json` identifies frozen built outputs. First build and
fixture failures, commands and raw outputs remain alongside successful attempts.
Frontier JSON evidence is gzip-compressed (mtime0); recorded frontier hashes
refer to decompressed bytes. Raw originals remain ignored in the workspace.
No binaries, expert caches, model files or build-cache trees are published.

Remaining C scope includes Engine startup/lifetime/cache admission, Session
creation/free, TP coordination, outer payload/suffix transactions, public
sampling/frontier facades and optional diagnostic/resident services. No algorithm,
approximation policy, token order or payload format was intentionally changed.
