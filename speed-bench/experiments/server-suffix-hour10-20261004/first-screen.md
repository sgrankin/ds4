# Server suffix reuse: first screen

Revision `a892cf0957f0`. This completed native→reuse screen uses ordinary
full-input Responses replay, `store:false`, a stable `prompt_cache_key`,
temperature 0, effort none and cap 32. Each policy evaluates the original context
before two fact-preserving deletions and entirely fresh user questions.
All **6/6 responses completed and answered correctly**: READY, total 416, and
archive code MAPLE-42 with reserve 73.

| HTTP phase | Native seconds | Reuse seconds |
| --- | ---: | ---: |
| Source warmup | 85.101 | 87.449 |
| First edit, 48 arrival lines removed | 67.656 | 68.096 |
| Second edit, 32 maintenance lines removed | 53.939 | 9.616 |
| Both edited turns | 121.596 | 77.713 |
| All three requests, including source warmup | 206.696 | 165.161 |

The first 48-line deletion displaced the source anchor by 809 tokens, an odd
shift. Reuse logged `suffix exact fallback ... reason=no-aligned-source` and
rebuilt that request. The second edit grafted **1568 tokens** from a native
SSD source (`source_approximate=0`, `source_ssd=1`, `pre_rope=1`): source 480:2048
became destination 36:1604, with 328 fresh tokens. The answer remained correct.
This exercises an approximate graft from an exact source; it does not prove
hidden-state or raw-logit equality.

Request-time savings were **20.1% across all three requests** and **36.1% across
the edited turns**. These totals include source warmup but exclude server
startup. This is one ordered native→reuse comparison of one fixed tiny workload,
with a parity fallback retained. It is not a balanced latency estimate or a
general task-quality claim. [First summary](first-summary.json),
[native responses](native/http/result.json), [reuse responses](reuse/http/result.json).

Both guards completed with child returncode 0. Native had 203 normal samples;
reuse had 161. Every sample was pressure level 1, with zero bad samples and zero
swap growth. Swap stayed at 0.012998046875 GiB. Peak wired+compressor memory was
87.4444 GiB native and 87.4756 GiB reuse. Large allocation alone did not stop
either job. [Native guard](native-guard/result.json),
[reuse guard](reuse-guard/result.json); the corresponding `pressure.jsonl`
files retain every measurement.

The completed 47-line follow-up at `a56347a8a9a3` is reported separately in
[README.md](README.md). It has warm disk-prefix hits and two successful grafts;
its edited-turn saving is 62.20%. The cold48 figures above are unchanged and
are not pooled with that later cohort.
