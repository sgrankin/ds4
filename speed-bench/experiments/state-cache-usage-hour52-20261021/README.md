# Item52: report restored SSD prefixes in HTTP usage

2026-10-21, 02:35–02:45 UTC ten-minute cap. Source base
`129aaa0c52f81b357a11e22884df7d8834da9790`.

Item50 proved an8616-token replay restore after restart, while HTTP reported
cached_tokens0. Usage was computed before the state-cache loader ran inside
session sync. Return only the verified newly restored range:

```text
new SSD credit = restored prefix length - already usable live prefix
cached tokens += SSD credits from cold-prefix and final sync
cache writes = prompt tokens - cached tokens
```

An optional sync out-parameter carries those credits; internal and image paths
use their existing behavior. Existing live/graft credit is retained, and newly
computed prefill tokens are not counted as reads. Refresh final request usage,
progress span and trace after successful sync. No model, cache format, lookup
policy or scheduler changes belong to this correction.

Reuse the [Item50 two-process driver](../replay-persistence-hour50-20261021/serve.py)
with a strict restart assertion: first-request HTTP cached_tokens equals the
actual replay-hit length. Both real search/read/final workflows must pass with
identical first requests and distinct server PIDs. The same1017 model,
K128/selected1, native routing, ctx16384/chunk8192 and hot0 SSD cache apply.
Build/server/agent checks pass; [commands and hashes](evidence/checks/checks.json).
Completed02:38 UTC, within budget. Both first workflows pass with actual file
reads, identical first requests and distinct server PIDs. Cold R1 reports
cached0/write8618; restart restores8618 and reports cached8618/write0, with no
replay sweep. R2 reports cached8675/write88 and R3 cached8827/write31 in BOTH
processes. All six retained responses satisfy reads+writes=input. This checks
the observed disk hit and retained live appends; it is not a new latency or
quality comparison. [First outputs and usage](evidence/serve/result.json),
[summary](summary.json). All67 pressure samples normal, peak
wired+compressed99.466GiB and zero swap growth. Both servers exit normally.

```sh
make -j2 ds4-server ds4-agent ds4_test ds4_agent_test
python3 speed-bench/ideas/pressure_run.py .workspaces/item52-guard -- \
  python3 speed-bench/experiments/replay-persistence-hour50-20261021/serve.py \
  --model "$PWD/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --output .workspaces/item52-server
```
