# Item49: announce discovered schemas once

2026-10-21; completed 01:33 UTC, within the 01:18–02:18 budget. **All eight first
workflows passed with actual file reads. The unset default saved 57.45% on
favorable parent history and was 4.00% slower on ordinary continuation.**
Source base `4e12e4f2b6cff8594167abb22651a44957637063`; all arms used binary SHA256
`b6869f2862bb20dcbf96363115d5187ac72ffd6ebd4ad36552390db7f05e820d`.
[Item46](../incremental-tools-hour46-20261020/README.md) retains its earlier results.

## Change and comparison

DeepSeek V4.1 Responses discovery should place the discovered definitions once
at their chronological discovery position, rather than also rendering the same
raw schema JSON as the tool-result body. With a generated announcement, an
implicit schema-only result becomes `Tool discovery complete.`; explicit caller
`output`/`result` stays verbatim. Full tool definitions remain private trusted
replay metadata and the protocol `tools` field is unchanged. There is no
cross-event schema/name equality deduplication.

Incremental rendering is now the V4.1 Responses default;
`DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS=0` retains the legacy initial-schema hoist
and full implicit tools-result body. Static tools and other model syntax are
unchanged. No cache exclusion or KV-layout change belongs to this item.

Reuse the existing [Item46 driver](../incremental-tools-hour46-20261020/run.py),
with its small `u`/`0uu0` extension and one changed server binary in `0,u,u,0`
order. `0` explicitly selects legacy rendering; `u` omits
`DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS` from the server environment and exercises
the new default. Each arm runs ordinary
full-input discovery first, then favorable `previous_response_id` discovery;
both search, call a discovered reader, read a real synthetic note and answer
with archived and current facts. Both timed candidate arms therefore qualify
the unset default; no separate default run is needed. Do not duplicate the
harness or select improved retries.

Workflow wall time includes the actual client file read; R2+R3 excludes
discovery and tool time. Apple M5 Max/128 GiB, verified 1017 IQ2_XXS/Q2_K weights,
native routing, suffix reuse off, application SSD KV cache off, temperature
0/no thinking, context 16,384, chunk 2,048, batched-session 1, expert-cache target
64 GB and MTP off match the Item46 method. Shared prefixes may be warm because
ordinary precedes favorable in each server.

## Results and limits

| Case / timed scope | Legacy bracket mean | Unset default mean | Saving | Legacy drift |
|---|---:|---:|---:|---:|
| Ordinary full-input workflow |29.502 s|30.683 s|**−4.00%**|−0.363%|
| Ordinary R2+R3 |9.744 s|10.157 s|−4.23%|−1.581%|
| Favorable parent-history workflow |188.734 s|80.300 s|**57.45%**|−1.550%|
| Favorable R2+R3 |133.201 s|24.479 s|**81.62%**|−1.716%|

Against the faster legacy control, favorable workflow saving is 57.12% and
post-discovery saving 81.46%; ordinary workflow is 4.19% slower. The
[summary](summary.json) retains each arm and the formula
`saving = 1 - default_mean / legacy_mean`.

Favorable inputs were 4,485/5,651/5,757 tokens with default cached counts
0/4,539/5,723; legacy R2/R3 reused zero tokens. Ordinary R2 cached 669 in both
variants and processed 79 legacy versus 88 default fresh tokens. R3 processed
34 fresh tokens in both, with cached counts 818/827. A nine-token formatting
cost remains. Of the ordinary 1.181 s workflow gap, 0.769 s occurred in R1,
whose prompt was unchanged; the entire difference cannot be assigned to late
formatting. Item46's duplicate path added 73 tokens, but the timing difference
between those separate brackets does not isolate deduplication's contribution.

All eight first workflows passed discovered-name/namespace/path checks, read
the real fixture file, returned the expected archived and current facts, and
completed without caps. All servers exited 0. Output lengths were identical
across arms: ordinary 48/70/29 tokens, favorable 54/72/29. The default favors
persistent parent-history reuse despite the small measured ordinary-path cost;
explicit `0` remains available. This is one ordered fixture per case and one
ABBA, not general coding quality, KL, ΔNLL or whole-agent evidence. Matching
graded answers do not establish equal output probabilities.

The earlier Item46 implementation saved 46.30% favorable but cost 8.17% ordinary.
Those historical duplication results remain unchanged and separately labelled.

## Retained evidence

- [First outputs and actual read grades](evidence/abba/result.json),
  [unset-default environment](evidence/abba/02-default/execution.json) and
  [synthetic fixture](evidence/abba/fixtures/favorable/fixture.json). Each case
  retains all three request/response/wire JSONs. Private response-store
  directories are excluded from publication.
- [Guard result](evidence/abba-guard/result.json) and
  [pressure samples](evidence/abba-guard/pressure.jsonl): all 633 normal,
  maximum swap growth 3.87 MiB, peak wired plus compressed memory 98.098 GiB.
- [Build/unit commands and results](evidence/checks/checks.json),
  [build](evidence/checks/build.log) and [server checks](evidence/checks/server-tests.log)
  passed. CPU cases cover unset/1/0, non-V4.1 behavior and result preservation;
  the timed candidate servers independently exercise the actual unset default.

The [portable recipe](../../../recipes/11-incremental-tool-schemas.md) gives the
current `--order 0uu0` reproduction command using the existing guarded driver.

[Portable recipe](../../../recipes/11-incremental-tool-schemas.md).
Credit the [official DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)
for announcing schemas at discovery rather than rewriting the initial preamble.
