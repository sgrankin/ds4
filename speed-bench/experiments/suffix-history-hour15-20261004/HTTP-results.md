(・ω・) All four completed retrieval records read the advertised policy file once. The original cap32 responses truncate before a final answer. The cap128 responses calculate the correct integers, but include explanations and therefore still fail the stored integer-only task grade.

This audit reads only the completed `http/` and `http-cap128/` raw records. Those are fresh question + actual policy tool-result controls, with `source_warmup=false` and `DS4_SERVER_CONTEXT_SUFFIX_REUSE=0`; their case names identify policy fixtures, not a measured suffix-reuse effect. Original responses and grades remain unchanged.

| Record and raw final response | Actual file read | Final HTTP / response status | Final cap / output tokens | Integer-only format | Stored `correct` / `pass` | Separate mathematical review |
| --- | --- | --- | --- | --- | --- | --- |
| [helpful-history, cap32](http/checks/helpful-history/response-1.json) | 1 × `fixtures/helpful-history/policy.json`; current multiple 8 | 200 / incomplete, `max_output_tokens` | 32 / 32 | Fail | [false / false](http/checks/helpful-history/result.json) | Correct net units 459; rounding and final integer not reached |
| [obsolete-history, cap32](http/checks/obsolete-history/response-1.json) | 1 × `fixtures/obsolete-history/policy.json`; current multiple 10 | 200 / incomplete, `max_output_tokens` | 32 / 32 | Fail | [false / false](http/checks/obsolete-history/result.json) | Correct net units 459; rounding and final integer not reached |
| [helpful-history, cap128](http-cap128/checks/helpful-history/response-1.json) | 1 × `fixtures/helpful-history/policy.json`; current multiple 8 | 200 / completed | 128 / 55 | Fail: explanation + integer | [false / false](http-cap128/checks/helpful-history/result.json) | Correct: `18 × 26 − 9 = 459`; `floor(459 / 8) × 8 = 456` |
| [obsolete-history, cap128](http-cap128/checks/obsolete-history/response-1.json) | 1 × `fixtures/obsolete-history/policy.json`; current multiple 10 | 200 / completed | 128 / 49 | Fail: explanation + integer | [false / false](http-cap128/checks/obsolete-history/result.json) | Correct: `18 × 26 − 9 = 459`; `floor(459 / 10) × 10 = 450` |

Each per-case `result.json` records `actual_read_count=1`, `read=true`, no read error, and the required path. Its read call matches the raw `response-0.json` function call (`whole=true`, `raw=true`) and the current JSON policy returned in `request-1.json`. The raw final text matches the stored result text in all four records. All stored parsed `answer` values remain `null`; no explanatory response is regraded as an integer-only pass. The requests explicitly require “reply only with the final integer.”

The two cap32 texts end at `rounding_multiple ` and `multiple ` respectively. The cap128 helpful text ends with a separate `456` after its calculation; the obsolete text ends with a separate `450` after its calculation. These support the mathematical review above, while the full response format fails. `stop_token_observed` is `null` for every reply record: cap128 completion is confirmed by the raw response status, without claiming separately observed EOS tokens.

| Record | Request 0 seconds | Request 1 seconds | Request 0 + 1 seconds | Stored whole-case seconds |
| --- | ---: | ---: | ---: | ---: |
| helpful-history, cap32 | 21.110 | 3.531 | 24.641 | 24.643 |
| obsolete-history, cap32 | 21.363 | 3.680 | 25.043 | 25.044 |
| helpful-history, cap128 | 20.976 | 4.744 | 25.721 | 25.722 |
| obsolete-history, cap128 | 21.205 | 4.617 | 25.822 | 25.823 |

Request 0 is the tool-call turn: all four return HTTP200/completed with cap128 and 108 output tokens. Request 1 is the answer turn reported in the first table. Whole-case seconds include both HTTP turns and host read/orchestration overhead; they are not final-answer-only or core edit timings.

The [original startup record](http/startup.json) reports 2.733 seconds separately from the [HTTP check runner](http/result.json), which reports 52.480 seconds and return code 3. The [cap128 startup record](http-cap128/startup.json) reports 2.728 seconds separately from its [HTTP check runner](http-cap128/result.json), which reports 54.355 seconds and return code 3. The runner values cover both cases and orchestration; startup is recorded separately. These fresh retrieval measurements do not establish a latency comparison between core fresh and suffix-reuse branches.
