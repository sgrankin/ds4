# Native DeepSeek and shared stack: final qualification

The handwritten rewrite covers DeepSeek V4.1 Flash on Metal, its engine and session/cache controllers, Agent, server, CLI, benchmark, Eval and Web helper. Actual Metal objects and pipeline catalogues remain Objective-C driver operations; MSL, third-party dependencies and C TP transport remain explicit libraries. GLM/Qwen, legacy profiles and other backends retain C implementations. Main stays at `ec5424de`.

Raw run output, source/binary manifests, pressure samples and transcripts are retained locally and ignored by version control. This report and reproduction scripts are tracked; a fresh checkout does not include the raw evidence.

Complete model runs use implementation checkpoint `5e5bb690`. The final clean Make build required one include spelling fix in `expert_cache.h`: both spellings resolve to the same existing public header, but the relative spelling also works without Zig's root include directory. No declaration or runtime expression changes. Both source manifests and this exact delta are retained. Installed applications are rebuilt with that fix and separately smoke-tested.

## Correctness and practical tasks

- All 98 host/CPU fixture invocations pass across C/native ReleaseFast and Debug, plus 45 C/native/Debug GPU invocations and six diagnostic record/replay/hotlist-merge calls. First failures and their corrections are retained; no numerical tolerance was weakened. [Closing owners](../zig-metal-lifecycle-20261022/README.md).
- All four ordinary runs have identical complete 129,280-logit prefill vectors. This compares the prefill frontier, rather than every teacher-decode output.
- Both six-turn Agent runs pass three context edits, three SSD/pre-RoPE suffix grafts and three fresh tool reads of the permanent archive. Every query returns416 and `RIVER-73`. Two grafts follow earlier approximate state. Arm-specific paths produce slightly different token/graft geometry; this is a practical task pair, not a perfectly token-matched speed experiment or an immutable-original-master proof.
- C→native server restart passes all 22 recorded protocol/durability checks: actual concurrent decode, disconnect cancellation, surviving request, OpenAI/Anthropic streaming, Responses/conversation CRUD and SSD recovery.
- Trained DSpark C/native/Debug matches all 16 cycle/RNG records, plus runtime cancellation/context markers. The C oracle changes drafter/cycle implementations while retaining the selected native core.
- Actual C and native HTTP vision servers each pass all six DeepSeek image cases. This is task grading, not full embedding-byte identity.
- An installed-server image follow-up answers `Red` correctly and reports 212 cached tokens out of 225 prompt tokens, after the first 211-token image request. Exact requests, responses and usage are retained.
- Eval's three-way grading/trace/terminal checks and Web's actual local HTTP/WebSocket recovery are recorded in their [Eval](../zig-eval-frontend-20261022/README.md) and [Web](../zig-web-20261022/README.md) reports. Actual Chrome/consent and two-host TP are not qualified.

## Observed runtime

| Workload | C | Native |
| --- | ---: | ---: |
| Ordinary whole process, bracket mean |20.700s|21.061s|
| Ordinary steady decode, bracket mean |18.445tokens/s|18.305tokens/s|
| Six-turn persistent Agent |229.193s|230.237s|
| Six vision cases, ordered client time |151.610s|158.780s|

The ordinary native process mean is 1.75% longer; steady decode differs by 0.76%. C prefill rate drifts from 37.02 to 34.37 tokens/s across the bracket, so this does not establish a speedup or prove universal non-regression. Agent time differs by 0.46%; the vision ordered point is 4.73% longer. These few observations qualify this model/hardware/workload, with pressure and individual points retained.

## Build cost

Same source, Apple toolchain and 12 jobs; warm source-wide content forcing, two no-ops and two F32-controller edits. Comments restore byte-for-byte; Make timestamp barriers remain outside timers.

| Operation | Make C | Native Zig |
| --- | ---: | ---: |
| Full rebuild, one point |9.221s|11.938s|
| No-op, two-point mean |0.014s|1.025s|
| Single F32 controller edit, two-point mean |3.319s|1.092s|

Full rebuild costs 29.5% more; the measured controller edit costs 67.1% less. No-op overhead remains about one second. These are concrete retained costs, not a blanket build-performance improvement. The initial Make include failure is retained separately and excluded from successful timing points.

## Run and reproduce

```sh
tools/zig-build
./ds4-local --bin-dir "$PWD/zig-out/bin" persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

Use `--server` for the serving preset. Root-level Make executables keep their independent selection. See the [module/build guide](../../../zig/README.md) for ownership and C fallback commands. Local `summary.json` and `evidence/` retain source/binary identities, metrics, pressure, failures and compressed transcripts. The tracked [campaign commands](evidence/commands.sh) describe the serial campaign from the repository root and its workspace prerequisites; choose fresh output directories. Useful binaries, actual cache payloads and source-only build copies remain under `.workspaces/`, outside version control.
