# Fixed broader qualification inputs

All data and model output remain local under `.workspaces/`. These are sampled
task qualifications, not complete Decision Index or public vision index runs.
The toolkit is pinned at `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`.

## Four text tasks: 2000 requests

```nu
nu speed-bench/experiments/clef-native-20261024/stage_text.nu
# To recreate the four source tasks as well:
nu speed-bench/experiments/clef-native-20261024/stage_text.nu --rebuild
```

The recipe calls the upstream edition 0.2.1 sampler with seed 20260919, retaining
complete case groups and its exclusions. The resulting sample has 500 requests
each from API-Bank, BANKING77, CLINC150+OOS and When2Call. All 2000 payloads were
checked equal after splitting them into the upstream Suite's base/added files.
`--count 100` reproduces the same selection as the initial 100-request screen.

```nu
let data = '.workspaces/clef-artifacts-20261024/decision-index'
(nu speed-bench/experiments/clef-native-20261024/evaluate.nu
    http://127.0.0.1:19285 clef-flash .workspaces/clef-quality-2000
    --rows ($data | path join sample-2000.jsonl.gz)
    --suite ($data | path join sample-2000-suite))
```

This reuses the existing runner and task scorers. The manifest records actual
sample counts. The sample Suite deliberately does not have the full release's
hash; it must not be presented as a verified full edition or headline index.

## Public images: 28 BLINK requests

```nu
nu speed-bench/experiments/clef-native-20261024/stage_vision.nu
```

The unchanged upstream BLINK builder downloads about 138 MB from
`BLINK-Benchmark/BLINK@a3666eb249237ba3d5eca8db21176cc47967e040`. All 961 items
and 1535 image bytes reproduce the toolkit's pinned digests. The staging helper
uses its existing per-set builder and row converter directly because its CLI's
partial rebuild subsequently tries to load all missing vision sets.

The fixed sample selects four PNG/JPEG cases from each of seven subtasks:
counting, depth, spatial relations, multiview reasoning and semantic, visual and
functional correspondence. It has 12 single-image and 16 two-image requests.
The seed is 20260919; images remain original source bytes. The subset directory
links to the locally rebuilt images; no data is uploaded or checked in.

Use the existing vision HTTP adapter with an explicit SystemOne path. Its
`protocol=decisions` option describes the body shape; its default URL differs
from this server's OpenAI Decisions route.

```nu
let kit = '.workspaces/clef-references/decision-index'
let python = '.workspaces/clef-artifacts-20261024/venv/bin/python'
let suite = '.workspaces/clef-artifacts-20261024/vision-suite/subset-28'
$env.PYTHONPATH = $kit
(^$python -m decision_index vision run --dir $suite --no-verify --workers 1
    --engine decide --option protocol=decisions --option path=/v1/systemone
    --option base_url=http://127.0.0.1:19285 --option model=clef-flash
    --out .workspaces/clef-vision-28)
(^$python -m decision_index vision score --dir $suite
    --results .workspaces/clef-vision-28/results.jsonl --out .workspaces/clef-vision-28)
```

`--no-verify` is necessary because the selected directory has 28 rows and omits
the other ten public benchmarks. The original BLINK source rebuild does verify.
Report BLINK's sample accuracy and answered count. The scorer's aggregate field
does not become a full public vision index on this subset (`complete` is false).
The runner caps images at 1,638,400 pixels, then each model's ordinary native
processor budget applies. Record those settings when comparing implementations.

The existing offline `mlx_reference:Clef` adapter now accepts `vision=true` and
the runner's image root, using the published checkpoint processor/tower with a
FP32 joint head. It also recognizes Omni's separate loader. This path is source
prepared and syntax checked; the new media adapter still needs an actual model
run. Reference jobs need the pressure guard and an exclusive model lease, just
like native jobs. Use `--engine mlx_reference:Clef --option vision=true
--option model=gguf/clef-flash-int4`, and add this experiment directory to
`PYTHONPATH` beside the toolkit. Do not compare a model-loaded reference against
a simultaneously running native model.
