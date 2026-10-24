# Selected expert read ordering on Metal — 2026-10-23

**Decision: not adopted.** File-order sorting produced small mean timing wins,
but the unchanged control and drifting baseline endpoints show variation of
the same size. The runtime source was restored; the experimental patch is kept
only as a reproducer. No runtime switch remains.

Source control: `before-upstream-ee7e840` (`9944e786`), unchanged Metal runtime
through the upstream adaptation. Inspiration: antirez's upstream CUDA
[ee7e8405232c](https://github.com/antirez/ds4/commit/ee7e8405232c8971000cc7ba2b945b00bf6d20d3),
which orders selected expert reads and coalesces adjacent transfers. Hardware:
Apple M5 Max, 128 GiB. Model: 1017 IQ2_XXS/Q2_K imatrix GGUF. Fixed prompt:
`speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt` (34,615 bytes).

## Change and measurements

The candidate sorts only `Core.Task` records by source offset immediately before
`IO.readTasks`. Expert IDs, destination buffers and cache publication stay put.
Metal allocates per-expert gate/up/down destinations; consecutive source-plane
reads do not have adjacent destinations. Directly copying CUDA's coalescing
would therefore require a different staging layout, beyond this experiment.

Each timing bracket is A/B/B/A, one guarded job at a time. A is unchanged;
B enables the patch's presence-gated sort flag. Ordinary uses a requested
64GB cache (actual 56.88 GiB / 6136 experts), default 8-row selected tiles,
512-token setup followed by two 128-token appends. Miss uses a requested 32GB
cache, forced tile128/limit1024, one-token setup and two 128-token appends.
Append seconds are reconstructed as the sum of `prefill_tokens/prefill_tps`
from the CSV, whose throughput values are rounded; setup is excluded.

| Case | Mean A seconds | Mean B seconds | Mean time change |
| --- | ---: | ---: | ---: |
| Ordinary appends | 7.152 | 7.000 | -2.13% |
| Cache-missing appends | 6.462 | 6.351 | -1.72% |
| Unaffected 2048-token prefill control | 14.998 | 15.431 | +2.89% |

Ordinary B's mean is within 0.1% of the first A (7.00507 s); miss B's mean is
within 0.2% of the final A (6.36108 s). The apparent improvements depend on the
other, drifting A endpoints. These samples do not establish a sorting speedup.
The control also generated eight teacher-forced tokens; steady decode was
22.38 versus 21.73 tokens/s, with exact frontier logits.

Every compared 129280-value frontier is identical: ordinary512/640/768,
miss1/129/257 and control2048. This checks read destinations/publication at
those prefixes; it is not a task-quality gain. Diagnostic logs confirm actual
selected batches and cache misses. The initial automatic-cache diagnostic was
pressure-stopped; matched 64GB ordinary and 32GB miss runs completed. The
large automatic-cache/default128-row regime remains unqualified.

There was no page-cache purge: “miss” describes the expert cache, not physical
cold SSD state. Logs report logical reads and I/O wall time, not physical disk
bytes. Raw logs, CSVs, full logits, guard records and `summary.json` stay ignored
under `.workspaces/upstream-selected-read-20261023/`.

## Reproduce without changing the serving checkout

Use an isolated jj workspace, the saved patch and existing pressure guard.
The commands below build both binaries, then reverse the source patch while
retaining the candidate executable. They require the installed Q2 GGUF.

```nu
let recipe = ($env.PWD | path join speed-bench/experiments/upstream-selected-read-20261023)
let models = ($env.PWD | path join gguf)
let checkout = ($env.PWD | path join .workspaces/selected-read-repro)
jj workspace add --revision 9944e786 $checkout
cd $checkout
cp -r $recipe speed-bench/experiments/
let runner = 'speed-bench/experiments/upstream-selected-read-20261023/run.nu'
ln -s $models gguf
let work = '.workspaces/upstream-selected-read-20261023'
mkdir $work
cp speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt ($work | path join prompt.txt)
nu tools/zig-build ds4-bench --prefix .workspaces/upstream-sync-20261023/baseline
cp .workspaces/upstream-sync-20261023/baseline/bin/ds4-bench .workspaces/upstream-sync-20261023/baseline/ds4-bench
git apply ($recipe | path join candidate.patch)
nu tools/zig-build ds4-bench --prefix ($work | path join candidate-build)
git apply -R ($recipe | path join candidate.patch)
for workload in [ordinary miss] {
    for arm in [[index variant]; [0 A] [1 B] [2 B] [3 A]] {
        nu $runner $'($workload)-($arm.index)-($arm.variant)' $arm.variant $workload
    }
}
nu $runner control-A A control
nu $runner control-B B control
```

The copied runner resolves its `ROOT` to the isolated source. Keep builds and
unrelated jobs idle during timing. Add `--profile` only for
diagnostic arms, equally on A/B; it changes instrumentation. Use fresh labels
on retries because each guard directory must be new.
