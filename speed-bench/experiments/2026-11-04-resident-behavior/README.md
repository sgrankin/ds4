# Resident routing: controlled coding behavior

All five controlled background-task trials passed the independent oracle. With the unrelated background removed, both the published and current CPU-routing arms produced no implementation, while all three GPU refresh arms passed. These trials do not establish whether GPU routing caused the earlier failure; a five-seed normal-sampling follow-up also found one GPU-arm completion failure. This experiment investigates the repetitive reasoning and missing implementation observed in the [resident package qualification](../2026-11-04-resident-agent-qualification/README.md). That comparison changed working directories along with the runtime and settings, so it did not isolate routing as the cause.

## Setup

The selected one-hour window is November 4, 2026, 18:33–19:33 Eastern. On an Apple M5 Max with 128 GiB memory, each arm runs the same DeepSeek V4.1 Flash IQ2_XXS/Q2_K checkpoint with a 64 GB expert cache. The task asks the reasoning-enabled agent to implement topological layers, including generator inputs and cycles, with temperature zero and a 2,048-token limit per model call. The agent may make multiple calls, so reaching that limit in one call does not necessarily end the task. The independent layers oracle checks the resulting file. One trial per arm describes behavior on this task; it does not establish a comparative speed or general task-success rate. Temperature zero selects greedy sampling; the normal agent default is temperature 1.0, so these results do not establish behavior under its default stochastic sampling.

The published arm uses `364ceeab` and its original settings. The current-code arms use runtime `1b692d25` and the experiment 7 prefill settings: ordinary CPU routing (artifact label `current_gpu_off`), then GPU routing with ordinary-route refresh intervals of 4, 8 and 16 positions. The GPU arms also enable address reuse and concurrent shared and routed experts. All arms run model inference on Metal; CPU routing refers to host-side expert selection and substitution. Its cap-five resident policy is already an approximation, not an unquantized or native-expert reference. Each binary runs from its matching source root. Every arm sees the same canonical empty task directory and fixed injected date and timezone; the completed directory is archived before the next arm. The system-prefix cache is fresh and isolated per arm. The OS file cache is not controlled. The driver records arguments, environment, binary hashes, output and exit codes outside the task directory.

Run the existing driver under the pressure guard with a fresh output directory:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/14-resident-behavior/guard \
  -- nu speed-bench/experiments/2026-11-04-resident-behavior/agent-check.nu
```

Raw artifacts remain local under `.workspaces/hillclimb-20261104/14-resident-behavior/agent-check/`. The driver runs arms sequentially and applies the same [layers oracle](check-layers.py) to each archived result. Use `--plain` with a fresh output directory to omit the unrelated background text.

## Routing interpretation

On ordinary-route refresh positions, the host reads the native selected IDs, applies the CPU resident policy and loads the resulting IDs; refresh does not necessarily load every native selection. The cap-five policy protects the strongest selection. On eligible intervening GPU-resident positions, the GPU instead replaces missing selected IDs with resident experts in score order, retaining the native slot weights. It can replace all six selections, including the strongest, and does not apply the CPU policy's cap, gap or rank protections. This is an intentional approximation, not by itself evidence of a weight or lifetime bug. See [route selection](../../../src/engine/deepseek41/moe.zig), [expert loading](../../../src/metal/expert_load.zig), and the [resident routing kernel](../../../src/metal/dsv41.metal).

The resident-only dispatch does not update cache hotness or last-used state, while token-based aging continues. These can affect later eviction choices, but whether they cause expert starvation or the observed behavior is unproven. The source review found no concrete weight or resource-lifetime defect. See [resident dispatch](../../../src/metal/routed_single_selected.zig) and [cache policy](../../../src/metal/expert_cache.zig).

## Findings

Each arm produced `layers.py`, exited successfully and passed all 174 oracle evaluations: 58 cases each exercised as lists, iterators and generators. These are input checks on one agent trajectory per arm, not independent agent trials. The CPU-routing arm encountered an interrupted write at a per-call limit, then recovered on a later call. The refresh-16 arm wrote and tested an implementation, then made a small cleanup and tested again; the earlier repeated sentence was not observed. The guarded run completed successfully.

The same five arms were repeated with the unrelated background removed and fresh isolated caches. The published and current CPU-routing agents both exited zero without creating `layers.py`, after repetitive planning. All three GPU refresh arms created implementations that passed all 174 checks. The guard returned 1 for the two missing files, not a pressure abort. A missing file is a bounded completion failure, not an incorrect finished algorithm. The published arm's repetitive planning resembles the prior failure but does not establish the same underlying mechanism.

The task directory, injected time, visible files and bootstrap cache conditions all changed from the earlier qualification. Their combined effect cannot be attributed to any one control. The background and plain prompts also produce different trajectories, so these single trials neither identify a GPU routing cause nor demonstrate a reliable advantage of the GPU arms. No concrete numerical or lifetime defect was identified.

A further plain-prompt comparison used temperature 1.0, top-p 1.0 and min-p 0.05, with seeds 1–5. CPU routing passed all 174 checks in all five runs. GPU refresh 16 passed in seeds 1–4; seed 5 exited zero without a file or tool call after spending its budget planning. Its final output repeated an endpoint-validation paragraph twice and explored alternate algorithms before truncating in thought; it did not repeat the exact sentence from the earlier qualification. This one failure is retained alongside the nine completed runs. All ten used the same task, path, fixed time and fresh per-run caches. Order alternated CPU-first on seeds 1, 3 and 5 and GPU-first on seeds 2 and 4. These results cover one coding problem, two prompt forms and, for the plain prompt, five sampled seeds. They do not establish general coding accuracy or comparative performance.

All planned sampled runs completed. The guards recorded pressure level 1 throughout, zero bad samples and swap between 28.8293457 and 28.829834 GiB. The background guard returned zero; the plain and sampled guards returned 1 for oracle failures, with no pressure abort.

## Decision

Keep GPU resident routing opt-in and leave the presets and installed binaries unchanged. Completion varied across prompt contexts and sampling settings; one normal-sampling run exhausted its budget without a file. The evidence does not isolate a routing cause. No concrete weight or lifetime defect was identified. Return to kernel profiling, with single-lane resident selection as the next target and identical routing semantics as the constraint.

## Sampled reproduction

The local ignored `.workspaces/hillclimb-20261104/14-resident-behavior/sampled-check.nu` driver reads the completed plain-arm manifests and runs the two current-code arms in the order above. It changes temperature to 1.0, adds the seed, and retains the default top-p and min-p; all other task arguments and per-arm settings are inherited from the plain runs. It archives each task directory and uses a fresh system-prefix cache for each run. The local `sampled-check/` manifests, logs and `oracle-result.json` retain every result. For a fresh reproduction, use fresh output and guard directories and run the driver under `speed-bench/ideas/pressure_run.py`; the historical driver refuses to overwrite its existing output and includes a 19:23 Eastern cutoff that must be updated for a later run.
