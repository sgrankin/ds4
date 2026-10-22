#!/bin/sh
# Run from /Volumes/Models/ds4, ONE block at a time after source/binary freeze.
# Prepared only: no command in this file has been launched by its author.
set -eu
final_dir=.workspaces/zig-qualified-20261022
target_model=gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
draft_model=gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
vision_model=gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf

# Build prerequisite ONLY (root owns compiler window; do not overlap models).
# The native fixture step is conditional on native=true. Its C oracle omits
# DS4_NATIVE_ENGINE_DSPARK and CYCLE in the fixture TU; the core stays native.
tools/zig-build -Dnative=true --prefix "$final_dir/native" dspark-c-oracle dspark-native-oracle
tools/zig-build -Dnative=true -Doptimize=Debug --prefix "$final_dir/native-debug" dspark-native-oracle

# 1. Ordinary performance bracket: 512 prefill /64 forced decode, 32GB cache,
# ctx1024/chunk512, F32 KV, replay/routing/DSpark off. Approximately 4-5min.
python3 -B speed-bench/ideas/pressure_run.py "$final_dir/guard-ordinary" -- \
  python3 -B "$final_dir/ordinary.py" \
  --control-bin-dir "$final_dir/c/bin" --native-bin-dir "$final_dir/native/bin" \
  --model "$target_model" --out "$final_dir/ordinary"

# 2. Actual six-turn Agent pair, about 7.5-8min: three context deletions,
# fresh read of permanent archive on every query, exact416/RIVER-73 gold.
python3 -B speed-bench/ideas/pressure_run.py "$final_dir/guard-agent" -- \
  python3 -B speed-bench/experiments/zig-final-20261022/agent_workflow.py \
  --model "$target_model" --binaries "$final_dir" --out "$final_dir/agent"

# 3. C writer -> native restart: concurrent two-slot decode, live SSE
# disconnect, surviving continuation, Responses/conversations durable CRUD.
# Existing complete OpenAI/Anthropic wire probes also run. About 1-3min.
python3 -B speed-bench/ideas/pressure_run.py "$final_dir/guard-http" -- \
  python3 -B speed-bench/experiments/zig-server-protocols-http-20261022/qualify_http.py \
  --server "$final_dir/c/bin/ds4-server" --restart-server "$final_dir/native/bin/ds4-server" \
  --model "$target_model" --output "$final_dir/http"

# 4. Trained16-cycle C/native/Debug comparison, about 2.5min, fixture40GB.
# Compare complete cycle/RNG records, cancellation and context clamp markers.
python3 -B speed-bench/ideas/pressure_run.py "$final_dir/guard-trained" -- \
  python3 -B "$final_dir/trained_cycle.py" \
  --control-bin-dir "$final_dir/native/bin" --native-bin-dir "$final_dir/native/bin" \
  --debug-bin-dir "$final_dir/native-debug/bin" \
  --target "$target_model" --draft "$draft_model" --out "$final_dir/trained"

# 5. Six actual image HTTP cases on each C/native server, about 5min total.
# The existing grader's historical GLM filename supports suite ds41-1017.
python3 -B speed-bench/ideas/pressure_run.py "$final_dir/guard-vision" -- \
  python3 -B .workspaces/zig-vision-qualification/launch.py \
  --model "$target_model" --vision "$vision_model" \
  --c-server "$final_dir/c/bin/ds4-server" --native-server "$final_dir/native/bin/ds4-server" \
  --c-source final-c --native-source final-native --output "$final_dir/vision"

# 6. Uncontended build cost AFTER models stop. Existing method and exact-byte
# restores; source-wide forcing includes every actual selected compiled unit.
python3 -B "$final_dir/measure_builds.py" --out .workspaces/zig-qualified-builds-20261022
