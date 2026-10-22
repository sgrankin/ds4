#!/usr/bin/env python3
"""Compare the existing bounded ds4-bench workload serially under pressure_run.py."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time

ROOT = Path.cwd().resolve()  # Run from repository root.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-execution-model-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {"control": args.control_bin_dir.resolve() / "ds4-bench", "native": args.native_bin_dir.resolve() / "ds4-bench"}
    prompt = ROOT / "speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    sources = [
        "ds4.c", "ds4_deepseek41_graph.h", "ds4_model_config.h", "ds4_model_weights.h",
        "build.zig", "zig/engine/model_api.h", "zig/engine/input.zig", "zig/engine/model.zig",
        "zig/engine/config.zig", "zig/engine/weights.zig", "zig/engine/graph.zig",
        "zig/engine/deepseek41/graph_api.h", "zig/engine/deepseek41/layout.zig",
        "zig/engine/deepseek41/resources.zig", "zig/engine/deepseek41/ops.zig",
        "zig/engine/deepseek41/attention.zig", "zig/engine/deepseek41/batch_attention.zig",
        "zig/engine/deepseek41/moe.zig", "zig/engine/deepseek41/executor.zig",
        "zig/engine/deepseek41/prefill.zig", "zig/engine/deepseek41/prefill_policy.zig",
        "zig/engine/deepseek41/prefill_buffers.zig", "zig/engine/deepseek41/engram_prefetch.zig",
        "zig/engine/deepseek41/streaming.zig", "zig/engine/deepseek41/expert_cache.zig",
        "zig/engine/deepseek41/session_batch.zig",
        "zig/engine/deepseek41/moe_services.h", "zig/engine/deepseek41/executor_services.h",
        "ds4_tp.h", "ds4_gpu_tp.h",
        "ds4_session_frontier.h", "zig/engine/deepseek41/frontier.zig",
        "zig/engine/deepseek41/session.zig", "zig/engine/deepseek41/encoder_residency.zig",
    ]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    result = {"scope": "Session owner: F32/F16 512-prefix/128-teacher control/native pairs; 8192/8 demand-workspace growth at context9216/default chunk; 8192/8 replay256+selected at context9216/chunk512. Full prefill frontiers compared within each case; not decode-step quality or matched timing curves.", "prompt": str(prompt.relative_to(ROOT)), "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(), "source_sha256": source_hashes, "runs": []}

    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    for mode, arms, extra, prefix_tokens, teacher_tokens, context, chunk in [
        ("f32", ["control", "native"], {"DS4_V41_COMPRESSED_KV_F16": "0"}, 512, 128, 1024, 512),
        ("f16", ["control", "native"], {"DS4_V41_COMPRESSED_KV_F16": "1"}, 512, 128, 1024, 512),
        ("growth", ["control", "native"], {"DS4_METAL_V41_DEMAND_WORKSPACE": "1"}, 8192, 8, 9216, None),
        ("replay", ["control", "native"], {"DS4_V41_DECODER_REPLAY_ROWS": "256", "DS4_V41_DECODER_REPLAY_SELECTED": "1"}, 8192, 8, 9216, 512),
    ]:
        reference = None
        for index, arm in enumerate(arms):
            label = f"{mode}-{index}-{arm}"
            folder = out / label
            folder.mkdir()
            (folder / "logits").mkdir()
            command = list(map(str, [
                binaries[arm], "--model", args.model.resolve(), "--metal", "--ssd-streaming",
                "--ssd-streaming-cache-experts", "48GB", "--prompt-file", prompt,
                "--ctx-start", prefix_tokens, "--ctx-max", prefix_tokens, "--ctx-alloc", context,
                *(("--prefill-chunk", chunk) if chunk is not None else ()), "--gen-tokens", teacher_tokens, "--teacher-forced-decode",
                "--csv", folder / "metrics.csv", "--dump-frontier-logits-dir", folder / "logits",
            ]))
            start = time.monotonic()
            with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, env=env | extra, stdout=stdout, stderr=stderr, timeout=600)
            row = {"label": label, "mode": mode, "arm": arm, "command": command, "env": extra, "seconds": time.monotonic() - start, "returncode": process.returncode, "binary_sha256": hashlib.sha256(binaries[arm].read_bytes()).hexdigest()}
            result["runs"].append(row)
            save()
            print(json.dumps({key: row[key] for key in ["label", "returncode", "seconds"]}), flush=True)
            assert process.returncode == 0, label
            files = list((folder / "logits").glob("*.json"))
            assert len(files) == 1, label
            saved = json.loads(files[0].read_text())
            logits = saved["logits"]
            assert len(logits) == saved["vocab"] and all(value is not None and math.isfinite(value) for value in logits), label
            if reference is None:
                reference = logits
            with (folder / "metrics.csv").open() as stream:
                row["metrics"] = list(csv.DictReader(stream))
            assert len(row["metrics"]) == 1, label
            metrics = row["metrics"][0]
            assert int(metrics["ctx_tokens"]) == int(metrics["prefill_tokens"]) == prefix_tokens, label
            assert int(metrics["gen_tokens"]) == teacher_tokens, label
            stderr_text = (folder / "stderr.log").read_text()
            if mode == "growth":
                assert "workspace grows 2048 -> 4096" in stderr_text, label + " did not grow workspace"
            if mode == "replay":
                assert "decoder replay rows=256" in stderr_text and "decoder replay selected_layers=" in stderr_text, label + " did not exercise selected replay"
            row["identical_logits_within_mode"] = logits == reference
            row["frontier_file_sha256"] = hashlib.sha256(files[0].read_bytes()).hexdigest()
            save()
            assert row["identical_logits_within_mode"], label + " frontier logits differ"
    result["source_after_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    assert result["source_after_sha256"] == source_hashes, "active source changed during qualification"
    result["all_pass"] = True
    save()
    print("V4.1 execution model qualification: PASS", flush=True)


if __name__ == "__main__":
    main()
