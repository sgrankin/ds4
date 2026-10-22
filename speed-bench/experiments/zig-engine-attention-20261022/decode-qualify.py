#!/usr/bin/env python3
"""Compare real scalar greedy decode through existing CLI top-logprob dumps."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-attention-decode-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {"control": args.control_bin_dir.resolve() / "ds4", "native": args.native_bin_dir.resolve() / "ds4"}
    prompt = HERE / "decode-prompt.txt"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    sources = [
        "ds4.c", "ds4_deepseek41_graph.h", "ds4_model_config.h", "ds4_model_weights.h",
        "build.zig", "zig/engine/model_api.h", "zig/engine/input.zig", "zig/engine/model.zig",
        "zig/engine/config.zig", "zig/engine/weights.zig", "zig/engine/graph.zig",
        "zig/engine/deepseek41/graph_api.h", "zig/engine/deepseek41/layout.zig",
        "zig/engine/deepseek41/resources.zig", "zig/engine/deepseek41/ops.zig",
        "zig/engine/deepseek41/attention.zig", "zig/engine/deepseek41/batch_attention.zig",
        "ds4_tp.h", "ds4_gpu_tp.h",
    ]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    result = {"scope": "Three ordered control/native scalar-greedy pairs, 32-step cap, top20 JSON. Float or token disagreement is reported, not an execution rejection. Top20 drift is not full-vocabulary KL.", "prompt": str(prompt.relative_to(ROOT)), "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(), "source_sha256": source_hashes, "runs": [], "comparisons": []}

    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    for mode, extra in [
        ("default", {}),
        ("f16", {"DS4_V41_COMPRESSED_KV_F16": "1"}),
        ("f16-legacy-gather", {"DS4_V41_COMPRESSED_KV_F16": "1", "DS4_METAL_V41_PACKED_F16_GATHER": "0"}),
    ]:
        records = []
        for arm in ["control", "native"]:
            label = mode + "-" + arm
            folder = out / label
            folder.mkdir()
            dump = folder / "logprobs.json"
            command = list(map(str, [
                binaries[arm], "--model", args.model.resolve(), "--metal", "--ssd-streaming",
                "--ssd-streaming-cache-experts", "48GB", "--ctx", "2048", "--prefill-chunk", "512",
                "--mtp-draft", "1", "--seed", "73", "--nothink", "--temp", "0", "--tokens", "32",
                "--raw-prompt", "--prompt-file", prompt, "--dump-logprobs", dump, "--logprobs-top-k", "20",
            ]))
            start = time.monotonic()
            with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, env=env | extra, stdout=stdout, stderr=stderr, timeout=600)
            row = {"label": label, "mode": mode, "arm": arm, "command": command, "env": extra, "seconds": time.monotonic() - start, "returncode": process.returncode, "binary_sha256": hashlib.sha256(binaries[arm].read_bytes()).hexdigest()}
            result["runs"].append(row)
            save()
            print(json.dumps({key: row[key] for key in ["label", "returncode", "seconds"]}), flush=True)
            assert process.returncode == 0, label
            data = json.loads(dump.read_text())
            assert 512 < data["prompt_tokens"] < 2016 and data["ctx"] == 2048 and data["top_k"] == 20, label
            assert 0 < len(data["steps"]) <= 32, label
            for step in data["steps"]:
                assert len(step["top_logprobs"]) == 20, label
                assert all(math.isfinite(score["logit"]) and math.isfinite(score["logprob"]) for score in step["top_logprobs"]), label
            row.update({"prompt_tokens": data["prompt_tokens"], "steps": len(data["steps"]), "complete_32_steps": len(data["steps"]) == 32, "selected_token_ids": [step["selected"]["id"] for step in data["steps"]], "dump_sha256": hashlib.sha256(dump.read_bytes()).hexdigest()})
            records.append(data)
            save()
        control, native = records
        shared_history = True
        first_mismatch = None
        maximum_logit = maximum_logprob = 0.0
        compared_steps = compared_scores = 0
        for index, (left, right) in enumerate(zip(control["steps"], native["steps"])):
            if shared_history:
                ltop = {score["token"]["id"]: score for score in left["top_logprobs"]}
                rtop = {score["token"]["id"]: score for score in right["top_logprobs"]}
                for token in ltop.keys() & rtop.keys():
                    maximum_logit = max(maximum_logit, abs(ltop[token]["logit"] - rtop[token]["logit"]))
                    maximum_logprob = max(maximum_logprob, abs(ltop[token]["logprob"] - rtop[token]["logprob"]))
                    compared_scores += 1
                compared_steps += 1
            if left["selected"]["id"] != right["selected"]["id"] and first_mismatch is None:
                first_mismatch = index
                shared_history = False
        if first_mismatch is None and len(control["steps"]) != len(native["steps"]):
            first_mismatch = min(len(control["steps"]), len(native["steps"]))
        result["comparisons"].append({"mode": mode, "full_json_equal": control == native, "steps_json_equal": control["steps"] == native["steps"], "selected_ids_equal": [s["selected"]["id"] for s in control["steps"]] == [s["selected"]["id"] for s in native["steps"]], "first_selected_mismatch": first_mismatch, "shared_history_compared_steps": compared_steps, "shared_history_common_top20_scores": compared_scores, "shared_history_max_abs_logit_drift": maximum_logit, "shared_history_max_abs_logprob_drift": maximum_logprob})
        save()
    result["source_after_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    assert result["source_after_sha256"] == source_hashes, "active source changed during qualification"
    result["all_valid_execution"] = True
    result["all_complete_32_steps"] = all(row["complete_32_steps"] for row in result["runs"])
    save()
    print("V4.1 scalar decode execution: PASS; identity/drift recorded in results.json", flush=True)


if __name__ == "__main__":
    main()
