#!/usr/bin/env python3
"""Run existing V4.1 graph planning/ownership fixtures serially under pressure_run.py."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--alias-prompt", type=Path, help="Optional expensive prefill-alias fixture: requires >26,624 encoded tokens and a fixed 64 GiB expert cache")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-graph-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bins = {"c": args.c_bin_dir.resolve(), "native": args.native_bin_dir.resolve()}
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    sources = [
        "ds4.c", "ds4_deepseek41_graph.h", "ds4_model_config.h", "ds4_model_weights.h",
        "build.zig", "tests/test_deepseek41_graph.c", "zig/engine/model_api.h",
        "zig/engine/input.zig", "zig/engine/model.zig", "zig/engine/config.zig",
        "zig/engine/weights.zig", "zig/engine/graph.zig",
        "zig/engine/deepseek41/graph_api.h", "zig/engine/deepseek41/layout.zig",
        "zig/engine/deepseek41/resources.zig",
    ]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    result = {"scope": "Existing memory-plan and session-accounting fixtures; optional existing wide prefill-alias fixture. No benchmark speed claim.", "source_sha256": source_hashes, "runs": [], "comparisons": []}

    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    modes = [("memory-plan", []), ("session-accounting", [])]
    if args.alias_prompt:
        modes.append(("prefill-alias", [args.alias_prompt.resolve()]))
    for mode, extra in modes:
        reference = None
        for arm in ["c", "native"]:
            label = arm + "-" + mode
            binary = bins[arm] / "test-deepseek41-graph"
            command = list(map(str, [binary, args.model.resolve(), "--" + mode, *extra]))
            start = time.monotonic()
            with (out / (label + ".stdout.log")).open("wb") as stdout, (out / (label + ".stderr.log")).open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=1200)
            row = {"label": label, "command": command, "seconds": time.monotonic() - start, "returncode": process.returncode, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
            result["runs"].append(row)
            save()
            print(json.dumps(row), flush=True)
            assert process.returncode == 0, label
            output = (out / (label + ".stdout.log")).read_bytes()
            assert b"PASS" in output, label
            if reference is None:
                reference = output
            result["comparisons"].append({"label": label, "same_success_output_as_c": output == reference})
            save()
            assert output == reference, label + " C/native success output differs"
    result["source_after_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    assert result["source_after_sha256"] == source_hashes, "source changed during qualification"
    result["all_pass"] = True
    save()
    print("V4.1 graph planning/ownership qualification: PASS", flush=True)


if __name__ == "__main__":
    main()
