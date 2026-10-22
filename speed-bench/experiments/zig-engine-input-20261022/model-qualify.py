#!/usr/bin/env python3
"""Run existing model ownership/layout fixtures serially under pressure_run.py."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
MODELS = [
    "DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf",
    "GLM-5.3-Flash-Q2.gguf",
    "Qwen3.8-Flash-Next-Q4.gguf",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-bin-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "gguf")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-input-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bins = {"c": args.c_bin_dir.resolve(), "native": args.native_bin_dir.resolve()}
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    rows = []
    source_files = [
        "ds4.c", "ds4_gguf.h", "ds4_model_config.h", "ds4_model_weights.h",
        "build.zig", "tests/test_deepseek41_gguf.c", "tests/test_qwen4_ngrams.c",
        "zig/gguf.zig", "zig/engine/model_api.h", "zig/engine/model.zig",
        "zig/engine/config.zig", "zig/engine/weights.zig", "zig/engine/input.zig",
    ]
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in source_files if (ROOT / name).exists()}
    summary = {"scope": "Existing synthetic negative/mapping fixtures plus actual three-family private/shared open-config-bind-close. No GPU, weight payload warming or generation.",
               "source_sha256": source, "runs": rows, "comparisons": []}

    def save():
        (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")

    def run(label, command):
        command = list(map(str, command))
        start = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=180)
        (out / f"{label}.stdout.log").write_bytes(process.stdout)
        (out / f"{label}.stderr.log").write_bytes(process.stderr)
        row = {"label": label, "command": command, "seconds": time.monotonic() - start,
               "returncode": process.returncode, "binary_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()}
        rows.append(row)
        save()
        print(json.dumps(row), flush=True)
        assert process.returncode == 0, label
        return process.stdout

    # Existing fixtures own their small invalid-file cases and child assertions;
    # compare their success, without requiring byte-identical system error text.
    for arm in ["c", "native"]:
        run(f"{arm}-sparse", [bins[arm] / "test-deepseek41-gguf"])
        run(f"{arm}-qwen-ngram", [bins[arm] / "test-qwen4-ngrams"])

    for filename in MODELS:
        model = args.models_dir.resolve() / filename
        reference = None
        for arm in ["c", "native"]:
            label = filename.split("-")[0] + "-" + arm
            raw = run(label, [bins[arm] / "test-deepseek41-gguf", "--model-lifecycle", model])
            records = [json.loads(line) for line in raw.splitlines()]
            identity = [record for record in records if record["kind"] == "identity"]
            timing = [record for record in records if record["kind"] == "timing"]
            assert len(identity) == len(timing) == 2, label
            assert all(record["close_reset"] for record in timing), label
            stable = [{key: value for key, value in record.items() if key not in {"pass", "shared"}} for record in identity]
            assert stable[0] == stable[1], label + " private/shared reopen changed identity"
            if reference is None:
                reference = stable
            comparison = {"label": label, "model": filename, "file_bytes": model.stat().st_size,
                          "private_shared_identity_equal": stable[0] == stable[1],
                          "matches_c_identity": stable == reference, "identity": identity, "timings": timing}
            summary["comparisons"].append(comparison)
            save()
            assert comparison["matches_c_identity"], label + " C/native identity mismatch"
    summary["source_after_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source}
    assert summary["source_sha256"] == summary["source_after_sha256"], "source changed during qualification"
    summary["all_pass"] = True
    save()
    print("Model ownership/configuration/binding qualification: PASS", flush=True)


if __name__ == "__main__":
    main()
