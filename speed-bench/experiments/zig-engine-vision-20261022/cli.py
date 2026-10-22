#!/usr/bin/env python3
"""Reuse the existing red-image REPL check with an explicit binary directory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    parser.add_argument("--sidecar", type=Path, default=ROOT / "gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf")
    parser.add_argument("--image", type=Path, default=ROOT / ".workspaces/zig-combined-20261022/vision/red.png")
    parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-engine-vision-cli-rerun")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.bin_dir.resolve() / "ds4"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
    env.pop("LINENOISE_ASSUME_TTY", None)
    profile = '(version 1)(allow default)(deny file-write* (literal ' + json.dumps(str(Path.home() / ".ds4_history")) + '))'
    command = list(map(str, ["/usr/bin/sandbox-exec", "-p", profile, binary, "--model", args.model.resolve(), "--vision", args.sidecar.resolve(), "--ssd-streaming", "--ssd-streaming-cache-experts", "32GB", "--ctx", "4096", "--prefill-chunk", "512", "--mtp-draft", "1", "--seed", "1", "--temp", "0", "--nothink", "--tokens", "16", "--system", "When shown an image, identify its main color. Reply with exactly one uppercase color word and nothing else."]))
    stdin = f"/nothink\n/read {args.image.resolve()}\nWhat was the main color? Reply only the uppercase color word.\n/exit\n".encode()
    (out / "stdin.txt").write_bytes(stdin)
    start = time.monotonic()
    with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
        process = subprocess.run(command, cwd=ROOT, input=stdin, env=env, stdout=stdout, stderr=stderr, timeout=300)
    stdout, stderr = (out / "stdout.log").read_bytes(), (out / "stderr.log").read_bytes()
    result = {"command": command, "returncode": process.returncode, "seconds": time.monotonic() - start, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "red_answers": len(re.findall(rb"\bRED\b", stdout)), "image_encoded": bool(re.search(rb"image 112x84, \d+ image tokens", stderr)), "scope": "One actual image read and contextual follow-up. Functional screen only, not a latency comparison."}
    result["pass"] = result["returncode"] == 0 and result["red_answers"] >= 2 and result["image_encoded"]
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert result["pass"]


if __name__ == "__main__":
    main()
