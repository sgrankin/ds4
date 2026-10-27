"""Summarize the paired prefill profile; standard-library numerical reduction.

Run profile-prefill.nu BINARY OUTPUT under the existing pressure guard, then:
    python3 speed-bench/experiments/index-serving-20261027/analyze-prefill.py OUTPUT

Encoder boundaries perturb execution. Compare kernel anatomy, not headline speed;
pipeline creation includes initialization and is not confined to the prefill window.
"""
import collections
import csv
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
result = {}
for arm in ("f32", "fp4"):
    directory = root / arm
    gpu = collections.defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for line in (directory / "timeline.txt").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "E":
            duration_us = float(fields[5])
            assert duration_us >= 0, line
            item = gpu[fields[-1]]
            item["calls"] += int(fields[8])
            item["gpu_ms"] += duration_us / 1000
    assert gpu, f"No GPU timestamp records: {directory}"
    creation = collections.defaultdict(float)
    for name, milliseconds in re.findall(r"Metal pipeline create (\S+) ([\d.]+) ms", (directory / "stderr.log").read_text()):
        creation[name] += float(milliseconds)
    with (directory / "timings.csv").open() as stream:
        benchmark = list(csv.DictReader(stream))
    assert len(benchmark) == 1 and benchmark[0]["gen_tokens"] == "0", benchmark
    result[arm] = {
        "benchmark": benchmark[0],
        "gpu_total_ms": sum(item["gpu_ms"] for item in gpu.values()),
        "gpu_by_kernel": dict(sorted(gpu.items(), key=lambda pair: -pair[1]["gpu_ms"])),
        "pipeline_create_total_ms": sum(creation.values()),
        "pipeline_create_ms": dict(sorted(creation.items())),
        "note": "Encoder pass boundaries perturb timings; use paired attribution, not absolute speed claims.",
    }
print(json.dumps(result, indent=2))
