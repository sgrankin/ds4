#!/usr/bin/env python3
"""Read paired profile.nu outputs; same 8K favorable task, no inference execution.

python3 speed-bench/experiments/integer-prefill-20261027/analyze-profile.py OUTPUT
GPU/expert totals cover both prefills plus teacher and normal-reasoning decode.
Encode/drain are wall times, not independent CPU/GPU utilization measurements.
"""
import collections
import csv
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
result = {}
stages = re.compile(r"V4\.1 prefill layer=(\d+) pos=\d+ rows=\d+ encoder_resident=\d+ "
                    r"map=([\d.]+) engram=([\d.]+) encode=([\d.]+) drain=([\d.]+) seed=([\d.]+) ms")
fields = ("map_ms", "engram_ms", "encode_wall_ms", "drain_wall_ms", "seed_ms")
for arm in ("control", "int8"):
    directory = root / arm
    with (directory / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["case"] == "favorable", rows
    assert rows[0]["rows"] == "0" and rows[0]["prompt_tokens"] == "8192", rows
    timing = {name: float(value) for name, value in rows[0].items() if name != "case"}
    timing["primary_ms"] = timing["greedy_sync_ms"] + timing["greedy_ms"]
    meta = json.loads((directory / "tokens.json").read_text())
    assert meta["thinking"] and meta["context"] == 16384 and meta["chunk"] == 8192 and meta["max_new"] == 384
    gpu = collections.defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for line in (directory / "timeline.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "E":
            duration = float(parts[5])
            assert duration >= 0, line
            gpu[parts[-1]]["calls"] += int(parts[8])
            gpu[parts[-1]]["gpu_ms"] += duration / 1000
    assert gpu, f"Missing GPU timestamps: {directory}"
    invoked = any("kernel_mul_mm_id_iq2_xxs_mpp_i8" in name for name in gpu)
    assert invoked == (arm == "int8"), f"Wrong GPU compute path: {directory}"
    groups = collections.defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for name, item in gpu.items():
        group = ("IQ2 routed prefill" if "kernel_mul_mm_id_iq2_xxs" in name else
                 "Q2 routed prefill" if "kernel_mul_mm_id_q2_K" in name else
                 "index" if "indexer" in name else
                 "attention" if "attn" in name or "attention" in name else "other")
        groups[group]["calls"] += item["calls"]
        groups[group]["gpu_ms"] += item["gpu_ms"]
    stderr = (directory / "stderr.log").read_text()
    creation = collections.defaultdict(float)
    for name, milliseconds in re.findall(r"Metal pipeline create (\S+) ([\d.]+) ms", stderr):
        creation[name] += float(milliseconds)
    host = collections.defaultdict(lambda: dict(records=0, **dict.fromkeys(fields, 0.0)))
    for layer, *values in stages.findall(stderr):
        host[layer]["records"] += 1
        for name, value in zip(fields, values):
            host[layer][name] += float(value)
    assert host, f"Missing native prefill profile: {directory}"
    expert = re.findall(r"streaming expert timing total (.*)", stderr)
    assert expert, f"Missing native expert timing summary: {directory}"
    expert = {name: float(value) for name, value in re.findall(r"(\w+)=([\d.]+)", expert[-1])}
    answers = [json.loads(line) for line in (directory / "answers.jsonl").read_text().splitlines()]
    result[arm] = dict(timing=timing, answers=answers,
        gpu_total_ms=sum(item["gpu_ms"] for item in gpu.values()),
        gpu_by_group=dict(groups),
        gpu_by_kernel=dict(sorted(gpu.items(), key=lambda item: -item[1]["gpu_ms"])),
        prefill_wall_by_layer=dict(sorted(host.items(), key=lambda item: int(item[0]))),
        prefill_wall_totals={name: sum(item[name] for item in host.values()) for name in fields},
        expert_timing_total=expert, pipeline_create_total_ms=sum(creation.values()),
        pipeline_create_ms=dict(sorted(creation.items())))
print(json.dumps(dict(primary_formula="greedy_sync_ms + greedy_ms", arms=result,
    note="Profiled scheduling differs. Compare paired anatomy; expert/GPU totals span both prefills and decode. Encode includes host work/I/O, drain is wait wall time."), indent=2))
