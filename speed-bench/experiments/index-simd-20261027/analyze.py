#!/usr/bin/env python3
"""Summarize fixture wall timings and the existing Metal CB timestamp log."""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sample = re.compile(r"index-simd sample rows=(\d+) repeat=(\d+) selection=(\w+) implementation=(\w+) dtype=(\w+) scope=(\w+) calls=(\d+)")
gpu = re.compile(r"^ds4: cb command batch: .*?, gpu (\d+) us,")
writer = csv.writer(sys.stdout)
writer.writerow(["rows", "selection", "implementation", "dtype", "scope", "wall_ms", "gpu_ms"])
for directory in map(Path, sys.argv[1:]):
    samples = defaultdict(list)
    pending = None
    for line in (directory / "checks.log").read_text().splitlines():
        if match := sample.fullmatch(line):
            assert pending is None, "missing command-buffer timestamp"
            rows, repeat, selection, implementation, dtype, scope, calls = match.groups()
            pending = ((rows, selection, implementation, dtype, scope), int(repeat), int(calls))
        elif pending and (match := gpu.search(line)):
            key, repeat, calls = pending
            if repeat:
                samples[key].append(int(match[1]) / (1000 * calls))
            pending = None
    assert pending is None, "truncated timestamp log"
    with (directory / "timings.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            key = tuple(row[name] for name in ("rows", "selection", "implementation", "dtype", "scope"))
            times = samples.get(key, [])
            if times:
                assert len(times) == 4, f"expected four measured GPU samples: {key}"
            writer.writerow([*key, row["ms"], f"{statistics.mean(times):.9f}" if times else ""])
