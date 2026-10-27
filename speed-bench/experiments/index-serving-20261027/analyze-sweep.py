#!/usr/bin/env python3
"""Paired SIMD timings over one saved prefix; repeat zero remains warmup."""
import csv
import json
import math
from pathlib import Path
import statistics as stats
import sys

source, output = map(Path, sys.argv[1:])
groups = {}
for row in csv.DictReader(source.open()):
    key = row['case'], int(row['repeat'])
    groups.setdefault(key, {})[int(row['simd'])] = row
assert groups and all(set(pair) == {0, 1} for pair in groups.values()), 'Incomplete paired sweep'
cases = []
for case in sorted({key[0] for key in groups}):
    pairs = []
    for (name, repeat), pair in sorted(groups.items()):
        if name != case or repeat == 0:
            continue
        assert pair[0]['tokens'] == pair[1]['tokens'], 'Token counts differ'
        old, new = (float(pair[mode]['eval_ms']) for mode in (0, 1))
        pairs.append(dict(repeat=repeat, tokens=int(pair[0]['tokens']), control_ms=old,
                          simd_ms=new, time_reduction=1 - new / old, log_speedup=math.log(old / new)))
    assert pairs, 'No measured pairs after warmup'
    old, new = ([pair[key] for pair in pairs] for key in ('control_ms', 'simd_ms'))
    cases.append(dict(case=case, measured_pairs=len(pairs), pairs=pairs,
                      control_mean_ms=stats.mean(old), simd_mean_ms=stats.mean(new),
                      control_median_ms=stats.median(old), simd_median_ms=stats.median(new),
                      mean_paired_time_reduction=stats.mean(p['time_reduction'] for p in pairs),
                      median_paired_time_reduction=stats.median(p['time_reduction'] for p in pairs),
                      geometric_speedup=math.exp(stats.mean(p['log_speedup'] for p in pairs))))
report = dict(source=str(source), excluded_warmup_repeat=0, cases=cases)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
