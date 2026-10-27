#!/usr/bin/env python3
"""Summarize the fixture's five paired leaf samples, keeping warmups separate."""
import csv
import json
from pathlib import Path
import statistics as stats
import sys


def distribution(values):
    return dict(mean=stats.mean(values), median=stats.median(values),
                minimum=min(values), maximum=max(values))


def summarize(path):
    lines = [line for line in path.read_text().splitlines()
             if line.startswith('packed_activation_leaf,')]
    rows = list(csv.DictReader(lines))
    assert rows, f'No leaf samples: {path}'
    shapes = {}
    for row in rows:
        shape = tuple(int(row[key]) for key in ('rows', 'input', 'middle', 'experts'))
        entry = dict(phase=row['phase'], pair=int(row['pair']), arm=row['arm'],
                     milliseconds=float(row['milliseconds']))
        assert entry['phase'] in ('sample', 'warmup'), entry
        assert entry['arm'] in ('control', 'compressed'), entry
        assert entry['milliseconds'] > 0, entry
        shapes.setdefault(shape, []).append(entry)

    results = []
    for shape, entries in shapes.items():
        warmups = [entry for entry in entries if entry['phase'] == 'warmup']
        measured = [entry for entry in entries if entry['phase'] == 'sample']
        assert len(measured) == 10, f'Expected five pairs: {path}, {shape}'
        pairs = []
        for pair in range(1, 6):
            samples = [entry for entry in measured if entry['pair'] == pair]
            expected = ['control', 'compressed'] if pair % 2 else ['compressed', 'control']
            assert [entry['arm'] for entry in samples] == expected, (path, shape, pair, samples)
            arms = {entry['arm']: entry['milliseconds'] for entry in samples}
            pairs.append(dict(pair=pair, order='AB' if pair % 2 else 'BA',
                              control_ms=arms['control'], compressed_ms=arms['compressed'],
                              time_change_percent=(arms['compressed'] / arms['control'] - 1) * 100))
        control = [pair['control_ms'] for pair in pairs]
        compressed = [pair['compressed_ms'] for pair in pairs]
        changes = [pair['time_change_percent'] for pair in pairs]
        results.append(dict(shape=dict(zip(('rows', 'input', 'middle', 'experts'), shape)),
                            warmups=warmups, measured_pairs=pairs,
                            control_ms=distribution(control), compressed_ms=distribution(compressed),
                            paired_time_change_percent=distribution(changes),
                            aggregate_time_change_percent=(stats.mean(compressed) / stats.mean(control) - 1) * 100,
                            compressed_slower_pairs=sum(change > 0 for change in changes),
                            by_order={order: distribution([pair['time_change_percent'] for pair in pairs
                                                           if pair['order'] == order])
                                      for order in ('AB', 'BA')}))
    return dict(log=str(path), shapes=results)


if len(sys.argv) < 2:
    raise SystemExit(f'Usage: {sys.argv[0]} LEAF_LOG [LEAF_LOG ...]')
print(json.dumps(dict(samples_per_arm=5,
                      note='Positive change means compressed storage slower. Warmups are separate and excluded. '
                           'Timed region is the routed MoE call plus GPU synchronization; allocation, upload and '
                           'numerical checks are untimed. A and B mean control and compressed.',
                      logs=[summarize(Path(arg)) for arg in sys.argv[1:]]), indent=2))
