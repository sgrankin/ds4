#!/usr/bin/env python3
"""Use upstream Decision Index task scorers without its full-index aggregator.

The upstream index expects every benchmark. A four-task smoke sample should
report its actual task metrics, not invent a comparable headline index.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from decision_index import editions
from decision_index.scoring import added, index02
from decision_index.scoring.report import benchmark_summary, load_results
from decision_index.suite.io import Suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    suite = Suite(args.suite, '0.2.1')
    results = load_results(args.results)
    spec = index02.spec('0.2.1')
    extra_ids = {int(n) for n in spec['added']}
    base, extra = [], defaultdict(list)
    rows = list(suite.rows(apply_exclusions=True))
    for row in rows:
        (extra[row['_evaluation']['catalog_id']] if row['_evaluation']['catalog_id'] in extra_ids else base).append(row)
    metrics = {int(n): (m['name'], m['key']) for n, m in spec.get('metrics', {}).items()}
    engine = next(iter(results.values()), {}).get('engine', 'unknown')
    summary = benchmark_summary(suite, results, engine, rows=base, metrics=metrics)
    for number, records in sorted(extra.items()):
        report = added.report(number, records, results)
        summary['benchmarks'].append({k: report[k] for k in ('catalog_id', 'dataset', 'requests', 'answered', 'unsupported', 'errors', 'abstained', 'pending', 'metric', 'score', 'median_ms')})
    # All current smoke tasks are categorical. Report unrounded distribution
    # statistics, independently of each protocol's transformed confidence.
    groups = defaultdict(list)
    for row in rows:
        result = results.get(row['_evaluation']['run_id'], {})
        if result.get('status') != 'ok':
            continue
        for name, expected in row['expected'].items():
            probabilities = result['response']['answers'][name]['probabilities']
            target = str(expected)
            nll = -math.log(max(float(probabilities[target]), 1e-30))
            brier = sum((float(p) - (option == target)) ** 2 for option, p in probabilities.items())
            predicted = max(probabilities, key=probabilities.get)
            value = (nll, brier, float(probabilities[predicted]), predicted == target)
            groups[str(row['_evaluation']['catalog_id'])].append(value)
            groups['all'].append(value)
    calibration = {}
    for name, values in groups.items():
        bins = defaultdict(list)
        for value in values:
            bins[min(9, int(value[2] * 10))].append(value)
        calibration[name] = {'fields': len(values), 'mean_nll': sum(v[0] for v in values) / len(values),
                             'mean_multiclass_brier': sum(v[1] for v in values) / len(values),
                             'ece_10_equal_width_bins': sum(abs(sum(v[2] - v[3] for v in bucket)) for bucket in bins.values()) / len(values)}
    summary.update(edition='0.2.1', sample_only=True, calibration=calibration,
                   note='Four-task smoke sample; no full Decision Index score is computed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'counts': summary['counts'], 'latency_ms': summary['successful_request_latency_ms'], 'benchmarks': summary['benchmarks'], 'calibration': calibration}, indent=2))


if __name__ == '__main__':
    main()
