#!/usr/bin/env python3
"""Five fresh paired runs: practical latency, order/drift visibility and quality."""
import importlib.util
import json
import math
from pathlib import Path
import statistics as stats
import sys

root = Path(sys.argv[1])
helper = Path(__file__).resolve().parents[1] / 'decoder-replay-hour47-20261020/analyze.py'
spec = importlib.util.spec_from_file_location('fixed_teacher', helper)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def distribution(values):
    return dict(mean=stats.mean(values), median=stats.median(values), minimum=min(values), maximum=max(values))


def quality(reference, arm):
    for key in ('vocab', 'context', 'chunk', 'cases'):
        assert reference['meta'][key] == arm['meta'][key], key
    coordinates = lambda run: [(r['case'], r['step'], r['target']) for r in run['rows']]
    assert coordinates(reference) == coordinates(arm), 'Fixed teacher histories differ'
    exact = reference['logits'].tobytes() == arm['logits'].tobytes()
    rows = [] if exact else probe.paired(reference, arm)
    answer, expected = arm['answers'][0], reference['answers'][0]
    return dict(teacher_rows=len(arm['rows']), byte_equal_rows=len(arm['rows']) if exact else sum(r['byte_equal'] for r in rows),
                mean_delta_nll=0.0 if exact else stats.mean(r['delta_nll'] for r in rows),
                mean_kl=0.0 if exact else stats.mean(r['kl'] for r in rows),
                argmax_flips=0 if exact else sum(r['argmax_flip'] for r in rows),
                same_generated_ids=answer['tokens'] == expected['tokens'],
                generated_tokens=len(answer['tokens']), correct=answer['exact'], stopped=answer['stopped'])


runs = {(pair, arm): probe.read_arm(root / f'pair-{pair}' / arm)
        for pair in range(1, 6) for arm in ('f32', 'fp4')}
reference = runs[1, 'f32']
pairs = []
for pair in range(1, 6):
    item = dict(pair=pair, order='AB' if pair % 2 else 'BA')
    for arm in ('f32', 'fp4'):
        run = runs[pair, arm]
        timing = run['timing']['favorable']
        metrics = {name: float(timing[name]) for name in ('sync_ms', 'greedy_sync_ms', 'greedy_ms', 'teacher_ms')}
        metrics['primary_ms'] = metrics['greedy_sync_ms'] + metrics['greedy_ms']
        item[arm] = dict(timing=metrics, quality=quality(reference, run))
    item['time_changes'] = {name: item['fp4']['timing'][name] / value - 1
                            for name, value in item['f32']['timing'].items()}
    pairs.append(item)

metrics = {}
for name in pairs[0]['time_changes']:
    old, new = ([p[arm]['timing'][name] for p in pairs] for arm in ('f32', 'fp4'))
    changes = [p['time_changes'][name] for p in pairs]
    metrics[name] = dict(f32_ms=distribution(old), fp4_ms=distribution(new),
                         paired_time_change=distribution(changes), packed_slower_pairs=sum(v > 0 for v in changes),
                         aggregate_time_change=stats.mean(new) / stats.mean(old) - 1,
                         geometric_time_ratio=math.exp(stats.mean(math.log(b / a) for a, b in zip(old, new))),
                         by_order={order: distribution([p['time_changes'][name] for p in pairs if p['order'] == order])
                                   for order in ('AB', 'BA')})

report = dict(samples_per_layout=5, primary_formula='greedy_sync_ms + greedy_ms',
              note='Positive change means packed slower. Five fresh processes per layout; no discarded run. '
                   'Second prefill has fresh KV but warmed experts after teachers; engine initialization is excluded. '
                   'Fixed teachers are short forced answers at a reasoning frontier; generated IDs include normal reasoning.',
              metrics=metrics, pairs=pairs)
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(primary=metrics['primary_ms'],
                      correct_runs=sum(p[a]['quality']['correct'] and p[a]['quality']['stopped'] for p in pairs for a in ('f32', 'fp4')),
                      same_generated_ids=all(p[a]['quality']['same_generated_ids'] for p in pairs for a in ('f32', 'fp4'))), indent=2))
