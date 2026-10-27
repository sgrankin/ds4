#!/usr/bin/env python3
"""Five paired integer-prefill runs; practical timing and measured approximation."""
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics as stats
import sys

root = Path(sys.argv[1])
helper = Path(__file__).resolve().parents[1] / 'decoder-replay-hour47-20261020/analyze.py'
spec = importlib.util.spec_from_file_location('fixed_teacher', helper)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def distribution(values):
    return dict(mean=stats.mean(values), median=stats.median(values), minimum=min(values), maximum=max(values))


def process_io(directory):
    stderr = (directory / 'stderr.log').read_text()
    caches = re.findall(r'^.*streaming expert cache budget=.*$', stderr, re.MULTILINE)
    assert caches, f'Missing streaming expert cache summary: {directory}'
    values = dict(re.findall(r'(\w+)=([\d.]+)', caches[-1]))
    required = ('hits', 'misses', 'live', 'miss_pread', 'pread_ms')
    assert all(key in values for key in required), f'Incomplete expert I/O summary: {caches[-1]}'
    totals = re.findall(r'^.*streaming expert timing total (.*)$', stderr, re.MULTILINE)
    assert totals, f'Missing streaming expert timing total: {directory}'
    return dict(hits=int(values['hits']), misses=int(values['misses']),
                live_gib=float(values['live']), miss_pread_gib=float(values['miss_pread']),
                pread_ms=float(values['pread_ms']),
                expert_timing_total={name: float(value) for name, value in
                                     re.findall(r'(\w+)=([\d.]+)', totals[-1])})


def quality(reference, arm):
    for key in ('vocab', 'context', 'chunk', 'cases'):
        assert reference['meta'][key] == arm['meta'][key], key
    coordinates = lambda run: [(r['case'], r['step'], r['target']) for r in run['rows']]
    assert coordinates(reference) == coordinates(arm), 'Fixed teacher histories differ'
    exact = reference['logits'].tobytes() == arm['logits'].tobytes()
    rows = [] if exact else probe.paired(reference, arm)
    expected = {a['case']: a for a in reference['answers']}
    result = {}
    for answer in arm['answers']:
        case = answer['case']
        selected = [r for r in rows if r['case'] == case]
        count = sum(r['case'] == case for r in arm['rows'])
        dnll = 0.0 if exact else stats.mean(r['delta_nll'] for r in selected)
        result[case] = dict(teacher_rows=count, byte_equal_rows=count if exact else sum(r['byte_equal'] for r in selected),
                            mean_delta_nll=dnll, teacher_perplexity_ratio=math.exp(dnll),
                            mean_kl=0.0 if exact else stats.mean(r['kl'] for r in selected),
                            max_kl=0.0 if exact else max(r['kl'] for r in selected),
                            argmax_flips=0 if exact else sum(r['argmax_flip'] for r in selected),
                            same_generated_ids=answer['tokens'] == expected[case]['tokens'],
                            generated_tokens=len(answer['tokens']), correct=answer['exact'], stopped=answer['stopped'])
    return result


runs = {(pair, arm): probe.read_arm(root / f'pair-{pair}' / arm)
        for pair in range(1, 6) for arm in ('control', 'int8')}
reference = runs[1, 'control']
cases = list(reference['timing'])
pairs = []
for pair in range(1, 6):
    item = dict(pair=pair, order='AB' if pair % 2 else 'BA')
    for arm in ('control', 'int8'):
        run = runs[pair, arm]
        timing = {}
        for case in cases:
            values = {name: float(run['timing'][case][name]) for name in ('sync_ms', 'greedy_sync_ms', 'greedy_ms', 'teacher_ms')}
            values['primary_ms'] = values['greedy_sync_ms'] + values['greedy_ms']
            timing[case] = values
        timing['all_cases'] = {name: sum(timing[c][name] for c in cases) for name in timing[cases[0]]}
        item[arm] = dict(timing=timing, quality=quality(reference, run), process_io=process_io(run['path']))
    pairs.append(item)

metrics = {}
for case in cases + ['all_cases']:
    metrics[case] = {}
    for name in pairs[0]['control']['timing'][case]:
        old, new = ([p[arm]['timing'][case][name] for p in pairs] for arm in ('control', 'int8'))
        changes = [b / a - 1 for a, b in zip(old, new)]
        metrics[case][name] = dict(control_ms=distribution(old), int8_ms=distribution(new),
                                  paired_time_change=distribution(changes), int8_slower_pairs=sum(v > 0 for v in changes),
                                  aggregate_time_change=stats.mean(new) / stats.mean(old) - 1,
                                  geometric_time_ratio=math.exp(stats.mean(math.log(b / a) for a, b in zip(old, new))),
                                  by_order={order: distribution([v for p, v in zip(pairs, changes) if p['order'] == order])
                                            for order in ('AB', 'BA')})

io_fields = ('hits', 'misses', 'live_gib', 'miss_pread_gib', 'pread_ms')
io_summary = dict(scope='Whole process: all teacher/answer prefills, teacher continuations and generated answers; not per-case or primary-only. GiB values are rounded by the native logger.',
                  arms={arm: {name: distribution([p[arm]['process_io'][name] for p in pairs])
                              for name in io_fields} for arm in ('control', 'int8')})
report = dict(samples_per_arm=5, cases=cases, primary_formula='greedy_sync_ms + greedy_ms',
              note='Positive change means INT8 slower. All five fresh processes retained. Second prefill has fresh KV '
                   'and warmed experts after teachers; initialization excluded. Fixed teachers are forced answers at '
                   'a reasoning frontier. Actual generation retains normal reasoning; divergence is measured, not rejected.',
              metrics=metrics, process_io_summary=io_summary, pairs=pairs)
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(primary={c: metrics[c]['primary_ms'] for c in cases + ['all_cases']},
                      correct_by_case={c: sum(p[a]['quality'][c]['correct'] and p[a]['quality'][c]['stopped']
                                              for p in pairs for a in ('control', 'int8')) for c in cases}), indent=2))
