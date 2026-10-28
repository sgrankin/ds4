#!/usr/bin/env python3
"""Combine five-pair checkpoint workflow summaries; optional population/edit plot."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics as stats


def distribution(values):
    return dict(samples=values, mean=stats.mean(values), median=stats.median(values),
                minimum=min(values), maximum=max(values), sample_sd=stats.stdev(values))


PREAD = re.compile(
    r'Metal streaming prefill layer pread layer=(?P<layer>\d+) tokens=(?P<tokens>\d+) '
    r'threads=\d+ ranges=\d+ bytes=(?P<planned>[\d.]+) GiB '
    r'wait=(?P<wait>[\d.]+) ms thread=(?P<worker>[\d.]+) ms '
    r'read=(?P<read>[\d.]+) GiB ok=(?P<ok>\d+)')
LAYER = re.compile(
    r'V4\.1 prefill layer=(?P<layer>\d+) pos=\d+ rows=(?P<rows>\d+) encoder_resident=\d+ '
    r'map=(?P<map>[\d.]+) engram=(?P<engram>[\d.]+) encode=(?P<encode>[\d.]+) '
    r'drain=(?P<drain>[\d.]+) seed=(?P<seed>[\d.]+) ms')
START = re.compile(r'ds4-server: (?:chat|completion) ctx=.* prompt start$')
FINISH = re.compile(r'ds4-server: (?:chat|completion) ctx=.* gen=\d+.* finish=\S+.* ([\d.]+)s$')
PROFILE_FIELDS = ('layer_records', 'layer0_sweeps', 'layer0_large_sweeps', 'layer0_small_sweeps', 'layer0_rows', 'map_ms', 'engram_ms',
                  'encode_ms', 'drain_ms', 'seed_ms', 'pread_jobs', 'planned_pread_gib',
                  'logical_pread_gib', 'pread_join_wait_ms', 'pread_worker_elapsed_ms', 'pread_failures')


def profile_metrics():
    return dict.fromkeys(PROFILE_FIELDS, 0) | dict(_last_layer=None, layers_seen=set())


def record_profile(metrics, layer, pread):
    if layer:
        number, rows = int(layer['layer']), int(layer['rows'])
        metrics['layer_records'] += 1
        metrics['layers_seen'].add(number)
        if number == 0:
            metrics['layer0_rows'] += rows
            if metrics['_last_layer'] != 0:
                metrics['layer0_sweeps'] += 1
                metrics['layer0_large_sweeps' if rows >= 256 else 'layer0_small_sweeps'] += 1
        metrics['_last_layer'] = number
        for field in ('map', 'engram', 'encode', 'drain', 'seed'):
            metrics[f'{field}_ms'] += float(layer[field])
    if pread:
        metrics['pread_jobs'] += 1
        metrics['planned_pread_gib'] += float(pread['planned'])
        metrics['logical_pread_gib'] += float(pread['read'])
        metrics['pread_join_wait_ms'] += float(pread['wait'])
        metrics['pread_worker_elapsed_ms'] += float(pread['worker'])
        metrics['pread_failures'] += int(pread['ok']) != 1


def read_profiles(path, cases, edits):
    total, requests, current = profile_metrics(), [], None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if START.search(line):
            assert current is None, f'Overlapping requests are outside this sequential campaign: {path}:{number}'
            index = len(requests)
            assert index < len(cases) * (edits + 1), (path, index)
            current = dict(case=cases[index // (edits + 1)], edit=index % (edits + 1),
                           start_line=number, metrics=profile_metrics())
            total['_last_layer'] = None
        layer, pread = LAYER.search(line), PREAD.search(line)
        record_profile(total, layer, pread)
        if current is not None:
            record_profile(current['metrics'], layer, pread)
        if finish := FINISH.search(line):
            assert current is not None, f'Finish without start: {path}:{number}'
            current.update(finish_line=number, generation_seconds=float(finish[1]))
            requests.append(current)
            current = None
    assert current is None and len(requests) == len(cases) * (edits + 1), path
    for metrics in [total] + [request['metrics'] for request in requests]:
        metrics.pop('_last_layer')
        metrics['layers_seen'] = sorted(metrics['layers_seen'])
    return dict(enabled=bool(total['layer_records'] or total['pread_jobs']), total=total, requests=requests)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--plot', type=Path)
parser.add_argument('--control-label', default='Control')
parser.add_argument('--candidate-label', default='Candidate')
args = parser.parse_args()
root = args.root
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['pairs'] == 5
arms = ['control', 'candidate']
labels = {'control': args.control_label, 'candidate': args.candidate_label}
cases = {case: json.loads((root / case / 'summary.json').read_text())
         for case in manifest['cases']}
processes = json.loads((root / 'processes/summary.json').read_text())
assert processes['order'] == ['AB', 'BA', 'AB', 'BA', 'AB']
combined = [dict(pair=index + 1, **{
    arm: {name: sum(case['pairs'][index][arm][name] for case in cases.values())
          for name in ('initial_seconds', 'edited_seconds', 'whole_seconds')}
    for arm in arms}) for index in range(5)]
report = dict(manifest=manifest, labels=labels, cases={}, quality={}, processes=processes)
for case, pairs in [(name, value['pairs']) for name, value in cases.items()] + [('combined', combined)]:
    metrics = {}
    for name in ('initial_seconds', 'edited_seconds', 'whole_seconds'):
        values = {arm: [pair[arm][name] for pair in pairs] for arm in arms}
        changes = [candidate / control - 1
                   for control, candidate in zip(values['control'], values['candidate'])]
        metrics[name] = dict(arms={arm: distribution(value) for arm, value in values.items()},
                             paired_change=distribution(changes),
                             aggregate_change=stats.mean(values['candidate']) / stats.mean(values['control']) - 1,
                             slower_pairs=sum(change > 0 for change in changes))
    report['cases'][case] = dict(metrics=metrics, pairs=pairs,
                                workflows_per_sample=len(cases) if case == 'combined' else 1)
for arm in arms:
    rows = [row for case in cases.values() for pair in case['pairs'] for row in pair[arm]['samples']]
    report['quality'][arm] = dict(correct=sum(row['correct'] for row in rows), replies=len(rows),
                                statuses=dict(Counter(row['status'] for row in rows)),
                                answers=dict(Counter(row['answer'] for row in rows)),
                                output_tokens=[row['usage']['output_tokens'] for row in rows])
    for sample in processes['samples']:
        if sample['arm'] != arm:
            continue
        directory = root / 'processes' / f"pair-{sample['pair']}" / arm
        sample['startup_seconds'] = json.loads((directory / 'startup.json').read_text())['startup_seconds']
        transient = [row for row in sample['captures'] if row['transient']]
        sample['capture_ms_total'] = sum(row['capture_ms'] for row in sample['captures'] if row['capture_ms'] >= 0)
        sample['inline_capture_count'] = sum(row['capture_ms'] < 0 for row in sample['captures'])
        sample['transient_capture_ms'] = sum(row['capture_ms'] for row in transient if row['capture_ms'] >= 0)
        sample['transient_inline_capture_count'] = sum(row['capture_ms'] < 0 for row in transient)
        sample['transient_publish_ms'] = sum(row['publish_ms'] for row in transient)
        sample['profile'] = read_profiles(Path(sample['log']), manifest['cases'], manifest['edits'])
for name in ('capture_ms_total', 'inline_capture_count', 'transient_capture_ms',
             'transient_inline_capture_count', 'transient_publish_ms'):
    processes['metrics'][name] = {
        arm: distribution([sample[name] for sample in processes['samples'] if sample['arm'] == arm])
        for arm in arms}
report['profiles'] = dict(
    enabled=any(sample['profile']['enabled'] for sample in processes['samples']),
    per_process={field: {arm: distribution([sample['profile']['total'][field]
                                          for sample in processes['samples'] if sample['arm'] == arm])
                         for arm in arms} for field in PROFILE_FIELDS},
    per_case={case: {
        phase: {field: {arm: distribution([
            sum(request['metrics'][field] for request in sample['profile']['requests']
                if request['case'] == case and (phase == 'whole' or
                    (phase == 'initial' and request['edit'] == 0) or
                    (phase == 'edits' and request['edit'] > 0)))
            for sample in processes['samples'] if sample['arm'] == arm])
            for arm in arms} for field in PROFILE_FIELDS}
        for phase in ('initial', 'edits', 'whole')} for case in manifest['cases']},
    interpretation='layer0_sweeps counts new layer-0 groups in the layer-major profile; multiple chunks '
                   'within one layer count once. Large/small classify the first layer-0 chunk at '
                   '256 rows, separating full-weight batches from short selected-cache tiles in '
                   'this fixed configuration. Inspect layers_seen for coverage; unprofiled scalar '
                   'paths are not counted. PREAD GiB are logical API read bytes rounded to two decimals '
                   'per log record, not physical SSD traffic. Join wait and layer map may overlap; '
                   'worker durations overlap GPU execution and must not be summed into wall time. '
                   'Requests span prompt-start through generation-finish, including suffix-graft work. '
                   'Process totals also retain profile events outside those markers. '
                   'capture_ms=-1 identifies capture already included in forward time, excluded from '
                   'isolated capture totals and counted separately.')
report['scope'] = ('Five measured processes per configuration, initial plus all edited requests. '
                   'Combined samples sum workflows within each round, not independent observations. '
                   'Answers are graded; this is not fixed-teacher NLL/KL or token-ID equivalence.')
(root / 'aggregate-summary.json').write_text(json.dumps(report, indent=2) + '\n')

if args.plot:
    os.environ.setdefault('MPLCONFIGDIR', str((root / 'matplotlib-cache').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((root / 'font-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype'] = 'none'
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.7),
                             constrained_layout=True, squeeze=False)
    for ax, case in zip(axes[0], cases):
        metrics = report['cases'][case]['metrics']
        for arm, color in [('control', '#3b5f8a'), ('candidate', '#217b59')]:
            initial = metrics['initial_seconds']['arms'][arm]
            edited = metrics['edited_seconds']['arms'][arm]
            edits = manifest['edits']
            x, y = initial['mean'], edited['mean'] / edits
            ax.scatter(initial['samples'], [value / edits for value in edited['samples']],
                       color=color, s=18, alpha=.45)
            ax.errorbar(x, y, xerr=initial['sample_sd'], yerr=edited['sample_sd'] / edits,
                        marker='o', markersize=7, color=color, capsize=4)
            ax.annotate(labels[arm], (x, y), xytext=((-6, 9) if arm == 'control' else (6, 9)),
                        ha=('right' if arm == 'control' else 'left'), textcoords='offset points',
                        color=color, fontsize=10)
        ax.set_title(case.capitalize())
        ax.set_xlabel('Initial population (seconds)')
        ax.set_ylabel('Mean cost per edit (seconds)')
        ax.grid(alpha=.2)
    fig.suptitle('Checkpoint capture — five processes per configuration')
    fig.supxlabel('Small dots: all rounds. Large dots: means. Whiskers: sample SD. Lower and left is better.', fontsize=9)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, metadata={'Date': None})
    plt.close(fig)
    if args.plot.suffix == '.svg':
        args.plot.write_text('\n'.join(line.rstrip() for line in args.plot.read_text().splitlines()) + '\n')

for case, data in report['cases'].items():
    print(case)
    for arm in arms:
        print(labels[arm], ', '.join(f"{name}={data['metrics'][name]['arms'][arm]['mean']:.6f}s"
                                   for name in ('initial_seconds', 'edited_seconds', 'whole_seconds')))
    print('paired whole %', [round(value * 100, 3)
                            for value in data['metrics']['whole_seconds']['paired_change']['samples']])
print('quality', {arm: [value['correct'], value['replies']] for arm, value in report['quality'].items()})
if report['profiles']['enabled']:
    print('profiles per-process means')
    for field in ('layer0_sweeps', 'logical_pread_gib', 'pread_join_wait_ms', 'drain_ms'):
        print(field, {arm: report['profiles']['per_process'][field][arm]['mean'] for arm in arms})
