#!/usr/bin/env python3
"""Combine the existing two case summaries and plot the measured tradeoff."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics as stats


def distribution(values):
    return dict(samples=values, mean=stats.mean(values), median=stats.median(values),
                minimum=min(values), maximum=max(values), sample_sd=stats.stdev(values))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--plot', type=Path)
args = parser.parse_args()
root = args.root
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['pairs'] == 5 and manifest['edits'] == 2
assert manifest['cases'] == ['ordinary', 'favorable']
arms = ['control', 'sparse', 'candidate']
labels = {'control': 'Off', 'sparse': 'Sparse1', 'candidate': 'Fuller4'}
cases = {case: json.loads((root / case / 'summary.json').read_text())
         for case in manifest['cases']}
processes = json.loads((root / 'processes/summary.json').read_text())
order = processes['order']
assert order == ['ABS', 'SBA', 'ABS', 'SBA', 'ABS']
combined = []
for index in range(5):
    combined.append(dict(pair=index + 1, **{
        arm: {name: sum(case['pairs'][index][arm][name] for case in cases.values())
              for name in ('initial_seconds', 'edited_seconds', 'whole_seconds')}
        for arm in arms}))

report = dict(manifest=manifest, order=order, cases={}, quality={}, processes=processes)
for case, pairs in [(name, value['pairs']) for name, value in cases.items()] + [('combined', combined)]:
    metrics = {}
    for name in ('initial_seconds', 'edited_seconds', 'whole_seconds'):
        values = {arm: [pair[arm][name] for pair in pairs] for arm in arms}
        comparisons = {}
        for arm in arms[1:]:
            changes = [candidate / control - 1 for control, candidate in zip(values['control'], values[arm])]
            comparisons[arm] = dict(
                paired_change=distribution(changes),
                aggregate_change=stats.mean(values[arm]) / stats.mean(values['control']) - 1,
                order_change={label: stats.mean([values[arm][i] for i in indices]) /
                              stats.mean([values['control'][i] for i in indices]) - 1
                              for label, indices in [('forward', [0, 2, 4]), ('reverse', [1, 3])]},
                slower_pairs=sum(change > 0 for change in changes))
        metrics[name] = dict(arms={arm: distribution(value) for arm, value in values.items()},
                             comparisons=comparisons)
    per_workflow = 2 if case == 'combined' else 1
    initial = metrics['initial_seconds']['arms']
    edits = metrics['edited_seconds']['arms']
    projection = {}
    for arm in arms:
        per_edit = edits[arm]['mean'] / 2
        projection[arm] = dict(initial_plus_eight_calculated_seconds=initial[arm]['mean'] + 8 * per_edit,
                               calculated_eight_change=(initial[arm]['mean'] + 8 * per_edit) /
                               (initial['control']['mean'] + 4 * edits['control']['mean']) - 1)
        if arm != 'control':
            saved = (edits['control']['mean'] - edits[arm]['mean']) / 2
            projection[arm]['calculated_break_even_edits'] = (
                (initial[arm]['mean'] - initial['control']['mean']) / saved if saved > 0 else None)
    report['cases'][case] = dict(metrics=metrics, pairs=pairs,
                                workflows_per_sample=per_workflow, calculated_projection=projection)

for arm in arms:
    rows = [row for case in cases.values() for pair in case['pairs'] for row in pair[arm]['samples']]
    assert len(rows) == 30
    report['quality'][arm] = dict(correct=sum(row['correct'] for row in rows), replies=len(rows),
                                statuses=dict(Counter(row['status'] for row in rows)),
                                answers=dict(Counter(row['answer'] for row in rows)),
                                output_tokens=[row['usage']['output_tokens'] for row in rows],
                                reasoning_tokens=[row['usage']['output_tokens_details']['reasoning_tokens'] for row in rows])
    for sample in processes['samples']:
        if sample['arm'] != arm:
            continue
        directory = root / 'processes' / f"pair-{sample['pair']}" / arm
        files = [path for path in (directory / 'cache').iterdir() if path.is_file()]
        sample['cache_file_count'] = len(files)
        sample['cache_file_logical_bytes'] = sum(path.stat().st_size for path in files)
        sample['startup_seconds'] = json.loads((directory / 'startup.json').read_text())['startup_seconds']
        transient = [row for row in sample['captures'] if row['transient']]
        sample['transient_capture_ms'] = sum(row['capture_ms'] for row in transient)
        sample['transient_publish_ms'] = sum(row['publish_ms'] for row in transient)
report['scope'] = ('Five measured samples/configuration; measured horizon is initial + two edits. '
                   'Combined values sum both workflows, not ten independent samples. '
                   'Eight-edit values are calculated from mean observed edit cost, not measured. '
                   'Correct answers are graded; no fixed-teacher NLL/KL or token-ID equivalence measurement.')
(root / 'aggregate-summary.json').write_text(json.dumps(report, indent=2) + '\n')

if args.plot:
    os.environ.setdefault('MPLCONFIGDIR', str((root / 'matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype'] = 'none'
    colors = {'control': '#3b5f8a', 'sparse': '#a87923', 'candidate': '#217b59'}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    for ax, case in zip(axes, ['ordinary', 'favorable']):
        metrics = report['cases'][case]['metrics']
        for arm in arms:
            initial = metrics['initial_seconds']['arms'][arm]
            edited = metrics['edited_seconds']['arms'][arm]
            x, y = initial['mean'], edited['mean'] / 2
            ax.scatter(initial['samples'], [value / 2 for value in edited['samples']],
                       color=colors[arm], s=18, alpha=.45, zorder=2)
            ax.errorbar(x, y, xerr=initial['sample_sd'], yerr=edited['sample_sd'] / 2,
                        marker='o', markersize=7, color=colors[arm], capsize=4, zorder=3)
            ax.annotate(labels[arm], (x, y), xytext=(7, -16 if arm == 'control' else 9),
                        textcoords='offset points', color=colors[arm], fontsize=10, weight='bold')
        ax.set_title('Ordinary serial layout' if case == 'ordinary' else 'Favorable parallel results')
        ax.set_xlabel('Initial context population (seconds)')
        ax.set_ylabel('Mean cost per edit (seconds; two measured edits)')
        ax.set_xlim(34, 86)
        ax.set_ylim(20, 44)
        ax.grid(alpha=.2)
        ax.text(.03, .04, 'Lower and left is better', transform=ax.transAxes, fontsize=9, color='#555555')
    fig.suptitle('Semantic checkpoint tradeoff — five fresh processes per configuration', fontsize=12)
    fig.supxlabel('Dots: all five rounds. Large markers: means. Whiskers: one sample SD.\n'
                  'M5 Max · 1017 IQ2_XXS/Q2_K · high reasoning · HOT512 · spacing512 · no MTP', fontsize=9)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, metadata={'Date': None})
    fig.savefig(root / 'pareto.png', dpi=150)
    plt.close(fig)

for case, data in report['cases'].items():
    print(case)
    for arm in arms:
        print(labels[arm], ', '.join(f"{name}={data['metrics'][name]['arms'][arm]['mean']:.6f}s"
                                   for name in ('initial_seconds', 'edited_seconds', 'whole_seconds')))
        if arm != 'control':
            print('  paired whole %', [round(value * 100, 3) for value in
                  data['metrics']['whole_seconds']['comparisons'][arm]['paired_change']['samples']])
    print('calculated projection', data['calculated_projection'])
print('quality', {arm: [value['correct'], value['replies']] for arm, value in report['quality'].items()})
