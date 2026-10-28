#!/usr/bin/env python3
"""Summarize five paired initial requests, with no edited requests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics as stats


def distribution(values):
    return dict(samples=values, mean=stats.mean(values), median=stats.median(values),
                minimum=min(values), maximum=max(values), sample_sd=stats.stdev(values))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['pairs'] == 5 and manifest['edits'] == 0
arms = ['control', 'candidate']
report = dict(manifest=manifest, cases={}, quality={})
requests = {arm: [] for arm in arms}
combined = {arm: [0.0] * 5 for arm in arms}
startup = {arm: [] for arm in arms}
for pair in range(1, 6):
    for arm in arms:
        process = root / 'processes' / f'pair-{pair}' / arm
        assert json.loads((process / 'result.json').read_text())['ok'], process
        startup[arm].append(json.loads((process / 'startup.json').read_text())['startup_seconds'])
for case in manifest['cases']:
    values = {arm: [] for arm in arms}
    for pair in range(1, 6):
        for arm in arms:
            directory = root / case / f'pair-{pair}' / arm
            rows = json.loads((directory / 'results.json').read_text())
            assert len(rows) == 1 and rows[0]['edit'] == 0, directory
            summary = json.loads((directory / 'summary.json').read_text())
            assert summary['edited_seconds'] == 0 and summary['responses'] == 1, directory
            seconds = rows[0]['seconds']
            values[arm].append(seconds)
            combined[arm][pair - 1] += seconds
            requests[arm].append(dict(case=case, pair=pair, **rows[0]))
    report['cases'][case] = dict(arms={arm: distribution(v) for arm, v in values.items()},
                                paired_change=distribution([b / a - 1 for a, b in zip(values['control'], values['candidate'])]))
for case, values in [('combined', combined)]:
    report['cases'][case] = dict(arms={arm: distribution(v) for arm, v in values.items()},
                                paired_change=distribution([b / a - 1 for a, b in zip(values['control'], values['candidate'])]))
for case in report['cases'].values():
    case['aggregate_change'] = case['arms']['candidate']['mean'] / case['arms']['control']['mean'] - 1
for arm in arms:
    rows = requests[arm]
    report['quality'][arm] = dict(correct=sum(row['correct'] for row in rows), replies=len(rows),
                                statuses=dict(Counter(row['status'] for row in rows)),
                                answers=dict(Counter(row['answer'] for row in rows)), requests=rows)
report['startup_seconds'] = {arm: distribution(v) for arm, v in startup.items()}
report['scope'] = ('Five fresh processes per arm, one initial request per workflow and no edits. '
                   'Combined samples sum the two workflows within each round. Startup is separate. '
                   'The control disables semantic checkpoints and ongoing-user suffix reuse; '
                   'ordinary prefix/suffix caching and cache-tier settings remain fixed.')
(root / 'no-edits-summary.json').write_text(json.dumps(report, indent=2) + '\n')
for case, data in report['cases'].items():
    print(case)
    for arm in arms:
        values = data['arms'][arm]
        print(f"  {arm}: {values['mean']:.3f} ± {values['sample_sd']:.3f} s "
              f"[{values['minimum']:.3f}–{values['maximum']:.3f}]")
    print(f"  Mean change: {data['aggregate_change'] * 100:+.2f}%")
    print('  Paired changes:', ', '.join(f'{value * 100:+.2f}%' for value in data['paired_change']['samples']))
print('Quality:', {arm: [value['correct'], value['replies']] for arm, value in report['quality'].items()})
