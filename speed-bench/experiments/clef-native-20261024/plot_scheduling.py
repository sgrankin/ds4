#!/usr/bin/env python3
"""Summarize the existing mixed-serving logs and plot measured scheduling points."""
import argparse
import json
import math
import os
from pathlib import Path
from statistics import median


def event_gaps(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    sent = next(e['at'] for e in events if e['direction'] == 'send')
    times = [e['at'] for e in events if e['direction'] == 'receive'
             and e['value']['type'].endswith('.delta') and e['value'].get('delta')]
    return max(b - a for a, b in zip([sent] + times, times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    rows = []
    probabilities = []
    for name in ('q0', 'q1', 'q4', 'q8', 'q0-closing'):
        directory = args.work / name
        result_path = next(p for p in sorted(directory.glob('client*/result.json'))
                           if json.loads(p.read_text())['pass'])
        result = json.loads(result_path.read_text())
        mixed = result['timings']['mixed']
        isolated = result['timings']['isolated']
        gaps = [event_gaps(result_path.parent / f'mixed-{i}-main.events.jsonl')
                for i in range(len(mixed))]
        row = {
            'name': name, 'layers': int(name[1:].split('-')[0]), 'repeats': len(mixed),
            'main_seconds': median(r['main_seconds'] for r in mixed),
            'helper_seconds': median(r['helper_seconds'] for r in mixed),
            'solo_main_seconds': median(r['main_seconds'] for r in isolated),
            'solo_helper_seconds': median(r['helper_seconds'] for r in isolated),
            'largest_observable_gap_seconds': median(gaps),
            'main_range': [min(r['main_seconds'] for r in mixed), max(r['main_seconds'] for r in mixed)],
            'helper_range': [min(r['helper_seconds'] for r in mixed), max(r['helper_seconds'] for r in mixed)],
            'helper_tokens': sorted({r['helper_input_tokens'] for r in mixed}),
            'main_output_tokens': sorted({r['main_output_tokens'] for r in mixed}),
        }
        for file in result_path.parent.glob('*helper.response.json'):
            response = json.loads(file.read_text())
            if 'answers' in response:
                probabilities.append(response['answers']['team']['probabilities'])
        rows.append(row)
    reference = probabilities[0]
    summary = {
        'rows': rows, 'probability_vectors': len(probabilities),
        'max_probability_gap': max(abs(p[k] - reference[k]) for p in probabilities for k in reference),
        'mean_delta_nll_billing': sum(math.log(reference['billing'] / p['billing']) for p in probabilities) / len(probabilities),
        'mean_kl_reference_to_candidate': sum(sum(reference[k] * math.log(reference[k] / p[k]) for k in reference) for p in probabilities) / len(probabilities),
        'note': 'Two repeats per point; error bars show observed ranges, not confidence intervals.'
    }
    (args.work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

    os.environ.setdefault('MPLCONFIGDIR', str(args.work.resolve() / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(args.work.resolve() / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    points = sorted(rows[:4], key=lambda r: r['main_seconds'])
    ax.plot([r['main_seconds'] for r in points], [r['helper_seconds'] for r in points], color='#94a3b8', linewidth=1)
    for row in rows:
        x, y = row['main_seconds'], row['helper_seconds']
        ax.errorbar(x, y, xerr=[[x-row['main_range'][0]], [row['main_range'][1]-x]],
                    yerr=[[y-row['helper_range'][0]], [row['helper_range'][1]-y]],
                    fmt='o' if row['layers'] else 's', capsize=4, color='#2563eb' if row['layers'] else '#475569')
        label = f"{row['layers']} layer" + ('' if row['layers'] == 1 else 's') if row['layers'] else 'One command'
        if row['name'].endswith('closing'):
            label = 'Closing control'
        offset = (8, 10) if row['name'] != 'q0-closing' else (-104, -22)
        if row['name'] == 'q0':
            offset = (-8, 34)
        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', fontsize=10,
                    ha='right' if row['name'] == 'q0' else 'left')
    ax.set(xlabel='Main model completion during overlap (seconds)',
           ylabel='Helper completion during overlap (seconds)',
           title='Flint + full Clef: GPU scheduling tradeoff\n2,881 helper tokens; same 57-token main answer')
    ax.grid(alpha=.2)
    ax.margins(x=.14, y=.35)
    for suffix in ('png', 'svg'):
        fig.savefig(args.work / f'pareto.{suffix}', dpi=160)


if __name__ == '__main__':
    main()
