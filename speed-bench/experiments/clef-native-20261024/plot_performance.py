#!/usr/bin/env python3
"""Plot existing four-call Clef screens; exclude each case's first call."""
import argparse
import json
import math
import os
from pathlib import Path
import re
from statistics import median


def load(directory):
    log = (directory / 'run.log').read_text()
    tokens = [int(x) for x in re.findall(r'clef-timing: tokens=(\d+)', log)]
    times = [float(x) for x in re.findall(r'clef-smoke: request \d+, ([\d.]+) ms', log)]
    answers = [json.loads(line) for line in (directory / 'answers.jsonl').read_text().splitlines() if line.strip()]
    answers = [row for row in answers if 'answers' in row]
    assert len(times) == len(tokens) == len(answers) and len(times) % 4 == 0, directory
    return [{'tokens': tokens[i], 'ms': median(times[i+1:i+4]), 'warm_ms': times[i+1:i+4],
             'answers': answers[i+1:i+4]} for i in range(0, len(times), 4)]


def distributions(response):
    return {key: value.get('probabilities', {'true': value.get('noul'), 'false': 1-value.get('noul', 0)})
            for key, value in response['answers'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    work = parser.parse_args().work
    specifications = [
        ('full-control-open', 'full-control-open', 'Opening control'),
        ('full-wide-ready', 'full-control-open', '64×128 tile'),
        ('full-tall', 'full-control-open', '128×64 tile'),
        ('full-control-close', 'full-control-open', 'Closing control'),
        ('full-ffn', 'full-control-open', 'BF16 SwiGLU'),
        ('full-tall-ffn', 'full-control-open', '128×64 + BF16 SwiGLU'),
        ('full-rows-first', 'full-control-open', 'Rows-first traversal'),
        ('full-expanded-success', 'full-control-open', 'Expanded BF16 matrices'),
        ('full-grouped', 'full-control-open', 'Eight-row-tile groups'),
        ('flash-control', 'flash-control', 'Control'),
        ('flash-ffn', 'flash-control', 'BF16 SwiGLU'),
        ('flash-expanded', 'flash-control', 'Expanded BF16 matrices'),
        ('flash-tall-ffn', 'flash-control', '128×64 + BF16 SwiGLU'),
        ('flash-rows-first', 'flash-control', 'Rows-first traversal'),
        ('full-length-tall', 'full-length-control', '128×64 length screen'),
        ('full-length-wide', 'full-length-control', '64×128 length screen'),
        ('full-length-rows-first', 'full-length-control', 'Rows-first length screen'),
        ('full-length-expanded', 'full-length-control', 'Expanded length screen'),
        ('full-length-grouped', 'full-length-control', 'Grouped length screen'),
    ]
    runs = {}
    for tag, reference, label in specifications:
        if not (work / tag / 'run.log').exists():
            continue
        cases, controls = load(work / tag), load(work / reference)
        assert len(cases) == len(controls), tag
        rows, gaps, kls, agreement = [], [], [], []
        for case, control in zip(cases, controls):
            assert case['tokens'] == control['tokens'], tag
            rows.append({key: case[key] for key in ('tokens', 'ms', 'warm_ms')} |
                        {'control_ms': control['ms'], 'speed_ratio': control['ms']/case['ms']})
            for actual, expected in zip(case['answers'], control['answers']):
                actual, expected = distributions(actual), distributions(expected)
                assert actual.keys() == expected.keys(), tag
                for key, p in expected.items():
                    q = actual[key]
                    assert p.keys() == q.keys(), (tag, key)
                    gaps.extend(abs(p[k]-q[k]) for k in p)
                    kls.append(sum(v*math.log(v/max(q[k], 1e-300)) for k, v in p.items() if v > 0))
                    agreement.append(max(p, key=p.get) == max(q, key=q.get))
        runs[tag] = {'label': label, 'reference': reference, 'rows': rows,
                     'probability_vectors': len(kls), 'same_top': sum(agreement),
                     'max_probability_gap': max(gaps), 'mean_kl': sum(kls)/len(kls)}
    summary = {'runs': runs, 'note': 'Median of three warm calls. Ratios above one are faster. Each screen uses its named control; no confidence intervals.'}
    (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(work.resolve() / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(work.resolve() / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    for ax, family in zip(axes, ('full', 'flash')):
        tags = [f'{family}-ffn', f'{family}-tall-ffn', f'{family}-rows-first',
                'full-expanded-success' if family == 'full' else 'flash-expanded']
        for tag in tags:
            if tag not in runs:
                continue
            run = runs[tag]
            rows = sorted(run['rows'], key=lambda row: row['tokens'])
            ax.semilogx([row['tokens'] for row in rows], [row['speed_ratio'] for row in rows], 'o-', label=run['label'])
        selected = sorted([row for row in runs[tags[0]]['rows'] if row['tokens'] < 2048] +
                          [row for row in runs[tags[-1]]['rows'] if row['tokens'] >= 2048], key=lambda row: row['tokens'])
        ax.semilogx([row['tokens'] for row in selected], [row['speed_ratio'] for row in selected], 's--', linewidth=2.5, label='Selected automatic policy')
        ax.axhline(1, color='black', linewidth=1, linestyle='--')
        ax.set_xticks([160, 266, 1000, 3000, 7000], ['160', '266', '1k', '3k', '7k'])
        ax.tick_params(axis='x', which='minor', labelbottom=False)
        ax.set(title='Full Clef' if family == 'full' else 'Clef-Flash', xlabel='Actual input tokens', ylabel='Speed ratio: control / candidate')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2)
    fig.suptitle('Clef kernel screens — higher is faster; 1.0 is unchanged')
    for suffix in ('png', 'svg'):
        fig.savefig(work / f'performance.{suffix}', dpi=160)
    print(f'Wrote {len(runs)} screens, summary.json and performance.png/svg in {work}')


if __name__ == '__main__':
    main()
