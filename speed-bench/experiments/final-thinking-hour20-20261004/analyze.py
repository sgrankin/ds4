#!/usr/bin/env python3
"""Actual first-attempt FINAL-policy scores; generated histories are not paired NLL."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

ARMS = ['high0-before', 'budget64', 'none', 'high0-after', 'budget1']


def read_json(path):
    return json.loads(path.read_text())


def emitted(reply, folder):
    response = read_json(folder / f"response-{reply}.json")
    saved = read_json(folder.parent.parent / 'store' / (response['id'] + '.json'))
    last = saved['replay'][-1]
    # IDs and creation times are server metadata; these strings are emitted bytes.
    return {k: last.get(k, '') for k in ['content', 'reasoning_content', 'raw_tool_text']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lowpoint', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, totals, summaries, records, folders = [], {}, {}, {}, {}
    for arm in ARMS:
        folder = (args.lowpoint if arm == 'budget1' and args.lowpoint else args.input) / arm
        if not (folder / 'checks/results.json').exists():
            continue
        records[arm] = read_json(folder / 'checks/results.json')
        folders[arm] = folder
        log = (folder / 'server.stderr.log').read_text(errors='replace')
        summaries[arm] = [dict(id=i, context=c, budget=int(b), thinking_tokens=int(t),
                               forced=int(f), deferred=int(d))
            for i, c, b, t, f, d in re.findall(
                r'reasoning_summary id=(\S+) ctx=(\S+) budget=(\d+) thinking_tokens=(\d+) forced=(\d+) deferred=(\d+)', log)]
        total = dict(cases=0, passes=0, code_passes=0, caps=0, wall_seconds=0,
                     output_tokens=0, reasoning_tokens=0, role_seconds={},
                     role_thinking={}, role_output_tokens={}, role_cached_tokens={},
                     role_fresh_input_tokens={}, forced=sum(s['forced'] for s in summaries[arm]),
                     deferred=sum(s['deferred'] for s in summaries[arm]))
        for row in records[arm]:
            flat = {k: row[k] for k in ['case', 'arm', 'protocol', 'write_budget',
                'final_policy', 'wall_seconds', 'complete', 'cap', 'pass',
                'actual_read_count', 'actual_write_count', 'http_output_tokens',
                'http_reasoning_tokens', 'final_text_length', 'first_code_bytes']}
            flat.update(code_correct=row['grade'].get('correct', False),
                        tests_passed=row['grade'].get('tests'), final_text=row['final_text'],
                        errors='; '.join(row['errors']))
            for role in ['READ', 'WRITE', 'FINAL']:
                reply = next(r for r in row['replies'] if r['role'] == role)
                usage = reply['usage']
                thinking = usage['output_tokens_details']['reasoning_tokens']
                cached = usage['input_tokens_details']['cached_tokens']
                values = dict(seconds=reply['seconds'], thinking_tokens=thinking,
                    output_tokens=usage['output_tokens'], cached_tokens=cached,
                    fresh_input_tokens=usage['input_tokens'] - cached)
                flat.update({role.lower() + '_' + k: v for k, v in values.items()})
                flat.update({role.lower() + '_effort': reply['effort'],
                             role.lower() + '_budget': reply['budget'],
                             role.lower() + '_status': reply['response_status']})
                for key, value in values.items():
                    target = dict(seconds='role_seconds', thinking_tokens='role_thinking',
                        output_tokens='role_output_tokens', cached_tokens='role_cached_tokens',
                        fresh_input_tokens='role_fresh_input_tokens')[key]
                    total[target][role] = total[target].get(role, 0) + value
            rows.append(flat)
            for key, value in dict(cases=1, passes=row['pass'], code_passes=flat['code_correct'],
                caps=row['cap'], wall_seconds=row['wall_seconds'],
                output_tokens=row['http_output_tokens'], reasoning_tokens=row['http_reasoning_tokens']).items():
                total[key] += value
        totals[arm] = total
    result = dict(rows=rows, by_arm=totals, budget_summaries=summaries,
                  scope='FINAL policy only; READ high/0 and WRITE high/128 fixed. No paired NLL/KL.',
                  pre_final_controls={}, native_repeat={})
    baseline = {r['case']: r for r in records['high0-before']}
    for arm, tasks in records.items():
        result['pre_final_controls'][arm] = {}
        for row in tasks:
            case = row['case']; base = baseline[case]
            a = folders['high0-before'] / 'checks' / case
            b = folders[arm] / 'checks' / case
            result['pre_final_controls'][arm][case] = dict(
                first_code_byte_equal=(a / 'first-solution.py').read_bytes() == (b / 'first-solution.py').read_bytes(),
                read_emitted_bytes_equal=emitted(0, a) == emitted(0, b),
                write_emitted_bytes_equal=emitted(1, a) == emitted(1, b),
                read_write_usage_equal=[r['usage'] for r in base['replies'][:2]] == [r['usage'] for r in row['replies'][:2]],
                quality_equal=base['grade'] == row['grade'],
                final_text_equal=base['final_text'].encode() == row['final_text'].encode(),
                final_emitted_bytes_equal=emitted(2, a) == emitted(2, b))
    for row in records.get('high0-after', []):
        base = baseline[row['case']]
        result['native_repeat'][row['case']] = dict(
            final_text_equal=base['final_text'] == row['final_text'],
            role_usage_equal=[r['usage'] for r in base['replies']] == [r['usage'] for r in row['replies']],
            quality_equal=base['grade'] == row['grade'] and base['pass'] == row['pass'])
    mean = (totals['high0-before']['wall_seconds'] + totals['high0-after']['wall_seconds']) / 2
    result['native_mean_wall_seconds'] = mean
    result['wall_change_vs_native_mean_percent'] = {a: 100 * (t['wall_seconds'] / mean - 1) for a, t in totals.items()}
    result['native_after_vs_before_wall_percent'] = 100 * (totals['high0-after']['wall_seconds'] / totals['high0-before']['wall_seconds'] - 1)
    result['case_wall_change_vs_native_mean_percent'] = {}
    for case in baseline:
        case_mean = sum(r['wall_seconds'] for r in rows if r['case'] == case and r['arm'] in ['high0-before', 'high0-after']) / 2
        result['case_wall_change_vs_native_mean_percent'][case] = {
            r['arm']: 100 * (r['wall_seconds'] / case_mean - 1) for r in rows if r['case'] == case}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'results.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['by_arm', 'pre_final_controls', 'native_repeat',
          'native_mean_wall_seconds', 'wall_change_vs_native_mean_percent', 'native_after_vs_before_wall_percent']}, indent=2))
    if args.plot:
        os.environ['MPLCONFIGDIR'] = '/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        arms = list(totals)
        labels = {'high0-before': 'High/0 before', 'budget64': 'High/64', 'none': 'Effort none',
                  'high0-after': 'High/0 after', 'budget1': 'High/1 follow-up'}
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        ax = axes[0]
        for index, arm in enumerate(arms):
            t = totals[arm]; ax.scatter(t['wall_seconds'], t['passes'], s=65)
            offsets = {'high0-before': (0, 42), 'high0-after': (0, -48),
                       'budget64': (0, 18), 'none': (0, 18), 'budget1': (0, -48)}
            ax.annotate(labels[arm] + f"\n{t['wall_seconds']:.1f} s", (t['wall_seconds'], t['passes']),
                        xytext=offsets[arm], textcoords='offset points', ha='center', fontsize=9,
                        arrowprops={'arrowstyle': '-', 'color': '#777'})
        ax.set(title='Completed practical tasks', xlabel='Whole READ + WRITE + FINAL wall (s)',
               ylabel='Correct tasks (161 checks each)', ylim=(0, 2.7)); ax.set_yticks([0, 1, 2]); ax.margins(x=.16)
        for ax, key, title, ylabel in [(axes[1], 'role_seconds', 'Wall by actual role', 'Seconds, two tasks'),
                                     (axes[2], 'role_thinking', 'Generated thinking by role', 'Thinking tokens, two tasks')]:
            bottom = [0.] * len(arms)
            for role, color in [('READ', '#247ba0'), ('WRITE', '#e89930'), ('FINAL', '#40916c')]:
                values = [totals[a][key][role] for a in arms]
                ax.bar([labels[a] for a in arms], values, bottom=bottom, label=role, color=color)
                bottom = [b + v for b, v in zip(bottom, values)]
            ax.set(title=title, ylabel=ylabel, ylim=(0, max(bottom) * 1.15)); ax.tick_params(axis='x', rotation=25); ax.legend(frameon=False)
        for ax in axes: ax.grid(axis='y', alpha=.15)
        fig.suptitle('FINAL policy: stable instructions, READ high/0, WRITE high/128; suffix off', fontsize=12)
        fig.savefig(args.output / 'final-policy-pareto.png', dpi=180)
        fig.savefig(args.output / 'final-policy-pareto.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
