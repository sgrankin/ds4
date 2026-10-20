#!/usr/bin/env python3
"""First-attempt task wall and quality with fixed caller budgets, varying routing."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('final_analysis', HERE.parent / 'final-thinking-hour20-20261004/analyze.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
ARMS = ['native-before', 'subs3', 'subs1', 'native-after']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows, records, totals, routes, closures = [], {}, {}, {}, {}
    for arm in ARMS:
        folder = args.input / arm
        records[arm] = previous.read_json(folder / 'checks/results.json')
        log = (folder / 'server.stderr.log').read_text()
        route_lines = re.findall(r'^ds4: resident-route (.*)$', log, re.M)
        routes[arm] = {key: float(value) if '.' in value else int(value)
                      for line in route_lines for key, value in re.findall(r'(\w+)=([\d.]+)', line)}
        routes[arm]['raw_lines'] = ['ds4: resident-route ' + line for line in route_lines]
        routes[arm]['native_weight_lines'] = re.findall(r'^ds4: resident-route-native-weights .+$', log, re.M)
        closures[arm] = [dict(context=c, budget=int(b), thinking_tokens=int(t), forced=int(f), deferred=int(d))
            for c, b, t, f, d in re.findall(r'reasoning_summary id=\S+ ctx=(\S+) budget=(\d+) thinking_tokens=(\d+) forced=(\d+) deferred=(\d+)', log)]
        total = dict(wall_seconds=0, passes=0, code_passes=0, caps=0,
                     output_tokens=0, thinking_tokens=0, roles={})
        for row in records[arm]:
            flat = {k: row[k] for k in ['case', 'arm', 'wall_seconds', 'pass', 'cap', 'complete',
                'actual_read_count', 'actual_write_count', 'http_output_tokens', 'http_reasoning_tokens',
                'first_code_bytes', 'final_text_length', 'final_text']}
            flat.update(code_pass=row['grade']['correct'], tests=row['grade'].get('tests'), errors='; '.join(row['errors']))
            for reply in row['replies']:
                role = reply['role']; usage = reply['usage']; cached = usage['input_tokens_details']['cached_tokens']
                values = dict(seconds=reply['seconds'], output_tokens=usage['output_tokens'],
                    thinking_tokens=usage['output_tokens_details']['reasoning_tokens'],
                    input_tokens=usage['input_tokens'], cached_tokens=cached, fresh_tokens=usage['input_tokens'] - cached)
                flat.update({role.lower() + '_' + k: v for k, v in values.items()})
                for k, v in values.items():
                    total['roles'].setdefault(role, {})[k] = total['roles'].get(role, {}).get(k, 0) + v
            rows.append(flat)
            for k, v in dict(wall_seconds=row['wall_seconds'], passes=row['pass'], code_passes=row['grade']['correct'],
                caps=row['cap'], output_tokens=row['http_output_tokens'], thinking_tokens=row['http_reasoning_tokens']).items(): total[k] += v
        total['forced'] = sum(s['forced'] for s in closures[arm])
        total['deferred'] = sum(s['deferred'] for s in closures[arm]); totals[arm] = total
    baseline = {row['case']: row for row in records['native-before']}
    controls = {}
    for arm, tasks in records.items():
        controls[arm] = {}
        for row in tasks:
            case = row['case']; base = baseline[case]
            a = args.input / 'native-before/checks' / case; b = args.input / arm / 'checks' / case
            controls[arm][case] = dict(first_code_byte_equal=(a / 'first-solution.py').read_bytes() == (b / 'first-solution.py').read_bytes(),
                final_text_byte_equal=base['final_text'].encode() == row['final_text'].encode(),
                role_usage_equal=[r['usage'] for r in base['replies']] == [r['usage'] for r in row['replies']],
                quality_equal=base['grade'] == row['grade'] and base['pass'] == row['pass'],
                role_emitted_bytes_equal={role: previous.emitted(i, a) == previous.emitted(i, b)
                    for i, role in enumerate(['READ', 'WRITE', 'FINAL'])})
    mean = (totals['native-before']['wall_seconds'] + totals['native-after']['wall_seconds']) / 2
    result = dict(rows=rows, by_arm=totals, native_mean_wall_seconds=mean,
        change_vs_native_mean_percent={a: 100 * (t['wall_seconds'] / mean - 1) for a, t in totals.items()},
        native_after_vs_before_percent=100 * (totals['native-after']['wall_seconds'] / totals['native-before']['wall_seconds'] - 1),
        route_summaries=routes, reasoning_summaries=closures, controls_vs_native_before=controls,
        scope='Resident profile only changes. READ high/0 WRITE high/128 FINAL high/1 fixed; suffix off. Generated histories differ; no paired NLL/KL or additive speed claim.')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'results.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['by_arm', 'change_vs_native_mean_percent', 'native_after_vs_before_percent', 'controls_vs_native_before', 'route_summaries']}, indent=2))
    if args.plot:
        os.environ['MPLCONFIGDIR'] = '/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ordered = ['native-before', 'subs1', 'subs3', 'native-after']
        labels = ['Native before', 'Max1', 'Max3', 'Native after']
        for arm, label, offset in zip(ordered, labels, [(0, 40), (0, 20), (0, 20), (0, -40)]):
            t = totals[arm]; axes[0].scatter(t['wall_seconds'], t['passes'], s=65)
            axes[0].annotate(label + f"\n{t['wall_seconds']:.1f}s", (t['wall_seconds'], t['passes']),
                xytext=offset, textcoords='offset points', ha='center', fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#777'})
        axes[0].set(title='Completed-task Pareto', xlabel='Whole READ + WRITE + FINAL wall (s)', ylabel='Correct tasks, 161 checks each', ylim=(0, 2.7)); axes[0].set_yticks([0, 1, 2]); axes[0].margins(x=.3)
        axes[1].plot([0, 1, 3], [mean, totals['subs1']['wall_seconds'], totals['subs3']['wall_seconds']], marker='o')
        axes[1].scatter([0, 0], [totals['native-before']['wall_seconds'], totals['native-after']['wall_seconds']], marker='x', color='#65717e', label='Native brackets')
        axes[1].set(title='Requested replacement cap', xlabel='Requested maximum replacements (0 = off)', ylabel='Whole-task pair wall (s)'); axes[1].set_xticks([0, 1, 3]); axes[1].legend(frameon=False)
        bottom = [0.] * 4
        for role, color in [('READ', '#247ba0'), ('WRITE', '#e89930'), ('FINAL', '#40916c')]:
            values = [totals[a]['roles'][role]['seconds'] for a in ordered]
            axes[2].bar(labels, values, bottom=bottom, label=role, color=color); bottom = [b + v for b, v in zip(bottom, values)]
        axes[2].set(title='Actual HTTP role wall', ylabel='Seconds, two tasks'); axes[2].tick_params(axis='x', rotation=25); axes[2].legend(frameon=False)
        for ax in axes: ax.grid(axis='y', alpha=.15)
        fig.suptitle('Resident routing on fixed natural instructions and READ0 / WRITE128 / FINAL1 budgets', fontsize=12)
        fig.savefig(args.output / 'resident-combined-pareto.png', dpi=180); fig.savefig(args.output / 'resident-combined-pareto.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
