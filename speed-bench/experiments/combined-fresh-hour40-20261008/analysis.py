#!/usr/bin/env python3
"""Collect fresh-task practical results; plot only after all latency runs end."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORDER = ('code-native-before', 'code-combined', 'code-suffix-only', 'code-combined-repeat', 'code-native-after')


def read(path):
    return json.loads(path.read_text())


def csv_out(directory, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def saving(candidate, control):
    return 100 * (1 - candidate / control)


def footer(lines, kind):
    matches = [line.split('ds4: ' + kind + ' ', 1)[1] for line in lines
               if 'ds4: ' + kind + ' ' in line]
    if not matches:
        return {}
    return {key: float(value) if '.' in value else int(value) for key, value in
            re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', matches[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    output = HERE / 'analysis'
    output.mkdir(exist_ok=True)
    result = dict(coding={}, comparisons={}, pressure=[], pending=[])
    task_rows, role_rows, graft_rows, route_rows = [], [], [], []
    for name in ORDER:
        directory = HERE / name
        source = directory / 'launcher-summary.json'
        if not source.exists():
            result['pending'].append(name)
            continue
        launcher = read(source)
        cases, grafts, fallbacks, routing = {}, [], [], {}
        for task_source in sorted(directory.glob('http/*/*/result.json')):
            case = task_source.parent.name
            task = read(task_source)
            cases[case] = task
            grade = task['grade']
            task_rows.append(dict(arm=name, case=case, wall_seconds=task['wall_seconds'],
                complete=task['complete'], cap=task['cap'], correct=grade.get('correct'),
                tests=grade.get('tests'), passed=grade.get('passed', 0), failed=grade.get('failed', 0),
                read_count=task['actual_read_count'], write_count=task['actual_write_count'],
                config_unchanged=task['config_unchanged'], errors=json.dumps(task['errors']),
                grading_error=json.dumps(grade.get('reason')), first_failure=json.dumps(grade.get('first_failure'))))
            for reply in task['replies']:
                usage = reply['usage']
                role_rows.append(dict(arm=name, case=case, role=reply['role'], seconds=reply['seconds'],
                    budget=reply['budget'], status=reply['response_status'],
                    input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                    cached_tokens=usage.get('input_tokens_details', {}).get('cached_tokens'),
                    fresh_tokens=usage.get('input_tokens_details', {}).get('cache_write_tokens'),
                    thinking_tokens=usage.get('output_tokens_details', {}).get('reasoning_tokens')))
            log = directory / f'{case}-server.stderr.log'
            lines = log.read_text().splitlines()
            routing[case] = {kind: footer(lines, kind) for kind in ('resident-route', 'resident-prefill')}
            for kind, counters in routing[case].items():
                route_rows.append(dict(arm=name, case=case, kind=kind, **counters))
            for line in lines:
                if 'suffix exact fallback' in line:
                    fallbacks.append(dict(case=case, line=line))
                if 'suffix APPROXIMATE' not in line:
                    continue
                fields = dict(re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)', line))
                row = dict(arm=name, case=case, line=line, fields=fields)
                if 'destination' in fields:
                    row.update(destination_start=int(fields['destination'].split(':')[0]),
                        destination_end=int(fields['destination'].split(':')[1]),
                        source_start=int(fields['source'].split(':')[0]))
                    row['shift_destination_minus_source'] = row['destination_start'] - row['source_start']
                if 'latest_user_start' in fields:
                    row['latest_user_start'] = int(fields['latest_user_start'])
                    row['latest_user_outside_graft'] = row['destination_end'] <= row['latest_user_start']
                grafts.append(row)
                graft_rows.append(dict(**{k: v for k, v in row.items() if k != 'fields'}, fields=json.dumps(fields)))
        expected = {phase['phase'] for phase in launcher['phases']}
        result['pending'].extend(name + '/' + case for case in sorted(expected - cases.keys()))
        result['coding'][name] = dict(launcher=launcher, cases=cases, grafts=grafts,
            exact_fallbacks=fallbacks, routing=routing,
            task_seconds=sum(task['wall_seconds'] for task in cases.values()),
            complete=sum(task['complete'] for task in cases.values()),
            caps=sum(task['cap'] for task in cases.values()),
            correct=sum(bool(task['grade'].get('correct')) for task in cases.values()),
            tests=sum(task['grade'].get('tests', 0) for task in cases.values()),
            passed=sum(task['grade'].get('passed', 0) for task in cases.values()),
            failed=sum(task['grade'].get('failed', 0) for task in cases.values()),
            output_tokens=sum(task['http_output_tokens'] for task in cases.values()),
            reasoning_tokens=sum(task['http_reasoning_tokens'] for task in cases.values()))
    if {'code-native-before', 'code-native-after'} <= result['coding'].keys():
        before, after = [result['coding'][label] for label in ('code-native-before', 'code-native-after')]
        controls = [before['task_seconds'], after['task_seconds']]
        result['native_control_drift_percent'] = 100 * (controls[1] / controls[0] - 1)
        result['native_control_mean_seconds'] = statistics.mean(controls)
        result['native_control_faster_seconds'] = min(controls)
        for label in ('code-combined', 'code-suffix-only', 'code-combined-repeat'):
            if label not in result['coding']:
                continue
            candidate = result['coding'][label]
            comparison = dict(task_saving_percent=saving(candidate['task_seconds'], statistics.mean(controls)),
                task_saving_vs_faster_control=saving(candidate['task_seconds'], min(controls)), cases={})
            for case, task in candidate['cases'].items():
                per_case = [arm['cases'][case]['wall_seconds'] for arm in (before, after)]
                comparison['cases'][case] = dict(saving_percent=saving(task['wall_seconds'], statistics.mean(per_case)),
                    saving_vs_faster_control=saving(task['wall_seconds'], min(per_case)))
            result['comparisons'][label] = comparison
        result['native_equality'], result['source_context_equality'], result['edited_context_equality'] = {}, {}, {}
        for case in before['cases']:
            tasks = [arm['cases'][case] for arm in (before, after)]
            codes = [HERE / label / 'http' / case / case / 'first-solution.py'
                     for label in ('code-native-before', 'code-native-after')]
            result['native_equality'][case] = dict(first_code_equal=all(path.exists() for path in codes)
                and codes[0].read_bytes() == codes[1].read_bytes(),
                role_usage_equal=[reply['usage'] for reply in tasks[0]['replies']]
                    == [reply['usage'] for reply in tasks[1]['replies']])
            for field, basename in (('source_context_equality', 'context.before.txt'),
                                    ('edited_context_equality', 'context.after.txt')):
                paths = [HERE / label / 'http' / case / case / basename for label in result['coding']]
                result[field][case] = all(path.exists() for path in paths) and all(
                    path.read_bytes() == paths[0].read_bytes() for path in paths)
    if {'code-combined', 'code-suffix-only'} <= result['coding'].keys():
        result['combined_vs_suffix_only_saving_percent'] = saving(result['coding']['code-combined']['task_seconds'],
            result['coding']['code-suffix-only']['task_seconds'])
    if {'code-combined', 'code-combined-repeat'} <= result['coding'].keys():
        primary = result['coding']['code-combined']['task_seconds']
        repeat = result['coding']['code-combined-repeat']['task_seconds']
        result['combined_repeat_change_percent'] = 100 * (repeat / primary - 1)
    for directory in sorted(HERE.glob('guard-*')):
        if not (directory / 'result.json').exists():
            result['pending'].append(directory.name)
            continue
        guard = read(directory / 'result.json')
        initial = read(directory / 'command.json')['initial']
        samples = [json.loads(line) for line in (directory / 'pressure.jsonl').read_text().splitlines()]
        result['pressure'].append(dict(guard=directory.name, status=guard['status'],
            child_returncode=guard['child_returncode'], samples=len(samples),
            adverse_samples=sum(bool(sample['reason']) for sample in samples),
            levels=sorted({sample['level'] for sample in samples}),
            peak_wired_compressor_gib=max(sample['wired_gb'] + sample['compressor_gb'] for sample in samples),
            initial_swap_gib=initial['swap_gb'], peak_swap_gib=max(sample['swap_gb'] for sample in samples),
            max_swap_growth_mib=1024 * (max(sample['swap_gb'] for sample in samples) - initial['swap_gb'])))
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, rows in (('tasks.csv', task_rows), ('roles.csv', role_rows), ('grafts.csv', graft_rows),
                       ('routing.csv', route_rows), ('pressure.csv', result['pressure'])):
        csv_out(output, name, rows)
    if args.plot and result['comparisons']:
        os.environ['MPLCONFIGDIR'] = str(ROOT / '.workspaces/bench-plots-venv/mpl-cache')
        os.environ['XDG_CACHE_HOME'] = str(ROOT / '.workspaces/recipe-20261008/font-cache')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        labels = [name for name in ORDER if name in result['coding']]
        colors = ['#777' if 'native' in name else '#3777ab' if 'combined' in name else '#ca8a2b' for name in labels]
        bars = axes[0].bar(range(len(labels)), [result['coding'][name]['task_seconds'] for name in labels], color=colors)
        for bar, label in zip(bars, labels):
            item = result['coding'][label]
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{item['passed']}/{item['tests']}\ncaps {item['caps']}", ha='center', va='bottom', fontsize=8)
        axes[0].set_xticks(range(len(labels)), [name.replace('code-', '') for name in labels], rotation=15, ha='right')
        axes[0].set(ylabel='Two-task wall (seconds)', title='Actual READ / edit / WRITE / FINAL')
        axes[0].set_ylim(0, max(bar.get_height() for bar in bars) * 1.18)
        for name, comparison in result['comparisons'].items():
            item = result['coding'][name]
            color = '#3777ab' if 'combined' in name else '#ca8a2b'
            axes[1].plot(comparison['task_saving_percent'], item['correct'],
                'D' if 'repeat' in name else 'o', label=name.replace('code-', ''),
                color=color, markerfacecolor='none' if 'repeat' in name else color,
                markersize=12 if 'repeat' in name else 8)
            axes[1].annotate(f"{comparison['task_saving_percent']:.1f}% mean / "
                f"{comparison['task_saving_vs_faster_control']:.1f}% faster", (comparison['task_saving_percent'], item['correct']),
                xytext=(-5, -22 if 'repeat' in name else 27 if 'combined' in name else 9),
                textcoords='offset points', fontsize=8,
                ha='right' if 'combined' in name else 'left')
        axes[1].set(xlabel='Whole-task saving vs bracketed control mean (%)',
            ylabel='Correct first writes (raw)', title='Practical speed / quality points', ylim=(-.15, 2.55), yticks=[0, 1, 2])
        axes[1].margins(x=.35)
        axes[1].legend(fontsize=8, loc='lower right')
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
        fig.text(.5, .02, 'Two fresh task families, one first write each; no retries or code-fence cleanup. '
            'Output paths may differ, so paired teacher delta-NLL / KL are not claimed.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, 1))
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'combined-pareto.{suffix}', dpi=160)
        plt.close(fig)
    print(json.dumps(dict(coding={key: {field: value[field] for field in
        ('task_seconds', 'correct', 'tests', 'passed', 'failed', 'complete', 'caps')}
        for key, value in result['coding'].items()}, comparisons=result['comparisons'], pending=result['pending']), indent=2))


if __name__ == '__main__':
    main()
