#!/usr/bin/env python3
"""Collect completed compaction screens and coding arms; plot only after timing."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


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


def relative_percent(candidate, control):
    return 100 * (candidate / control - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    output = HERE / 'analysis'
    output.mkdir(exist_ok=True)
    result = dict(screens=[], coding={}, comparisons={}, pressure=[], pending=[])
    screen_rows, grade_rows, task_rows, role_rows, graft_rows = [], [], [], [], []
    for source in sorted(HERE.glob('screen*/summary.json')):
        if not source.stat().st_size:
            result['pending'].append(source.parent.name)
            continue
        summary = read(source)
        directory = source.parent
        execution = read(directory / 'execution.json')
        command = execution['command']
        tail = int(command[command.index('--fresh-tail') + 1])
        grades = read(directory / 'grades.json') if (directory / 'grades.json').exists() else {}
        runs = {r['mode']: r for r in summary['results']}
        fresh, reuse = runs['fresh'], runs['reuse']
        delta = summary['mean_delta_nll']
        row = dict(screen=directory.name, case=summary['case'], fresh_tail_requested=tail,
            teacher_rows=summary['rows'], delta_nll_nats_per_token=delta,
            perplexity_change_fraction=math.expm1(delta), mean_kl=summary['mean_kl'],
            mean_tv=summary['mean_tv'], argmax_flips=summary['argmax_flips'],
            byte_equal_rows=summary['byte_equal_rows'], fresh_edit_ms=fresh['edit_ms'],
            reuse_edit_ms=reuse['edit_ms'], edit_saving_percent=-relative_percent(reuse['edit_ms'], fresh['edit_ms']),
            source_capture_ms=summary['source_capture_ms'], setup_ms=summary['setup_ms'],
            reused_tokens=summary['reused_tokens'], fresh_tail_actual=summary['fresh_tail_tokens'],
            query_fully_fresh=summary['query_fully_fresh'],
            destination_end=summary['destination_end'], latest_user_start=summary['latest_user_start'],
            fresh_greedy_cap=fresh['at_cap'], reuse_greedy_cap=reuse['at_cap'])
        screen_rows.append(row)
        diagnostic = directory / 'fresh-unfenced-grade.json'
        result['screens'].append(dict(**row, raw_summary=summary, grades=grades,
            fresh_unfenced_diagnostic=read(diagnostic) if diagnostic.exists() else None,
            execution=execution, result=read(directory / 'result.json') if (directory / 'result.json').exists() else None))
        if not grades:
            result['pending'].append(directory.name + ': external grades')
        for mode in ('fresh', 'reuse'):
            grade = grades.get(mode, {}).get('grade', {})
            grade_rows.append(dict(screen=directory.name, case=summary['case'], tail=tail, mode=mode,
                correct=grade.get('correct'), tests=grade.get('tests', 161), passed=grade.get('passed', 0),
                failed=grade.get('failed', 0), unresolved=grade.get('tests', 161) - grade.get('passed', 0) - grade.get('failed', 0),
                returncode=grades.get(mode, {}).get('returncode'), cap=runs[mode]['at_cap'],
                stopped=runs[mode]['stopped'], greedy_tokens=runs[mode]['greedy_tokens'],
                first_failure=json.dumps(grade.get('first_failure')), error=json.dumps(grade.get('reason'))))
    for directory in sorted(HERE.glob('code-*')):
        source = directory / 'launcher-summary.json'
        if not source.exists():
            result['pending'].append(directory.name)
            continue
        launcher = read(source)
        cases, grafts = {}, []
        for case in ('dedupe', 'bucket'):
            task_source = directory / 'http' / case / case / 'result.json'
            if not task_source.exists():
                result['pending'].append(directory.name + '/' + case)
                continue
            task = read(task_source)
            cases[case] = task
            grade = task['grade']
            task_rows.append(dict(arm=directory.name, mode=launcher['mode'], tail=launcher['fresh_tail'], case=case,
                wall_seconds=task['wall_seconds'], complete=task['complete'], cap=task['cap'],
                correct=grade.get('correct'), passed=grade.get('passed', 0), failed=grade.get('failed', 0),
                unresolved=grade.get('tests', 161) - grade.get('passed', 0) - grade.get('failed', 0),
                read_count=task['actual_read_count'], write_count=task['actual_write_count'],
                config_unchanged=task['config_unchanged'], errors=json.dumps(task['errors']),
                grading_error=json.dumps(grade.get('reason')), first_failure=json.dumps(grade.get('first_failure'))))
            for reply in task['replies']:
                usage = reply['usage']
                role_rows.append(dict(arm=directory.name, case=case, role=reply['role'], seconds=reply['seconds'],
                    budget=reply['budget'], status=reply['response_status'],
                    input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                    cached_tokens=usage.get('input_tokens_details', {}).get('cached_tokens'),
                    fresh_tokens=usage.get('input_tokens_details', {}).get('cache_write_tokens'),
                    thinking_tokens=usage.get('output_tokens_details', {}).get('reasoning_tokens')))
            log = directory / f'{case}-server.stderr.log'
            for line in log.read_text().splitlines():
                if 'suffix APPROXIMATE' in line or 'context_suffix' in line or 'context suffix' in line:
                    fields = dict(re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)', line))
                    row = dict(arm=directory.name, case=case, tail_requested=launcher['fresh_tail'], line=line,
                        fields=fields, fresh_actual=int(fields['fresh']) if 'fresh' in fields else None,
                        graft_fresh_actual=int(fields['graft_fresh']) if 'graft_fresh' in fields else None)
                    if 'destination' in fields and 'latest_user_start' in fields:
                        row['destination_end'] = int(fields['destination'].split(':')[1])
                        row['latest_user_start'] = int(fields['latest_user_start'])
                        row['latest_user_outside_graft'] = row['destination_end'] <= row['latest_user_start']
                    grafts.append(row)
                    graft_rows.append(dict(**{k: v for k, v in row.items() if k != 'fields'}, fields=json.dumps(fields)))
        result['coding'][directory.name] = dict(launcher=launcher, cases=cases,
            grafts=grafts,
            task_seconds=sum(t['wall_seconds'] for t in cases.values()),
            complete=sum(t['complete'] for t in cases.values()), caps=sum(t['cap'] for t in cases.values()),
            passed=sum(t['grade'].get('passed', 0) for t in cases.values()),
            failed=sum(t['grade'].get('failed', 0) for t in cases.values()))
    if {'code-native-before', 'code-native-after'} <= result['coding'].keys():
        before, after = [result['coding'][label] for label in ('code-native-before', 'code-native-after')]
        result['native_control_drift_percent'] = relative_percent(after['task_seconds'], before['task_seconds'])
        for label, candidate in result['coding'].items():
            if candidate['launcher']['mode'] != 'reuse':
                continue
            controls = [before['task_seconds'], after['task_seconds']]
            comparison = dict(task_change_percent=relative_percent(candidate['task_seconds'], statistics.mean(controls)),
                task_change_vs_faster_control=relative_percent(candidate['task_seconds'], min(controls)), cases={})
            for case, task in candidate['cases'].items():
                controls = [c['cases'][case]['wall_seconds'] for c in (before, after)]
                comparison['cases'][case] = dict(change_percent=relative_percent(task['wall_seconds'], statistics.mean(controls)),
                    change_vs_faster_control=relative_percent(task['wall_seconds'], min(controls)))
            result['comparisons'][label] = comparison
        result['native_equality'] = {}
        for case in ('dedupe', 'bucket'):
            tasks = [c['cases'][case] for c in (before, after)]
            codes = [(HERE / label / 'http' / case / case / 'first-solution.py')
                for label in ('code-native-before', 'code-native-after')]
            result['native_equality'][case] = dict(first_code_equal=all(p.exists() for p in codes) and codes[0].read_bytes() == codes[1].read_bytes(),
                role_usage_equal=[r['usage'] for r in tasks[0]['replies']] == [r['usage'] for r in tasks[1]['replies']])
        result['edited_context_equality'] = {}
        for case in ('dedupe', 'bucket'):
            paths = [(HERE / label / 'http' / case / case / 'context.after.txt') for label in result['coding']]
            result['edited_context_equality'][case] = all(p.read_bytes() == paths[0].read_bytes() for p in paths)
    for directory in sorted(HERE.glob('guard-*')):
        if not (directory / 'result.json').exists():
            result['pending'].append(directory.name)
            continue
        guard = read(directory / 'result.json')
        initial = read(directory / 'command.json')['initial']
        samples = [json.loads(line) for line in (directory / 'pressure.jsonl').read_text().splitlines()]
        result['pressure'].append(dict(guard=directory.name, status=guard['status'],
            child_returncode=guard['child_returncode'], samples=len(samples),
            adverse_samples=sum(bool(s['reason']) for s in samples),
            levels=sorted({s['level'] for s in samples}),
            peak_wired_compressor_gib=max(s['wired_gb'] + s['compressor_gb'] for s in samples),
            initial_swap_gib=initial['swap_gb'], peak_swap_gib=max(s['swap_gb'] for s in samples),
            max_swap_growth_mib=1024 * (max(s['swap_gb'] for s in samples) - initial['swap_gb'])))
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, rows in (('screens.csv', screen_rows), ('screen-grades.csv', grade_rows),
                       ('tasks.csv', task_rows), ('roles.csv', role_rows), ('grafts.csv', graft_rows),
                       ('pressure.csv', result['pressure'])):
        csv_out(output, name, rows)
    if args.plot:
        os.environ['MPLCONFIGDIR'] = str(ROOT / '.workspaces/bench-plots-venv/mpl-cache')
        os.environ['XDG_CACHE_HOME'] = str(ROOT / '.workspaces/recipe-20261008/font-cache')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        ax = axes[0]
        rows = sorted([r for r in screen_rows if r['case'] == 'ordinary-summary'], key=lambda r: r['fresh_tail_requested'])
        ax.plot([r['edit_saving_percent'] for r in rows], [r['delta_nll_nats_per_token'] for r in rows], 'o-')
        for row in rows:
            ax.annotate(f"{row['fresh_tail_requested']} ({row['delta_nll_nats_per_token']:+.4f})",
                (row['edit_saving_percent'], row['delta_nll_nats_per_token']),
                xytext=(4, 6), textcoords='offset points', fontsize=8)
        ax.set(xlabel='Same-screen edit saving (%)', ylabel='Teacher delta-NLL (nats/token)',
            title='Ordinary summary: requested tail curve')
        ax.margins(x=.2, y=.25)
        ax = axes[1]
        for row in screen_rows:
            if row['fresh_tail_requested'] != 64:
                continue
            ax.plot(row['edit_saving_percent'], row['delta_nll_nats_per_token'], 'o', label=row['case'])
            ax.annotate(f"{row['delta_nll_nats_per_token']:+.4f}",
                (row['edit_saving_percent'], row['delta_nll_nats_per_token']),
                xytext=(4, 6), textcoords='offset points', fontsize=8)
        ax.set(xlabel='Same-screen edit saving (%)', ylabel='Teacher delta-NLL (nats/token)',
            title='Qualifications: requested tail 64')
        ax.margins(x=.25, y=.3)
        ax.legend(fontsize=7, loc='best')
        ax = axes[2]
        labels = sorted(result['coding'], key=lambda s: (s != 'code-native-before', s == 'code-native-after',
            result['coding'][s]['launcher']['fresh_tail']))
        colors = ['#777' if result['coding'][s]['launcher']['mode'] == 'native' else '#4172ad' for s in labels]
        bars = ax.bar(range(len(labels)), [result['coding'][s]['task_seconds'] for s in labels], color=colors)
        for bar, label in zip(bars, labels):
            item = result['coding'][label]
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{item['passed']}/322\ncaps {item['caps']}",
                ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(labels)), [s.replace('code-', '') for s in labels], rotation=20, ha='right')
        ax.set(ylabel='Two-task wall (seconds)', title='Separate actual READ / edit / WRITE / FINAL')
        if bars:
            ax.set_ylim(0, max(bar.get_height() for bar in bars) * 1.15)
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
        fig.text(.5, .02, 'Oracle screens are diagnostic edit timings; coding bars are a separate workload. '
            'Equivalent baseline raw fence failure retained; unfenced diagnostic 161 passed.\n'
            'Server requested tails 64/128/256 all freshened 259 dedupe / 264 bucket tokens (latest User kept fresh).',
            ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, 1))
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'compaction-pareto.{suffix}', dpi=160)
        plt.close(fig)
    print(json.dumps(dict(screens=screen_rows, coding={k: {field: v[field] for field in
        ('task_seconds', 'passed', 'failed', 'complete', 'caps')} for k, v in result['coding'].items()},
        comparisons=result['comparisons'], pending=result['pending']), indent=2))


if __name__ == '__main__':
    main()
