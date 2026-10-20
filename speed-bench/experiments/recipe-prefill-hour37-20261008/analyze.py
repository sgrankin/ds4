#!/usr/bin/env python3
"""Summarize the owned warm-prefill experiment; --plot only after timing ends."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / '.workspaces/recipe-20261008'


def read(path):
    return json.loads(path.read_text())


def footer(path, kind):
    matches = re.findall(r'ds4: ' + re.escape(kind) + r' ([^\n]+)', path.read_text())
    if not matches:
        return {}
    return {k: float(v) if '.' in v else int(v) for k, v in
        re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', matches[-1])}


def cache_io(path):
    matches = re.findall(r'streaming expert cache budget=([^\n]+)', path.read_text())
    if not matches:
        return {}
    line = matches[-1]
    return dict(logical_miss_pread_gib=float(re.search(r'miss_pread=([0-9.]+) GiB', line)[1]),
        pread_ms=float(re.search(r'pread_ms=([0-9.]+)', line)[1]),
        hits=int(re.search(r'\bhits=(\d+)', line)[1]), misses=int(re.search(r'\bmisses=(\d+)', line)[1]))


def csv_out(output, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def change(candidate, control):
    return 100 * (candidate / control - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plot', action='store_true')
    a = p.parse_args()
    output = HERE / 'analysis'
    output.mkdir(exist_ok=True)
    result = dict(runtime_revision='9c6918f0',
        server_sha256='247e2f2fa76a32647ce6ac6c14a2a36b645e4776a3a45bc53ca2fba3b948f5ee',
        bench_sha256='826d6f2ea7ad6ed9c941ac6602a0e1f7c3278c5add1d71e5993db64b3b9cbabd',
        pending=[], warm={}, coding={}, quality={}, guards={})
    warm_rows, role_rows, task_rows, quality_rows, route_rows, phase_rows = [], [], [], [], [], []
    for subs in (3, 1):
        directory = RAW / f'warm-subs{subs}'
        source = directory / 'summary.json'
        if not source.exists():
            result['pending'].append(str(source.relative_to(ROOT)))
            continue
        runs = read(source)
        complete = [r['variant'] for r in runs] == ['control', 'candidate', 'candidate', 'control']
        if not complete:
            result['pending'].append(f'warm-subs{subs}: incomplete ABBA')
        for i, run in enumerate(runs):
            stats = footer(directory / f'{i}-{run["variant"]}' / 'run.log', 'resident-prefill')
            for phase in run['phases']:
                seconds = int(phase['prefill_tokens']) / float(phase['prefill_tps'])
                warm_rows.append(dict(max_subs=subs, trial=i, variant=run['variant'],
                    context_tokens=int(phase['ctx_tokens']), prefill_tokens=int(phase['prefill_tokens']),
                    printed_tps=float(phase['prefill_tps']), seconds_from_rounded_tps=seconds))
            route_rows.append(dict(source=f'warm-subs{subs}', trial=i, variant=run['variant'], **stats))
        if complete:
            group = {}
            for label in ('control', 'candidate'):
                selected = [r for r in runs if r['variant'] == label]
                group[label] = dict(
                    warm_seconds=statistics.mean(int(r['phases'][-1]['prefill_tokens']) /
                        float(r['phases'][-1]['prefill_tps']) for r in selected),
                    initial_seconds=statistics.mean(int(r['phases'][0]['prefill_tokens']) /
                        float(r['phases'][0]['prefill_tps']) for r in selected))
            first, last = [int(runs[i]['phases'][-1]['prefill_tokens']) /
                float(runs[i]['phases'][-1]['prefill_tps']) for i in (0, 3)]
            group.update(warm_change_percent=change(group['candidate']['warm_seconds'], group['control']['warm_seconds']),
                warm_change_vs_faster_control=change(group['candidate']['warm_seconds'], min(first, last)),
                warm_control_drift_percent=change(last, first),
                initial_change_percent=change(group['candidate']['initial_seconds'], group['control']['initial_seconds']),
                control_frontiers_exact=all(v['exact'] for v in runs[-1]['frontiers'].values()),
                candidate_frontiers=runs[1]['frontiers'],
                candidate_frontiers_repeat=runs[1]['frontiers'] == runs[2]['frontiers'])
            result['warm'][str(subs)] = group
    code_data = {}
    for label in ('code-native-before', 'code-prefill3', 'code-prefill1', 'code-native-after'):
        directory = HERE / label
        source = directory / 'launcher-summary.json'
        if not source.exists():
            result['pending'].append(label)
            continue
        launcher = read(source)
        cases = {}
        for case in ('dedupe', 'bucket'):
            task = read(directory / 'http' / case / case / 'result.json')
            cases[case] = task
            grade = task['grade']
            task_rows.append(dict(arm=label, case=case, task_seconds=task['wall_seconds'],
                passed=grade.get('passed', 0), failed=grade.get('failed', 0),
                unresolved=grade['tests'] - grade.get('passed', 0) - grade.get('failed', 0),
                complete=task['complete'], cap=task['cap'], read_count=task['actual_read_count'],
                write_count=task['actual_write_count'], config_unchanged=task['config_unchanged'],
                first_code_bytes=task['first_code_bytes'], errors=json.dumps(task['errors'])))
            for reply in task['replies']:
                usage = reply['usage']
                role_rows.append(dict(arm=label, case=case, role=reply['role'], seconds=reply['seconds'],
                    budget=reply['budget'], status=reply['response_status'],
                    input_tokens=usage['input_tokens'], cached_tokens=usage['input_tokens_details']['cached_tokens'],
                    fresh_tokens=usage['input_tokens_details']['cache_write_tokens'],
                    output_tokens=usage['output_tokens'], thinking_tokens=usage['output_tokens_details']['reasoning_tokens']))
            stats = footer(directory / f'{case}-server.stderr.log', 'resident-prefill')
            route_rows.append(dict(source=label, case=case, **stats))
        code_data[label] = cases
        result['coding'][label] = dict(task_seconds=sum(t['wall_seconds'] for t in cases.values()),
            launcher_seconds=launcher['seconds'], passed=sum(t['grade'].get('passed', 0) for t in cases.values()),
            failed=sum(t['grade'].get('failed', 0) for t in cases.values()),
            complete=sum(t['complete'] for t in cases.values()), caps=sum(t['cap'] for t in cases.values()))
    if {'code-native-before', 'code-prefill3', 'code-native-after'} <= code_data.keys():
        before, after = [result['coding'][label] for label in ('code-native-before', 'code-native-after')]
        for metric in ('task_seconds', 'launcher_seconds'):
            controls = [before[metric], after[metric]]
            for label in ('code-prefill3', 'code-prefill1'):
                if label in result['coding']:
                    candidate = result['coding'][label]
                    candidate[metric + '_change_percent'] = change(candidate[metric], statistics.mean(controls))
                    candidate[metric + '_change_vs_faster_control'] = change(candidate[metric], min(controls))
            result['coding'][metric + '_control_drift_percent'] = change(controls[1], controls[0])
        equality = {}
        for case in ('dedupe', 'bucket'):
            controls = [code_data[label][case] for label in ('code-native-before', 'code-native-after')]
            codes = [(HERE / label / 'http' / case / case / 'first-solution.py').read_bytes()
                for label in ('code-native-before', 'code-native-after', 'code-prefill3')]
            public = lambda t: dict(calls=[{k: c[k] for k in ('role', 'name', 'arguments', 'output', 'executed')}
                for c in t['calls']], final_text=t['final_text'])
            equality[case] = dict(native_first_code_equal=codes[0] == codes[1],
                native_role_usage_equal=[r['usage'] for r in controls[0]['replies']] == [r['usage'] for r in controls[1]['replies']],
                native_public_equal_excluding_ids_and_timings=public(controls[0]) == public(controls[1]),
                candidate_code_equals_native=codes[2] == codes[0],
                candidate_public_equals_native=public(code_data['code-prefill3'][case]) == public(controls[0]))
            if 'code-prefill1' in code_data:
                weak_code = (HERE / 'code-prefill1' / 'http' / case / case / 'first-solution.py').read_bytes()
                equality[case].update(candidate1_code_equals_native=weak_code == codes[0],
                    candidate1_public_equals_native=public(code_data['code-prefill1'][case]) == public(controls[0]))
        result['coding']['equality'] = equality
    for subs in (3, 1):
        directory = RAW / f'quality-subs{subs}'
        source = directory / 'summary.json'
        if not source.exists():
            result['pending'].append(str(source.relative_to(ROOT)))
            continue
        summary = read(source)
        equivalence_complete = {'off', 'baseline-repeat'} <= {r['name'] for r in summary}
        result['quality'][str(subs)] = dict(manifest=read(directory / 'manifest.json'), variants=summary,
            native_off_equivalence=all(r['tokens'] == r['bit_equal_rows'] for r in summary
                if r['name'] in ('off', 'baseline-repeat')) if equivalence_complete else None,
            complete=equivalence_complete)
        if not result['quality'][str(subs)]['complete']:
            result['pending'].append(f'quality-subs{subs}: incomplete variants')
        controls = [r for r in summary if r['name'] in ('baseline', 'off', 'baseline-repeat')]
        candidate = next((r for r in summary if r['name'] == 'candidate'), None)
        result['quality'][str(subs)]['io'] = {r['name']: cache_io(directory / (r['name'] + '.log')) for r in summary}
        if equivalence_complete and candidate is not None:
            control_ms = statistics.mean(r['prefill_ms'] for r in controls)
            result['quality'][str(subs)]['diagnostic_prefill'] = dict(control_variants=[r['name'] for r in controls],
                control_mean_ms=control_ms, candidate_ms=candidate['prefill_ms'],
                change_percent=change(candidate['prefill_ms'], control_ms))
        for row in summary:
            quality_rows.append(dict(max_subs=subs, **{k: v for k, v in row.items()
                if k not in ('per_prompt', 'prompt_bootstrap_95')}))
            if row['name'] == 'candidate':
                phase_rows.extend(dict(max_subs=subs, phase=phase, **values)
                    for phase, values in row['per_prompt'].items())
                with (directory / 'candidate.csv').open() as f:
                    teacher_rows = list(csv.DictReader(f))
                worst = max(teacher_rows, key=lambda r: float(r['delta_raw_nll']))
                result['quality'][str(subs)]['worst_target'] = worst
    for label in ('guard-code-before', 'guard-code3', 'guard-code1', 'guard-code-after', 'guard-warm3', 'guard-warm1', 'guard-quality3', 'guard-quality1'):
        directory = HERE / label
        source = directory / 'result.json'
        if not source.exists():
            continue
        rows = [json.loads(line) for line in (directory / 'pressure.jsonl').read_text().splitlines()]
        result['guards'][label] = dict(result=read(source), samples=len(rows),
            all_normal=all(r['level'] == 1 for r in rows),
            peak_wired_plus_compressor_gib=max(r['wired_gb'] + r['compressor_gb'] for r in rows),
            swap_growth_gib=rows[-1]['swap_gb'] - read(directory / 'command.json')['initial']['swap_gb'])
    result['guard_totals'] = dict(samples=sum(r['samples'] for r in result['guards'].values()),
        all_normal=all(r['all_normal'] for r in result['guards'].values()),
        all_child_zero=all(r['result']['child_returncode'] == 0 for r in result['guards'].values()),
        peak_wired_plus_compressor_gib=max(r['peak_wired_plus_compressor_gib'] for r in result['guards'].values()),
        largest_job_swap_growth_mib=1024 * max(r['swap_growth_gib'] for r in result['guards'].values()))
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, rows in (('warm.csv', warm_rows), ('tasks.csv', task_rows), ('roles.csv', role_rows), ('quality.csv', quality_rows), ('quality-phases.csv', phase_rows), ('routes.csv', route_rows)):
        csv_out(output, name, rows)
    if a.plot:
        os.environ['MPLCONFIGDIR'] = str(ROOT / '.workspaces/bench-plots-venv/mpl-cache')
        os.environ['XDG_CACHE_HOME'] = str(RAW / 'font-cache')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        ax = axes[0]
        ax.scatter([0], [0], color='black', label='Native relative reference')
        for subs, warm in result['warm'].items():
            quality = result['quality'].get(subs, {})
            candidate = next((r for r in quality.get('variants', []) if r['name'] == 'candidate'), None)
            if candidate is not None and 'diagnostic_prefill' in quality:
                x, y = -quality['diagnostic_prefill']['change_percent'], candidate['delta_nll']
                ax.scatter([x], [y], label=f'Max {subs} substitutions')
                ax.annotate(subs, (x, y), xytext=(5, 5), textcoords='offset points')
        ax.set(xlabel='Instrumented prefill saving (%)', ylabel='Token-weighted delta-NLL (nats/token)',
            title='One teacher trajectory: diagnostics')
        ax.margins(y=.18)
        ax.legend()
        ax = axes[1]
        limits = sorted(result['warm'], key=int)
        ax.bar(limits, [-result['warm'][s]['warm_change_percent'] for s in limits], color='#4172ad', label='vs control mean')
        ax.scatter(limits, [-result['warm'][s]['warm_change_vs_faster_control'] for s in limits], color='black', label='vs faster control')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xlabel='Maximum substitutions', ylabel='Warm 512-token time saving (%)', title='Separate warm ABBA prompt')
        ax.legend()
        ax = axes[2]
        labels = [s for s in ('code-native-before', 'code-prefill3', 'code-native-after', 'code-prefill1') if s in code_data]
        walls = [result['coding'][s]['task_seconds'] for s in labels]
        bars = ax.bar(range(len(labels)), walls, color=['#777' if 'native' in s else '#4172ad' for s in labels])
        for bar, label in zip(bars, labels):
            r = result['coding'][label]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{r['passed']}/322", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(labels)), [s.replace('code-', '') + (' (later)' if s == 'code-prefill1' else '')
            for s in labels], rotation=20, ha='right')
        ax.set(ylabel='Two-task wall (seconds)', title='Separate first-code qualification')
        ax.set_ylim(0, max(walls) * 1.14)
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'prefill-pareto.{suffix}', dpi=160)
        plt.close(fig)
    print(json.dumps(dict(warm=result['warm'], coding=result['coding'], pending=result['pending']), indent=2))


if __name__ == '__main__':
    main()
