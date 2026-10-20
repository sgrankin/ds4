#!/usr/bin/env python3
"""Summarize completed teacher-forced resident-routing arms; never read raw arrays."""
import argparse
import csv
import json
import math
import re
from pathlib import Path


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite numeric result')
    return value


def counters(stderr, prefix, integer_fields):
    lines = [line for line in stderr.splitlines() if line.startswith(prefix)]
    if not lines:
        return {}
    result = {}
    for key, value in re.findall(r'(\w+)=([-+0-9.eE]+)', lines[-1]):
        result[key] = int(value) if key in integer_fields else number(value)
    return result


def means(rows):
    count = len(rows)
    return {
        'tokens': count,
        'mean_delta_nll_nats_per_token': math.fsum(number(r['delta_raw_nll']) for r in rows) / count,
        'mean_kl_nats_per_token': math.fsum(number(r['raw_kl']) for r in rows) / count,
        'mean_tv': math.fsum(number(r['raw_tv']) for r in rows) / count,
        'mean_base_nll_nats_per_token': math.fsum(number(r['base_nll']) for r in rows) / count,
        'mean_raw_nll_nats_per_token': math.fsum(number(r['raw_nll']) for r in rows) / count,
        'sum_delta_nll_nats': math.fsum(number(r['delta_raw_nll']) for r in rows),
        'max_delta_nll_nats': max(number(r['delta_raw_nll']) for r in rows),
        'max_kl_nats': max(number(r['raw_kl']) for r in rows),
        'argmax_flips': sum(int(r['argmax_flip']) for r in rows),
        'decode_ms': math.fsum(number(r['decode_ms']) for r in rows),
    }


def read_arm(directory, label, dataset):
    summary = json.loads((directory / (label + '.stdout')).read_text().strip())
    execution_file = directory / (label + '.execution.json')
    execution = json.loads(execution_file.read_text()) if execution_file.exists() else {}
    if execution and execution.get('returncode') != 0:
        raise ValueError('helper returncode=' + str(execution.get('returncode')))
    with (directory / (label + '.csv')).open(newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows or len(rows) != summary['tokens']:
        raise ValueError(f'incomplete CSV: {len(rows)} rows, summary expects {summary.get("tokens")}')
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case'], []).append(row)
    if len(by_case) != summary['cases']:
        raise ValueError('case count differs from completed helper summary')
    env = execution.get('environment', {})
    algorithm = 'functional' if 'functional' in label else 'affinity' if 'affinity' in label else 'native'
    gap = env.get('DS4_METAL_V41_RESIDENT_ROUTE_GAP')
    if gap is None:
        match = re.search(r'gap[-=]?([.0-9]+)', label)
        gap = match.group(1) if match else None
    result = {'dataset': dataset, 'arm': label, 'algorithm': algorithm,
              'route_gap': number(gap) if gap is not None else None,
              'cases': len(by_case), **means(rows),
              'prefill_ms': number(summary['prefill_ms']),
              'startup_ms': number(summary['startup_ms']),
              'metrics_ms': number(summary.get('metrics_ms', 0)),
              'helper_wall_s': execution.get('seconds'),
              'bit_equal_rows': summary.get('bit_equal_rows')}
    for field, suffix, convert in (('route_max_mass', 'MAX_MASS', number), ('route_max_rank', 'MAX_RANK', int), ('route_max_subs', 'MAX_SUBS', int), ('route_max_total_mass', 'MAX_TOTAL_MASS', number)):
        value = env.get('DS4_METAL_V41_RESIDENT_ROUTE_' + suffix)
        result[field] = convert(value) if value is not None else None
    # The helper's phase totals cover model execution; wall includes startup/metrics/IO.
    result['decode_ms'] = number(summary['decode_ms'])
    result['model_ms'] = result['prefill_ms'] + result['decode_ms']
    result['model_s'] = result['model_ms'] / 1000
    result['decode_s'] = result['decode_ms'] / 1000
    stderr_file = directory / (label + '.stderr')
    stderr = stderr_file.read_text(errors='replace') if stderr_file.exists() else ''
    groups = {
        'io': counters(stderr, 'PROBE_IO ', {'enabled', 'pread_calls', 'requested_bytes', 'returned_bytes', 'cache_loader_bytes', 'cache_hits', 'cache_misses', 'cache_evictions'}),
        'route': counters(stderr, 'ds4: resident-route ', {'layers', 'substitutions', 'native_misses', 'skipped_loads', 'skipped_bytes', 'full_snapshots', 'double_layers', 'planned_substitutions', 'drops', 'triple_layers'}),
        'functional': counters(stderr, 'FUNCTIONAL_MAP ', {'eligible', 'planned', 'committed', 'declined', 'owned', 'page_charge', 'failed'}),
    }
    for group, values in groups.items():
        result.update({group + '_' + key: value for key, value in values.items()})
    result['io_requested_GB_decimal'] = result.get('io_requested_bytes', 0) / 1e9 if 'io_requested_bytes' in result else None
    result['functional_cpu_rank_ms'] = groups['functional'].get('cpu_rank_ms')
    result['missing_counters'] = ','.join(group for group, values in groups.items()
        if not values and (group == 'io' or group == 'route' and algorithm != 'native' or group == 'functional' and algorithm == 'functional'))
    per_case = [{'dataset': dataset, 'arm': label, 'algorithm': algorithm, 'case': case,
                 'prompt_tokens': int(case_rows[0]['prompt_tokens']), **means(case_rows)}
                for case, case_rows in by_case.items()]
    return result, per_case


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparisons(results):
    for dataset in sorted({r['dataset'] for r in results}):
        arms = [r for r in results if r['dataset'] == dataset]
        controls = [r for r in arms if r['arm'] in ('native-before', 'native-after')]
        if not controls:
            continue
        model = math.fsum(r['model_ms'] for r in controls) / len(controls)
        decode = math.fsum(r['decode_ms'] for r in controls) / len(controls)
        io_controls = [r['io_requested_bytes'] for r in controls if 'io_requested_bytes' in r]
        io = math.fsum(io_controls) / len(io_controls) if io_controls else None
        for row in arms:
            row['native_controls_used'] = ','.join(r['arm'] for r in controls)
            row['native_mean_model_ms'] = model
            row['model_time_change_percent'] = 100 * (row['model_ms'] / model - 1) if model else None
            row['decode_time_change_percent'] = 100 * (row['decode_ms'] / decode - 1) if decode else None
            row['requested_bytes_change_percent'] = 100 * (row['io_requested_bytes'] / io - 1) if io and 'io_requested_bytes' in row else None
        late = next((r for r in arms if r['arm'] == 'native-late'), None)
        if late:
            for row in arms:
                row['native_late_model_ms'] = late['model_ms']
                row['native_late_decode_ms'] = late['decode_ms']
                row['model_time_vs_native_late_change_percent'] = 100 * (row['model_ms'] / late['model_ms'] - 1) if late['model_ms'] else None
                row['decode_time_vs_native_late_change_percent'] = 100 * (row['decode_ms'] / late['decode_ms'] - 1) if late['decode_ms'] else None


def plot(results, output, time_field='model_s', filename='pareto', time_label='Prefill + decode model time (seconds)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = sorted({r['dataset'] for r in results})
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 5.2), squeeze=False)
    colors = {'native': '#777777', 'affinity': '#247ba0', 'functional': '#df7c26'}
    offsets = {'gap .1': (-42, 18), 'gap .3': (8, -27), 'gap .6': (9, 15), 'gap .9': (-43, -28),
               'count1': (-48, 30), 'count2': (14, 30), 'count3': (13, -34),
               'functional': (-54, -30), 'B=.025': (-95, 50), 'B=.05': (-50, 25),
               'B=.10': (10, -34), 'count3 (.20 mass)': (14, 28), '3, mass .25': (-60, 44)}
    def alias(row, dataset_rows):
        name = row['arm']
        if name == 'affinity-three':
            return None if any(r['arm'] == 'affinity-wide3' for r in dataset_rows) else 'count3 (.20 mass)'
        if name == 'affinity-two':
            return None  # duplicate old sweep point remains plotted and in CSV
        if name == 'affinity':
            return 'gap .3'
        if name.startswith('affinity-gap'):
            value = f"{float(name.removeprefix('affinity-gap')):g}"
            return 'gap ' + (value[1:] if value.startswith('0.') else value)
        if name.startswith('affinity-wide'):
            return 'count' + name.removeprefix('affinity-wide')
        if name == 'affinity-three-mass0.25':
            return '3, mass .25'
        if name.startswith('functional-band'):
            value = name.removeprefix('functional-band')
            return 'B=' + (value[1:] if value.startswith('0.') else value)
        return name
    for ax, dataset in zip(axes[0], datasets):
        rows = [r for r in results if r['dataset'] == dataset]
        for row in rows:
            x, y = row[time_field], row['mean_delta_nll_nats_per_token']
            ax.scatter(x, y, color=colors[row['algorithm']], s=45, zorder=3)
            if row['algorithm'] == 'native':
                continue
            label = alias(row, rows)
            if label:
                ax.annotate(label, (x, y), xytext=offsets.get(label, (8, 12)), textcoords='offset points', fontsize=9,
                            arrowprops={'arrowstyle': '-', 'color': '#888888', 'linewidth': .6}, zorder=4)
        natives = [r for r in rows if r['arm'] in ('native-before', 'native-after', 'native-late')]
        if natives:
            low, high = min(r[time_field] for r in natives), max(r[time_field] for r in natives)
            y = math.fsum(r['mean_delta_nll_nats_per_token'] for r in natives) / len(natives)
            ax.plot([low, high], [y, y], color=colors['native'], linewidth=2.5, zorder=2)
            ax.annotate('native', ((low + high) / 2, y), xytext=(0, -25), textcoords='offset points',
                        ha='center', fontsize=9, color='#555555',
                        arrowprops={'arrowstyle': '-', 'color': '#888888', 'linewidth': .6})
        frontier = [r for r in rows if not any(q[time_field] <= r[time_field] and
            q['mean_delta_nll_nats_per_token'] <= r['mean_delta_nll_nats_per_token'] and
            (q[time_field] < r[time_field] or q['mean_delta_nll_nats_per_token'] < r['mean_delta_nll_nats_per_token']) for q in rows)]
        frontier.sort(key=lambda r: r[time_field])
        ax.plot([r[time_field] for r in frontier], [r['mean_delta_nll_nats_per_token'] for r in frontier], '--', color='#aaaaaa', linewidth=1)
        ax.axhline(0, color='#cccccc', linewidth=1)
        ax.set_title(f"{max(r['cases'] for r in rows)}-case screen")
        ax.set_xlabel(time_label)
        ax.set_ylabel('Token-weighted mean ΔNLL (nats/token)')
        ax.margins(x=.18, y=.28)
        ax.grid(alpha=.15)
    fig.suptitle('Resident routing: time / teacher-forced likelihood tradeoff')
    fig.text(.5, .02, 'B= functional band. Grey native span: measured native-control range, not a confidence interval.', ha='center', fontsize=8, color='#666666')
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(output / (filename + '.png'), dpi=170)
    fig.savefig(output / (filename + '.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    root = args.directory
    output = args.output or root
    results, cases, skipped = [], [], []
    # Supports initial four arms, additional affinity gaps and a nested24-case set.
    candidates = set(root.rglob('*.stdout'))
    candidates.update(p.with_suffix('.stdout') for p in root.rglob('*.csv') if p.name not in ('results.csv', 'per_case.csv'))
    for stdout in sorted(candidates):
        if any(part.startswith('generation') for part in stdout.relative_to(root).parts):
            continue
        directory, label = stdout.parent, stdout.stem
        dataset = str(directory.relative_to(root)) if directory != root else root.name
        try:
            row, case_rows = read_arm(directory, label, dataset)
            results.append(row)
            cases.extend(case_rows)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = str(error)
            skipped.append({'arm': str(stdout.relative_to(root)), 'reason': reason})
            print('SKIP', stdout.relative_to(root), reason)
    comparisons(results)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'results.csv', results)
    write_csv(output / 'per_case.csv', cases)
    warnings = []
    if not args.no_plot:
        try:
            plot(results, output)
            plot(results, output, 'decode_s', 'pareto_decode', 'Decode model time (seconds)')
        except ImportError as error:
            warnings.append('Plot skipped; use ROOT plotting environment: ' + str(error))
            print(warnings[-1])
    document = {'units': {'times': 'milliseconds unless *_s', 'nll_kl': 'nats/token', 'IO_bytes': 'logical bytes; decimalGB=*bytes/1e9', 'counter_counts': 'integers'},
                'scope': 'Teacher-forced replay; no generated task-success measurement. Timing is diagnostic, not balanced final latency evidence.',
                'aggregation': 'Each CSV row is one teacher token. ΔNLL/KL are token-weighted, not an unweighted mean of case means. Baseline comparison uses available native-before/native-after model-time mean.',
                'late_control_scope': 'native-late is a separate timing-drift comparison. Primary native-before/native-after baseline calculations are unchanged; the plotted native range includes late when present.',
                'completed_arms': len(results), 'results': results, 'per_case': cases, 'skipped': skipped, 'warnings': warnings}
    (output / 'results.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    for row in results:
        print(f"{row['dataset']} {row['arm']}: model={row['model_s']:.3f}s ΔNLL={row['mean_delta_nll_nats_per_token']:+.6g} KL={row['mean_kl_nats_per_token']:.6g} flips={row['argmax_flips']}/{row['tokens']}")


if __name__ == '__main__':
    main()
