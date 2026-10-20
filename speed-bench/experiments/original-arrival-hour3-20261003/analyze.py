#!/usr/bin/env python3
"""Summarize completed original-arrival replay arms without reading raw arrays."""
import argparse
import importlib.util
import json
import math
import re
from pathlib import Path


def replay_reader():
    path = Path(__file__).resolve().parents[1] / 'resident-functional-hour2-20261003' / 'analyze.py'
    spec = importlib.util.spec_from_file_location('resident_replay_analyze', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrival_counters(stderr):
    records = []
    for line in stderr.splitlines():
        if line.startswith('ds4: ORIGINAL_DEADLINE '):
            records.append({key: int(value) for key, value in re.findall(r'(\w+)=(-?\d+)', line)})
    if not records:
        return {}
    totals = {'deadline_footer_records': len(records),
              'deadline_generations': len({r['generation'] for r in records if 'generation' in r})}
    summed = ('requests', 'started', 'completed', 'failed', 'promoted', 'future', 'expired',
              'deadline_tries', 'deadline_misses', 'logical_requested',
              'pread_calls', 'pread_requested', 'pread_returned')
    for key in summed:
        if any(key in r for r in records):
            totals['deadline_' + key] = sum(r.get(key, 0) for r in records)
    for key in ('reader_host', 'adapter_reserve', 'payload_peak'):
        if any(key in r for r in records):
            totals['deadline_max_' + key] = max(r.get(key, 0) for r in records)
    totals['deadline_any_overflow'] = any(r.get('overflow', 0) != 0 for r in records)
    return totals


def read_arm(reader, directory, label, dataset):
    row, cases = reader.read_arm(directory, label, dataset)
    if label.startswith('deadline'):
        row['algorithm'] = 'deadline'
        for case in cases:
            case['algorithm'] = 'deadline'
    stderr = (directory / (label + '.stderr')).read_text(errors='replace')
    stage = arrival_counters(stderr)
    row.update(stage)
    if row['algorithm'] == 'deadline' and not stage:
        row['missing_counters'] = ','.join(filter(None, (row['missing_counters'], 'ORIGINAL_DEADLINE')))
    for suffix, demand_key, stage_key in (
            ('pread_calls', 'io_pread_calls', 'deadline_pread_calls'),
            ('requested_bytes', 'io_requested_bytes', 'deadline_pread_requested'),
            ('returned_bytes', 'io_returned_bytes', 'deadline_pread_returned')):
        demand = row.get(demand_key)
        staged = row.get(stage_key, 0 if row['algorithm'] != 'deadline' else None)
        row['demand_' + suffix] = demand
        row['staging_' + suffix] = staged
        row['total_' + suffix] = demand + staged if demand is not None and staged is not None else None
    for name in ('demand', 'staging', 'total'):
        value = row[name + '_requested_bytes']
        row[name + '_requested_GB_decimal'] = value / 1e9 if value is not None else None
    return row, cases


def comparisons(results):
    for dataset in sorted({r['dataset'] for r in results}):
        rows = [r for r in results if r['dataset'] == dataset]
        controls = [r for r in rows if r['arm'] in ('affinity-before', 'affinity-after')]
        if not controls:
            continue
        for row in rows:
            row['affinity_controls_used'] = ','.join(r['arm'] for r in controls)
            row['affinity_control_count'] = len(controls)
        for field in ('decode_ms', 'model_ms', 'demand_requested_bytes', 'total_requested_bytes'):
            values = [r[field] for r in controls if r.get(field) is not None]
            if not values:
                continue
            baseline = math.fsum(values) / len(values)
            for row in rows:
                row['affinity_mean_' + field] = baseline
                row[field + '_vs_affinity_change_percent'] = 100 * (row[field] / baseline - 1) if baseline and row.get(field) is not None else None


def plots(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = sorted({r['dataset'] for r in results})
    if not datasets:
        return
    colors = {'native': '#777777', 'affinity': '#247ba0', 'deadline': '#df7c26'}
    labels = {'native-before': 'N1', 'native-after': 'N2', 'native-late': 'N3',
              'affinity-before': 'A1', 'affinity-after': 'A2', 'affinity-late': 'A3',
              'deadline': 'age80', 'deadline-age1': 'age1', 'deadline-age8': 'age8'}
    offsets = {'N1': (-25, 17), 'N2': (8, -24), 'N3': (-20, -36),
               'A1': (8, -26), 'A2': (10, 15), 'A3': (-28, 25),
               'age80': (-40, -30), 'age1': (10, 18), 'age8': (-38, 18)}
    key = 'N1/N2: native before/after. A1/A2/A3: affinity before/after/late. age: original-staging TTL.'
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        rows = [r for r in results if r['dataset'] == dataset]
        for i, row in enumerate(rows):
            x, y = row['decode_s'], row['mean_delta_nll_nats_per_token']
            ax.scatter(x, y, color=colors.get(row['algorithm'], '#555555'), s=45)
            label = labels.get(row['arm'], row['arm'])
            ax.annotate(label, (x, y), xytext=offsets.get(label, (8, 12 if i % 2 == 0 else -22)),
                        textcoords='offset points', fontsize=9,
                        arrowprops={'arrowstyle': '-', 'color': '#999999', 'linewidth': .5})
        ax.axhline(0, color='#cccccc', linewidth=1)
        ax.set_title(f"{max(r['cases'] for r in rows)} cases: {dataset}")
        ax.set_xlabel('Decode model time (seconds)')
        ax.set_ylabel('Token-weighted ΔNLL (nats/token, native reference)')
        ax.grid(alpha=.15)
        ax.margins(x=.18, y=.3)
    fig.suptitle('Original staging: decode time / teacher-forced likelihood')
    fig.text(.5, .015, key, ha='center', fontsize=8, color='#666666')
    fig.tight_layout(rect=(0, .045, 1, .96))
    for extension in ('png', 'svg'):
        fig.savefig(output / ('pareto_decode.' + extension), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        rows = [r for r in results if r['dataset'] == dataset and r['total_requested_bytes'] is not None]
        positions = list(range(len(rows)))
        demand = [r['demand_requested_GB_decimal'] for r in rows]
        stage = [r['staging_requested_GB_decimal'] for r in rows]
        ax.bar(positions, demand, label='Demand', color='#247ba0')
        ax.bar(positions, stage, bottom=demand, label='Original staging', color='#df7c26')
        ax.set_xticks(positions, [labels.get(r['arm'], r['arm']) for r in rows], rotation=0)
        ax.set_title(dataset)
        ax.set_ylabel('Requested pread bytes (decimal GB)')
        ax.grid(axis='y', alpha=.15)
        ax.legend()
    fig.suptitle('Demand + original staging IO (logical reads, not physical disk traffic)')
    fig.text(.5, .015, key, ha='center', fontsize=8, color='#666666')
    fig.tight_layout(rect=(0, .045, 1, .96))
    for extension in ('png', 'svg'):
        fig.savefig(output / ('io.' + extension), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    reader = replay_reader()
    results, cases, skipped = [], [], []
    root, output = args.directory, args.output or args.directory
    candidates = set(root.rglob('*.stdout'))
    candidates.update(p.with_suffix('.stdout') for p in root.rglob('*.csv') if p.name not in ('results.csv', 'per_case.csv'))
    for stdout in sorted(candidates):
        if any(part.startswith('generation') for part in stdout.relative_to(root).parts):
            continue
        directory, label = stdout.parent, stdout.stem
        dataset = str(directory.relative_to(root)) if directory != root else root.name
        try:
            row, per_case = read_arm(reader, directory, label, dataset)
            results.append(row)
            cases.extend(per_case)
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append({'arm': str(stdout.relative_to(root)), 'reason': str(error)})
            print('SKIP', stdout.relative_to(root), error)
    comparisons(results)
    output.mkdir(parents=True, exist_ok=True)
    reader.write_csv(output / 'results.csv', results)
    reader.write_csv(output / 'per_case.csv', cases)
    scope = {
        'quality': 'Teacher-forced token-weighted ΔNLL/KL against the native reference; no generated-task success measured.',
        'latency': 'Diagnostic decode/model timing; affinity-before/after arithmetic mean is the timing comparator, not a confidence interval.',
        'io': 'Demand PROBE_IO totals and ALL ORIGINAL_DEADLINE pread footers are separate, then added. Requested bytes count logical reads, not physical disk traffic.',
        'staging': 'logical_requested counts admitted staging payloads; pread_requested/returned are actual worker read counters. Peaks are maxima, not sums.',
        'routes': 'Route substitution counters precede deadline promotion; do not equate planned skipped bytes with net avoided IO.',
    }
    (output / 'results.json').write_text(json.dumps({'scope': scope, 'results': results, 'per_case': cases, 'skipped': skipped}, indent=2) + '\n')
    if results and not args.no_plot:
        try:
            plots(results, output)
        except ImportError:
            print('Plots skipped: matplotlib is not installed in this Python environment.')
    print(f'Wrote {len(results)} completed arms; skipped {len(skipped)} incomplete arms.')


if __name__ == '__main__':
    main()
