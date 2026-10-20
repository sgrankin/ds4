#!/usr/bin/env python3
"""Reuse item3 CSV/log summaries and plots; never read reference/candidate arrays."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=HERE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    root, output = args.directory, args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output.resolve() / '.mplcache')
    path = HERE.parent / 'original-arrival-hour3-20261003' / 'analyze.py'
    spec = importlib.util.spec_from_file_location('arrival_analyze', path)
    arrival = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arrival)
    reader = arrival.replay_reader()
    results, cases, skipped = [], [], []
    candidates = set(root.rglob('*.stdout'))
    candidates.update(p.with_suffix('.stdout') for p in root.rglob('*.csv')
                      if p.name not in ('results.csv','per_case.csv'))
    for stdout in sorted(candidates):
        if any(part.startswith('generation') for part in stdout.relative_to(root).parts):
            continue
        dataset = str(stdout.parent.relative_to(root)) if stdout.parent != root else root.name
        try:
            row, per_case = arrival.read_arm(reader,stdout.parent,stdout.stem,dataset)
            results.append(row)
            cases.extend(per_case)
        except (OSError,ValueError,KeyError,TypeError) as error:
            skipped.append(dict(arm=str(stdout.relative_to(root)),reason=str(error)))
            print('SKIP',stdout.relative_to(root),error)
    reader.comparisons(results)  # Native-before/after mean; original primary comparison.
    arrival.comparisons(results)  # Affinity-before/after if those controls exist.
    for dataset in {r['dataset'] for r in results}:
        rows = [r for r in results if r['dataset'] == dataset]
        if any('affinity_controls_used' in r for r in rows):
            continue
        control = next((r for r in rows if r['arm'] == 'affinity-strong'),None)
        if control:
            for row in rows:
                row.update(affinity_controls_used='affinity-strong',affinity_control_count=1)
                for field in ('decode_ms','model_ms','demand_requested_bytes','total_requested_bytes'):
                    baseline = control.get(field)
                    if baseline is not None:
                        row['affinity_mean_'+field] = baseline
                        row[field+'_vs_affinity_change_percent'] = 100*(row[field]/baseline-1) if baseline and row.get(field) is not None else None
    reader.write_csv(output/'results.csv',results)
    reader.write_csv(output/'per_case.csv',cases)
    scope = {
        'quality': 'New authored fixed teacher histories; token-weighted delta-NLL/KL against native reference. Task generation is graded separately.',
        'latency': 'Diagnostic timing versus native-before/after mean; staging comparison uses affinity-before/after mean when present, otherwise the single affinity-strong arm.',
        'io': 'Demand PROBE_IO and all ORIGINAL_DEADLINE pread footers are separate and summed; logical requested bytes are not physical disk traffic.',
        'input_files': 'CSV, stdout JSON, stderr counters and execution JSON only; no raw binary arrays.',
    }
    (output/'results.json').write_text(json.dumps(dict(scope=scope,results=results,per_case=cases,skipped=skipped),indent=2)+'\n')
    if results and not args.no_plot:
        try:
            arrival.plots(results,output)
        except ImportError:
            print('Plots skipped: matplotlib is unavailable.')
    print(f'Wrote {len(results)} completed arms; skipped {len(skipped)} incomplete arms.')


if __name__ == '__main__':
    main()
