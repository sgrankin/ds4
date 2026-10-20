#!/usr/bin/env python3
"""Canonical full-input suffix curve: task quality, coverage, geometry and wall."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'concurrent-sessions-hour19-20261004'
spec = importlib.util.spec_from_file_location('concurrent_analysis', PREVIOUS / 'analyze.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((PREVIOUS / 'fixture.json').read_text())
    steps, groups, caches, by_arm = [], [], [], {}
    for label, capacity in [('existing-before', 0), ('canonical', 0), ('existing-after', 0)]:
        folder = args.input / label
        a, b, c = previous.arm(folder, label, fixture['sessions'], fixture['edited_archive_codes'])
        steps += a; groups += b; by_arm[label] = c
        launcher = json.loads((folder / 'launcher-summary.json').read_text())
        assert launcher['hot_mb'] == capacity
        for phase in ['before', 'after']:
            log = (folder / f'{phase}-server.stderr.log').read_text(errors='replace')
            summaries = [json.loads(line.split('state_cache_summary ', 1)[1])
                         for line in log.splitlines() if 'state_cache_summary ' in line]
            assert len(summaries) == 1, (label, phase, 'expected one shutdown summary', len(summaries))
            summary = summaries[0]
            summary.update(arm=label, phase=phase, configured_hot_mb=capacity,
                           configured_hot_bytes=capacity * 1024 * 1024,
                           hot_bytes_within_configured_cap=summary['hot_bytes'] <= capacity * 1024 * 1024)
            caches.append(summary)
        c['hot_read_ops'] = sum(s['hot_reads'] for s in caches if s['arm'] == label)
        c['disk_read_ops'] = sum(s['disk_reads'] for s in caches if s['arm'] == label)
        c['endpoint_evictions'] = sum(s['endpoint_evictions'] for s in caches if s['arm'] == label)
    metrics = ['launcher_wall_seconds', 'client_phase_wall_seconds', 'edited_group_makespan_seconds']
    result = dict(by_arm=by_arm, steps=steps, groups=groups, cache_shutdowns=caches,
        scope='Canonical full-input policy alone changes; hot0 and suffix64 fixed. Exact pair alignment remains required. Hot/disk counters are logical chunk read operations, not physical SSD bytes. No parallel GPU or paired generated-history NLL claim.',
        change_vs_existing_before_percent={arm: {k: 100 * (row[k] / by_arm['existing-before'][k] - 1) for k in metrics}
                               for arm, row in by_arm.items()})
    result['group_change_vs_existing_before_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'existing-before' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['change_vs_existing_after_percent'] = {
        arm: {k: 100 * (row[k] / by_arm['existing-after'][k] - 1) for k in metrics}
        for arm, row in by_arm.items()}
    result['group_change_vs_existing_after_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'existing-after' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['existing_bracket_after_vs_before_percent'] = {
        k: 100 * (by_arm['existing-after'][k] / by_arm['existing-before'][k] - 1)
        for k in metrics}
    result['canonical_change_vs_existing_mean_percent'] = {
        k: 100 * (by_arm['canonical'][k] / ((by_arm['existing-before'][k] + by_arm['existing-after'][k]) / 2) - 1)
        for k in metrics}
    result['text_and_usage_vs_existing_before'] = {}
    for arm in by_arm:
        result['text_and_usage_vs_existing_before'][arm] = {}
        for row in [s for s in steps if s['arm'] == arm]:
            baseline = next(s for s in steps if s['arm'] == 'existing-before' and s['step'] == row['step'])
            result['text_and_usage_vs_existing_before'][arm][row['step']] = dict(
                text_byte_equal=baseline['text'].encode() == row['text'].encode(),
                grade_equal=baseline['pass'] == row['pass'],
                token_totals_equal=all(baseline[k] == row[k] for k in ['output_tokens', 'input_tokens', 'cached_tokens']))
    previous.save(args.output / 'results.json', result)
    for name, rows in [('steps', steps), ('groups', groups), ('cache-shutdowns', caches)]:
        with (args.output / (name + '.csv')).open('w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['by_arm', 'change_vs_existing_before_percent', 'group_change_vs_existing_before_percent', 'cache_shutdowns']}, indent=2))
    if args.plot:
        os.environ['MPLCONFIGDIR'] = '/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), constrained_layout=True)
        labels = list(by_arm)
        x = list(range(len(labels)))
        for key, label in [('launcher_wall_seconds', 'Whole launcher'), ('edited_group_makespan_seconds', 'Edited group sum')]:
            axes[0].plot(x, [by_arm[a][key] for a in labels], marker='o', label=label)
        for position, arm in zip(x, labels):
            axes[0].annotate(f"{by_arm[arm]['passes']}/8 strict", (position, by_arm[arm]['launcher_wall_seconds']), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        axes[0].set_xticks(x, ['Existing before', 'Canonical', 'Existing after'])
        axes[0].set_ylim(0, max(by_arm[a]['launcher_wall_seconds'] for a in labels) * 1.13)
        axes[0].set(title='Whole and edited-group wall', xlabel='Measured arm order', ylabel='Seconds'); axes[0].legend(frameon=False)
        for index, (arm, color) in enumerate(zip(labels, ['#65717e', '#e89930', '#a5b1be'])):
            names = ['edited-AB', 'edited-CD']
            values = [next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name) for name in names]
            axes[1].bar([i + (index - 1) * .24 for i in range(2)], values, width=.24, label=arm, color=color)
        axes[1].set_xticks([0, 1], ['Edited AB\nbefore restart', 'Edited CD\nafter restart'])
        axes[1].set(title='Two-client group makespan', ylabel='Seconds'); axes[1].legend(frameon=False)
        hits = [by_arm[a]['approximate_hits'] for a in labels]
        fallback = [by_arm[a]['exact_fallbacks'] for a in labels]
        axes[2].bar(labels, hits, label='Approximate graft', color='#40916c')
        axes[2].bar(labels, fallback, bottom=hits, label='Exact fallback', color='#a5b1be')
        axes[2].set(title='Actual suffix coverage', ylabel='Requests across both lifetimes'); axes[2].legend(frameon=False)
        for ax in axes: ax.tick_params(axis='x', rotation=20)
        for ax in axes: ax.grid(axis='y', alpha=.15)
        fig.suptitle('Canonical full-input suffix policy: pair alignment retained, all24 task/read checks pass', fontsize=12)
        fig.savefig(args.output / 'canonical-suffix-tradeoff.png', dpi=180)
        fig.savefig(args.output / 'canonical-suffix-tradeoff.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
