#!/usr/bin/env python3
"""Hot-cache curve: unchanged concurrent tasks plus shutdown-owned cache counters."""
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
    for label, capacity in [('hot0', 0), ('hot128', 128), ('hot512', 512), ('hot0-after', 0)]:
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
        scope='Hot capacity changes; suffix64 fixed. Closing hot0 is a diagnostic drift control chosen after early results. Warm outputs and endpoint working sets differ. Hot/disk counters are logical chunk read operations, not physical SSD bytes. RAM entries disappear at restart. No parallel GPU claim.',
        change_vs_hot0_percent={arm: {k: 100 * (row[k] / by_arm['hot0'][k] - 1) for k in metrics}
                               for arm, row in by_arm.items()})
    result['group_change_vs_hot0_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'hot0' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['change_vs_hot0_after_percent'] = {
        arm: {k: 100 * (row[k] / by_arm['hot0-after'][k] - 1) for k in metrics}
        for arm, row in by_arm.items()}
    result['group_change_vs_hot0_after_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'hot0-after' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['text_and_usage_vs_hot0'] = {}
    for arm in by_arm:
        result['text_and_usage_vs_hot0'][arm] = {}
        for row in [s for s in steps if s['arm'] == arm]:
            baseline = next(s for s in steps if s['arm'] == 'hot0' and s['step'] == row['step'])
            result['text_and_usage_vs_hot0'][arm][row['step']] = dict(
                text_byte_equal=baseline['text'].encode() == row['text'].encode(),
                grade_equal=baseline['pass'] == row['pass'],
                token_totals_equal=all(baseline[k] == row[k] for k in ['output_tokens', 'input_tokens', 'cached_tokens']))
    previous.save(args.output / 'results.json', result)
    for name, rows in [('steps', steps), ('groups', groups), ('cache-shutdowns', caches)]:
        with (args.output / (name + '.csv')).open('w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['by_arm', 'change_vs_hot0_percent', 'group_change_vs_hot0_percent', 'cache_shutdowns']}, indent=2))
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
        axes[0].set_xticks(x, ['0 before', '128', '512', '0 after'])
        axes[0].set_ylim(0, max(by_arm[a]['launcher_wall_seconds'] for a in labels) * 1.13)
        axes[0].set(title='Capacity in measured arm order', xlabel='Configured hot capacity (MiB)', ylabel='Seconds'); axes[0].legend(frameon=False)
        for index, (arm, color) in enumerate(zip(labels, ['#65717e', '#e89930', '#247ba0', '#a5b1be'])):
            names = ['edited-AB', 'edited-CD']
            values = [next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name) for name in names]
            axes[1].bar([i + (index - 1.5) * .20 for i in range(2)], values, width=.20, label=arm, color=color)
        axes[1].set_xticks([0, 1], ['Edited AB\nbefore restart', 'Edited CD\nafter restart'])
        axes[1].set(title='Two-client group makespan', ylabel='Seconds'); axes[1].legend(frameon=False)
        hot = [by_arm[a]['hot_read_ops'] for a in labels]
        disk = [by_arm[a]['disk_read_ops'] for a in labels]
        axes[2].bar(labels, hot, label='Hot', color='#40916c')
        axes[2].bar(labels, disk, bottom=hot, label='Disk path', color='#a5b1be')
        axes[2].set(title='Logical chunk read operations', ylabel='Operations across both lifetimes'); axes[2].legend(frameon=False)
        for ax in axes: ax.grid(axis='y', alpha=.15)
        fig.suptitle('Hot-cache capacity: suffix64, four histories, two slots and two clients in flight', fontsize=12)
        fig.savefig(args.output / 'hot-cache-curve.png', dpi=180)
        fig.savefig(args.output / 'hot-cache-curve.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
