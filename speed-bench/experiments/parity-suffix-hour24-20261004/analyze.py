#!/usr/bin/env python3
"""Optional parity padding and bounded canonical-plus-padding follow-up."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re

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
    for label, capacity in [('existing-before', 0), ('padded', 0), ('canonical-padded', 0), ('existing-after', 0)]:
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
        scope='Primary PAD0/1/0 comparison, plus canonical1/PAD1 follow-up chosen after primary. Hot0/suffix64 fixed. Padding changes model input; exact pair alignment retained. Logical chunk reads, not physical SSD bytes. No paired generated-history NLL or additive claim.',
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
    result['candidate_change_vs_existing_mean_percent'] = {
        arm: {k: 100 * (by_arm[arm][k] / ((by_arm['existing-before'][k] + by_arm['existing-after'][k]) / 2) - 1)
              for k in metrics} for arm in ['padded', 'canonical-padded']}
    result['canonical_padded_change_vs_padded_percent'] = {
        k: 100 * (by_arm['canonical-padded'][k] / by_arm['padded'][k] - 1) for k in metrics}
    result['text_and_usage_vs_existing_before'] = {}
    for arm in by_arm:
        result['text_and_usage_vs_existing_before'][arm] = {}
        for row in [s for s in steps if s['arm'] == arm]:
            baseline = next(s for s in steps if s['arm'] == 'existing-before' and s['step'] == row['step'])
            result['text_and_usage_vs_existing_before'][arm][row['step']] = dict(
                text_byte_equal=baseline['text'].encode() == row['text'].encode(),
                grade_equal=baseline['pass'] == row['pass'],
                token_totals_equal=all(baseline[k] == row[k] for k in ['output_tokens', 'input_tokens', 'cached_tokens']))
    grafts = []
    for group in groups:
        for line in group['route_log_lines']:
            if 'suffix APPROXIMATE' not in line: continue
            fields = dict(re.findall(r'(\w+)=([\d:]+)', line))
            row = {k: group[k] for k in ['arm', 'phase', 'group']}
            for key, value in fields.items(): row[key] = value if ':' in value else int(value)
            row['graft_seconds'] = float(re.search(r'([\d.]+) ms$', line)[1]) / 1000
            # This fixed fixture's tool-result continuation starts at1531; all
            # primary queries start1344..1384. Retain the actual coordinates.
            row['request_part'] = 'tool-result continuation' if row['latest_user_start'] > 1400 else 'edited primary'
            row['raw_log'] = line; grafts.append(row)
    result['grafts'] = grafts
    for arm in by_arm:
        selected = [g for g in grafts if g['arm'] == arm]
        by_arm[arm]['padded_grafts'] = sum(g['prefix_pad'] for g in selected)
        by_arm[arm]['primary_grafts'] = sum(g['request_part'] == 'edited primary' for g in selected)
        by_arm[arm]['continuation_grafts'] = sum(g['request_part'] == 'tool-result continuation' for g in selected)
        by_arm[arm]['approximate_source_grafts'] = sum(g['source_approximate'] for g in selected)
    result['actual_reply_usage'] = {}
    for arm in by_arm:
        result['actual_reply_usage'][arm] = {}
        for phase in ['before', 'after']:
            raw = json.loads((args.input / arm / 'http' / phase / 'result.json').read_text())
            for row in raw['steps']:
                result['actual_reply_usage'][arm][row['step']] = [r['usage'] for r in row['replies']]
    previous.save(args.output / 'results.json', result)
    for name, rows in [('steps', steps), ('groups', groups), ('cache-shutdowns', caches), ('grafts', grafts)]:
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
        axes[0].set_xticks(x, ['Existing before', 'PAD', 'Canonical + PAD', 'Existing after'])
        axes[0].set_ylim(0, max(by_arm[a]['launcher_wall_seconds'] for a in labels) * 1.13)
        axes[0].set(title='Whole and edited-group wall', xlabel='Measured arm order', ylabel='Seconds'); axes[0].legend(frameon=False)
        for index, (arm, color) in enumerate(zip(labels, ['#65717e', '#e89930', '#247ba0', '#a5b1be'])):
            names = ['edited-AB', 'edited-CD']
            values = [next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name) for name in names]
            axes[1].bar([i + (index - 1.5) * .20 for i in range(2)], values, width=.20, label=arm, color=color)
        axes[1].set_xticks([0, 1], ['Edited AB\nbefore restart', 'Edited CD\nafter restart'])
        axes[1].set(title='Two-client group makespan', ylabel='Seconds'); axes[1].legend(frameon=False)
        for arm, offset in zip(labels, [(0, 18), (-35, 18), (0, -45), (0, -35)]):
            value = by_arm[arm]
            axes[2].scatter(value['launcher_wall_seconds'], value['passes'], s=65)
            axes[2].annotate(arm, (value['launcher_wall_seconds'], value['passes']),
                xytext=offset, textcoords='offset points', ha='center', fontsize=8,
                arrowprops={'arrowstyle': '-', 'color': '#777'})
        axes[2].set(title='Whole-cost / strict-quality Pareto', xlabel='Launcher wall (s)', ylabel='Strict passing steps of 8', ylim=(0, 9.5))
        axes[2].set_yticks([0, 4, 7, 8]); axes[2].margins(x=.25)
        for ax in axes: ax.tick_params(axis='x', rotation=20)
        for ax in axes: ax.grid(axis='y', alpha=.15)
        fig.suptitle('Parity padding: all 32 fresh reads and 16 edited answers pass; combined warm format failure retained', fontsize=12)
        fig.savefig(args.output / 'parity-suffix-tradeoff.png', dpi=180)
        fig.savefig(args.output / 'parity-suffix-tradeoff.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
