#!/usr/bin/env python3
"""Same-profile visible continuation of approximate checkpoints."""
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
    for label, capacity in [('before', 0), ('fixed', 0), ('after', 0)]:
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
        scope='Only visible continuation on same-profile approximate graphs changes. Canonical1/PAD1/hot0/suffix64 fixed. Native sources remain canonical; ID-only bypass unchanged. Logical chunk reads are not physical SSD traffic. No paired generated-history NLL or additive claim.',
        change_vs_old_before_percent={arm: {k: 100 * (row[k] / by_arm['before'][k] - 1) for k in metrics}
                               for arm, row in by_arm.items()})
    result['group_change_vs_old_before_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'before' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['change_vs_old_after_percent'] = {
        arm: {k: 100 * (row[k] / by_arm['after'][k] - 1) for k in metrics}
        for arm, row in by_arm.items()}
    result['group_change_vs_old_after_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'after' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['old_bracket_after_vs_before_percent'] = {
        k: 100 * (by_arm['after'][k] / by_arm['before'][k] - 1)
        for k in metrics}
    result['fixed_change_vs_old_mean_percent'] = {
        k: 100 * (by_arm['fixed'][k] / ((by_arm['before'][k] + by_arm['after'][k]) / 2) - 1)
        for k in metrics}
    result['text_and_usage_vs_old_before'] = {}
    for arm in by_arm:
        result['text_and_usage_vs_old_before'][arm] = {}
        for row in [s for s in steps if s['arm'] == arm]:
            baseline = next(s for s in steps if s['arm'] == 'before' and s['step'] == row['step'])
            result['text_and_usage_vs_old_before'][arm][row['step']] = dict(
                text_byte_equal=baseline['text'].encode() == row['text'].encode(),
                grade_equal=baseline['pass'] == row['pass'],
                token_totals_equal=all(baseline[k] == row[k] for k in ['output_tokens', 'input_tokens', 'cached_tokens']))
    grafts, continuations = [], []
    for group in groups:
        raw_group = next(g for g in json.loads((args.input / group['arm'] / 'http' / group['phase'] / 'result.json').read_text())['groups'] if g['group'] == group['group'])
        for line in raw_group['server_log_lines']:
            if 'suffix APPROXIMATE' in line:
                fields = dict(re.findall(r'(\w+)=([\d:]+)', line))
                row = {k: group[k] for k in ['arm', 'phase', 'group']}
                row.update({k: v if ':' in v else int(v) for k, v in fields.items()})
                row['graft_seconds'] = float(re.search(r'([\d.]+) ms$', line)[1]) / 1000
                row['request_part'] = 'tool-result continuation' if row['latest_user_start'] > 1400 else 'edited primary'
                row['raw_log'] = line; grafts.append(row)
            if 'responses live continuation' in line:
                match = re.search(r'match=(\S+) ids=(\d+) cached=(\d+) prompt=(\d+) approximate=(\d+)', line)
                if match:
                    kind, ids, cached, prompt, approximate = match.groups()
                    row = {k: group[k] for k in ['arm', 'phase', 'group']}
                    row.update(match=kind, ids=int(ids), cached=int(cached), prompt=int(prompt),
                               fresh=int(prompt)-int(cached), approximate=int(approximate), raw_log=line)
                    continuations.append(row)
    result['grafts'] = grafts; result['visible_continuations'] = continuations
    result['actual_reply_usage'] = {}
    result['public_reply_controls_vs_old_before'] = {}
    for arm in by_arm:
        selected = [g for g in grafts if g['arm'] == arm]
        by_arm[arm]['primary_grafts'] = sum(g['request_part'] == 'edited primary' for g in selected)
        by_arm[arm]['completion_grafts'] = sum(g['request_part'] == 'tool-result continuation' for g in selected)
        by_arm[arm]['approximate_source_grafts'] = sum(g['source_approximate'] for g in selected)
        by_arm[arm]['approximate_visible_continuations'] = sum(c['arm'] == arm and c['approximate'] for c in continuations)
        by_arm[arm]['phase_log_graft_count'] = sum((args.input / arm / (phase + '-server.stderr.log')).read_text().count('suffix APPROXIMATE') for phase in ['before', 'after'])
        assert by_arm[arm]['phase_log_graft_count'] == len(selected), arm
        result['actual_reply_usage'][arm] = {}
        result['public_reply_controls_vs_old_before'][arm] = {}
        for phase in ['before', 'after']:
            raw = json.loads((args.input / arm / 'http' / phase / 'result.json').read_text())
            for row in raw['steps']:
                result['actual_reply_usage'][arm][row['step']] = [r['usage'] for r in row['replies']]
                public = []
                for index in range(len(row['replies'])):
                    name = row['step'] + '-' + str(index) + '-response.json'
                    outputs = []
                    for label in ['before', arm]:
                        reply = json.loads((args.input / label / 'http' / phase / row['session'] / name).read_text())
                        outputs.append([{k: v for k, v in item.items() if k not in ['id', 'call_id']} for item in reply['output']])
                    public.append(outputs[0] == outputs[1])
                result['public_reply_controls_vs_old_before'][arm][row['step']] = dict(reply_output_equal=public,
                    reply_usage_equal=result['actual_reply_usage']['before'][row['step']] == result['actual_reply_usage'][arm][row['step']])
    previous.save(args.output / 'results.json', result)
    for name, rows in [('steps', steps), ('groups', groups), ('cache-shutdowns', caches), ('grafts', grafts), ('visible-continuations', continuations)]:
        with (args.output / (name + '.csv')).open('w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['by_arm', 'change_vs_old_before_percent', 'group_change_vs_old_before_percent', 'cache_shutdowns']}, indent=2))
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
        axes[0].set_xticks(x, ['Old before', 'Fixed', 'Old after'])
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
        axes[2].set(title='Actual suffix coverage', ylabel='Logged graft / fallback events'); axes[2].legend(frameon=False)
        for ax in axes: ax.tick_params(axis='x', rotation=20)
        for ax in axes: ax.grid(axis='y', alpha=.15)
        fig.suptitle('Approximate visible continuation: all 24 task/read checks pass, canonical1/PAD1 fixed', fontsize=12)
        fig.savefig(args.output / 'approximate-continuation-tradeoff.png', dpi=180)
        fig.savefig(args.output / 'approximate-continuation-tradeoff.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
