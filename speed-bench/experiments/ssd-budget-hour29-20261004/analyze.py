#!/usr/bin/env python3
"""SSD quota tradeoff on concurrent suffix reuse."""
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
    for label, capacity in [('control-before', 4096), ('ssd512', 512), ('ssd128', 128), ('control-after', 4096)]:
        folder = args.input / label
        a, b, c = previous.arm(folder, label, fixture['sessions'], fixture['edited_archive_codes'])
        steps += a; groups += b; by_arm[label] = c
        launcher = json.loads((folder / 'launcher-summary.json').read_text())
        assert launcher['hot_mb'] == 0 and launcher['ssd_mb'] == capacity
        for phase in ['before', 'after']:
            log = (folder / f'{phase}-server.stderr.log').read_text(errors='replace')
            summaries = [json.loads(line.split('state_cache_summary ', 1)[1])
                         for line in log.splitlines() if 'state_cache_summary ' in line]
            assert len(summaries) == 1, (label, phase, 'expected one shutdown summary', len(summaries))
            summary = summaries[0]
            summary.update(arm=label, phase=phase, configured_ssd_mb=capacity,
                           configured_ssd_bytes=capacity * 1024 * 1024,
                           ssd_bytes_within_configured_cap=summary['ssd_bytes'] <= capacity * 1024 * 1024,
                           admission_warning_lines=[line for line in log.splitlines() if 'cache' in line.lower() and any(word in line.lower() for word in ['warning', 'store failed', 'insufficient', 'quota', 'admission', 'store skipped'])])
            caches.append(summary)
        c['hot_read_ops'] = sum(s['hot_reads'] for s in caches if s['arm'] == label)
        c['disk_read_ops'] = sum(s['disk_reads'] for s in caches if s['arm'] == label)
        c['endpoint_evictions'] = sum(s['endpoint_evictions'] for s in caches if s['arm'] == label)
    metrics = ['launcher_wall_seconds', 'client_phase_wall_seconds', 'edited_group_makespan_seconds']
    result = dict(by_arm=by_arm, steps=steps, groups=groups, cache_shutdowns=caches,
        scope='Only SSD cache capacity changes:4096/512/128/4096MiB. Canonical1/PAD1/hot0/suffix64/GAP0 fixed. Preserve128warmformatfailure and source-coverage loss. Logical chunk reads are not physical SSD traffic. No paired generated-history NLL or additive claim.',
        change_vs_control_before_percent={arm: {k: 100 * (row[k] / by_arm['control-before'][k] - 1) for k in metrics}
                               for arm, row in by_arm.items()})
    result['group_change_vs_control_before_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'control-before' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['change_vs_control_after_percent'] = {
        arm: {k: 100 * (row[k] / by_arm['control-after'][k] - 1) for k in metrics}
        for arm, row in by_arm.items()}
    result['group_change_vs_control_after_percent'] = {
        name: {arm: 100 * (next(g['makespan_seconds'] for g in groups if g['arm'] == arm and g['group'] == name)
                            / next(g['makespan_seconds'] for g in groups if g['arm'] == 'control-after' and g['group'] == name) - 1)
               for arm in by_arm}
        for name in ['warm-AB', 'warm-CD', 'edited-AB', 'edited-CD']}
    result['control_bracket_after_vs_before_percent'] = {
        k: 100 * (by_arm['control-after'][k] / by_arm['control-before'][k] - 1)
        for k in metrics}
    result['change_vs_control_mean_percent'] = {arm: {k:100*(by_arm[arm][k]/((by_arm['control-before'][k]+by_arm['control-after'][k])/2)-1) for k in metrics} for arm in by_arm}
    result['text_and_usage_vs_control_before'] = {}
    for arm in by_arm:
        result['text_and_usage_vs_control_before'][arm] = {}
        for row in [s for s in steps if s['arm'] == arm]:
            baseline = next(s for s in steps if s['arm'] == 'control-before' and s['step'] == row['step'])
            result['text_and_usage_vs_control_before'][arm][row['step']] = dict(
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
    result['public_reply_controls_vs_control_before'] = {}
    for arm in by_arm:
        selected = [g for g in grafts if g['arm'] == arm]
        by_arm[arm]['primary_grafts'] = sum(g['request_part'] == 'edited primary' for g in selected)
        by_arm[arm]['completion_grafts'] = sum(g['request_part'] == 'tool-result continuation' for g in selected)
        by_arm[arm]['approximate_source_grafts'] = sum(g['source_approximate'] for g in selected)
        by_arm[arm]['approximate_visible_continuations'] = sum(c['arm'] == arm and c['approximate'] for c in continuations)
        by_arm[arm]['phase_log_graft_count'] = sum((args.input / arm / (phase + '-server.stderr.log')).read_text().count('suffix APPROXIMATE') for phase in ['before', 'after'])
        assert by_arm[arm]['phase_log_graft_count'] == len(selected), arm
        result['actual_reply_usage'][arm] = {}
        result['public_reply_controls_vs_control_before'][arm] = {}
        for phase in ['before', 'after']:
            raw = json.loads((args.input / arm / 'http' / phase / 'result.json').read_text())
            for row in raw['steps']:
                result['actual_reply_usage'][arm][row['step']] = [r['usage'] for r in row['replies']]
                public = []
                for index in range(len(row['replies'])):
                    name = row['step'] + '-' + str(index) + '-response.json'
                    outputs = []
                    for label in ['control-before', arm]:
                        reply = json.loads((args.input / label / 'http' / phase / row['session'] / name).read_text())
                        outputs.append([{k: v for k, v in item.items() if k not in ['id', 'call_id']} for item in reply['output']])
                    public.append(outputs[0] == outputs[1])
                result['public_reply_controls_vs_control_before'][arm][row['step']] = dict(reply_output_equal=public,
                    reply_usage_equal=result['actual_reply_usage']['control-before'][row['step']] == result['actual_reply_usage'][arm][row['step']])
    result['guard_summaries'] = []
    for arm in by_arm:
        by_arm[arm]['edited_passes'] = sum(s['pass'] for s in steps if s['arm'] == arm and 'edited' in s['step'])
        guard = args.input / ('guard-'+arm)
        samples = [json.loads(x) for x in (guard/'pressure.jsonl').read_text().splitlines()]
        result['guard_summaries'].append(dict(arm=arm, **json.loads((guard/'result.json').read_text()), samples=len(samples),
            levels=sorted(set(x['level'] for x in samples)), swap_first_gb=samples[0]['swap_gb'], swap_last_gb=samples[-1]['swap_gb'],
            peak_wired_compressor_gb=max(x['wired_gb']+x['compressor_gb'] for x in samples), max_consecutive_bad=max(x['consecutive_bad'] for x in samples)))
    previous.save(args.output / 'results.json', result)
    for name, rows in [('steps', steps), ('groups', groups), ('cache-shutdowns', caches), ('grafts', grafts), ('visible-continuations', continuations)]:
        with (args.output / (name + '.csv')).open('w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['by_arm', 'change_vs_control_before_percent', 'group_change_vs_control_before_percent', 'cache_shutdowns']}, indent=2))
    if args.plot:
        os.environ['MPLCONFIGDIR'] = '/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15.5,5),constrained_layout=True)
        labels=['4096 before','512','128','4096 after']; arms=list(by_arm)
        for key,label in [('launcher_wall_seconds','Whole launcher'),('edited_group_makespan_seconds','Edited group sum')]:
            mean=(by_arm['control-before'][key]+by_arm['control-after'][key])/2
            axes[0].plot([128,512,4096],[by_arm['ssd128'][key],by_arm['ssd512'][key],mean],marker='o',label=label)
            axes[0].scatter([4096,4096],[by_arm['control-before'][key],by_arm['control-after'][key]],marker='x',color='#65717e')
        axes[0].set_xscale('log',base=2);axes[0].set_xticks([128,512,4096],['128','512','4096'])
        axes[0].set(title='Capacity vs wall (4096 mean + brackets)',xlabel='SSD-owned capacity (MiB)',ylabel='Seconds');axes[0].legend(frameon=False)
        for index,(arm,color) in enumerate(zip(arms,['#65717e','#e89930','#40916c','#a5b1be'])):
            values=[next(g['makespan_seconds'] for g in groups if g['arm']==arm and g['group']==name) for name in ['edited-AB','edited-CD']]
            axes[1].bar([i+(index-1.5)*.20 for i in range(2)],values,width=.20,label=labels[index],color=color)
        axes[1].set_xticks([0,1],['Edited AB','Edited CD\nafter restart']);axes[1].set(title='Two-client group makespan',ylabel='Seconds');axes[1].legend(frameon=False)
        x=list(range(4))
        axes[2].bar([v-.15 for v in x],[by_arm[arm]['passes'] for arm in arms],width=.30,label='Strict tasks /8',color='#247ba0')
        axes[2].bar([v+.15 for v in x],[by_arm[arm]['primary_grafts'] for arm in arms],width=.30,label='Edited source grafts /4',color='#40916c')
        axes[2].set_xticks(x,labels);axes[2].set(title='Retained quality and source coverage',ylabel='Actual count',ylim=(0,9));axes[2].legend(frameon=False)
        for ax in axes:ax.grid(axis='y',alpha=.15)
        fig.suptitle('SSD quota: all 32 reads / 16 edited facts correct; 128MiB loses sources and one warm format grade',fontsize=12)
        fig.savefig(args.output/'ssd-budget-tradeoff.png',dpi=180);fig.savefig(args.output/'ssd-budget-tradeoff.svg');plt.close(fig)


if __name__=='__main__':
    main()
