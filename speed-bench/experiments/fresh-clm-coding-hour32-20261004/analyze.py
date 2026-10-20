#!/usr/bin/env python3
"""Retained first attempts, role costs and actual suffix geometry; no model retries."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

ARMS = ['native-before', 'context-fast', 'persistent-fast', 'native-after']
CASES = ['dedupe', 'bucket']


def read(path):
    return json.loads(path.read_text())


def public_output(body):
    return [{k: v for k, v in item.items() if k not in ['id', 'call_id']} for item in body['output']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    records, rows, totals, grafts, continuations, reasoning, caches, guards, routes = {}, [], {}, [], [], [], [], [], []
    for arm in ARMS:
        total = dict(task_wall_seconds=0, launcher_wall_seconds=read(a.input/arm/'launcher-summary.json')['seconds'],
            raw_driver_passes=0, saved_code_passes=0, qualified_passes=0, pipeline_completions=0, grade_passed_checks=0, grade_failed_checks=0, grade_unresolved_checks=0, caps=0, reads=0, writes=0, output_tokens=0,
            thinking_tokens=0, role_seconds={}, role_thinking={}, role_cached={}, role_fresh={})
        records[arm] = {}
        for case in CASES:
            folder = a.input/arm/'http'/case/case
            row = read(folder/'result.json'); records[arm][case] = row
            saved_grade = row['grade']
            passed_checks = saved_grade.get('passed', 0)
            failed_checks = saved_grade.get('failed', 0)
            unresolved_checks = 161 - passed_checks - failed_checks
            code = (folder/'first-solution.py').read_bytes() if (folder/'first-solution.py').exists() else None
            write_call = next((c for c in row['calls'] if c['role'] == 'WRITE'), None)
            if code is not None:
                assert write_call and code == json.loads(write_call['arguments'])['content'].encode()
            read_call = next((c for c in row['calls'] if c['role'] == 'READ'), None)
            if read_call and read_call.get('executed'):
                assert json.loads(read_call['output']) == read(folder/'task.json')['config']
            pipeline = bool(row['complete'] and not row['cap'] and not row['errors'] and row['config_unchanged']
                and row['actual_read_count'] == row['actual_write_count'] == 1
                and all(c['executed'] for c in row['calls']))
            qualified = bool(pipeline and saved_grade.get('correct') and saved_grade.get('tests') == 161)
            flat = dict(arm=arm, case=case, wall_seconds=row['wall_seconds'], raw_driver_pass=row['pass'],
                raw_grade_correct=row['grade'].get('correct'), raw_grade_reason=row['grade'].get('reason'),
                grade_source='original driver grade', grade_expected_checks=161, grade_passed_checks=passed_checks, grade_failed_checks=failed_checks, grade_unresolved_checks=unresolved_checks,
                saved_code_correct=saved_grade.get('correct'), saved_code_tests=saved_grade.get('tests'),
                pipeline_complete=pipeline, qualified_pass=qualified, cap=row['cap'], reads=row['actual_read_count'],
                writes=row['actual_write_count'], first_code_bytes=len(code) if code is not None else None, output_tokens=row['http_output_tokens'],
                thinking_tokens=row['http_reasoning_tokens'], final_text=row['final_text'])
            total['task_wall_seconds'] += row['wall_seconds']
            for k, v in dict(raw_driver_passes=row['pass'], saved_code_passes=saved_grade.get('correct', False),
                qualified_passes=qualified, pipeline_completions=pipeline, grade_passed_checks=passed_checks, grade_failed_checks=failed_checks, grade_unresolved_checks=unresolved_checks, caps=row['cap'], reads=row['actual_read_count'], writes=row['actual_write_count'],
                output_tokens=row['http_output_tokens'], thinking_tokens=row['http_reasoning_tokens']).items(): total[k] += v
            for reply in row['replies']:
                role, usage = reply['role'], reply['usage']; cached = usage['input_tokens_details']['cached_tokens']
                values = dict(seconds=reply['seconds'], thinking=usage['output_tokens_details']['reasoning_tokens'],
                    cached=cached, fresh=usage['input_tokens']-cached, input_tokens=usage['input_tokens'], output_tokens=usage['output_tokens'])
                flat.update({role.lower()+'_'+k: v for k, v in values.items()})
                for key in ['seconds', 'thinking', 'cached', 'fresh']:
                    total['role_'+key][role] = total['role_'+key].get(role, 0) + values[key]
            rows.append(flat)
            log = (a.input/arm/f'{case}-server.stderr.log').read_text()
            for line in log.splitlines():
                common = dict(arm=arm, case=case)
                if 'suffix APPROXIMATE' in line:
                    fields = {k: v if ':' in v else int(v) for k, v in re.findall(r'(\w+)=([\d:]+)', line)}
                    fields['graft_seconds'] = float(re.search(r'([\d.]+) ms$', line)[1])/1000
                    grafts.append(dict(**common, **fields, raw_log=line))
                if 'responses live continuation' in line:
                    m = re.search(r'match=(\S+) ids=(\d+) cached=(\d+) prompt=(\d+) approximate=(\d+)', line)
                    if m:
                        kind, ids, cached, prompt, approximate = m.groups()
                        continuations.append(dict(**common, match=kind, ids=int(ids), cached=int(cached), prompt=int(prompt),
                            fresh=int(prompt)-int(cached), approximate=int(approximate), raw_log=line))
                if 'reasoning_summary ' in line:
                    m = re.search(r'ctx=(\S+) budget=(\d+) thinking_tokens=(\d+) forced=(\d+) deferred=(\d+)', line)
                    context, budget, thinking, forced, deferred = m.groups()
                    reasoning.append(dict(**common, context=context, budget=int(budget), thinking=int(thinking), forced=int(forced), deferred=int(deferred)))
                if line.startswith('ds4: resident-route '):
                    fields = {k: float(v) if '.' in v else int(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
                    routes.append(dict(**common, **fields, raw_log=line,
                        native_weight_lines=[x for x in log.splitlines() if x.startswith('ds4: resident-route-native-weights ')]))
                if 'state_cache_summary ' in line:
                    caches.append(dict(**common, **json.loads(line.split('state_cache_summary ',1)[1])))
        total['forced'] = sum(r['forced'] for r in reasoning if r['arm'] == arm)
        total['deferred'] = sum(r['deferred'] for r in reasoning if r['arm'] == arm)
        total['grafts'] = sum(r['arm'] == arm for r in grafts)
        total['approximate_visible_continuations'] = sum(r['arm'] == arm and r['approximate'] for r in continuations)
        total['logical_disk_read_ops'] = sum(r['disk_reads'] for r in caches if r['arm'] == arm)
        totals[arm] = total
        guard = a.input/('guard-'+arm); samples = [json.loads(x) for x in (guard/'pressure.jsonl').read_text().splitlines()]
        guards.append(dict(arm=arm, **read(guard/'result.json'), samples=len(samples), levels=sorted(set(x['level'] for x in samples)),
            swap_first_gb=samples[0]['swap_gb'], swap_last_gb=samples[-1]['swap_gb'],
            peak_wired_compressor_gb=max(x['wired_gb']+x['compressor_gb'] for x in samples), max_consecutive_bad=max(x['consecutive_bad'] for x in samples)))
    controls = {}
    for arm in ARMS:
        controls[arm] = {}
        for case in CASES:
            first, other = records['native-before'][case], records[arm][case]
            folders = [a.input/label/'http'/case/case for label in ['native-before',arm]]
            controls[arm][case] = dict(first_code_bytes_equal=((folders[0]/'first-solution.py').read_bytes() == (folders[1]/'first-solution.py').read_bytes()) if all((f/'first-solution.py').exists() for f in folders) else None,
                final_text_equal=first['final_text'] == other['final_text'],
                role_usage_equal=[r['usage'] for r in first['replies']] == [r['usage'] for r in other['replies']],
                public_output_equal=[public_output(read(folders[0]/f'response-{i}.json')) == public_output(read(folders[1]/f'response-{i}.json')) for i in range(min(len(first['replies']),len(other['replies'])))],
                reply_count_equal=len(first['replies'])==len(other['replies']),
                read_request_equal=read(folders[0]/'request-0.json') == read(folders[1]/'request-0.json'),
                source_and_edited_context_equal=all((folders[0]/name).read_bytes() == (folders[1]/name).read_bytes() for name in ['context.before.txt','context.after.txt']))
    profiles = {}
    for arm in ARMS:
        executions = [read(a.input/arm/f'{case}-execution.json') for case in CASES]
        envs = [e['environment'] for e in executions]
        fields = ['DS4_METAL_V41_RESIDENT_ROUTE_GAP', 'DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS', 'DS4_STATE_CACHE_DIR']
        assert all(envs[0][k] == envs[1][k] for k in fields)
        profiles[arm] = dict(cache=envs[0]['DS4_STATE_CACHE_DIR'],
            route_environment={k:v for k,v in envs[0].items() if 'RESIDENT_ROUTE' in k},
            route_profile_same_after_restart=True)
    assert len({p['cache'] for p in profiles.values()}) == len(ARMS)
    route_totals = {arm:{key:sum(r[key] for r in routes if r['arm'] == arm) for key in
        ['layers','substitutions','native_misses','skipped_loads','skipped_bytes','double_layers','triple_layers','drops']}
        for arm in ARMS}
    mean = {k:(totals['native-before'][k]+totals['native-after'][k])/2 for k in ['task_wall_seconds','launcher_wall_seconds']}
    result = dict(rows=rows, by_arm=totals, controls_vs_native_before=controls, grafts=grafts, visible_continuations=continuations,
        reasoning_summaries=reasoning, cache_shutdowns=caches, guard_summaries=guards, route_summaries=routes, route_totals=route_totals, source_profiles=profiles, native_mean=mean,
        change_vs_native_mean_percent={arm:{k:100*(t[k]/mean[k]-1) for k in mean} for arm,t in totals.items()},
        native_after_vs_before_percent={k:100*(totals['native-after'][k]/totals['native-before'][k]-1) for k in mean},
        scope='Fresh dedupe/bucket first attempts. Native suffix0/GAP0; context-fast suffix64/GAP0; persistent-fast suffix64/strong max3. READ0/WRITE128/FINAL1 fixed. Actual passed and failed checks are separate from unresolved checks, pipeline completion, caps and all-case task pass. No model retries, paired NLL, additive gain or general quality claim.')
    result['change_vs_faster_native_percent'] = {arm:{key:100*(totals[arm][key]/min(totals['native-before'][key],totals['native-after'][key])-1)
        for key in mean} for arm in ARMS}
    result['case_change_vs_native_mean_percent'] = {arm:{case:100*(records[arm][case]['wall_seconds']/
        ((records['native-before'][case]['wall_seconds']+records['native-after'][case]['wall_seconds'])/2)-1)
        for case in CASES} for arm in ARMS}
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in [('tasks',rows),('grafts',grafts),('visible-continuations',continuations),('reasoning',reasoning),('cache-shutdowns',caches),('routes',routes)]:
        if not data:
            (a.output/(name+'.csv')).write_text('')
            continue
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps({k:result[k] for k in ['by_arm','change_vs_native_mean_percent','native_after_vs_before_percent','controls_vs_native_before','guard_summaries']},indent=2))
    if a.plot:
        os.environ['MPLCONFIGDIR']='/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15.5,5),constrained_layout=True)
        labels=['Native before','Context fast','Persistent fast','Native after']
        for arm,label,offset in zip(ARMS,labels,[(-70,-40),(5,20),(5,25),(0,25)]):
            t=totals[arm];axes[0].scatter(t['task_wall_seconds'],t['qualified_passes'],s=65)
            axes[0].annotate(f"{label}\n{t['task_wall_seconds']:.1f}s",(t['task_wall_seconds'],t['qualified_passes']),xytext=offset,textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':'#777'})
        axes[0].set(title='Retained first-attempt task Pareto',xlabel='Whole READ + edit + WRITE + FINAL (s)',ylabel='Correct pipelines / saved 161-check grades',ylim=(0,2.7));axes[0].set_yticks([0,1,2]);axes[0].margins(x=.25)
        for ax,key,title,ylabel in [(axes[1],'role_fresh','Actual fresh input by role','Tokens, both tasks'),(axes[2],'role_seconds','HTTP role wall','Seconds, both tasks')]:
            bottom=[0.]*len(ARMS)
            for role,color in [('READ','#247ba0'),('WRITE','#e89930'),('FINAL','#40916c')]:
                values=[totals[arm][key].get(role,0) for arm in ARMS];ax.bar(labels,values,bottom=bottom,label=role,color=color);bottom=[b+v for b,v in zip(bottom,values)]
            ax.set(title=title,ylabel=ylabel,ylim=(0,max(bottom)*1.15));ax.tick_params(axis='x',rotation=20);ax.legend(frameon=False)
        for ax in axes:ax.grid(axis='y',alpha=.15)
        fig.suptitle('Fresh CLM coding: dedupe + bucket, READ0 / WRITE128 / FINAL1 fixed',fontsize=12)
        fig.savefig(a.output/'fresh-clm-pareto.png',dpi=180);fig.savefig(a.output/'fresh-clm-pareto.svg');plt.close(fig)


if __name__ == '__main__':
    main()
