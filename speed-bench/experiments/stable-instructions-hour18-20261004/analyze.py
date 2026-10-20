#!/usr/bin/env python3
"""Actual first-attempt task/role scores; no paired NLL on divergent histories."""
import argparse
import csv
import json
import os
import os
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];totals={};summaries={}
    for arm in ['staged-before','natural','staged-after']:
        folder=args.input/arm
        if not (folder/'checks/results.json').exists():continue
        stderr=(folder/'server.stderr.log').read_text(errors='replace')
        summaries[arm]=[dict(id=i,context=c,budget=int(b),thinking_tokens=int(t),forced=int(f),deferred=int(d))
            for i,c,b,t,f,d in re.findall(r'reasoning_summary id=(\S+) ctx=(\S+) budget=(\d+) thinking_tokens=(\d+) forced=(\d+) deferred=(\d+)',stderr)]
        total={'cases':0,'passes':0,'code_passes':0,'caps':0,'wall_seconds':0,
               'output_tokens':0,'reasoning_tokens':0,'role_seconds':{},'role_thinking':{},
               'role_cached_tokens':{},'role_fresh_input_tokens':{},
               'forced':sum(r['forced'] for r in summaries[arm]),
               'deferred':sum(r['deferred'] for r in summaries[arm])}
        for row in json.loads((folder/'checks/results.json').read_text()):
            flat={k:row[k] for k in ['case','arm','protocol','write_budget','wall_seconds','complete','cap','pass',
                                     'actual_read_count','actual_write_count','http_output_tokens','http_reasoning_tokens']}
            flat.update(code_correct=row['grade'].get('correct',False),tests_passed=row['grade'].get('tests'),
                        tests_expected=161,errors='; '.join(row['errors']))
            for role in ['READ','WRITE','FINAL']:
                reply=next((r for r in row['replies'] if r['role']==role),{})
                usage=reply.get('usage') or {}
                thinking=(usage.get('output_tokens_details') or {}).get('reasoning_tokens',0)
                flat[role.lower()+'_seconds']=reply.get('seconds')
                flat[role.lower()+'_thinking_tokens']=thinking
                flat[role.lower()+'_status']=reply.get('response_status')
                flat[role.lower()+'_input_tokens']=usage.get('input_tokens')
                cached=(usage.get('input_tokens_details') or {}).get('cached_tokens',0)
                flat[role.lower()+'_cached_tokens']=cached
                flat[role.lower()+'_fresh_input_tokens']=usage.get('input_tokens',0)-cached
                total['role_seconds'][role]=total['role_seconds'].get(role,0)+reply.get('seconds',0)
                total['role_thinking'][role]=total['role_thinking'].get(role,0)+thinking
                total['role_cached_tokens'][role]=total['role_cached_tokens'].get(role,0)+cached
                total['role_fresh_input_tokens'][role]=total['role_fresh_input_tokens'].get(role,0)+usage.get('input_tokens',0)-cached
            rows.append(flat)
            total['cases']+=1;total['passes']+=row['pass'];total['code_passes']+=flat['code_correct']
            total['caps']+=row['cap'];total['wall_seconds']+=row['wall_seconds']
            total['output_tokens']+=row['http_output_tokens'];total['reasoning_tokens']+=row['http_reasoning_tokens']
        totals[arm]=total
    result={'rows':rows,'by_arm':totals,'budget_summaries':summaries,
            'scope':'One protocol change: staged vs stable task.system, all WRITE128, same frozen tasks. Actual first-attempt histories, no paired NLL/KL.'}
    result['staged_repeat']={}
    before=args.input/'staged-before/checks/results.json'
    after=args.input/'staged-after/checks/results.json'
    if before.exists() and after.exists():
        for first,last in zip(json.loads(before.read_text()),json.loads(after.read_text())):
            a=Path(first['first_code_path']).read_bytes() if first['first_code_path'] else None
            b=Path(last['first_code_path']).read_bytes() if last['first_code_path'] else None
            result['staged_repeat'][first['case']]={'same_case':first['case']==last['case'],
                'first_code_equal':a==b,'final_equal':first['final_text']==last['final_text'],
                'quality_equal':first['pass']==last['pass'] and first['grade']==last['grade'],
                'role_usage_equal':[(r['role'],r['usage']) for r in first['replies']]==[(r['role'],r['usage']) for r in last['replies']]}
        result['staged_after_vs_before_wall_percent']=100*(totals['staged-after']['wall_seconds']/totals['staged-before']['wall_seconds']-1)
        mean=(totals['staged-before']['wall_seconds']+totals['staged-after']['wall_seconds'])/2
        result['attempted_wall_change_vs_staged_mean_percent']={a:100*(v['wall_seconds']/mean-1) for a,v in totals.items()}
    result['completed_natural_wall_change_vs_staged_case_mean_percent']={}
    for case in ['ledger','lru']:
        groups=[next((r for r in rows if r['case']==case and r['arm']==arm),None) for arm in ['staged-before','natural','staged-after']]
        if all(r and r['pass'] for r in groups):
            result['completed_natural_wall_change_vs_staged_case_mean_percent'][case]=100*(groups[1]['wall_seconds']/((groups[0]['wall_seconds']+groups[2]['wall_seconds'])/2)-1)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'results.csv').open('w',newline='') as fp:
        if rows:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))
    if args.plot and len(rows)==6:
        os.environ['MPLCONFIGDIR']='/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        arms=['staged-before','natural','staged-after'];labels=['Staged before','Natural','Staged after']
        fig,axes=plt.subplots(1,3,figsize=(15.5,5),constrained_layout=True)
        ax=axes[0]
        for arm,label,color,offset in zip(arms,labels,['#65717e','#247ba0','#9aa5af'],[(-90,-33),(-20,20),(-90,25)]):
            x,y=totals[arm]['wall_seconds'],totals[arm]['passes']
            ax.scatter(x,y,s=65,color=color)
            ax.annotate(label+'\n'+f'{x:.1f} s',(x,y),xytext=offset,textcoords='offset points',fontsize=9,
                        arrowprops={'arrowstyle':'-','color':color})
        ax.set(title='Completed task Pareto',xlabel='Summed READ + WRITE + FINAL wall (s)',ylabel='Completed correct tasks (161 checks each)',ylim=(0,2.6))
        ax.set_yticks([0,1,2]);ax.margins(x=.2);ax.grid(alpha=.15)
        for ax,key,title,ylabel in [(axes[1],'role_fresh_input_tokens','Actual fresh prefill by role','Fresh input tokens, both tasks'),
                                   (axes[2],'role_seconds','HTTP wall by role','Seconds, both tasks')]:
            bottom=[0.]*3
            for role,color in [('READ','#247ba0'),('WRITE','#e89930'),('FINAL','#40916c')]:
                values=[totals[a][key][role] for a in arms]
                ax.bar(labels,values,bottom=bottom,label=role,color=color)
                bottom=[b+v for b,v in zip(bottom,values)]
            ax.set(title=title,ylabel=ylabel,ylim=(0,max(bottom)*1.15))
            ax.tick_params(axis='x',rotation=20);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
        fig.suptitle('Stable instructions vs stage directives: all arms WRITE128, suffix approximation off',fontsize=12)
        fig.savefig(args.output/'protocol-pareto.png',dpi=180)
        fig.savefig(args.output/'protocol-pareto.svg')
        plt.close(fig)


if __name__=='__main__':main()
