#!/usr/bin/env python3
"""Actual first-attempt task/role scores; no paired NLL on divergent histories."""
import argparse
import csv
import json
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
    for arm in ['budget128','budget256','native']:
        folder=args.input/arm
        if not (folder/'checks/results.json').exists():continue
        stderr=(folder/'server.stderr.log').read_text(errors='replace')
        summaries[arm]=[dict(id=i,context=c,budget=int(b),thinking_tokens=int(t),forced=int(f),deferred=int(d))
            for i,c,b,t,f,d in re.findall(r'reasoning_summary id=(\S+) ctx=(\S+) budget=(\d+) thinking_tokens=(\d+) forced=(\d+) deferred=(\d+)',stderr)]
        total={'cases':0,'passes':0,'code_passes':0,'caps':0,'wall_seconds':0,
               'output_tokens':0,'reasoning_tokens':0,'role_seconds':{},'role_thinking':{},
               'forced':sum(r['forced'] for r in summaries[arm]),
               'deferred':sum(r['deferred'] for r in summaries[arm])}
        for row in json.loads((folder/'checks/results.json').read_text()):
            flat={k:row[k] for k in ['case','arm','write_budget','wall_seconds','complete','cap','pass',
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
                total['role_seconds'][role]=total['role_seconds'].get(role,0)+reply.get('seconds',0)
                total['role_thinking'][role]=total['role_thinking'].get(role,0)+thinking
            rows.append(flat)
            total['cases']+=1;total['passes']+=row['pass'];total['code_passes']+=flat['code_correct']
            total['caps']+=row['cap'];total['wall_seconds']+=row['wall_seconds']
            total['output_tokens']+=row['http_output_tokens'];total['reasoning_tokens']+=row['http_reasoning_tokens']
        totals[arm]=total
    result={'rows':rows,'by_arm':totals,'budget_summaries':summaries,
            'scope':'Two new contracts, first attempt, no repair; whole READ/WRITE/FINAL excludes startup/grading. No paired NLL/KL.'}
    if 'native' in totals:
        result['attempted_wall_change_vs_native_percent']={a:100*(v['wall_seconds']/totals['native']['wall_seconds']-1) for a,v in totals.items()}
        result['completed_task_wall_change_vs_native_percent']={}
        for row in rows:
            baseline=next((r for r in rows if r['arm']=='native' and r['case']==row['case']),None)
            if row['pass'] and baseline and baseline['pass']:
                result['completed_task_wall_change_vs_native_percent'][row['case']+'/'+row['arm']]=100*(row['wall_seconds']/baseline['wall_seconds']-1)
    (args.output/'practical-results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'practical-results.csv').open('w',newline='') as fp:
        if rows:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))
    if args.plot and len(rows)==6:
        os.environ['MPLCONFIGDIR']='/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,5.3),constrained_layout=True)
        arms=['budget128','budget256','native'];labels=['WRITE128','WRITE256','Native']
        lru=[next(r for r in rows if r['case']=='lru' and r['arm']==arm) for arm in arms]
        ledger=[next(r for r in rows if r['case']=='ledger' and r['arm']==arm) for arm in arms]
        ax=axes[0]
        bars=ax.bar(labels,[r['wall_seconds'] for r in lru],color=['#247ba0','#e89930','#65717e'])
        for b,r in zip(bars,lru):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+3,f"{r['wall_seconds']:.1f} s\n161/161, complete",ha='center',fontsize=9)
        ax.set(title='LRU: completed-task latency',ylabel='READ + WRITE + FINAL task wall (s)',ylim=(0,205))
        ax.grid(axis='y',alpha=.15)
        ax=axes[1];bottom=[0.]*3
        for role,color in [('READ','#247ba0'),('WRITE','#e89930'),('FINAL','#40916c')]:
            values=[r[role.lower()+'_seconds'] or 0 for r in ledger]
            ax.bar(labels,values,bottom=bottom,label=role,color=color)
            bottom=[b+v for b,v in zip(bottom,values)]
        for i,(r,status) in enumerate(zip(ledger,['161/161\nFINAL complete','161/161 code\nFINAL capped','WRITE capped\nno code'])):
            ax.text(i,bottom[i]+3,f"{r['wall_seconds']:.1f} s\n{status}",ha='center',fontsize=9)
        ax.set(title='Ledger: fixed-budget attempts',ylabel='Attempted task wall (s)',ylim=(0,280))
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,loc='upper left')
        fig.suptitle('Two new first-attempt tasks; 2048 output allowance shared across request roles',fontsize=12)
        fig.savefig(args.output/'practical-pareto.png',dpi=180)
        fig.savefig(args.output/'practical-pareto.svg')
        plt.close(fig)


if __name__=='__main__':main()
