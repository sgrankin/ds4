#!/usr/bin/env python3
"""Score the recorded first attempts without opening a model."""
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
    rows=[];by_arm={};logs={};native={};budget_summaries={}
    for arm in ['native-before','budget512','budget256','native-after']:
        folder=args.input/arm
        if not (folder/'checks/results.json').exists():continue
        stderr=(folder/'server.stderr.log').read_text(errors='replace')
        # Preserve the complete matching records; no guessed fired/deferred count.
        logs[arm]=[line for line in stderr.splitlines() if re.search(r'(reasoning[_ -]?budget|budget.*(?:fired|deferred|forced)|forced.*budget)',line,re.I)]
        summary_pattern=r'reasoning_summary id=(\S+) ctx=(\S+) budget=(\d+) thinking_tokens=(\d+) forced=(\d+) deferred=(\d+)'
        summaries=[dict(response_id=i,context=c,budget=int(b),thinking_tokens=int(t),forced=int(f),deferred=int(d))
                   for i,c,b,t,f,d in re.findall(summary_pattern,stderr)]
        budget_summaries[arm]=summaries
        totals={'cases':0,'passes':0,'code_passes':0,'caps':0,'wall_seconds':0,
                'output_tokens':0,'reasoning_tokens':0,'role_tokens':{},'role_seconds':{},
                'budget_forced':sum(s['forced'] for s in summaries),
                'budget_deferred':sum(s['deferred'] for s in summaries)}
        for row in json.loads((folder/'checks/results.json').read_text()):
            roles={r['role']:r for r in row['replies']}
            flat={'arm':arm,'case':row['case'],'write_budget':row['write_budget'],
                'pass':row['pass'],'code_correct':row['grade'].get('correct',False),
                'tests_passed':row['grade'].get('tests') if row['grade'].get('correct') else None,
                'tests_expected':161,'complete':row['complete'],'cap':row['cap'],
                'actual_reads':row['actual_read_count'],'actual_writes':row['actual_write_count'],
                'wall_seconds':row['wall_seconds'],'output_tokens':row['http_output_tokens'],
                'reasoning_tokens':row['http_reasoning_tokens'],'errors':'; '.join(row['errors'])}
            for role in ['READ','WRITE','FINAL']:
                r=roles.get(role,{})
                usage=r.get('usage') or {}
                flat[role.lower()+'_seconds']=r.get('seconds')
                flat[role.lower()+'_output_tokens']=usage.get('output_tokens')
                flat[role.lower()+'_reasoning_tokens']=(usage.get('output_tokens_details') or {}).get('reasoning_tokens')
                flat[role.lower()+'_status']=r.get('response_status')
                counts=totals['role_tokens'].setdefault(role,{'output':0,'reasoning':0})
                counts['output']+=usage.get('output_tokens',0)
                counts['reasoning']+=(usage.get('output_tokens_details') or {}).get('reasoning_tokens',0)
                totals['role_seconds'][role]=totals['role_seconds'].get(role,0)+r.get('seconds',0)
            rows.append(flat)
            totals['cases']+=1;totals['passes']+=row['pass'];totals['code_passes']+=flat['code_correct']
            totals['caps']+=row['cap'];totals['wall_seconds']+=row['wall_seconds']
            totals['output_tokens']+=row['http_output_tokens'];totals['reasoning_tokens']+=row['http_reasoning_tokens']
            if arm.startswith('native'):
                code=Path(row['first_code_path']).read_bytes() if row['first_code_path'] else None
                native[(arm,row['case'])]=(code,row['final_text'],[(r['role'],r['usage']) for r in row['replies']])
        by_arm[arm]=totals
    native_repeat={case: native.get(('native-before',case))==native.get(('native-after',case))
                   for case in ['join','window_argmax']
                   if ('native-before',case) in native and ('native-after',case) in native}
    result={'rows':rows,'by_arm':by_arm,'native_code_final_usage_repeat':native_repeat,
        'budget_log_lines':logs,'budget_summaries':budget_summaries,
        'scope':'Actual divergent READ/WRITE/FINAL histories: task/code/token/wall scores; no paired teacher NLL or KL.'}
    if 'native-before' in by_arm and 'native-after' in by_arm:
        mean=(by_arm['native-before']['wall_seconds']+by_arm['native-after']['wall_seconds'])/2
        result['native_bracket_mean_wall_seconds']=mean
        result['native_bracket_after_vs_before_percent']=100*(by_arm['native-after']['wall_seconds']/by_arm['native-before']['wall_seconds']-1)
        result['wall_change_vs_native_mean_percent']={arm:100*(v['wall_seconds']/mean-1) for arm,v in by_arm.items()}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'results.csv').open('w',newline='') as fp:
        if rows:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'by_arm':by_arm,'native_repeat':native_repeat,'budget_log_lines':logs},indent=2))
    if args.plot and by_arm:
        os.environ['MPLCONFIGDIR']=str(Path('/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        arms=list(by_arm)
        labels=['Native before' if a=='native-before' else 'Native after' if a=='native-after' else a.replace('budget','WRITE ') for a in arms]
        fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
        colors={'native-before':'#65717e','native-after':'#9aa5af','budget512':'#e89930','budget256':'#247ba0'}
        ax=axes[0]
        for arm,label in zip(arms,labels):
            x,y=by_arm[arm]['wall_seconds'],by_arm[arm]['passes']
            ax.scatter(x,y,s=65,color=colors[arm])
            offset={'native-before':(-100,-32),'native-after':(-100,22),'budget512':(-45,22),'budget256':(-30,-32)}[arm]
            ax.annotate(label+'\n'+f'{x:.1f} s', (x,y),xytext=offset,textcoords='offset points',fontsize=9,
                        arrowprops={'arrowstyle':'-','color':colors[arm]})
        ax.set(title='Practical task Pareto',xlabel='Summed READ + WRITE + FINAL wall (s)',ylabel='Completed correct tasks (161 checks each)')
        ax.set_ylim(0,2.6);ax.set_yticks([0,1,2]);ax.margins(x=.2);ax.grid(alpha=.18)
        role_colors={'READ':'#247ba0','WRITE':'#e89930','FINAL':'#40916c'}
        for ax,metric,title,ylabel in [(axes[1],'seconds','HTTP wall by request role','Seconds, both tasks'),
                                     (axes[2],'thinking','Actual thinking by request role','Generated thinking tokens, both tasks')]:
            bottom=[0]*len(arms)
            for role in ['READ','WRITE','FINAL']:
                values=[by_arm[a]['role_seconds'][role] if metric=='seconds' else by_arm[a]['role_tokens'][role]['reasoning'] for a in arms]
                ax.bar(range(len(arms)),values,bottom=bottom,label=role,color=role_colors[role])
                bottom=[b+v for b,v in zip(bottom,values)]
            ax.set_xticks(range(len(arms)),labels,rotation=25,ha='right')
            ax.set(title=title,ylabel=ylabel);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
        fig.suptitle('WRITE-only thinking budget: two fixed first-attempt Python tasks, four serial HTTP arms',fontsize=13)
        fig.savefig(args.output/'task-pareto.png',dpi=180)
        fig.savefig(args.output/'task-pareto.svg')
        plt.close(fig)


if __name__=='__main__':main()
