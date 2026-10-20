#!/usr/bin/env python3
"""Four-session task isolation, queue-inclusive latency and group makespan."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def arm(directory,label,facts,edited_codes):
    steps,groups=[],[]
    phase_wall=0
    for phase in ['before','after']:
        result=json.loads((directory/'http'/phase/'result.json').read_text())
        phase_wall+=result['seconds']
        for row in result['steps']:
            session=row['session'];warm='warm' in row['group'];f=facts[session]
            gold='READY' if warm else {'active_total':f['batches']*f['items_per_batch']-f['reserved_items'],
                                     'archived_code':edited_codes[session]}
            try:answer=row['text'].strip() if warm else json.loads(row['text'])
            except ValueError:answer=None
            reads=[c for c in row['calls'] if c.get('read')]
            read_isolated=len(reads)==1 and reads[0].get('path')==row['required_read'] and Path(row['required_read']).parent.name==session
            replies=row['replies'];cap=any(r.get('response_status')=='incomplete' for r in replies)
            correct=answer==gold
            independent_pass=correct and read_isolated and row['complete'] and not row['errors']
            steps.append({'arm':label,'phase':phase,'group':row['group'],'step':row['step'],'session':session,
                'queue_inclusive_seconds':row['queue_inclusive_seconds'],'http_task_seconds':row['seconds'],
                'edit_seconds':row.get('edit_seconds',0),'correct':correct,'read_isolated':read_isolated,
                'actual_reads':len(reads),'complete':row['complete'],'cap':cap,'pass':independent_pass,
                'stored_pass':row['pass'],'grade_agrees':independent_pass==row['pass'],
                'output_tokens':sum((r.get('usage') or {}).get('output_tokens',0) for r in replies),
                'input_tokens':sum((r.get('usage') or {}).get('input_tokens',0) for r in replies),
                'cached_tokens':sum(((r.get('usage') or {}).get('input_tokens_details') or {}).get('cached_tokens',0) for r in replies),
                'text':row['text'],'errors':'; '.join(row['errors'])})
        for group in result['groups']:
            lines=group['server_log_lines']
            approx=[line for line in lines if 'suffix APPROXIMATE' in line]
            fallback=[line for line in lines if 'suffix exact fallback' in line]
            members=[s for s in steps if s['phase']==phase and s['group']==group['group']]
            groups.append({'arm':label,'phase':phase,'group':group['group'],'makespan_seconds':group['seconds'],
                'passes':sum(s['pass'] for s in members),'steps':len(members),
                'mean_queue_inclusive_seconds':statistics.mean(s['queue_inclusive_seconds'] for s in members),
                'max_queue_inclusive_seconds':max(s['queue_inclusive_seconds'] for s in members),
                'approximate_hits':len(approx),'ssd_approximate_hits':sum('source_ssd=1' in line for line in approx),
                'approximate_source_hits':sum('source_approximate=1' in line for line in approx),
                'exact_fallbacks':len(fallback),'exact_hit_log_count':len(group['exact_hit_log_lines']),
                'slot_eviction_log_count':len(group['slot_eviction_log_lines']),
                'route_log_lines':approx+fallback,'slot_eviction_log_lines':group['slot_eviction_log_lines']})
    launcher=json.loads((directory/'launcher-summary.json').read_text())
    summary={'steps':len(steps),'passes':sum(r['pass'] for r in steps),'caps':sum(r['cap'] for r in steps),
        'actual_reads':sum(r['actual_reads'] for r in steps),'grades_agree':all(r['grade_agrees'] for r in steps),
        'client_phase_wall_seconds':phase_wall,'launcher_wall_seconds':launcher['seconds'],
        'all_group_makespan_seconds':sum(g['makespan_seconds'] for g in groups),
        'edited_group_makespan_seconds':sum(g['makespan_seconds'] for g in groups if 'edited' in g['group']),
        'sum_queue_inclusive_seconds':sum(r['queue_inclusive_seconds'] for r in steps),
        'approximate_hits':sum(g['approximate_hits'] for g in groups),
        'ssd_approximate_hits':sum(g['ssd_approximate_hits'] for g in groups),
        'exact_fallbacks':sum(g['exact_fallbacks'] for g in groups),
        'slot_eviction_log_count':sum(g['slot_eviction_log_count'] for g in groups)}
    return steps,groups,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--reuse',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fixture=json.loads((HERE/'fixture.json').read_text());steps=[];groups=[];by_arm={}
    for directory,label in [(args.native,'native'),(args.reuse,'reuse64')]:
        a,b,c=arm(directory,label,fixture['sessions'],fixture['edited_archive_codes'])
        steps+=a;groups+=b;by_arm[label]=c
    result={'by_arm':by_arm,'steps':steps,'groups':groups,
        'scope':'Two clients in flight, two resident slots, four identities. Group makespan and queue-inclusive HTTP task latency do not prove parallel GPU execution. Logs counted once per group.'}
    result['attempted_reuse_vs_native_percent']={k:100*(by_arm['reuse64'][k]/by_arm['native'][k]-1)
        for k in ['client_phase_wall_seconds','launcher_wall_seconds','all_group_makespan_seconds','edited_group_makespan_seconds']}
    save(args.output/'results.json',result)
    for name,rows in [('steps',steps),('groups',groups)]:
        with (args.output/(name+'.csv')).open('w',newline='') as fp:
            if rows:
                writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'by_arm':by_arm,'changes':result['attempted_reuse_vs_native_percent']},indent=2))
    if args.plot:
        os.environ['MPLCONFIGDIR']='/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12.5,5.3),constrained_layout=True)
        names=['warm-AB','warm-CD','edited-AB','edited-CD']
        for label,color,offset in [('native','#65717e',-.18),('reuse64','#e89930',.18)]:
            selected=[next(g for g in groups if g['arm']==label and g['group']==name) for name in names]
            bars=axes[0].bar([i+offset for i in range(4)],[g['makespan_seconds'] for g in selected],width=.36,color=color,label=label)
            for bar,g in zip(bars,selected):
                axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f"{g['makespan_seconds']:.1f}\n{g['passes']}/2",ha='center',fontsize=8)
        axes[0].set_xticks(range(4),names,rotation=20)
        axes[0].set(title='Group makespan and strict grades',ylabel='Two-client group wall (s)',ylim=(0,112))
        axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.15)
        names=[s['step'] for s in steps if s['arm']=='native']
        for label,color,offset in [('native','#65717e',-.18),('reuse64','#e89930',.18)]:
            selected=[next(s for s in steps if s['arm']==label and s['step']==name) for name in names]
            axes[1].bar([i+offset for i in range(len(names))],[s['queue_inclusive_seconds'] for s in selected],width=.36,color=color,label=label)
        axes[1].set_xticks(range(len(names)),names,rotation=45,ha='right')
        axes[1].set(title='Individual queue-inclusive task latency',ylabel='Task submission to finish (s)',ylim=(0,105))
        axes[1].legend(frameon=False);axes[1].grid(axis='y',alpha=.15)
        fig.suptitle('Four sessions, two resident slots, two clients in flight; one reuse64 warm format failure',fontsize=12)
        fig.savefig(args.output/'concurrency-tradeoff.png',dpi=180)
        fig.savefig(args.output/'concurrency-tradeoff.svg')
        plt.close(fig)


if __name__=='__main__':main()
