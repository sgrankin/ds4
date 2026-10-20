#!/usr/bin/env python3
"""Actual repeated deletions, fresh reads and suffix-source chains; model-free."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

ARMS = ['native-before','reuse','native-after']
FACTS = {'B':(31,8,17,[f'ORBIT-{n}' for n in range(57,65)])}


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows,edits,grafts,continuations,caches,guards=[],[],[],[],[],[]; totals={}; controls={}
    for arm in ARMS:
        t=dict(passes=0,reads=0,caps=0,deletions=0,warm_seconds=0,edited_seconds_including_deletion=0,
            launcher_seconds=read(a.input/arm/'launcher-summary.json')['seconds'],phase_seconds={},round_seconds={},round_passes={})
        for phase in ['before','after']:
            folder=a.input/arm/'http'/phase; raw=read(folder/'result.json'); t['phase_seconds'][phase]=raw['seconds']
            for edit in raw['edits']:
                name=f"{edit['session']}-edit{edit['round']}"
                before=read(folder/(name+'-before-snapshot.json')); after=read(folder/(name+'-after-snapshot.json'))
                assert after['version']==before['version']+1
                assert after['items']==[item for item in before['items'] if item['id']!=edit['item_id']]
                assert len(before['items'])==len(after['items'])+1
                edits.append(dict(arm=arm,phase=phase,**edit));t['deletions']+=1
            for row in raw['steps']:
                round_number=row.get('round',0); session=row['session']; batches,items,reserved,codes=FACTS[session]
                gold='READY' if not round_number else dict(active_total=batches*items-reserved,archived_code=codes[round_number-1])
                try: answer=row['text'].strip() if not round_number else json.loads(row['text'])
                except ValueError: answer=None
                calls=[c for c in row['calls'] if c.get('read')]
                isolated=len(calls)==1 and Path(calls[0]['path']).parent.name==session and Path(calls[0]['path']).name==('archive.json' if round_number else 'active.json')
                if isolated:
                    observed= json.loads(calls[0]['text'])
                    fresh=observed.get('archived_code')==codes[round_number-1] if round_number else all(observed.get(k)==v for k,v in zip(['batches','items_per_batch','reserved_items'],[batches,items,reserved]))
                else: fresh=False
                independent=bool(answer==gold and isolated and fresh and row['complete'] and not row['cap'] and not row['errors'])
                assert independent==row['pass']
                values=dict(arm=arm,phase=phase,step=row['step'],session=session,round=round_number,
                    seconds=row['seconds'],seconds_including_deletion=row.get('seconds_including_edit',row['seconds']),
                    edit_seconds=row.get('edit_seconds',0),pass_=independent,correct=answer==gold,read_isolated=isolated,
                    fresh_read=fresh,cap=row['cap'],text=row['text'],usage=[r['usage'] for r in row['replies']])
                rows.append(values);t['passes']+=independent;t['reads']+=fresh;t['caps']+=row['cap']
                if round_number:
                    t['edited_seconds_including_deletion']+=values['seconds_including_deletion']
                    t['round_seconds'][round_number]=t['round_seconds'].get(round_number,0)+values['seconds_including_deletion']
                    t['round_passes'][round_number]=t['round_passes'].get(round_number,0)+independent
                else:t['warm_seconds']+=row['seconds']
                log=(folder/(row['step']+'-server.log')).read_text()
                for line in log.splitlines():
                    common=dict(arm=arm,phase=phase,step=row['step'],round=round_number)
                    if 'suffix APPROXIMATE' in line:
                        fields={k:v if ':' in v else int(v) for k,v in re.findall(r'(\w+)=([\d:]+)',line)}
                        fields['graft_seconds']=float(re.search(r'([\d.]+) ms$',line)[1])/1000
                        assert int(fields['destination'].split(':')[1])<=fields['latest_user_start']
                        grafts.append(dict(**common,**fields,raw_log=line))
                    if 'responses live continuation' in line:
                        m=re.search(r'match=(\S+) ids=(\d+) cached=(\d+) prompt=(\d+) approximate=(\d+)',line)
                        kind,ids,cached,prompt,approx=m.groups()
                        continuations.append(dict(**common,match=kind,ids=int(ids),cached=int(cached),prompt=int(prompt),fresh=int(prompt)-int(cached),approximate=int(approx),raw_log=line))
            log=(a.input/arm/f'{phase}-server.stderr.log').read_text()
            for line in log.splitlines():
                if 'state_cache_summary ' in line:caches.append(dict(arm=arm,phase=phase,**json.loads(line.split('state_cache_summary ',1)[1])))
        t['client_phase_seconds']=sum(t['phase_seconds'].values())
        t['grafts']=sum(g['arm']==arm for g in grafts);t['approximate_sources']=sum(g['arm']==arm and g['source_approximate'] for g in grafts)
        t['approximate_continuations']=sum(g['arm']==arm and g['approximate'] for g in continuations)
        t['logical_disk_read_ops']=sum(c['disk_reads'] for c in caches if c['arm']==arm)
        t['phase_log_grafts']=sum((a.input/arm/f'{ph}-server.stderr.log').read_text().count('suffix APPROXIMATE') for ph in ['before','after'])
        assert t['phase_log_grafts']==t['grafts'];totals[arm]=t
        guard=a.input/('guard-'+arm);samples=[json.loads(x) for x in (guard/'pressure.jsonl').read_text().splitlines()]
        guards.append(dict(arm=arm,**read(guard/'result.json'),samples=len(samples),levels=sorted(set(x['level'] for x in samples)),
            swap_first_gb=samples[0]['swap_gb'],swap_last_gb=samples[-1]['swap_gb'],peak_wired_compressor_gb=max(x['wired_gb']+x['compressor_gb'] for x in samples),max_consecutive_bad=max(x['consecutive_bad'] for x in samples)))
    for arm in ARMS:
        controls[arm]={}
        for row in [r for r in rows if r['arm']==arm]:
            base=next(r for r in rows if r['arm']=='native-before' and r['step']==row['step'])
            outputs=[]
            for label in ['native-before',arm]:
                outputs.append([[{k:v for k,v in item.items() if k not in ['id','call_id']} for item in read(a.input/label/'http'/row['phase']/(row['step']+f'-{i}-response.json'))['output']] for i in range(len(row['usage']))])
            controls[arm][row['step']]=dict(final_text_equal=base['text']==row['text'],usage_equal=base['usage']==row['usage'],output_tokens_equal=[u['output_tokens'] for u in base['usage']]==[u['output_tokens'] for u in row['usage']],public_outputs_equal=outputs[0]==outputs[1])
    metrics=['launcher_seconds','client_phase_seconds','edited_seconds_including_deletion','warm_seconds']
    mean={k:(totals['native-before'][k]+totals['native-after'][k])/2 for k in metrics}
    result=dict(rows=rows,by_arm=totals,edits=edits,grafts=grafts,visible_continuations=continuations,cache_shutdowns=caches,guard_summaries=guards,
        controls_vs_native_before=controls,native_mean=mean,
        change_vs_native_mean_percent={arm:{k:100*(t[k]/mean[k]-1) for k in metrics} for arm,t in totals.items()},
        native_after_vs_before_percent={k:100*(totals['native-after'][k]/totals['native-before'][k]-1) for k in metrics},
        round_change_vs_native_mean_percent={n:100*(totals['reuse']['round_seconds'][n]/((totals['native-before']['round_seconds'][n]+totals['native-after']['round_seconds'][n])/2)-1) for n in list(range(1,9))},
        scope='One Conversation, eight real item deletions. Restart after edit4. Actual approximate SSD source selections are recorded without assuming unique or immediately previous source identities. No isolated relocation/additive/physicalSSD/private-identity/NLL claim.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in [('steps',rows),('deletions',edits),('grafts',grafts),('visible-continuations',continuations),('cache-shutdowns',caches)]:
        with (a.output/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({k:result[k] for k in ['by_arm','change_vs_native_mean_percent','native_after_vs_before_percent','round_change_vs_native_mean_percent','guard_summaries','controls_vs_native_before']},indent=2))
    if a.plot:
        os.environ['MPLCONFIGDIR']='/Volumes/Models/ds4/.workspaces/server-agent-day-20261004/plot-cache-hour16'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15.5,5),constrained_layout=True)
        for arm,label,color in zip(ARMS,['Native before','Reuse','Native after'],['#65717e','#e89930','#a5b1be']):
            axes[0].plot(list(range(1,9)),[totals[arm]['round_seconds'][n] for n in list(range(1,9))],marker='o',label=label,color=color)
        axes[0].set(title='Repeated deletion wall (one Conversation)',xlabel='Deleted public items per Conversation',ylabel='Read + final + deletion wall (s)');axes[0].set_xticks(list(range(1,9)));axes[0].legend(frameon=False)
        axes[1].plot(list(range(1,9)),[totals['reuse']['round_passes'][n] for n in list(range(1,9))],marker='o',label='Correct replies /1',color='#247ba0')
        axes[1].bar(list(range(1,9)),[sum(g['round']==n and g['source_approximate'] for g in grafts) for n in list(range(1,9))],width=.3,label='Approximate SSD source /1',color='#40916c',alpha=.6)
        axes[1].set(title='Actual repeated-source coverage and quality',xlabel='Deletion round',ylabel='Actual count',ylim=(0,1.6));axes[1].set_xticks(list(range(1,9)));axes[1].legend(frameon=False)
        labels=['Native before','Reuse','Native after'];axes[2].bar(labels,[totals[x]['launcher_seconds'] for x in ARMS],color=['#65717e','#e89930','#a5b1be'])
        axes[2].set(title='Whole launcher (warm + edits + restart)',ylabel='Seconds');axes[2].tick_params(axis='x',rotation=20)
        for ax in axes:ax.grid(axis='y',alpha=.15)
        fig.suptitle('Eight real deletions of one durable Conversation: all 27 reads/replies pass; restart after edit 4',fontsize=12)
        fig.savefig(a.output/'repeated-conversations-curve.png',dpi=180);fig.savefig(a.output/'repeated-conversations-curve.svg');plt.close(fig)


if __name__=='__main__':main()
