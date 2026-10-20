#!/usr/bin/env python3
"""Summarize this fixed actual-agent screen from ordinary JSON records."""
import csv, json, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[]
for p in sorted((HERE/'runs').glob('*/result.json')):
 if p.parent.name=='pilot-off': continue
 r=json.loads(p.read_text()); fixture=json.loads((p.parent/'fixture.json').read_text())
 grafts=[re.search(r'APPROXIMATE reused=(\d+) fresh=(\d+)',s) for s in r['suffix_trace'] if 'APPROXIMATE reused=' in s]
 repeat='fourth_answer' in r
 edit=r.get('fourth_turn_including_edit_seconds') if repeat else r['final_turn_including_edit_seconds']
 fifth=r.get('fifth_turn_including_edit_seconds')
 rows.append(dict(arm=p.parent.name,cohort=(('repeat-short' if fixture['notes']==24 else 'repeat-long') if repeat else ('short' if fixture['notes']==24 else 'long')),
  mode=r['mode'],tail=r['tail_lower_bound'],task_pass=r['task_pass'],answer_correct=r['answer_correct'],
  fourth_answer_correct=r.get('fourth_answer_correct'),workflow_seconds=r['seconds'],edit_turn_seconds=edit,
  fifth_turn_seconds=fifth,edit_cycle_seconds=(edit+fifth if fifth is not None else edit),
  reused_tokens=sum(int(m[1]) for m in grafts),fresh_tokens=sum(int(m[2]) for m in grafts),
  approximate_edits=len(grafts),accepted_edits=len(r['context_edit_trace']),
  archived_fact_removed=r.get('archived_fact_removed_before_query'),incoming_question_retained=r.get('incoming_question_retained')))
with (HERE/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
comparisons=[]
for cohort in sorted({r['cohort'] for r in rows}):
 owned=[r for r in rows if r['cohort']==cohort]
 native=[r for r in owned if r['mode']=='native']
 if not native: continue
 means={k:sum(r[k] for r in native)/len(native) for k in ['workflow_seconds','edit_turn_seconds','edit_cycle_seconds']}
 for r in owned:
  if r['mode']=='native': continue
  comparisons.append(dict(arm=r['arm'],native_controls=len(native),task_pass=r['task_pass'],
   **{k+'_saving_percent':100*(1-r[k]/means[k]) for k in means},
   saving_vs_fastest_native_percent=100*(1-r['workflow_seconds']/min(n['workflow_seconds'] for n in native))))
(HERE/'results.json').write_text(json.dumps(dict(rows=rows,comparisons=comparisons),indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
try:
 import os
 os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplcache'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except ImportError:
 raise SystemExit(0)
fig,axes=plt.subplots(2,2,figsize=(10,7))
colors={'short':'#2674a6','long':'#d57528','repeat-short':'#55852c','repeat-long':'#804a99'}
for ax,cohort,title in zip(axes.flat,colors,['Short: one edit','Long: one edit','Short: two edits','Long: two edits']):
 owned=[r for r in rows if r['cohort']==cohort]
 native=[r for r in owned if r['mode']=='native']
 mean=sum(r['workflow_seconds'] for r in native)/len(native)
 passed=sum(r['task_pass'] for r in native)
 spread=[[mean-min(r['workflow_seconds'] for r in native)],[max(r['workflow_seconds'] for r in native)-mean]]
 ax.errorbar(mean,100*passed/len(native),xerr=spread,fmt='o',color='#444444',capsize=4)
 ax.annotate('native '+str(passed)+'/'+str(len(native)),(mean,100*passed/len(native)),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
 for r in owned:
  if r['mode']=='native': continue
  x=r['workflow_seconds'];y=int(r['task_pass'])*100
  ax.scatter(x,y,c=colors[cohort],marker='x',s=65)
  ax.annotate('tail≥'+str(r['tail'])+' '+str(int(r['task_pass']))+'/1',(x,y),xytext=(0,10 if r['tail']==128 else 27),textcoords='offset points',ha='center',fontsize=8)
 ax.set_title(title);ax.set_xlabel('Whole workflow wall time (s)');ax.set_ylabel('Workflow passes (%)')
 ax.set_ylim(-5,130);ax.set_yticks([0,100]);ax.grid(alpha=.2)
 ax.set_xlim(min(r['workflow_seconds'] for r in owned)*.88,max(r['workflow_seconds'] for r in owned)*1.12)
fig.suptitle('Storage-backed agent: native brackets and suffix candidates')
fig.text(.5,.015,'Small screen: 13 workflows, 19 correct graded answers; tail values are minimums.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.95))
for ext in ['png','svg']: fig.savefig(HERE/('tradeoffs.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig)
