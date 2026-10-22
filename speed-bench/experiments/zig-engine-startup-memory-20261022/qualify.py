#!/usr/bin/env python3
"""Compare real startup cache policy and full frontier through public CLI."""
from pathlib import Path
import argparse,hashlib,json,math,subprocess,sys,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace;out=w/'model';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=['ds4.c','build.zig','ds4_engine_memory.h','zig/engine/deepseek41/memory.zig','zig/engine/graph.zig','zig/engine/deepseek41/graph_api.h','tests/test_deepseek41_graph.c']
report={'source_before':{p:sha(p) for p in files},'runs':[],'comparisons':{}}
model=str(Path('gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf').resolve())
for arm in ['control','native']:
 binary=(w/arm/'bin/test-deepseek41-graph').resolve();cmd=[sys.executable,'-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings','{}',str(binary),model,'--memory-plan']
 start=time.monotonic()
 with (out/(arm+'-plan.stdout')).open('wb') as so,(out/(arm+'-plan.stderr')).open('wb') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=180)
 report['runs'].append({'label':arm+'-plan','command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start,'binary_sha256':sha(binary)});assert r.returncode==0
for case,budget in [('auto',[]),('bytes',['--ssd-streaming-cache-experts','32GB']),('count',['--ssd-streaming-cache-experts','512'])]:
 reference=None;reference_log=None
 for arm in ['control','native']:
  label=case+'-'+arm;binary=(w/arm/'bin/ds4').resolve();dump=out/(label+'.json')
  cmd=[sys.executable,'-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings','{}',str(binary),'--model',model,'--metal','--ssd-streaming',*budget,'--ctx','1024','--prefill-chunk','512','--temp','0','--nothink','--prompt','Return only the word MAPLE.','--dump-logits',str(dump.resolve())]
  start=time.monotonic()
  with (out/(label+'.stdout')).open('wb') as so,(out/(label+'.stderr')).open('wb') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=300)
  row={'label':label,'command':cmd,'seconds':time.monotonic()-start,'returncode':r.returncode,'binary_sha256':sha(binary)};report['runs'].append(row)
  (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['label','seconds','returncode']}),flush=True);assert r.returncode==0,label
  data=json.loads(dump.read_text());assert data['vocab']==len(data['logits'])==129280 and all(isinstance(x,(float,int)) and math.isfinite(x) for x in data['logits'])
  lines=[x for x in (out/(label+'.stderr')).read_text().splitlines() if x.startswith(('ds4: SSD streaming auto','ds4:   ','ds4: metal SSD streaming cache','ds4: memory:','ds4: memory detail:','ds4: SSD streaming initial metal model map'))]
  row['startup_diagnostics']=lines;row['frontier_sha256']=sha(dump)
  if reference is None:reference=data;reference_log=lines
  else:
   report['comparisons'][case]={'all_frontier_logits_identical':data==reference,'startup_diagnostics_identical':lines==reference_log}
   assert data==reference and lines==reference_log,case
report['source_after']={p:sha(p) for p in files};assert report['source_before']==report['source_after'];report['all_pass']=True
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
