from pathlib import Path
import argparse,hashlib,json,subprocess,time
p=argparse.ArgumentParser();p.add_argument('arm');a=p.parse_args();w=Path('.workspaces/zig-engine-core-20261022');out=w/'integration'/a.arm;out.mkdir(exist_ok=False)
reference=json.loads((w/'model/f16-bytes-control.json').read_text());runs=[]
for label,args in [('frontier',None),('accounting',None)]:
 if label=='frontier':
  r=json.loads((w/'model/results.json').read_text());cmd=next(x['command'] for x in r['runs'] if x['label']=='f16-bytes-native');cmd=[x.replace('/native/bin/ds4','/'+a.arm+'/bin/ds4').replace('/model/f16-bytes-native.json','/integration/'+a.arm+'/frontier.json') for x in cmd]
 else:
  model=str(Path('gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf').resolve());cmd=['python3','-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py','--settings','{}',str((w/a.arm/'bin/test-deepseek41-graph').resolve()),model,'--session-accounting']
 start=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as so,(out/(label+'.stderr')).open('wb') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=300)
 runs.append({'label':label,'command':cmd,'seconds':time.monotonic()-start,'returncode':r.returncode});(out/'results.json').write_text(json.dumps(runs,indent=2)+'\n');assert r.returncode==0
 if label=='frontier':assert json.loads((out/'frontier.json').read_text())==reference
print(a.arm+' F16 frontier and failed-startup/reopen accounting: PASS')
