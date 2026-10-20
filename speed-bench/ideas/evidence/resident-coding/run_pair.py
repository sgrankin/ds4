#!/usr/bin/env python3
import datetime,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/Volumes/Models/ds4')
out=Path(__file__).resolve().parent
prep=out/'prepared'
results=[]
started=time.time()
hard_deadline=datetime.datetime(2026,10,2,17,14,50,tzinfo=datetime.timezone.utc).timestamp()
for arm,gap in [('baseline-ds41','0'),('resident-ds41','.05')]:
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('DS4_METAL_V41_RESIDENT_ROUTE') or key.startswith('DS4_OVERTHINK') or key.startswith('DS4_METAL_V41_DROP_WEAK'):
            del env[key]
    env['DS4_METAL_V41_RESIDENT_ROUTE_GAP']=gap
    env['DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS']='.12'
    env['DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS']='1'
    env['DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS']='.20'
    for shader in (prep/'metal').glob('*.metal'):
        env['DS4_METAL_'+shader.stem.upper()+'_SOURCE']=str(shader)
    cmd=[sys.executable,str(prep/'live_first.py'),str(out/arm),'--binary',str(prep/'ds4-agent'),'--model',str(root/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),'--ctx','8192','--cache-gb','64','--prefill-chunk','512','--timeout','275']
    remaining=int(hard_deadline-time.time())
    if remaining < 30:
        print('INSUFFICIENT DEADLINE',arm,remaining,flush=True); break
    cmd[-1]=str(min(275,remaining))
    print('START',arm,'wall deadline',cmd[-1],flush=True)
    wall=time.monotonic()
    with (out/(arm+'-runner.log')).open('w') as log:
        proc=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    summary=out/arm/'summary.json'
    checks=out/arm/'checks.json'
    results.append(dict(arm=arm,rc=proc.returncode,wall_sec=time.monotonic()-wall,summary=json.loads(summary.read_text()) if summary.exists() else None,checks=json.loads(checks.read_text()) if checks.exists() else None))
    (out/'pair-results.json').write_text(json.dumps(dict(started=started,scope='first reconciliation task only',results=results,completed=False),indent=2)+'\n')
    print('DONE',arm,'rc',proc.returncode,flush=True)
(out/'pair-results.json').write_text(json.dumps(dict(started=started,scope='first reconciliation task only',results=results,completed=True),indent=2)+'\n')
print(json.dumps(results),flush=True)
sys.exit(0 if len(results)==2 and all(x['rc']==0 for x in results) else 1)
