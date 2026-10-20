import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/Volumes/Models/ds4');out=Path(__file__).resolve().parent;prep=out/'prepared'
env=os.environ.copy()
for key in list(env):
    if key.startswith('DS4_METAL_V41_RESIDENT_ROUTE') or key.startswith('DS4_OVERTHINK') or key.startswith('DS4_METAL_V41_DROP_WEAK'):del env[key]
env.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP='.05',DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS='.12',DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS='1',DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS='.20')
for shader in (prep/'metal').glob('*.metal'):env['DS4_METAL_'+shader.stem.upper()+'_SOURCE']=str(shader)
cmd=[sys.executable,str(prep/'live_first.py'),str(out/'resident-ds41-full275'),'--binary',str(prep/'ds4-agent'),'--model',str(root/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),'--ctx','8192','--cache-gb','64','--prefill-chunk','512','--timeout','275']
start=time.time();print('START candidate retry275',flush=True)
with (out/'resident-ds41-full275-runner.log').open('w') as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'candidate-retry-result.json').write_text(json.dumps(dict(rc=p.returncode,started=start,ended=time.time(),scope='first original reconciliation task; fresh unchanged fixture',timeout_sec=275),indent=2)+'\n')
print('DONE candidate retry275 rc',p.returncode,flush=True);sys.exit(p.returncode)
