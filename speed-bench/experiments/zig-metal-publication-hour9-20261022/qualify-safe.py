from pathlib import Path
import json, os, subprocess, time
root=Path.cwd();base=root/'.workspaces/zig-metal-publication-20261022';out=base/'safe';out.mkdir()
rows=[]
for arm in ['c','native']:
    binary=base/arm/'bin/test-deepseek41-metal'
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
    env['DS4_METAL_MATH_SAFE']='1'
    start=time.monotonic();p=subprocess.run([str(binary)],env=env,capture_output=True,timeout=180)
    (out/(arm+'.stdout')).write_bytes(p.stdout);(out/(arm+'.stderr')).write_bytes(p.stderr)
    rows.append(dict(arm=arm,command=[str(binary)],environment={'DS4_METAL_MATH_SAFE':'1'},seconds=time.monotonic()-start,returncode=p.returncode))
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
    if p.returncode:raise SystemExit('strict suite failed: '+arm)
