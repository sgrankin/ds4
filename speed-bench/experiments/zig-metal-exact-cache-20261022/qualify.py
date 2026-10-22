#!/usr/bin/env python3
"""Exact cached/owned/transient lifetime and existing broad GPU checks, serially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
rows=[]
cases=[]
for unretained in ['0','1']:
    for mib,mode in [('1','exact-cache-evict'),('0','exact-cache-unlimited')]:
        settings={'MTL_DEBUG_LAYER':'1','DS4_METAL_EXACT_VIEW_CACHE_MIB':mib}
        if unretained == '1': settings['DS4_METAL_UNRETAINED_COMMAND_BUFFERS']='1'
        cases.append((mode+'-unretained'+unretained,'test-metal-command-memory',[mode],settings))
cases += [('ssd','test-metal-ssd-experts',[],{'MTL_DEBUG_LAYER':'1'}),
          ('numeric-safe','test-deepseek41-metal',[],{'DS4_METAL_MATH_SAFE':'1'})]
for label,binary,args,settings in cases:
    executable=(a.prefix/'bin'/binary).resolve()
    cmd=[sys.executable,'-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py',
         '--settings',json.dumps(settings),str(executable),*args]
    start=time.monotonic()
    with (a.out/(label+'.stdout')).open('wb') as so,(a.out/(label+'.stderr')).open('wb') as se:
        result=subprocess.run(cmd,stdout=so,stderr=se,timeout=600)
    row={'label':label,'command':cmd,'binary_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
         'seconds':time.monotonic()-start,'returncode':result.returncode}
    rows.append(row);(a.out/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
