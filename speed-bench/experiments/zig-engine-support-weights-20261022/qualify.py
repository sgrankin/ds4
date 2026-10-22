#!/usr/bin/env python3
"""Existing inspect and trained DSpark fixtures for strict support binding."""
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
p.add_argument('--inspect-only',action='store_true')
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
rows=[]
model=str(Path('gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf').resolve())
for label,binary,args,settings in [
    ('inspect','ds4',['--model',model,'--metal','--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','1024','--prefill-chunk','512','--inspect','--mtp-model',str(Path('gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf').resolve())],{}),
    ('dspark-runtime','dspark-native-oracle',[model,str(Path('gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf').resolve()),'--runtime'],{})]:
    if a.inspect_only and label!='inspect':continue
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
