#!/usr/bin/env python3
"""Separate-process ABBA with frozen session tokens, binary, and Metal sources."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'experiments/ds41-ssd'))
from snapshot_shaders import snapshot_shaders

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--replay',type=Path,default=HERE/'session-v1.txt')
p.add_argument('--model',type=Path,default=Path('ds4flash.gguf'))
p.add_argument('--binary',type=Path,default=HERE/'replay')
p.add_argument('--candidate-env',required=True)
p.add_argument('--cache-gb',type=int)
p.add_argument('--order',default='ABBA',choices=['ABBA','AB','A'])
a=p.parse_args(); a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
binary=a.output/'replay'; shutil.copy2(a.binary,binary)
recording=a.output/'session.txt'; shutil.copy2(a.replay,recording)
env=os.environ.copy(); env.update(snapshot_shaders(a.output))
name,value=a.candidate_env.split('=',1); env.pop(name,None)
if a.cache_gb: env['DS4_REPLAY_CACHE_GB']=str(a.cache_gb)
(a.output/'manifest.json').write_text(json.dumps(dict(replay_sha256=digest(recording),binary_sha256=digest(binary),model=str(a.model.resolve()),candidate=a.candidate_env,cache_gb=a.cache_gb,order=a.order,env={k:v for k,v in env.items() if k.startswith('DS4_')}),indent=2)+'\n')
results=[]; reference=None
for i,variant in enumerate(a.order):
    dest=a.output/f'{i}-{variant}'; dest.mkdir()
    runenv=env.copy()
    if variant=='B': runenv[name]=value
    logits=dest/'logits.bin'; snapshot=dest/'snapshot.bin'
    with (dest/'timings.csv').open('w') as out, (dest/'stderr.log').open('w') as err:
        subprocess.run([str(binary),str(a.model.resolve()),str(recording),str(logits),str(snapshot)],env=runenv,stdout=out,stderr=err,check=True)
    hashes=dict(logits_sha256=digest(logits),snapshot_sha256=digest(snapshot))
    if reference is None: reference=hashes
    assert hashes==reference, f'Numerical mismatch in {dest}'
    rows=list(csv.DictReader((dest/'timings.csv').open()))
    for r in rows:
        for k in ['phase','context','tokens']: r[k]=int(r[k])
        for k in ['ms','first_decode_ms']: r[k]=float(r[k])
    result=dict(variant=variant,**hashes,phases=rows,
        startup_prefill_ms=rows[0]['ms'],
        append_prefill_ms=sum(r['ms'] for r in rows[1:] if r['kind']=='P'),
        decode_ms=sum(r['ms'] for r in rows if r['kind']=='D'),
        model_ms=sum(r['ms'] for r in rows))
    results.append(result)
    (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='phases'}),flush=True)
    logits.unlink(); snapshot.unlink()
