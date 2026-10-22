#!/usr/bin/env python3
"""One fixed short real-model generation through C/native exact-cache policy."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
w=Path('.workspaces/zig-metal-exact-cache-20261022');out=w/'model';out.mkdir(exist_ok=False)
files=['ds4_metal.m','build.zig',*[str(p) for p in Path('zig/metal').glob('exact_cache*')]]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'source_before':{p:sha(p) for p in files},'runs':[]}
for arm in ['control','native']:
    binary=(w/arm/'bin/ds4').resolve()
    cmd=[sys.executable,'-B','speed-bench/experiments/zig-session-frontier-20261022/clean_exec.py',
         '--settings','{}',str(binary),'--model',str(Path('gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf').resolve()),
         '--ssd-streaming','--ssd-streaming-cache-experts','32GB','--ctx','1024','--prefill-chunk','512',
         '--tokens','16','--temp','0','--nothink','--prompt','Return only the word MAPLE.']
    started=time.monotonic()
    with (out/(arm+'.stdout')).open('wb') as so,(out/(arm+'.stderr')).open('wb') as se:
        result=subprocess.run(cmd,stdout=so,stderr=se,timeout=180)
    row={'arm':arm,'command':cmd,'seconds':time.monotonic()-started,'returncode':result.returncode,'binary_sha256':sha(binary)}
    report['runs'].append(row);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
report['stdout_identical']=(out/'control.stdout').read_bytes()==(out/'native.stdout').read_bytes()
report['source_after']={p:sha(p) for p in files};assert report['source_before']==report['source_after']
report['all_pass']=True;(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
