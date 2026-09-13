#!/usr/bin/env python3
"""Exercise real short agent turns with an isolated KV directory and fixed binary."""
import argparse, json, os
from pathlib import Path
import re, selectors, shutil, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('output',type=Path)
p.add_argument('--env',action='append',default=[],help='NAME=VALUE')
p.add_argument('--cache-gb',type=int)
p.add_argument('--binary',default='./ds4-agent')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
binary=a.output.resolve()/'ds4-agent'; shutil.copy2(a.binary,binary)
cache=a.output.resolve()/'kv'; cache.mkdir()
results=[]
for mode in ['fresh','restored']:
    # Retain only the fixed system checkpoint; avoid full-turn cache hits.
    for f in cache.glob('*.kv'):
        if f.name!='sysprompt.kv': f.unlink()
    dest=a.output.resolve()/mode; dest.mkdir()
    env=os.environ.copy(); env['DS4_AGENT_CACHE_DIR']=str(cache)
    for item in a.env:
        k,v=item.split('=',1); env[k]=v
    cmd=[str(binary),'--ssd-streaming','--non-interactive','-n','1',
         '--trace',str(dest/'trace.log')]
    if a.cache_gb: cmd+=['--ssd-streaming-cache-experts',f'{a.cache_gb}GB']
    (dest/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith('DS4_')}},indent=2))
    prompts=['hello','What is 2 + 2?','hello']
    with (dest/'stdout.log').open('wb') as out:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        sel=selectors.DefaultSelector(); sel.register(proc.stdout,selectors.EVENT_READ)
        pending=b''; submitted=0; deadline=time.monotonic()+600
        while sel.get_map():
            if time.monotonic()>deadline:
                proc.kill(); proc.wait(); raise RuntimeError('agent timeout')
            for key,_ in sel.select(1):
                data=os.read(key.fd,65536)
                if not data: sel.unregister(key.fileobj); break
                out.write(data); out.flush(); pending+=data
                marker=b'+DWARFSTAR_WAITING'
                while marker in pending:
                    _,pending=pending.split(marker,1)
                    if submitted<len(prompts):
                        proc.stdin.write((prompts[submitted]+'\n').encode()); proc.stdin.flush(); submitted+=1
                    elif not proc.stdin.closed: proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError(f'agent failed: {dest}')
    trace=(dest/'trace.log').read_text()
    rows=[]
    for match in re.finditer(r'prefill sync done (?:tool_round=\d+ )?prompt=(\d+) cached=(\d+) suffix=(\d+) rc=(\d+) ([\d.]+) ms',trace):
        n,c,s,rc,ms=match.groups(); rows.append(dict(prompt=int(n),cached=int(c),suffix=int(s),rc=int(rc),ms=float(ms)))
    result=dict(mode=mode,system_kv_hit='sysprompt kv hit' in trace,turns=rows)
    results.append(result); print(json.dumps(result),flush=True)
(a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
