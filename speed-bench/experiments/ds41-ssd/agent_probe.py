#!/usr/bin/env python3
"""Exercise real short agent turns with an isolated KV directory and fixed binary."""
import argparse, json, os
from pathlib import Path
from datetime import datetime
import hashlib
import re, selectors, shutil, subprocess, time
from snapshot_shaders import snapshot_shaders
p=argparse.ArgumentParser()
p.add_argument('output',type=Path)
p.add_argument('--env',action='append',default=[],help='NAME=VALUE')
p.add_argument('--cache-gb',type=int)
p.add_argument('--prefill-chunk',type=int)
p.add_argument('--binary',default='./ds4-agent')
p.add_argument('--gen-tokens',type=int,default=1)
p.add_argument('--prompts-json',type=Path,help='JSON array of single-line prompts')
a=p.parse_args()
prompts=json.loads(a.prompts_json.read_text()) if a.prompts_json else ['hello','What is 2 + 2?','hello']
if a.gen_tokens < 1 or not isinstance(prompts,list) or not prompts or any(not isinstance(x,str) or not x or '\n' in x for x in prompts):
    p.error('positive generation limit and nonempty single-line prompts required')
a.output.mkdir(parents=True,exist_ok=False)
binary=a.output.resolve()/'ds4-agent'; shutil.copy2(a.binary,binary)
shaders=snapshot_shaders(a.output)
cache=a.output.resolve()/'kv'; cache.mkdir()
results=[]
for mode in ['fresh','restored']:
    # Retain only the fixed system checkpoint; avoid full-turn cache hits.
    for f in cache.glob('*.kv'):
        if f.name!='sysprompt.kv': f.unlink()
    dest=a.output.resolve()/mode; dest.mkdir()
    env=os.environ.copy(); env['DS4_AGENT_CACHE_DIR']=str(cache)
    env.update(shaders)
    for item in a.env:
        k,v=item.split('=',1); env[k]=v
    cmd=[str(binary),'--ssd-streaming','--non-interactive','-n',str(a.gen_tokens),
         '--trace',str(dest/'trace.log')]
    if a.gen_tokens > 1: cmd += ['--seed','1234']
    if a.cache_gb: cmd+=['--ssd-streaming-cache-experts',f'{a.cache_gb}GB']
    if a.prefill_chunk: cmd+=['--prefill-chunk',str(a.prefill_chunk)]
    (dest/'command.json').write_text(json.dumps({'argv':cmd,'env':{k:v for k,v in env.items() if k.startswith('DS4_')}},indent=2))
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
    generations=[]
    started=None; prefilled=None; token_ids=[]; first=None
    for line in trace.splitlines():
        if len(line)<24: continue
        stamp=datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S.%f')
        if ' prefill tool_round=' in line:
            started=stamp; prefilled=None; token_ids=[]; first=None
        if ' prefill sync done tool_round=' in line: prefilled=stamp
        if prefilled and ' token index=' in line:
            token_ids.append(int(re.search(r' id=(\d+)',line)[1]))
            if first is None: first=stamp
        if started and ' generation finished ' in line:
            generated=int(re.search(r' generated=(\d+)',line)[1])
            generations.append(dict(generated=generated,
                response_ms=(stamp-started).total_seconds()*1000,
                decode_ms=(stamp-prefilled).total_seconds()*1000 if prefilled else None,
                first_output_ms=(first-started).total_seconds()*1000 if first else None,
                output_ids_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()))
            started=None
    result['generations']=generations
    result['generation_limit']=a.gen_tokens
    results.append(result); print(json.dumps(result),flush=True)
(a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
