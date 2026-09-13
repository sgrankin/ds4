#!/usr/bin/env python3
"""Run the real SSD agent on an isolated, verified three-turn coding session."""
import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
from trace_session import extract

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'experiments/ds41-ssd'))
from snapshot_shaders import snapshot_shaders
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--binary',type=Path,default=Path('./ds4-agent'))
p.add_argument('--env',action='append',default=[])
p.add_argument('--cache-gb',type=int)
p.add_argument('--timeout',type=int,default=1200)
a=p.parse_args()
a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
binary=a.output/'ds4-agent'; shutil.copy2(a.binary,binary)
project=a.output/'project'; shutil.copytree(HERE/'fixture',project,ignore=shutil.ignore_patterns('__pycache__'))
cache=a.output/'kv'; cache.mkdir()
trace=a.output/'trace.log'
env=os.environ.copy(); env.update(snapshot_shaders(a.output))
env.update(DS4_AGENT_CACHE_DIR=str(cache),DS4_AGENT_TEST_TIME='1789319891',TZ='America/New_York',PYTHONDONTWRITEBYTECODE='1')
for value in a.env:
    k,v=value.split('=',1); env[k]=v
cmd=[str(binary),'--ssd-streaming','--non-interactive','--seed','1234','--temp','0','-n','4096',
     '--chdir',str(project),'--trace',str(trace)]
if a.cache_gb: cmd+=['--ssd-streaming-cache-experts',f'{a.cache_gb}GB']
(a.output/'command.json').write_text(json.dumps(dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith('DS4_') or k=='TZ'}),indent=2)+'\n')
shutil.copy2(HERE/'prompts.json',a.output/'prompts.json')
prompts=json.loads((a.output/'prompts.json').read_text())
provenance=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    prompts_sha256=hashlib.sha256((a.output/'prompts.json').read_bytes()).hexdigest(),
    shaders=json.loads((a.output/'metal/sha256.json').read_text()))
(a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
checks=[]; turns=[]; submitted=0; last=None; start=time.monotonic(); startup=None
proc=None
try:
    with (a.output/'stdout.log').open('wb') as out:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        sel=selectors.DefaultSelector(); sel.register(proc.stdout,selectors.EVENT_READ)
        pending=b''
        while sel.get_map():
            if time.monotonic()-start>a.timeout: raise TimeoutError('session deadline')
            for key,_ in sel.select(1):
                data=os.read(key.fd,65536)
                if not data: sel.unregister(key.fileobj); break
                out.write(data); out.flush(); pending+=data
                marker=b'+DWARFSTAR_WAITING'
                while marker in pending:
                    _,pending=pending.split(marker,1)
                    now=time.monotonic()
                    if startup is None: startup=(now-start)*1000
                    if last is not None:
                        turns.append((now-last)*1000)
                        stage=submitted-1
                        for relative in ['data.csv']:
                            assert (project/relative).read_bytes()==(HERE/'fixture'/relative).read_bytes(), f'Changed fixture: {relative}'
                        original=ast.parse((HERE/'fixture/tests/test_report.py').read_text())
                        actual=ast.parse((project/'tests/test_report.py').read_text())
                        def definitions(tree):
                            return {n.name: ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
                        expected=definitions(original); observed=definitions(actual)
                        assert all(observed.get(k)==v for k,v in expected.items()), 'Existing test definitions changed'
                        check=subprocess.run([sys.executable,str(HERE/'check.py'),str(project),str(stage)],capture_output=True,text=True,env=env)
                        tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=project,capture_output=True,text=True,env=env)
                        checks.append(dict(stage=stage,held_out_rc=check.returncode,tests_rc=tests.returncode,held_out=check.stdout+check.stderr,tests=tests.stdout+tests.stderr))
                        (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
                        assert check.returncode==tests.returncode==0, f'Task validation failed at stage {stage}'
                        if stage >= 1:
                            count=re.search(r'Ran (\d+) tests?',tests.stderr)
                            assert count and int(count[1])>=7, 'Expected at least two added tests'
                        print(f'PASS turn {stage+1}: {turns[-1]/1000:.3f}s',flush=True)
                    if submitted<len(prompts):
                        last=time.monotonic(); proc.stdin.write((prompts[submitted]+'\n').encode()); proc.stdin.flush(); submitted+=1
                    elif not proc.stdin.closed: proc.stdin.close(); last=None
        assert proc.wait()==0 and len(turns)==len(prompts)
    result,replay=extract(trace)
    assert result['tool_calls']>=6, 'Insufficient tool use'
    result.update(startup_ms=startup,turn_wall_ms=turns,total_turn_ms=sum(turns),success=True,
                  fixture_sha256=hashlib.sha256(b''.join(f.relative_to(HERE/'fixture').as_posix().encode()+f.read_bytes() for f in sorted((HERE/'fixture').rglob('*')) if f.is_file() and '__pycache__' not in str(f))).hexdigest())
    (a.output/'replay.txt').write_text(replay)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
