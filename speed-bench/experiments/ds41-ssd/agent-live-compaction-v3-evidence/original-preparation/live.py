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
import traceback
from trace_session import extract

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'experiments/ds41-ssd'))
from snapshot_shaders import snapshot_shaders
class FixtureMutation(AssertionError):
    pass


def stop_process(proc):
    """Only an unreaped owned leader protects its process-group identity."""
    if proc.returncode is not None: return
    try: os.killpg(proc.pid,signal.SIGTERM)
    except (ProcessLookupError,PermissionError): pass
    # Do not poll or wait during grace: either could reap the leader, allowing
    # its PID/PGID to be reused before the final signal.
    time.sleep(.5)
    if proc.returncode is None:
        try: os.killpg(proc.pid,signal.SIGKILL)
        except (ProcessLookupError,PermissionError): pass
    proc.wait(timeout=5)
    # Residual groups after a reaped leader belong to the birth-tracked MAIN
    # guard; this private frontend never signals a remembered reaped PGID.


def remaining(deadline):
    value=deadline-time.monotonic()
    if value<=0: raise TimeoutError('session deadline')
    return value


def checked_command(argv,deadline,**kwargs):
    child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           text=True,start_new_session=True,**kwargs)
    try:
        stdout,stderr=child.communicate(timeout=remaining(deadline))
        return subprocess.CompletedProcess(argv,child.returncode,stdout,stderr)
    except BaseException:
        stop_process(child)
        raise


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--binary',type=Path,default=Path('./ds4-agent'))
p.add_argument('--env',action='append',default=[])
p.add_argument('--cache-gb',type=int)
p.add_argument('--prefill-chunk',type=int)
p.add_argument('--timeout',type=float,default=900)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--project-dir',type=Path,required=True)
a=p.parse_args()
a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
binary=a.output/'ds4-agent'; shutil.copy2(a.binary,binary)
project=a.project_dir.resolve(); shutil.copytree(HERE/'fixture',project,ignore=shutil.ignore_patterns('__pycache__'))
cache=a.output/'kv'; cache.mkdir()
trace=a.output/'trace.log'
env=os.environ.copy(); env.update(snapshot_shaders(a.output))
env.update(DS4_AGENT_CACHE_DIR=str(cache),DS4_AGENT_TEST_TIME='1789319891',TZ='America/New_York',PYTHONDONTWRITEBYTECODE='1')
for value in a.env:
    k,v=value.split('=',1); env[k]=v
cmd=[str(binary),'--model',str(a.model.resolve()),'--ssd-streaming','--non-interactive','--seed','20261002','--temp','0','--nothink','--ctx','8192','-n','1024',
     '--chdir',str(project),'--trace',str(trace)]
if a.cache_gb: cmd+=['--ssd-streaming-cache-experts',f'{a.cache_gb}GB']
if a.prefill_chunk: cmd+=['--prefill-chunk',str(a.prefill_chunk)]
(a.output/'command.json').write_text(json.dumps(dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith('DS4_') or k=='TZ'}),indent=2)+'\n')
shutil.copy2(HERE/'prompts.json',a.output/'prompts.json')
prompts=json.loads((a.output/'prompts.json').read_text())
provenance=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    prompts_sha256=hashlib.sha256((a.output/'prompts.json').read_bytes()).hexdigest(),
    shaders=json.loads((a.output/'metal/sha256.json').read_text()))
(a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
checks=[]; turns=[]; submitted=0; last=None; start=time.monotonic(); startup=None
deadline=start+a.timeout
outcome=dict(status='RUNNING',phase='agent_start',stage=None)
proc=None
try:
    with (a.output/'stdout.log').open('wb') as out:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        sel=selectors.DefaultSelector(); sel.register(proc.stdout,selectors.EVENT_READ)
        pending=b''
        while sel.get_map():
            outcome.update(phase='agent_output')
            for key,_ in sel.select(min(1,remaining(deadline))):
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
                        outcome.update(phase='fixture_validation',stage=stage)
                        for relative in ['data.csv','tests/test_report.py']:
                            if (project/relative).read_bytes()!=(HERE/'fixture'/relative).read_bytes():
                                raise FixtureMutation('Changed fixture: '+relative)
                        outcome.update(phase='held_out_check')
                        check=checked_command([sys.executable,str(HERE/'check.py'),str(project),str(stage)],deadline,env=env)
                        checks.append(dict(stage=stage,held_out_rc=check.returncode,tests_rc=None,held_out=check.stdout+check.stderr,tests=''))
                        (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
                        outcome.update(phase='unit_tests')
                        tests=checked_command([sys.executable,'-m','unittest','discover','-s','tests','-v'],deadline,cwd=project,env=env)
                        checks[-1].update(tests_rc=tests.returncode,tests=tests.stdout+tests.stderr)
                        (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
                        outcome.update(phase='task_validation')
                        assert check.returncode==tests.returncode==0, f'Task validation failed at stage {stage}'
                        if stage >= 1:
                            count=re.search(r'Ran (\d+) tests?',tests.stderr)
                            assert count and int(count[1])>=7, 'Expected at least two added tests'
                        print(f'PASS turn {stage+1}: {turns[-1]/1000:.3f}s',flush=True)
                    if submitted<len(prompts):
                        last=time.monotonic(); proc.stdin.write((prompts[submitted]+'\n').encode()); proc.stdin.flush(); submitted+=1
                    elif not proc.stdin.closed: proc.stdin.close(); last=None
        outcome.update(phase='agent_exit')
        assert proc.wait(timeout=remaining(deadline))==0 and len(turns)==len(prompts)
    outcome.update(phase='trace_validation')
    result,replay=extract(trace)
    assert result['tool_calls']>=6, 'Insufficient tool use'
    result.update(startup_ms=startup,turn_wall_ms=turns,total_turn_ms=sum(turns),success=True,
                  fixture_sha256=hashlib.sha256(b''.join(f.relative_to(HERE/'fixture').as_posix().encode()+f.read_bytes() for f in sorted((HERE/'fixture').rglob('*')) if f.is_file() and '__pycache__' not in str(f))).hexdigest())
    (a.output/'replay.txt').write_text(replay)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    outcome.update(status='SUCCESS',phase='complete')
except BaseException as error:
    status='TIMEOUT' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else 'FAILED'
    if isinstance(error,FixtureMutation): status='FIXTURE_MUTATION'
    outcome.update(status=status,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    if isinstance(error,subprocess.TimeoutExpired):
        for label,value in [('command_stdout',error.stdout),('command_stderr',error.stderr)]:
            outcome[label]=value.decode(errors='replace') if isinstance(value,bytes) else value
    raise
finally:
    if proc:
        outcome['agent_pid']=proc.pid
        if outcome['status']!='SUCCESS' or proc.poll() is None: stop_process(proc)
        outcome['agent_returncode']=proc.poll()
    outcome.update(turns_submitted=submitted,turns_observed=len(turns),checks_recorded=len(checks),
                   elapsed_seconds=time.monotonic()-start,cleanup_grace_seconds=5.5)
    (a.output/'harness-outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
