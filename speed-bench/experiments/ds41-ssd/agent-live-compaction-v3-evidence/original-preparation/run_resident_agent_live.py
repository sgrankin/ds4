#!/usr/bin/env python3
"""Run a frozen tool-using agent comparison beneath the sole MAIN guard."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT=Path('/Volumes/Models/ds4')


def sha(path):
    p=Path(path)
    if p.suffix.lower()=='.gguf': raise ValueError('No model hashing')
    with p.open('rb') as s: return hashlib.file_digest(s,'sha256').hexdigest()


def put(path,value):
    with Path(path).open('x') as s: json.dump(value,s,indent=2); s.write('\n')


def check_assets(plan):
    for p,h in plan['asset_sha256'].items():
        if sha(p)!=h: raise ValueError('Frozen asset differs: '+p)


def require_guard(plan):
    if sha(plan['guard'])!=plan['guard_sha256']: raise ValueError('Guard changed')
    args=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip())
    # argv0 can be the macOS framework interpreter instead of its launch shim.
    expected_tail=[plan['guard'],plan['pressure'],'--',plan['launch_python'],
                   str(Path(__file__).resolve()),str(Path(plan['build_dir'])/'plan.json'),sha(Path(plan['build_dir'])/'plan.json')]
    if not args or args[1:]!=expected_tail: raise RuntimeError('Exact MAIN guard parent required')
    interpreter=plan['guard_interpreter_identity']
    try: same=os.path.samefile(args[0],interpreter['path'])
    except OSError: same=False
    if not same or sha(interpreter['path'])!=interpreter['sha256']:
        raise RuntimeError('Pinned MAIN guard framework interpreter required')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN,errno.EWOULDBLOCK): raise
            return dict(parent_argv=args,exact_suffix_argv=True,interpreter_samefile=True,
                        interpreter_sha256=interpreter['sha256'],guard_lock_held=True)
        fcntl.flock(f,fcntl.LOCK_UN)
    raise RuntimeError('MAIN model lock must be held')


def model_identity(plan):
    p=Path(plan['model']['path'])
    if p.resolve()!=p: raise ValueError('Model alias is forbidden')
    s=p.stat(); actual=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if actual!=plan['model']: raise ValueError('Q2 model identity changed')
    return actual


def summarize_arm(dest,name,rc):
    checks=json.loads((dest/'checks.json').read_text()) if (dest/'checks.json').exists() else []
    successful=sum(x['held_out_rc']==0 and x['tests_rc']==0 for x in checks)
    summary=json.loads((dest/'summary.json').read_text()) if (dest/'summary.json').exists() else None
    outcome=json.loads((dest/'harness-outcome.json').read_text()) if (dest/'harness-outcome.json').exists() else None
    ordered_stages=len(checks)==3 and all(type(x.get('stage')) is int for x in checks) and [x['stage'] for x in checks]==[0,1,2]
    required=['README.md','data.csv','ledger/__init__.py','ledger/io.py','ledger/money.py',
              'ledger/report.py','ledger/cli.py','tests/test_report.py','result.json','recent.json']
    missing=[p for p in required if not (dest/'project-final'/p).is_file()]
    complete=(rc==0 and ordered_stages and isinstance(summary,dict) and summary.get('success') is True
              and isinstance(outcome,dict) and outcome.get('status')=='SUCCESS' and not missing)
    log=(dest/'stdout.log').read_text(errors='replace') if (dest/'stdout.log').exists() else ''
    trace=(dest/'trace.log').read_text(errors='replace') if (dest/'trace.log').exists() else ''
    round_limits=[dict(tool_round=int(m[1]),generated=int(m[2]),carried=int(m[3]),context_limited=bool(int(m[4])))
        for m in re.finditer(r'generation finished tool_round=(\d+) generated=(\d+) carried=(\d+) context_limited=(\d+)',trace)
        if int(m[2])+int(m[3])>=1024 or int(m[4])]
    footers=[x for x in log.splitlines() if x.startswith('ds4: resident-route layers=')]
    route=None
    if footers:
        if len(footers)!=1: raise ValueError('Repeated route footer')
        raw=dict(re.findall(r'([a-z_]+)=([^ ]+)',footers[0]))
        route={k:int(raw[k]) for k in ['layers','substitutions','native_misses','skipped_loads','skipped_bytes','full_snapshots','double_layers','planned_substitutions','drops','triple_layers']}
        if route['drops'] or route['substitutions']!=route['skipped_loads'] or route['substitutions']!=route['planned_substitutions']:
            raise ValueError('Replacement route contract differs')
    if name=='broader3' and rc==0 and (route is None or not route['substitutions']):
        raise ValueError('Completed candidate must exercise replacement routing')
    files={str(p.relative_to(dest/'project-final')):sha(p) for p in sorted((dest/'project-final').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    return dict(arm=name,process_returncode=rc,complete=complete,
        turns_checked=len(checks),turns_correct=successful,all_three_turns_correct=complete and successful==3,
        ordered_stages_valid=ordered_stages,missing_final_project_files=missing,harness_outcome=outcome,
        checks=checks,agent_summary=summary,resident_route=route,project_file_sha256=files,
        rounds_at_configured_limit_or_context_boundary=round_limits,
        limit_observation='Trace records generated/carried/context_limited, not an explicit EOS-versus-cap flag; at-limit rounds are reported without inventing an exact termination cause.',
        stdout_sha256=sha(dest/'stdout.log') if (dest/'stdout.log').exists() else None,
        trace_sha256=sha(dest/'trace.log') if (dest/'trace.log').exists() else None)


def run(plan_path,expected):
    if sha(plan_path)!=expected: raise ValueError('Plan changed')
    plan=json.loads(plan_path.read_text());parent_proof=require_guard(plan);check_assets(plan)
    model=model_identity(plan)
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    put(out/'model-identity.json',model);put(out/'plan.json',plan);put(out/'parent-proof.json',parent_proof)
    build=Path(plan['build_dir']);shared=out/'shared-project'; results=[]
    base={k:v for k,v in os.environ.items() if not k.startswith('DS4')}
    base.update(plan['shader_environment'])
    for arm in plan['arms']:
        require_guard(plan);check_assets(plan);model_identity(plan)
        if shared.exists(): raise RuntimeError('Common project must be fresh')
        dest=out/arm['name'];env=base|arm['environment']
        argv=[sys.executable,str(build/'live.py'),str(dest),'--binary',str(build/'ds4-agent'),
            '--model',model['path'],'--project-dir',str(shared),
            '--cache-gb','64','--prefill-chunk','512','--timeout','900']
        put(out/(arm['name']+'-command.json'),dict(argv=argv,cwd=str(build),
             ds4_environment={k:v for k,v in env.items() if k.startswith('DS4')}))
        print(json.dumps(dict(event='arm_start',arm=arm['name'])),flush=True)
        with (out/(arm['name']+'-harness.log')).open('x') as log:
            rc=subprocess.run(argv,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if shared.exists():
            shutil.copytree(shared,dest/'project-final',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.rmtree(shared)
        require_guard(plan);check_assets(plan);model_identity(plan)
        result=summarize_arm(dest,arm['name'],rc); results.append(result)
        put(out/(arm['name']+'-summary.json'),result)
        print(json.dumps({k:result[k] for k in ['arm','process_returncode','complete','turns_checked','turns_correct','all_three_turns_correct','resident_route']}),flush=True)
    put(out/'summary.json',dict(status='COMPLETE',arms=results,
        timing_claim=plan['timing'],first_attempt_only=True,defaults_promoted=False))
    return 0 if all(x['all_three_turns_correct'] for x in results) else 3


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('plan_sha256')
    a=p.parse_args();sys.exit(run(a.plan,a.plan_sha256))
