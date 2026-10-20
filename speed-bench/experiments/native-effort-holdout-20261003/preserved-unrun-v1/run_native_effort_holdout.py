#!/usr/bin/env python3
"""Frozen balanced native-effort holdout; model access requires exact MAIN guard."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import analyze_native_effort_holdout as analysis

ROOT=Path('/Volumes/Models/ds4')
LEVELS=[0,25,75,75,25,0]
NAMES=['effort0-before','effort25-before','effort75-before','effort75-after','effort25-after','effort0-after']
SETTINGS=dict(cap=2048,seed=20261002,temperature_float_value=analysis.TEMP,top_p_float_value=analysis.TOP_P,
              min_p=0,cache_gib=64,context_size=8192,prefill_chunk=512,n_threads=2,power_percent=100,native_penalty=0,cases=6)
GOLDS={'fixed_points9':'22260','necklaces12_weight4':'43','onto7to4':'8400','domino3by8':'153',
       'partitions20_5_max7':'32','connected_graph6_edges7':'5700'}
INPUT_SHA='ad2f0510ebc2684a77f21ebb0f22fe3cb2eb9918890826108192064926dfcc3a'
GUARD_SHA='461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'

def sha(path):return analysis.digest(path)

def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def validate_plan(plan):
    analysis.require(plan['settings']==SETTINGS,'frozen geometry/sampler')
    analysis.require(all(type(plan['settings'][k]) is type(v) for k,v in SETTINGS.items()),'typed geometry/sampler')
    analysis.require([a['name'] for a in plan['arms']]==NAMES and [a['level'] for a in plan['arms']]==LEVELS,'fixed balanced arm order')
    analysis.require(all(type(a['level']) is int for a in plan['arms']),'typed native level')
    analysis.require(plan['guard']==str(ROOT/'speed-bench/ideas/pressure_run.py') and plan['guard_sha256']==GUARD_SHA,'literal MAIN pressure guard')
    analysis.require(plan['run_cwd']==str(Path(plan['workspace']).resolve()) and plan['root_cwd']==str(ROOT),'fixed working directories')
    analysis.require(plan['model']==dict(path=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),bytes=365713686528,mtime_ns=1789237692837653163),'retained explicit Q2 identity')
    analysis.require(plan['asset_sha256'].get(plan['fixture'])==INPUT_SHA and plan['input_sha256']==INPUT_SHA,'immutable six inputs')
    mandatory={str(Path(__file__).resolve()),str(Path(analysis.__file__).resolve()),plan['binary'],plan['fixture'],plan['guard'],
               plan['build_manifest'],plan['source_proof'],plan['guard_interpreter_identity']['path']}
    mandatory.update(plan['core_objects']);mandatory.update(v for k,v in plan['environment'].items() if k.startswith('DS4_METAL_'))
    analysis.require(mandatory<=set(plan['asset_sha256']),'mandatory source/helper/guard/input/shader inventory')
    analysis.require(plan['asset_sha256'][plan['binary']]==plan['binary_sha256'] and
                     plan['asset_sha256'][plan['guard']]==plan['guard_sha256'],'binary/guard pins')
    analysis.require(all(plan['asset_sha256'][p]==h for p,h in plan['core_objects'].items()),'native core object pins')
    env=plan['environment']
    analysis.require({k:v for k,v in env.items() if not k.startswith('DS4_METAL_')}==
        {'DS4_OVERTHINK_PENALTY':'0','DS4_OVERTHINK_SCOPE':'think','DS4_OVERTHINK_AFTER':'0','DS4_OVERTHINK_MARKERS':'hesitation'},'native policy environment')
    for a in plan['arms']:
        analysis.require(a['argv']==[plan['binary'],plan['model']['path'],a['run_name'],str(a['level'])] and
            a['results_dir']==str(Path(plan['workspace'])/'speed-bench/ideas/artifacts'/a['run_name']),'exact helper arguments/output')

def check_assets(plan):
    for path,want in plan['asset_sha256'].items():analysis.require(sha(path)==want,'frozen asset differs '+path)

def require_guard(plan,plan_path,expected_sha):
    analysis.require(sha(plan['guard'])==plan['guard_sha256'],'MAIN guard pin')
    args=shlex.split(subprocess.check_output(['/bin/ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip())
    tail=[plan['guard'],plan['pressure'],'--',plan['launch_python'],str(Path(__file__).resolve()),str(Path(plan_path).resolve()),expected_sha]
    analysis.require(bool(args) and args[1:]==tail,'exact MAIN guard parent suffix')
    interpreter=plan['guard_interpreter_identity']
    try:same=os.path.samefile(args[0],interpreter['path'])
    except OSError:same=False
    analysis.require(same and sha(interpreter['path'])==interpreter['sha256'],'pinned MAIN framework interpreter')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN,errno.EWOULDBLOCK):raise
            return dict(parent_argv=args,exact_parent_suffix=True,framework_samefile=True,guard_lock_held=True)
        fcntl.flock(lock,fcntl.LOCK_UN)
    raise RuntimeError('MAIN model lock must be held')

def model_identity(plan):
    model=Path(plan['model']['path']);analysis.require(model.resolve()==model,'no model alias')
    stat=model.stat();actual=dict(path=str(model),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    analysis.require(actual==plan['model'],'Q2 model identity changed');return actual

def run(plan_path,expected_sha):
    analysis.require(Path(plan_path).suffix.lower()!='.gguf','No model as manifest')
    payload=Path(plan_path).read_bytes()
    analysis.require(hashlib.sha256(payload).hexdigest()==expected_sha,'explicit authoritative manifest SHA')
    plan=json.loads(payload);validate_plan(plan)
    proof=require_guard(plan,plan_path,expected_sha);check_assets(plan)
    # First current GGUF stat occurs only after exact direct parent/held-lock/assets.
    identity=model_identity(plan)
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    put(output/'plan.json',plan);put(output/'parent-proof.json',proof);put(output/'model-identity.json',identity)
    base={k:v for k,v in os.environ.items() if not k.startswith('DS4')};base.update(plan['environment'])
    fixture=json.loads(Path(plan['fixture']).read_text())
    analysis.require({c['case']:c['answer'] for c in fixture['cases']}==GOLDS,'six exact gold identities')
    reports=[];first={};repeat_checks=[]
    try:
        for arm in plan['arms']:
            analysis.require(sha(plan_path)==expected_sha,'manifest retained through arms')
            require_guard(plan,plan_path,expected_sha);check_assets(plan);model_identity(plan)
            put(output/(arm['name']+'-command.json'),dict(argv=arm['argv'],cwd=plan['run_cwd'],ds4_environment={k:v for k,v in base.items() if k.startswith('DS4')}))
            print(json.dumps(dict(event='arm_start',arm=arm['name'],level=arm['level'])),flush=True)
            with (output/(arm['name']+'-stdout.log')).open('x') as so,(output/(arm['name']+'-stderr.log')).open('x') as se:
                process=subprocess.Popen(arm['argv'],cwd=plan['run_cwd'],env=base,stdout=so,stderr=se)
                put(output/(arm['name']+'-started.json'),dict(pid=process.pid));rc=process.wait()
            put(output/(arm['name']+'-process.json'),dict(returncode=rc))
            analysis.require(rc in (0,3),'producer infrastructure failure; first attempt retained')
            validated=analysis.validate_arm(arm['results_dir'],arm['level'],fixture,rc)
            report,rows,infos=validated;report.update(arm=arm['name'])
            if arm['level'] in first:
                analysis.require_repeat(first[arm['level']],validated);repeat_checks.append(arm['level'])
            else:first[arm['level']]=validated
            put(output/(arm['name']+'-summary.json'),report);reports.append(report)
            print(json.dumps(report,allow_nan=False),flush=True)
        require_guard(plan,plan_path,expected_sha);check_assets(plan);model_identity(plan)
        analysis.require(repeat_checks==[75,25,0],'all repeated native controls complete')
        put(output/'summary.json',dict(status='COMPLETE',arms=reports,all_first_attempts_retained=True,
            exact_repeated_native_levels=repeat_checks,unique_problem_count=6,first_attempts=36,
            first_blocks_by_effort={str(level):first[level][0] for level in [0,25,75]},
            timing='Diagnostic only, no speed claim.',interpretation='Native effort prefixes differ. No quantization causality or cross-prefix KL/NLL. All original wrong/capped/INVALID attempts retained.'))
        return 3 if any(r['invalid'] for r in reports) else 0
    except BaseException as error:
        put(output/'first-failure.json',dict(type=type(error).__name__,message=str(error),completed_arms=reports));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path);parser.add_argument('plan_sha256')
    args=parser.parse_args();sys.exit(run(args.plan,args.plan_sha256))
