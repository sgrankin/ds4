#!/usr/bin/env python3
"""Run a preregistered medium nonzero-position coding continuation screen.

All model metadata and opens happen only beneath the MAIN pressure guard.
Four first completions per arm are scored without repair; time is diagnostic.
"""
import argparse
from array import array
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path('/Volumes/Models/ds4')
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
FAST = {'DS4_METAL_V41_SELECTED_TILE':'64', 'DS4_METAL_V41_SELECTED_LIMIT':'1024'}
NATIVE = FAST | {'DS4_METAL_V41_RESIDENT_ROUTE_GAP':'0'}
THREE = FAST | {
    'DS4_METAL_V41_RESIDENT_ROUTE_GAP':'.30',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS':'.20',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS':'.40',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK':'48',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS':'3',
    'DS4_METAL_V41_RESIDENT_ROUTE_MODE':'apply',
    'DS4_METAL_V41_RESIDENT_ROUTE_POLICY':'replace'}
ARMS = [('native-before',64,NATIVE), ('broader3',64,THREE), ('native-after',64,NATIVE)]


def sha(path):
    if str(path).lower().endswith('.gguf'):
        raise ValueError('model hashing forbidden')
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def require_guard(plan=None):
    parent = subprocess.check_output(['ps','-p',str(os.getppid()),'-o','command='],text=True,timeout=2).strip()
    args=shlex.split(parent)
    if str(GUARD) not in args or (plan is not None and
            (args.index(str(GUARD))+1>=len(args) or args[args.index(str(GUARD))+1]!=plan['pressure_output'])):
        raise RuntimeError('MAIN pressure guard must be direct parent')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock,fcntl.LOCK_UN)
    raise RuntimeError('MAIN model-job lock is not held')


def check_assets(plan):
    for path, digest in plan['asset_sha256'].items():
        if sha(path) != digest:
            raise ValueError('frozen asset changed: '+path)


def clean_environment(plan):
    return {k:v for k,v in os.environ.items() if not k.startswith('DS4')} | plan['metal_environment']


def decision_hash(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    native=[(r['token_id'],r['stop'],r['eos'],r['piece_hex']) for r in rows]
    return hashlib.sha256(json.dumps(native,separators=(',',':')).encode()).hexdigest()


def require_native_control(directory, plan):
    for task in plan['task_order']:
        expected=plan['native_controls'][task]
        for suffix in ('warm-logits.bin','initial-logits.bin','prompt.json','txt'):
            if sha(directory/(task+'.'+suffix))!=expected[suffix]:
                raise ValueError('native historical control differs: '+task+'.'+suffix)
        if decision_hash(directory/(task+'.tokens.jsonl'))!=expected['native_decision_sha256']:
            raise ValueError('native historical decision control differs: '+task)


def route_footer(path):
    rows=[line for line in Path(path).read_text().splitlines()
          if line.startswith('ds4: resident-route layers=')]
    if len(rows)!=1:
        raise ValueError('candidate resident route footer missing or repeated')
    values=dict(re.findall(r'([a-z_]+)=([^ ]+)',rows[0]))
    required=('layers','substitutions','native_misses','skipped_loads','skipped_bytes',
              'full_snapshots','double_layers','planned_substitutions','drops','triple_layers')
    for key in required:
        if key not in values or not values[key].isdigit():
            raise ValueError('candidate resident route footer invalid')
        values[key]=int(values[key])
    if values['substitutions']<=0 or values['drops'] or values['skipped_loads']!=values['substitutions']:
        raise ValueError('candidate did not exercise replacement routing')
    return values


def guarded_model_identity(plan):
    path = Path(plan['model']['path'])
    if path.resolve() != path:
        raise ValueError('Q2 model must not resolve to another model')
    stat = path.stat()
    actual = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    expected = {key:plan['model'][key] for key in actual}
    if actual != expected:
        raise ValueError('model identity differs from prior prepared Q2 identity')
    return actual


def validate_arm(directory, plan, expected_cache=None):
    records = [json.loads(s) for s in (directory/'generation.jsonl').read_text().splitlines()]
    if [r['task'] for r in records] != plan['task_order']:
        raise ValueError('arm must cover frozen task order exactly once')
    metadata=[]
    for r in records:
        if expected_cache is not None and r['cache_gib']!=expected_cache:
            raise ValueError('cache argv/record drift')
        task=r['task']; prompt=json.loads((directory/(task+'.prompt.json')).read_text())
        if (r['warm_tokens'] <= 0 or r['warm_position'] != r['warm_tokens'] or
            not r['prefix_exact'] or r['actual_prefill_cap']<64 or not 256 <= r['append_tokens'] < 1024 or
            len(prompt['ids']) != r['warm_tokens']+r['append_tokens'] or
            len(prompt['ids'])+512 > 4096 or prompt['warm_tokens'] != r['warm_tokens'] or
            prompt['append_tokens'] != r['append_tokens'] or
            prompt['warm_position'] != r['warm_position'] or not prompt['prefix_exact']):
            raise ValueError('medium append eligibility or recorded prompt differs')
        if any(r[k]!=v for k,v in {'context':4096,'prefill_chunk':512,'max_tokens':512,
                                  'think_mode':'none','temperature':0,'seed':20261002}.items()):
            raise ValueError('generation options drift')
        trace=[json.loads(s) for s in (directory/r['token_trace']).read_text().splitlines()]
        if [v['step'] for v in trace] != list(range(r['sampling_decisions'])):
            raise ValueError('sampling trace decisions drift')
        emitted=[v for v in trace if not v['stop']]
        if len(emitted)!=r['generated_tokens'] or bytes.fromhex(''.join(v['piece_hex'] for v in emitted)) != (directory/r['raw_trace']).read_bytes():
            raise ValueError('emitted native stream differs from raw code')
        if r['stopped']:
            if not trace or not trace[-1]['stop'] or r['cap_hit'] or trace[-1]['token_id']!=r['stop_token_id'] or trace[-1]['eos']!=r['eos']:
                raise ValueError('stop/EOS metadata drift')
        elif not r['cap_hit'] or r['generated_tokens']!=512 or any(v['stop'] for v in trace):
            raise ValueError('cap metadata drift')
        warm=directory/(task+'.warm-logits.bin')
        if warm.stat().st_size!=r['vocab']*4:
            raise ValueError('warm full vocabulary logit bytes differ')
        logits=directory/(task+'.initial-logits.bin')
        if logits.stat().st_size!=r['vocab']*4 or r['vocab']!=129280:
            raise ValueError('initial full vocabulary logit bytes differ')
        metadata.append(dict(task=task,warm_tokens=r['warm_tokens'],append_tokens=r['append_tokens'],
            prompt_sha256=sha(directory/(task+'.prompt.json')),warm_logits_sha256=sha(warm),initial_logits_sha256=sha(logits),
            native_stream_sha256=sha(directory/r['token_trace']),raw_code_sha256=sha(directory/r['raw_trace'])))
    return records,metadata


def compare_vectors(left, right):
    a=array('f'); a.frombytes(Path(left).read_bytes())
    b=array('f'); b.frombytes(Path(right).read_bytes())
    if len(a)!=129280 or len(a)!=len(b) or not all(math.isfinite(x) for x in a) or not all(math.isfinite(x) for x in b):
        raise ValueError('invalid full vocabulary vectors')
    def logz(v):
        m=max(v); return m+math.log(math.fsum(math.exp(x-m) for x in v))
    za,zb=logz(a),logz(b)
    pa=[math.exp(x-za) for x in a]; pb=[math.exp(x-zb) for x in b]
    kl=math.fsum(p*((x-za)-(y-zb)) for p,x,y in zip(pa,a,b))
    return dict(full_raw_bytes_equal=Path(left).read_bytes()==Path(right).read_bytes(),
                kl_temperature_one=max(0.0,kl),tv_temperature_one=.5*math.fsum(abs(p-q) for p,q in zip(pa,pb)),
                argmax_flip=max(range(len(a)),key=a.__getitem__)!=max(range(len(b)),key=b.__getitem__))


def run(plan_path, plan_sha, output):
    if sha(plan_path)!=plan_sha:
        raise ValueError('frozen plan hash differs')
    plan=json.loads(Path(plan_path).read_text())
    require_guard(plan)  # Must precede even model stat.
    if plan['arms']!=[dict(name=n,cache_gib=c,policy=p) for n,c,p in ARMS]:
        raise ValueError('frozen arm policy differs')
    check_assets(plan)
    model=guarded_model_identity(plan)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',plan)
    write(output/'model-identity.json',model)
    base=clean_environment(plan)
    summaries=[]; prior_prompts=None
    for name,cache,policy in ARMS:
        check_assets(plan); guarded_model_identity(plan)
        child=output/name
        env=base|policy
        argv=[plan['binary'],model['path'],str(child),plan['prompts_dir'],str(cache),'512']
        write(output/(name+'-command.json'),dict(argv=argv,cwd=plan['build_dir'],
            ds4_environment={k:v for k,v in env.items() if k.startswith('DS4_')}))
        print(json.dumps(dict(event='arm_start',arm=name,cache_gib=cache)),flush=True)
        with (output/(name+'-stdout.log')).open('x') as out, (output/(name+'-stderr.log')).open('x') as err:
            rc=subprocess.run(argv,cwd=plan['build_dir'],env=env,stdout=out,stderr=err).returncode
        check_assets(plan); guarded_model_identity(plan)
        if rc:
            raise RuntimeError(name+' generation failed rc='+str(rc))
        records,metadata=validate_arm(child,plan,cache)
        prompts=[m['prompt_sha256'] for m in metadata]
        if prior_prompts is not None and prompts!=prior_prompts:
            raise ValueError('native prompt IDs differ across arms')
        prior_prompts=prompts
        if name.startswith('native-'):
            require_native_control(child,plan)
        scorer=[sys.executable,'-I','-S',plan['scorer'],plan['fixture'],str(child)]
        with (output/(name+'-score.log')).open('x') as log:
            score_rc=subprocess.run(scorer,cwd=plan['build_dir'],env=base,stdout=log,stderr=subprocess.STDOUT).returncode
        if score_rc not in (0,3):
            raise RuntimeError(name+' scorer execution failed rc='+str(score_rc))
        score=json.loads((child/'score.json').read_text())
        if score['fixture_sha256']!=plan['fixture_sha256'] or score['total']!=4:
            raise ValueError('scorer fixture/count differs')
        result=dict(arm=name,cache_gib=cache,correct=score['correct'],total=score['total'],score_rc=score_rc,
                    generated_tokens=sum(r['generated_tokens'] for r in records),
                    cap_hits=sum(r['cap_hit'] for r in records),tasks=score['tasks'],trace_metadata=metadata)
        if name=='broader3':
            result['resident_route']=route_footer(output/(name+'-stderr.log'))
        summaries.append(result)
        write(output/(name+'-summary.json'),result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','trace_metadata')}),flush=True)
    comparisons=[]
    for name,_,_ in ARMS[1:]:
        for task in plan['task_order']:
            a,b=output/'native-before',output/name
            vectors=compare_vectors(a/(task+'.initial-logits.bin'),b/(task+'.initial-logits.bin'))
            warm_vectors=compare_vectors(a/(task+'.warm-logits.bin'),b/(task+'.warm-logits.bin'))
            ta=[json.loads(s) for s in (a/(task+'.tokens.jsonl')).read_text().splitlines()]
            tb=[json.loads(s) for s in (b/(task+'.tokens.jsonl')).read_text().splitlines()]
            def stream(t):return [(r['token_id'],r['stop'],r['eos']) for r in t]
            sa,sb=stream(ta),stream(tb)
            first=next((i for i,(x,y) in enumerate(zip(sa,sb)) if x!=y),min(len(sa),len(sb)) if len(sa)!=len(sb) else None)
            comparisons.append(dict(arm=name,task=task,**vectors,warm_comparison=warm_vectors,exact_native_stream=sa==sb,
                first_decision_difference=first,exact_raw_code=(a/(task+'.txt')).read_bytes()==(b/(task+'.txt')).read_bytes()))
    check_assets(plan); guarded_model_identity(plan)
    summary=dict(status='COMPLETE',arms=summaries,comparisons_to_native_fast64=comparisons,
        fixture_sha256=plan['fixture_sha256'],assembly_sha256=plan['assembly_sha256'],
        all_native_prompt_ids_equal=True,both_native_controls_exact_historical_vectors_code_decisions=True,medium_nonzero_position_prefill_eligible_all_tasks=True,
        timing_claim='none; startup/order, CPU tracing and correctness checks are diagnostic',
        interpretation='Four first completions, unchanged34 gold checks; no repair/best-of. Whole resident policy, including any eligible warm or scalar prefill, is evaluated. Warm drift is measured, not forced equal for the candidate. Eligibility is native token count/source-policy evidence, not a profiler trace.')
    write(output/'summary.json',summary)
    return 0 if all(s['correct']==4 for s in summaries) else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan',type=Path);p.add_argument('plan_sha256');p.add_argument('output',type=Path)
    a=p.parse_args();sys.exit(run(a.plan,a.plan_sha256,a.output))
if __name__=='__main__':main()
