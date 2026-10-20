#!/usr/bin/env python3
"""CPU-only frozen launch preparation; never access the model or outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import run_native_effort_holdout as runner

ROOT=Path('/Volumes/Models/ds4')
WORKSPACE=Path(__file__).resolve().parents[1]
SEED=ROOT/'.workspaces/idea-quant-closing/speed-bench/ideas/artifacts/closing-prepared-20261002-v1'
SEED_SHA='cca9ffbc4e8f5d77c77f7b2c68f622dc144cbd54d426adc80b854418ba40ed65'
BUILD=WORKSPACE/'speed-bench/ideas/artifacts/native-effort-holdout-helper-build-v1'
BUILD_SHA='e259b62754cc4d66f2b5c262db42fc2fb7e421f7c9385a1a607a5e7f90db6a04'

def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def prepare(output):
    sha=runner.sha
    if sha(SEED/'plan.json')!=SEED_SHA or sha(BUILD/'build-manifest.json')!=BUILD_SHA:raise ValueError('Frozen seed/build differs')
    seed=json.loads((SEED/'plan.json').read_text());build=json.loads((BUILD/'build-manifest.json').read_text())
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name,src in [('run_native_effort_holdout.py',WORKSPACE/'speed-bench/run_native_effort_holdout.py'),
                     ('analyze_native_effort_holdout.py',WORKSPACE/'speed-bench/analyze_native_effort_holdout.py'),
                     ('native-effort-holdout',Path(build['binary']))]:shutil.copy2(src,output/name)
    binary=output/'native-effort-holdout';binary.chmod(0o755)
    fixture=WORKSPACE/'speed-bench/experiments/ds41-ssd/native-effort-holdout/input-freeze-v1.json'
    source_proof=WORKSPACE/'speed-bench/experiments/ds41-ssd/native-effort-holdout/source-seed-proof.json'
    env=seed['environment'];paths=set(build['objects'])|set(build['source_sha256'])|{fixture,source_proof,
        BUILD/'build-manifest.json',SEED/'plan.json',Path(seed['guard']),Path(seed['launch_python']),
        Path(seed['guard_interpreter_identity']['path']),binary,output/'run_native_effort_holdout.py',output/'analyze_native_effort_holdout.py',
        WORKSPACE/'speed-bench/experiments/ds41-ssd/native-effort-holdout/cpu-proof-v1.json',
        WORKSPACE/'speed-bench/experiments/ds41-ssd/native-effort-holdout/PREREGISTERED.md',
        WORKSPACE/'speed-bench/native_effort_holdout_gold.py',WORKSPACE/'speed-bench/prepare_native_effort_holdout.py',
        WORKSPACE/'tests/test_native_effort_holdout_gold.py',WORKSPACE/'tests/test_analyze_native_effort_holdout.py',
        WORKSPACE/'tests/test_native_effort_holdout_driver.py'}
    paths.update(Path(v) for k,v in env.items() if k.startswith('DS4_METAL_'))
    assets={str(Path(p).resolve()):sha(p) for p in sorted(paths,key=str)}
    for p,h in build['objects'].items():
        if assets[p]!=h:raise ValueError('Native object changed')
    for p,h in build['source_sha256'].items():
        if assets[p]!=h:raise ValueError('Helper build source changed')
    if sha(fixture)!=runner.INPUT_SHA:raise ValueError('Six inputs changed')
    if sha(binary)!=build['binary_sha256']:raise ValueError('Helper binary changed')
    study='native-effort-holdout-six-20261003-v1'
    arms=[]
    for name,level in zip(runner.NAMES,runner.LEVELS):
        run_name=study+'-'+name
        arms.append(dict(name=name,level=level,run_name=run_name,argv=[str(binary),seed['model']['path'],run_name,str(level)],
                         results_dir=str(WORKSPACE/'speed-bench/ideas/artifacts'/run_name)))
    plan=dict(workspace=str(WORKSPACE),run_cwd=str(WORKSPACE),root_cwd=str(ROOT),build_dir=str(output),
        output=str(WORKSPACE/'speed-bench/ideas/artifacts'/study),pressure=str(WORKSPACE/'speed-bench/ideas/artifacts'/(study+'-pressure')),
        arms=arms,settings=runner.SETTINGS,environment=env,fixture=str(fixture),input_sha256=runner.INPUT_SHA,
        model=seed['model'],model_identity_source='Retained immutable closing-prepared plan; no current stat performed during preparation.',
        binary=str(binary),binary_sha256=sha(binary),guard=seed['guard'],guard_sha256=seed['guard_sha256'],launch_python=seed['launch_python'],
        guard_interpreter_identity=seed['guard_interpreter_identity'],asset_sha256=assets,core_objects=build['objects'],
        build_manifest=str(BUILD/'build-manifest.json'),source_proof=str(source_proof),estimated_minutes=[90,110],
        runtime_source_commit='685ff0ba8864d66e7ca7ef60107536ac00fd2cb2',helper_input_source_commit='dde6f238a4edc569a698ba84c32216229f94d2f0')
    # validate_plan uses __file__ of the executable source; use copied equivalent.
    original=runner.__file__;runner.__file__=str(output/'run_native_effort_holdout.py')
    old_analysis=runner.analysis.__file__;runner.analysis.__file__=str(output/'analyze_native_effort_holdout.py')
    try:runner.validate_plan(plan)
    finally:runner.__file__=original;runner.analysis.__file__=old_analysis
    put(output/'plan.json',plan);plan_sha=sha(output/'plan.json')
    argv=[plan['launch_python'],plan['guard'],plan['pressure'],'--',plan['launch_python'],str(output/'run_native_effort_holdout.py'),str(output/'plan.json'),plan_sha]
    put(output/'prepared-command.json',dict(argv=argv,cwd=str(ROOT),environment=env,plan_sha256=plan_sha))
    command_sha=sha(output/'prepared-command.json')
    launcher='''#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from pathlib import Path
import run_native_effort_holdout as runner
BUILD=Path(%r)
PLAN_SHA=%r
COMMAND_SHA=%r
def preflight():
    if runner.sha(BUILD/'plan.json')!=PLAN_SHA or runner.sha(BUILD/'prepared-command.json')!=COMMAND_SHA:raise ValueError('Frozen authority differs')
    plan=json.loads((BUILD/'plan.json').read_text());command=json.loads((BUILD/'prepared-command.json').read_text())
    runner.validate_plan(plan);runner.check_assets(plan)
    expected=[plan['launch_python'],plan['guard'],plan['pressure'],'--',plan['launch_python'],str(BUILD/'run_native_effort_holdout.py'),str(BUILD/'plan.json'),PLAN_SHA]
    if command['argv']!=expected or command['cwd']!=str(runner.ROOT) or command['environment']!=plan['environment']:raise ValueError('Exact prepared command required')
    return plan,command
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    plan,command=preflight();print(json.dumps(dict(status='PREFLIGHT_PASS',assets=len(plan['asset_sha256']),model_access=False)),flush=True)
    if not a.execute:return 0
    env={k:v for k,v in os.environ.items() if not k.startswith('DS4')};env.update(command['environment'])
    return subprocess.run(command['argv'],cwd=command['cwd'],env=env).returncode
if __name__=='__main__':sys.exit(main())
'''%(str(output),plan_sha,command_sha)
    (output/'launch_native_effort_holdout.py').write_text(launcher)
    print(json.dumps(dict(prepared=str(output),plan_sha256=plan_sha,command_sha256=command_sha,
                         launcher_sha256=sha(output/'launch_native_effort_holdout.py'),model_access=False),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();prepare(a.output)
