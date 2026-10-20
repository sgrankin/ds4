#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from pathlib import Path
import run_native_effort_holdout as runner
BUILD=Path('/Volumes/Models/ds4/.workspaces/idea-native-effort-holdout/speed-bench/ideas/artifacts/native-effort-holdout-prepared-20261003-v1')
PLAN_SHA='e7c3c8a0e960a8630f872ccfe3963503b08dbffbc305af926e8fe66c9322dbd6'
COMMAND_SHA='cb168d647421abf296bbe3640c3b82ec24234f6ba43b06a4ad0a625716b12c2c'
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
