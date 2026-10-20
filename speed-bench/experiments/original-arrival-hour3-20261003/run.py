"""One guarded, serial fixed-input comparison; see each execution.json."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Volumes/Models/ds4')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.workspaces/idea-resident-original-deadline-native'
BUILD = WORK / 'speed-bench/ideas/artifacts/original-deadline-native-adapter-build-v1'
MODEL = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
FIXTURES = ROOT / 'speed-bench/experiments/resident-functional-hour2-20261003'
parser = argparse.ArgumentParser()
parser.add_argument('--all-cases', action='store_true')
parser.add_argument('--strong', action='store_true')
parser.add_argument('--group', default='screen')
parser.add_argument('--helper', default=str(BUILD / 'quality-original-deadline'))
parser.add_argument('arms', nargs='+')
args = parser.parse_args()
out = HERE / args.group
out.mkdir(parents=True, exist_ok=True)
st = MODEL.stat()  # Called only under the held MAIN model-job guard.
assert (st.st_size, st.st_mtime_ns) == (365713686528, 1789237692837653163)
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
env.update({
    'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.30',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.12',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '24',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': '1',
    'DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS': '1',
    'DS4_METAL_V41_RESIDENT_ROUTE_POLICY': 'replace',
    'DS4_METAL_V41_SELECTED_TILE': '64',
    'DS4_METAL_V41_SELECTED_LIMIT': '1024',
    'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1',
    'DS4_METAL_V41_RESIDENT_ROUTE_PROFILE': '1',
})
if args.strong:
    env.update({'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.90',
                'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.25',
                'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '64',
                'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': '3',
                'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS': '.50'})
for arm in args.arms:
    arm_env = dict(env)
    arm_env['DS4_METAL_V41_RESIDENT_ROUTE_MODE'] = 'native' if arm.startswith('native') else 'apply'
    arm_env['DS4_METAL_V41_ORIGINAL_DEADLINE'] = '1' if arm.startswith('deadline') else '0'
    helper = args.helper + '-' + arm.rsplit('-', 1)[1] if arm.endswith(('-age1', '-age8')) else args.helper
    command = [helper, str(MODEL), str(FIXTURES / ('all.tsv' if args.all_cases else 'four.tsv')),
               str(out / f'{arm}.csv'), str(out / 'reference.bin'),
               'record' if arm == 'native-before' else 'compare',
               '--cases', '24' if args.all_cases else '4', '--tokens', '32',
               '--ctx', '8192', '--cache-gb', '64', '--chunk', '512']
    if arm != 'native-before':
        command += ['--candidate-record', str(out / f'{arm}.bin')]
    record = {'argv': command, 'cwd': str(WORK),
              'environment': {k: v for k, v in arm_env.items() if k.startswith('DS4')},
              'model_bytes': st.st_size, 'model_mtime_ns': st.st_mtime_ns}
    print('Running', arm, flush=True)
    start = time.monotonic()
    with (out / f'{arm}.stdout').open('w') as stdout, (out / f'{arm}.stderr').open('w') as stderr:
        result = subprocess.run(command, cwd=WORK, env=arm_env, stdout=stdout, stderr=stderr, timeout=600)
    record.update(returncode=result.returncode, seconds=time.monotonic() - start)
    (out / f'{arm}.execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'arm': arm, 'returncode': result.returncode, 'seconds': record['seconds']}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    if arm.startswith('deadline'):
        assert 'ORIGINAL_DEADLINE generation=' in (out / f'{arm}.stderr').read_text()
