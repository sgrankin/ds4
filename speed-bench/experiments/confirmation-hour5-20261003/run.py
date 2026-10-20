"""Run new confirmation inputs serially inside the held MAIN pressure guard."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Volumes/Models/ds4')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.workspaces/idea-resident-original-deadline-native'
BUILD = WORK / 'speed-bench/ideas/artifacts/original-deadline-admitted-direct-v1'
MODEL = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, default=HERE / 'teacher.tsv')
parser.add_argument('--cases', type=int, default=16)
parser.add_argument('--tokens', type=int, default=32)
parser.add_argument('--substitutions', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--group', default='teacher')
parser.add_argument('--helper', default=str(BUILD / 'quality-original-deadline'))
parser.add_argument('--generate', action='store_true')
parser.add_argument('--cli', help='Explicit compiled TTL8 CLI for all generation arms; deadline env is0 for controls/affinity.')
parser.add_argument('arms', nargs='*', default=['native-before', 'deadline-age8', 'affinity-strong', 'native-after'])
args = parser.parse_args()
if args.generate and not args.cli:
    parser.error('--generate requires --cli')
with args.manifest.open(newline='') as file:
    inputs = list(csv.reader(file, delimiter='\t'))
if not 0 < args.cases <= len(inputs):
    parser.error('--cases must select existing manifest rows')
out = HERE / args.group
out.mkdir(parents=True, exist_ok=True)
st = MODEL.stat()  # ROOT executes this runner only inside the held MAIN guard.
assert (st.st_size, st.st_mtime_ns) == (365713686528, 1789237692837653163)
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
env.update({
    'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.90',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.25',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '64',
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': str(args.substitutions),
    'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS': '.50',
    'DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS': '1',
    'DS4_METAL_V41_RESIDENT_ROUTE_POLICY': 'replace',
    'DS4_METAL_V41_SELECTED_TILE': '64',
    'DS4_METAL_V41_SELECTED_LIMIT': '1024',
    'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1',
    'DS4_METAL_V41_RESIDENT_ROUTE_PROFILE': '1',
})

for arm in args.arms:
    arm_env = dict(env)
    arm_env['DS4_METAL_V41_RESIDENT_ROUTE_MODE'] = 'native' if arm.startswith('native') else 'apply'
    arm_env['DS4_METAL_V41_ORIGINAL_DEADLINE'] = '1' if arm.startswith('deadline') else '0'
    if args.generate:
        arm_env['DS4_CLI_FORCE_SESSION'] = '1'
        commands = [(case, [args.cli, '--model', str(MODEL), '--prompt-file', prompt,
                    '--ctx', '8192', '--ssd-streaming', '--ssd-streaming-cache-experts', '64GB',
                    '--prefill-chunk', '512', '--tokens', str(args.tokens), '--temp', '0',
                    '--seed', '20261003', '--nothink'])
                    for case, prompt, _ in inputs[:args.cases]]
    else:
        helper = args.helper + '-' + arm.rsplit('-', 1)[1] if arm.endswith(('-age1', '-age8')) else args.helper
        command = [helper, str(MODEL), str(args.manifest), str(out / f'{arm}.csv'),
                   str(out / 'reference.bin'), 'record' if arm == 'native-before' else 'compare',
                   '--cases', str(args.cases), '--tokens', str(args.tokens),
                   '--ctx', '8192', '--cache-gb', '64', '--chunk', '512']
        if arm != 'native-before':
            command += ['--candidate-record', str(out / f'{arm}.bin')]
        commands = [(None, command)]
    for case, command in commands:
        stem = f'{arm}-{case}' if case else arm
        record = {'argv': command, 'cwd': str(WORK),
                  'environment': {k: v for k, v in arm_env.items() if k.startswith('DS4')},
                  'model_bytes': st.st_size, 'model_mtime_ns': st.st_mtime_ns}
        print('Running', stem, flush=True)
        start = time.monotonic()
        with (out / f'{stem}.stdout').open('w') as stdout, (out / f'{stem}.stderr').open('w') as stderr:
            result = subprocess.run(command, cwd=WORK, env=arm_env, stdout=stdout, stderr=stderr, timeout=600)
        record.update(returncode=result.returncode, seconds=time.monotonic() - start)
        (out / f'{stem}.execution.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'arm': arm, 'case': case, 'returncode': result.returncode,
                          'seconds': record['seconds']}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        stderr = (out / f'{stem}.stderr').read_text(errors='replace')
        if not arm.startswith('native'):
            assert 'ds4: resident-route ' in stderr, 'resident routing footer missing'
        if arm.startswith('deadline'):
            assert 'ORIGINAL_DEADLINE generation=' in stderr
        if args.generate:
            assert 'GENERATION_SUMMARY ' in stderr
