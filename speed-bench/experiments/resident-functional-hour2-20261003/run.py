"""Run fixed-input resident comparisons under MAIN pressure_run.py."""
import json
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/Volumes/Models/ds4')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.workspaces/idea-resident-functional-runtime-native'
HELPER = WORK / 'speed-bench/ideas/artifacts/resident-functional-native-direct-v1/quality-functional'
MODEL = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
MAPDIR = ROOT / '.workspaces/idea-resident-functional-map-publication/speed-bench/ideas/artifacts/resident-functional-map-publication-v2'
FIXTURES = ROOT / '.workspaces/idea-resident-functional-validation24/speed-bench/ideas/artifacts/functional-validation24-input-freeze-20261003-v1'
IDS = ['functional24_01_cursor_pages', 'functional24_07_little_endian_words',
       'functional24_16_weighted_median', 'functional24_24_literal_unicode_note']

if sys.argv[1:] == ['prepare']:
    import shutil
    dest = HERE / 'fixtures'
    dest.mkdir(exist_ok=True)
    rows = []
    for name in IDS:
        for suffix in ('prompt.txt', 'continuation.txt', 'gold.json'):
            shutil.copy2(FIXTURES / f'{name}.{suffix}', dest)
        rows.append('\t'.join([name, str(dest / f'{name}.prompt.txt'), str(dest / f'{name}.continuation.txt')]))
    (HERE / 'four.tsv').write_text('\n'.join(rows) + '\n')
    raise SystemExit(0)

parser = argparse.ArgumentParser()
parser.add_argument('--all-cases', action='store_true')
parser.add_argument('--group', default='.')
parser.add_argument('--tokens', type=int, default=32)
parser.add_argument('--generate', action='store_true')
parser.add_argument('arms', nargs='*')
args = parser.parse_args()
out = HERE / args.group
out.mkdir(exist_ok=True)
manifest = HERE / 'all.tsv' if args.all_cases else HERE / 'four.tsv'
case_count = 24 if args.all_cases else 4
st = MODEL.stat()  # This path is executed only inside the held model-job guard.
assert (st.st_size, st.st_mtime_ns) == (365713686528, 1789237692837653163)
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
pins = (WORK / 'ds4_resident_functional_kernel_pins.h').read_text()
env.update(dict(re.findall(r'\{"(DS4_[^"]+)","([^"]+)"', pins)))
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
    'DS4_METAL_V41_RESIDENT_FUNCTIONAL_PROFILE': '1',
})

for arm in args.arms or ['native-before', 'affinity', 'functional', 'native-after']:
    arm_env = dict(env)
    arm_env['DS4_METAL_V41_RESIDENT_ROUTE_MODE'] = 'native' if arm.startswith('native') else 'apply'
    arm_env['DS4_METAL_V41_RESIDENT_FUNCTIONAL'] = '1' if arm.startswith('functional') else '0'
    if arm.startswith('affinity-gap'):
        arm_env['DS4_METAL_V41_RESIDENT_ROUTE_GAP'] = arm.removeprefix('affinity-gap')
    if arm in ('affinity-two', 'affinity-three', 'affinity-three-mass0.25'):
        arm_env.update({'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.90',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '64',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.20',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': '2' if arm.endswith('two') else '3',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS': '.40' if arm.endswith('two') else '.50'})
        if arm.endswith('mass0.25'):
            arm_env['DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS'] = '.25'
    if arm.startswith('affinity-wide'):
        arm_env.update({'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.90',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '64',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.25',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': arm.removeprefix('affinity-wide'),
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS': '.50'})
    helper = HELPER.with_name('quality-' + arm) if arm.startswith('functional-band') else HELPER
    if arm.startswith('functional'):
        arm_env.update({
            'DS4_METAL_V41_RESIDENT_FUNCTIONAL_MAP': str(MAPDIR / 'functional.ds4fmap1'),
            'DS4_METAL_V41_RESIDENT_FUNCTIONAL_QUALIFICATION': str(MAPDIR / 'qualification.ds4fqal1'),
            'DS4_METAL_V41_RESIDENT_FUNCTIONAL_QUALIFICATION_SHA256': 'da87dd4f73aa80eb2e029e9ab1c05f6005abb644f76eb8516ebb476a0a33a82e',
        })
    if args.generate:
        arm_env['DS4_CLI_FORCE_SESSION'] = '1'
        cli = HELPER.with_name('ds4-' + arm) if arm.startswith('functional-band') else HELPER.with_name('ds4-functional')
        for name in IDS:
            command = [str(cli), '--model', str(MODEL), '--prompt-file', str(HERE / 'fixtures' / f'{name}.prompt.txt'),
                       '--ctx', '8192', '--ssd-streaming', '--ssd-streaming-cache-experts', '64GB',
                       '--prefill-chunk', '512', '--tokens', str(args.tokens), '--temp', '0', '--seed', '20261003', '--nothink']
            record = {'argv': command, 'environment': {k: v for k, v in arm_env.items() if k.startswith('DS4')}}
            start = time.monotonic()
            print('Generating', arm, name, flush=True)
            stem = f'{arm}-{name}'
            with (out / f'{stem}.stdout').open('w') as stdout, (out / f'{stem}.stderr').open('w') as stderr:
                result = subprocess.run(command, cwd=ROOT, env=arm_env, stdout=stdout, stderr=stderr, timeout=300)
            record.update(returncode=result.returncode, seconds=time.monotonic() - start)
            (out / f'{stem}.execution.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({'arm': arm, 'case': name, 'returncode': result.returncode, 'seconds': record['seconds']}), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
            stderr_text = (out / f'{stem}.stderr').read_text()
            assert 'GENERATION_SUMMARY ' in stderr_text, 'session generation footer missing'
            if arm.startswith('functional'):
                assert 'FUNCTIONAL_MAP eligible=' in stderr_text, 'functional selector was not activated'
        continue
    command = [str(helper), str(MODEL), str(manifest), str(out / f'{arm}.csv'),
               str(out / 'reference.bin'), 'record' if arm == 'native-before' else 'compare',
               '--cases', str(case_count), '--tokens', str(args.tokens), '--ctx', '8192', '--cache-gb', '64', '--chunk', '512']
    if arm != 'native-before':
        command += ['--candidate-record', str(out / f'{arm}.bin')]
    record = {'argv': command, 'environment': {k: v for k, v in arm_env.items() if k.startswith('DS4')},
              'model_bytes': st.st_size, 'model_mtime_ns': st.st_mtime_ns}
    start = time.monotonic()
    print('Running', arm, flush=True)
    with (out / f'{arm}.stdout').open('w') as stdout, (out / f'{arm}.stderr').open('w') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=arm_env, stdout=stdout, stderr=stderr, timeout=600)
    record.update(returncode=result.returncode, seconds=time.monotonic() - start)
    (out / f'{arm}.execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'arm': arm, 'returncode': result.returncode, 'seconds': record['seconds']}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
