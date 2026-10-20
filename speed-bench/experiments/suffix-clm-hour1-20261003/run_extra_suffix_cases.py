"""Small serial sweep using the already-built relocated suffix oracle."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Volumes/Models/ds4')
HERE = Path(__file__).resolve().parent
source = ROOT / '.workspaces/idea-clm-block-oracle-ordinary/speed-bench/clm-block-oracle-ordinary-v2/run_clm_block_oracle.py'
spec = importlib.util.spec_from_file_location('existing_oracle', source)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
env.update(oracle.POLICY)
env.update({k: str(ROOT / 'metal' / v) for k, v in oracle.SHADERS.items()})
env['DS4_METAL_TIMING_SUMMARY'] = '1'
model = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
st = model.stat()
assert st.st_size == 365713686528 and st.st_mtime_ns == 1789237692837653163
helper = ROOT / 'speed-bench/ideas/artifacts/suffix-diagnostic-native-wrap-build-20261002/suffix_fact_oracle'
for family in ['ordinary-arithmetic', 'ordinary-python']:
    for tail in [128]:
        output = HERE / f'{family}-tail{tail}'
        output.mkdir(exist_ok=False)
        command = [str(helper), '--model', str(model), '--fixtures', str(HERE / 'fixtures' / family),
                   '--case', 'code', '--out', str(output), '--prefix', '1024', '--source-edit', '256',
                   '--edit', '128', '--unchanged', '512', '--fresh-tail', str(tail), '--greedy', '16',
                   '--ctx', '4096', '--chunk', '512', '--cache-gb', '64', '--gold-answer', '42' if family == 'ordinary-arithmetic' else '206',
                   '--stale-answer', '41' if family == 'ordinary-arithmetic' else '205', '--require-fresh-query']
        record = {'argv': command, 'cwd': str(ROOT), 'family': family, 'fresh_tail': tail,
                  'settings': {k: v for k, v in env.items() if k.startswith('DS4')}}
        started = time.monotonic()
        print('Running', output.name, flush=True)
        try:
            with (output / 'helper.stdout').open('w') as stdout, (output / 'helper.stderr').open('w') as stderr:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600)
            record['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            record['returncode'] = 124
        record['seconds'] = time.monotonic() - started
        (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        if (output / 'summary.json').exists() and record['returncode'] == 0:
            summary = json.loads((output / 'summary.json').read_text())
            print(json.dumps({'case': output.name, 'seconds': record['seconds'],
                              'mean_delta_nll': summary['sum_delta_nll'] / summary['rows'],
                              'mean_kl': summary['mean_kl'],
                              'answers': [x['greedy'] for x in summary['results']],
                              'edit_ms': [x['edit_ms'] for x in summary['results']]}), flush=True)
        else:
            print(json.dumps({'case': output.name, 'returncode': record['returncode']}), flush=True)
            raise SystemExit(record['returncode'])
