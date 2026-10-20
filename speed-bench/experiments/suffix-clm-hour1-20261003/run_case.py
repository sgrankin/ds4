"""Run one prebuilt caller inside MAIN pressure_run.py; keep ordinary logs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Volumes/Models/ds4')
CONTROLLER = ROOT / '.workspaces/idea-clm-block-oracle-ordinary/speed-bench/clm-block-oracle-ordinary-v2/run_clm_block_oracle.py'
spec = importlib.util.spec_from_file_location('existing_oracle', CONTROLLER)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

case = Path(sys.argv[1]).resolve()
output = case / 'run'
output.mkdir(exist_ok=False)
model = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
st = model.stat()
assert st.st_size == 365713686528 and st.st_mtime_ns == 1789237692837653163
env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
env.update(oracle.POLICY)
env.update({k: str(ROOT / 'metal' / v) for k, v in oracle.SHADERS.items()})
command = [str(case / 'probe'), '--guarded-run', str(model), str(output)]
record = {'argv': command, 'cwd': str(ROOT), 'settings': {k: v for k, v in env.items() if k.startswith('DS4')},
          'model_bytes': st.st_size, 'model_mtime_ns': st.st_mtime_ns}
started = time.monotonic()
print('Running', case.name, flush=True)
try:
    with (output / 'helper.stdout').open('w') as stdout, (output / 'helper.stderr').open('w') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600)
    record['returncode'] = result.returncode
    if result.returncode == 0:
        record['arms'] = oracle.controls(output, (output / 'helper.stdout').read_bytes())
        quality = oracle.quality(output)
        (output / 'quality.json').write_text(json.dumps(quality, indent=2) + '\n')
        record['mean_delta_nll'] = quality['four_teacher_sum_delta_nll'] / 4
except Exception as error:
    record.update(returncode=124 if isinstance(error, subprocess.TimeoutExpired) else 1,
                  error=repr(error))
finally:
    record['seconds'] = time.monotonic() - started
    (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'case': case.name, **{k: record[k] for k in ('returncode', 'seconds')},
                      'mean_delta_nll': record.get('mean_delta_nll'), 'error': record.get('error')}), flush=True)
raise SystemExit(record['returncode'])
