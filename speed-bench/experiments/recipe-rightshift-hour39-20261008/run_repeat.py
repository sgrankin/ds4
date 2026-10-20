#!/usr/bin/env python3
"""Run the existing immutable-source probe with right/left/right offsets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-revision', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / 'ds4_metal.m').read_text()))
    env.update({k: str(ROOT / v) for k, v in shaders.items()})
    env.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
               DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0',
               DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024')
    command = [str(a.binary.resolve()), '--model', str(ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),
               '--out', str(a.output.resolve()), '--moves', '3', '--prerope', '1',
               '--ctx', '8192', '--chunk', '512', '--cache-gb', '64']
    record = {'command': command, 'cwd': str(ROOT), 'runtime_revision': a.runtime_revision,
              'binary_sha256': hashlib.sha256(a.binary.read_bytes()).hexdigest(),
              'environment': {k: v for k, v in env.items() if k.startswith('DS4')}}
    (a.output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    with (a.output / 'stdout.log').open('wb') as out, (a.output / 'stderr.log').open('wb') as err:
        rc = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err).returncode
    (a.output / 'result.json').write_text(json.dumps({'returncode': rc}) + '\n')
    print(json.dumps({'repeat_right_left_right_returncode': rc}), flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
