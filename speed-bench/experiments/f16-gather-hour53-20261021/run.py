#!/usr/bin/env python3
"""Fixed F16 storage, paired scalar-gather settings; run under pressure_run.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--bench', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.model, a.bench, a.output = a.model.resolve(), a.bench.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4_', 'PT_'))}
    env.update(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024')
    digest = hashlib.sha256(a.bench.read_bytes()).hexdigest()
    arms = [('ordinary', n) for n in [0, 1, 1, 0]] + [('favorable', n) for n in [0, 1]]
    results = []
    for i, (case, packed) in enumerate(arms, 1):
        out = a.output/f'{i:02d}-{case}-packed{packed}'
        out.mkdir()
        env['DS4_METAL_V41_PACKED_F16_GATHER'] = str(packed)
        cmd = [str(a.bench), '--model', str(a.model), '--out', str(out),
               '--rows', '0', '--tokens', '8192', '--chunk', '512', '--ctx', '16384',
               '--compressed-kv-f16', '1', '--case', case]
        save(out/'execution.json', {'command':cmd, 'base_revision':'a2c47483ffd2ffac58726da0f07da5df4cf9accd',
            'binary_sha256':digest, 'environment':{k:v for k,v in env.items() if k.startswith('DS4_')},
            'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        start = time.monotonic()
        with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=720).returncode
        result = {'directory':out.name, 'case':case, 'packed':packed,
                  'returncode':rc, 'seconds':time.monotonic()-start}
        results.append(result)
        save(a.output/'result.json', results)
        print(json.dumps(result), flush=True)
        if rc:
            raise RuntimeError(f'First failure retained in {out}')


if __name__ == '__main__':
    main()
