#!/usr/bin/env python3
"""Run the existing relocation probe across source/destination storage layouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]

def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.model, a.binary, a.output = a.model.resolve(), a.binary.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    env.update(DS4_V41_DECODER_REPLAY_ROWS='0', DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',
        DS4_METAL_V41_RESIDENT_PREFILL_GAP='0', DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0',
        DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024')
    arms = [('01-f32', 0, 0, 1), ('02-f16', 1, 1, 1), ('03-f32-to-f16', 0, 1, 1),
            ('04-f16-to-f32', 1, 0, 1), ('05-f32', 0, 0, 1), ('06-legacy-f16', 1, 1, 0)]
    results = []
    for name, source, destination, prerope in arms:
        out = a.output/name
        out.mkdir()
        command = [str(a.binary), '--model', str(a.model), '--out', str(out),
            '--source-f16', str(source), '--destination-f16', str(destination),
            '--moves', '4' if prerope else '1', '--prerope', str(prerope),
            '--ctx', '8192', '--chunk', '512', '--cache-gb', '64']
        save(out/'execution.json', {'command': command, 'cwd': str(ROOT),
            'env': {k:v for k,v in env.items() if k.startswith('DS4')},
            'base_revision':'2436f414f083b7b04a9b0dedec4788e6f6bd62d0',
            'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest()})
        with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        results.append({'arm': name, 'returncode': rc, 'source_f16': source,
                        'destination_f16': destination, 'prerope': prerope})
        save(a.output/'result.json', results)
        print(json.dumps(results[-1]), flush=True)
        if rc: return rc
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
