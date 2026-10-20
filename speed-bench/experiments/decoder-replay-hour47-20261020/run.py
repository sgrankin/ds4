#!/usr/bin/env python3
"""Sequential replay curve; run under the existing pressure_run.py guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bench', type=Path, default=ROOT/'speed-bench/decoder_replay_bench')
    p.add_argument('--order', default='0,128,256,0')
    p.add_argument('--case', choices=['all','ordinary','favorable','changed'], default='all')
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--timeout', type=int, default=1500)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--selected', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4_', 'PT_'))}
    overrides = ({'DS4_METAL_GRAPH_PREFILL_PROFILE':'1',
                  'DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE':'1'} if a.profile else {})
    if a.selected:
        overrides['DS4_V41_DECODER_REPLAY_SELECTED']='1'
    env.update(overrides)
    results = []
    for index, rows in enumerate(map(int, a.order.split(',')), 1):
        out = a.output/f'{index:02d}-rows{rows}'
        out.mkdir()
        cmd = [str(a.bench.resolve()), '--model', str(Path(a.model).resolve()),
               '--out', str(out.resolve()), '--rows', str(rows), '--tokens', str(a.tokens),
               '--chunk', str(a.chunk), '--case', a.case]
        save(out/'execution.json', {'command':cmd,
            'environment':'Inherited except DS4_/PT_ variables cleared; bench sets replay rows',
            'profile_overrides':overrides,
            'binary_sha256':hashlib.sha256(a.bench.read_bytes()).hexdigest(),
            'base_revision':subprocess.check_output(['jj','--ignore-working-copy','log',
                '-r','@-','--no-graph','-T','commit_id'], cwd=ROOT, text=True).strip(),
            'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        started = time.monotonic()
        with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            try:
                code = proc.wait(timeout=a.timeout)
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                raise
        row = {'rows':rows, 'directory':out.name, 'seconds':time.monotonic()-started,
               'returncode':code}
        results.append(row)
        save(a.output/'result.json', results)
        print(json.dumps(row), flush=True)
        if code: raise RuntimeError(f'first failure retained in {out}')

if __name__ == '__main__': main()
