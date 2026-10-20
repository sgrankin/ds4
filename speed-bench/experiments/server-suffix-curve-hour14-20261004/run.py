#!/usr/bin/env python3
"""Nine pre-RoPE suffix points; ROOT launches this inside the MAIN pressure guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILIES = ['ordinary-deletion', 'best-self-contained', 'stress-changed-fact']
TAILS = [64, 128, 256]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--helper', type=Path, default=ROOT / '.workspaces/server-agent-day-20261004/curve-build/oracle')
    p.add_argument('--output', type=Path, default=HERE)
    args = p.parse_args()
    parent = subprocess.check_output(['ps', '-p', str(os.getppid()), '-o', 'command='], text=True)
    if 'speed-bench/ideas/pressure_run.py' not in parent:
        p.error('launch this loop as the direct child of the MAIN speed-bench/ideas/pressure_run.py')
    model = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
    # Model metadata is inspected only after the guard-parent check.
    st = model.stat()
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                             (ROOT / 'ds4_metal.m').read_text()))
    env.update({k:str(ROOT/v) for k,v in shaders.items()})
    env.update(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024',
               DS4_METAL_V41_RESIDENT_ROUTE_GAP='0')
    helper = args.helper.resolve()
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        for tail in TAILS:
            output = args.output / f'{family}-tail{tail}'
            output.mkdir(exist_ok=False)
            command = [str(helper), '--model', str(model), '--fixtures',
                       str(ROOT/'speed-bench/experiments/suffix-clm-hour1-20261003/fixtures'/family),
                       '--case', 'code', '--out', str(output), '--prefix', '1024',
                       '--source-edit', '256', '--edit', '128', '--unchanged', '512',
                       '--fresh-tail', str(tail), '--greedy', '16', '--ctx', '4096',
                       '--chunk', '512', '--cache-gb', '64', '--gold-answer', '3141',
                       '--stale-answer', '2718', '--require-fresh-query']
            record = {'argv':command, 'cwd':str(ROOT), 'family':family, 'fresh_tail':tail,
                      'settings':{k:v for k,v in env.items() if k.startswith('DS4')},
                      'helper_sha256':helper_sha, 'model_bytes':st.st_size,
                      'model_mtime_ns':st.st_mtime_ns, 'first_attempt':True,
                      'timing_scope':'Diagnostic core edit timing; previous-source setup/capture separate. No HTTP task timing.'}
            print('Running '+output.name, flush=True)
            start = time.monotonic()
            try:
                with (output/'helper.stdout').open('x') as stdout, (output/'helper.stderr').open('x') as stderr:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=300)
                record['returncode'] = completed.returncode
            except subprocess.TimeoutExpired:
                record['returncode'] = 124
            record['seconds'] = time.monotonic()-start
            (output/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
            if record['returncode']:
                print(json.dumps({'point':output.name, 'returncode':record['returncode']}), flush=True)
                return record['returncode']
            summary = json.loads((output/'summary.json').read_text())
            print(json.dumps({'point':output.name, 'seconds':record['seconds'],
                  'mean_delta_nll':summary['sum_delta_nll']/summary['rows'], 'mean_kl':summary['mean_kl'],
                  'fresh_answer':summary['results'][0]['greedy'], 'reuse_answer':summary['results'][1]['greedy'],
                  'fresh_correct':summary['results'][0]['correct_answer'], 'reuse_correct':summary['results'][1]['correct_answer'],
                  'edit_ms':[a['edit_ms'] for a in summary['results']]}), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
