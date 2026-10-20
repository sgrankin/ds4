#!/usr/bin/env python3
"""One fixed-teacher ABBA, run under pressure_run.py. No profiling or retries."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / '.workspaces/upgrade-20261020/kernel-recheck')
parser.add_argument('--model', type=Path, default=ROOT / 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf')
a = parser.parse_args()
OUT = a.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
MODEL = a.model.resolve()
PROMPT = ROOT / 'speed-bench/promessi_sposi.txt'

def dump(p, x):
    p.write_text(json.dumps(x, indent=2) + '\n')

def compare(a, b):
    assert len(a) == len(b) and a and all(math.isfinite(x) for x in a+b)
    def lp(v):
        m = max(v)
        z = m + math.log(math.fsum(math.exp(x-m) for x in v))
        return [x-z for x in v]
    la, lb = lp(a), lp(b)
    ta = max(range(len(a)), key=a.__getitem__)
    tb = max(range(len(b)), key=b.__getitem__)
    return {'exact': a == b, 'kl_reference_to_candidate': math.fsum(math.exp(x)*(x-y) for x,y in zip(la,lb)),
            'max_abs_logit_difference': max(abs(x-y) for x,y in zip(a,b)), 'top1_reference': ta,
            'top1_candidate': tb, 'top1_agrees': ta == tb}

rows = []
reference = None
for i, variant in enumerate(('baseline','candidate','candidate','baseline'), 1):
    dest = OUT / f'{i}-{variant}'
    dest.mkdir()
    shader = BASE / ('baseline.metal' if variant == 'baseline' else 'moe.metal')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))}
    env.update(DS4_METAL_MOE_SOURCE=str(shader), DS4_METAL_V41_RESIDENT_ROUTE_GAP='0',
               DS4_METAL_V41_RESIDENT_PREFILL_GAP='0', DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')
    command = [str(ROOT/'ds4-bench'), '--metal', '-m', str(MODEL), '--prompt-file', str(PROMPT),
               '--ctx-start','2048','--ctx-max','2048','--ctx-alloc','4096','--gen-tokens','128',
               '--prefill-chunk','512','--ssd-streaming','--ssd-streaming-cache-experts','64GB',
               '--teacher-forced-decode','--csv',str(dest/'speed.csv'),
               '--dump-frontier-logits-dir',str(dest)]
    dump(dest/'execution.json', {'command':command,'cwd':str(ROOT),'env':{k:v for k,v in env.items() if k.startswith('DS4')},
                                'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),
                                'prompt_sha256':hashlib.sha256(PROMPT.read_bytes()).hexdigest()})
    start = time.monotonic()
    with (dest/'run.log').open('w') as log:
        rc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    row = {'arm':i,'variant':variant,'returncode':rc,'seconds':time.monotonic()-start}
    if rc == 0:
        row['speed'] = list(csv.DictReader((dest/'speed.csv').open()))
        logits = {p.name:json.loads(p.read_text())['logits'] for p in sorted(dest.glob('*.logits.json'))}
        assert logits
        if reference is None:
            reference = logits
        assert logits.keys() == reference.keys()
        row['frontiers'] = {k:compare(reference[k],v) for k,v in logits.items()}
    rows.append(row)
    dump(OUT/'result.json', rows)
    print(json.dumps(row), flush=True)
    if rc:
        raise SystemExit(rc)
