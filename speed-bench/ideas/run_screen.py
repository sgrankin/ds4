#!/usr/bin/env python3
"""Freeze binaries/shaders; reuse baseline distributions for a quick gap sweep."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'speed-bench/experiments/ds41-ssd'))
from snapshot_shaders import snapshot_shaders


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def summarize(path):
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    groups = {}
    for row in rows:
        groups.setdefault(row['case'], []).append(row)
    case_delta = [statistics.mean(float(r['delta_raw_nll']) for r in rs)
                  for rs in groups.values()]
    rng = random.Random(20261002)
    boots = sorted(statistics.mean(rng.choices(case_delta, k=len(case_delta)))
                   for _ in range(2000))
    kl = sorted(float(r['raw_kl']) for r in rows)
    return dict(
        prompt_mean_delta_nll=statistics.mean(case_delta),
        prompt_bootstrap_95=[boots[49], boots[1949]],
        worst_prompt_delta_nll=max(case_delta),
        kl_p99=kl[min(len(kl)-1, int(len(kl)*.99))],
        kl_max=max(kl),
        per_prompt={name: dict(tokens=len(rs),
            delta_nll=statistics.mean(float(r['delta_raw_nll']) for r in rs),
            kl=statistics.mean(float(r['raw_kl']) for r in rs))
            for name, rs in groups.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model', type=Path,
                   default=ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf')
    p.add_argument('--cases', type=int, default=16)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--cache-gb', type=int, default=64)
    p.add_argument('--ctx', type=int, default=4096)
    p.add_argument('--chunk', type=int, default=512)
    p.add_argument('--gaps', default='0.02,0.05,0.10')
    p.add_argument('--mass', help='Optional cap on replaced original mixture mass')
    p.add_argument('--candidate-env', action='append', default=[],
                   help='Additional DS4_NAME=value applied only to candidate runs')
    p.add_argument('--reference', type=Path, help='Reuse an existing aligned baseline record')
    a = p.parse_args()
    candidate_env = dict(item.split('=', 1) for item in a.candidate_env)
    if any(not name.startswith('DS4_') for name in candidate_env):
        p.error('candidate environment names must start with DS4_')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    baseline, candidate = a.output/'baseline', a.output/'candidate'
    shutil.copy2(a.baseline, baseline); shutil.copy2(a.candidate, candidate)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('DS4_'):
            del env[name]
    env.update(snapshot_shaders(a.output))
    model = a.model.resolve()
    with model.open('rb') as f:
        header_sha = hashlib.sha256(f.read(65536)).hexdigest()
    provenance = dict(baseline_sha256=sha(baseline), candidate_sha256=sha(candidate),
        model_path=str(model), model_size=model.stat().st_size,
        model_mtime_ns=model.stat().st_mtime_ns,
        model_header_sha256=header_sha, manifest_sha256=sha(a.manifest),
        settings=dict(cases=a.cases, tokens=a.tokens, ctx=a.ctx,
                      cache_gb=a.cache_gb, chunk=a.chunk, gaps=a.gaps, mass=a.mass,
                      candidate_env=candidate_env),
        baseline_source=str(a.baseline.resolve()),
        candidate_source=str(a.candidate.resolve()),
        timing_scope='Inference calls exclude logit copying, KL arithmetic and reference I/O; screening timings are exploratory.')
    inputs = []
    for line in a.manifest.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split('\t')
        inputs.append(dict(case=fields[0],
            prompt_path=fields[1], prompt_sha256=sha(fields[1]),
            continuation_path=fields[2], continuation_sha256=sha(fields[2])))
        if len(inputs) == a.cases:
            break
    provenance['inputs'] = inputs
    (a.output/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    ref = a.output/'reference.bin'
    if a.reference:
        shutil.copy2(a.reference, ref)
        provenance['reused_reference_sha256'] = sha(ref)
        (a.output/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    results = []
    variants = ([] if a.reference else [('baseline', None, True)]) + [('off', '0', False)]
    variants += [('gap-'+gap, gap, False) for gap in a.gaps.split(',')]
    variants += [('baseline-repeat', None, False)]
    for name, gap, record in variants:
        exe = baseline if gap is None else candidate
        runenv = env.copy()
        if gap is not None:
            if name != 'off':
                runenv.update(candidate_env)
            runenv['DS4_METAL_V41_RESIDENT_ROUTE_GAP'] = gap
            if a.mass is not None:
                runenv['DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS'] = a.mass
        args = [str(exe), str(model), str(a.manifest.resolve()),
                str(a.output/(name+'.csv')), str(ref), 'record' if record else 'compare',
                '--cases', str(a.cases), '--tokens', str(a.tokens),
                '--ctx', str(a.ctx), '--cache-gb', str(a.cache_gb), '--chunk', str(a.chunk)]
        with (a.output/(name+'.json')).open('w') as out, (a.output/(name+'.log')).open('w') as err:
            subprocess.run(args, cwd=ROOT, env=runenv, stdout=out, stderr=err, check=True)
        result = json.loads((a.output/(name+'.json')).read_text())
        result.update(name=name, gap=gap, **summarize(a.output/(name+'.csv')))
        results.append(result)
        (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'per_prompt'}), flush=True)
    if any(r['bit_equal_rows'] != r['tokens']
           for r in results if r['name'] in ('off', 'baseline-repeat')):
        raise RuntimeError('mode-off or repeated baseline does not reproduce reference exactly')


if __name__ == '__main__':
    main()
