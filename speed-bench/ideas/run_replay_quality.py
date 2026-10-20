#!/usr/bin/env python3
"""Compare full distributions along a frozen, evolving agent tool trajectory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from run_screen import ROOT, sha, snapshot_shaders, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--replay', type=Path, default=ROOT/'speed-bench/ideas/latency-session.txt')
    p.add_argument('--model', type=Path, default=ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf')
    p.add_argument('--gap', default='0.05')
    p.add_argument('--mass', default='0.12')
    p.add_argument('--candidate-env', action='append', default=[])
    p.add_argument('--reference', type=Path)
    a = p.parse_args()
    extra = dict(item.split('=', 1) for item in a.candidate_env)
    if any(not name.startswith('DS4_') for name in extra):
        p.error('candidate environment names must start with DS4_')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    baseline, candidate = a.output/'baseline', a.output/'candidate'
    replay = a.output/'session.txt'
    shutil.copy2(a.baseline, baseline)
    shutil.copy2(a.candidate, candidate)
    shutil.copy2(a.replay, replay)
    env = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
    env.update(snapshot_shaders(a.output))
    ref = a.output/'reference.bin'
    if a.reference:
        shutil.copy2(a.reference, ref)
    provenance = dict(model=str(a.model.resolve()), baseline_sha256=sha(baseline),
        candidate_sha256=sha(candidate), replay_sha256=sha(replay),
        candidate_env=extra, gap=a.gap, mass=a.mass,
        settings=dict(ctx=8192, cache_gb=64, prefill_chunk=512),
        reused_reference_sha256=sha(ref) if a.reference else None,
        timing_scope='Instrumented quality screen; use separate replay for latency claims.')
    (a.output/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    variants = ([] if a.reference else [('baseline', None, True)]) + [
        ('off', '0', False), ('candidate', a.gap, False), ('baseline-repeat', None, False)]
    results = []
    for name, gap, record in variants:
        runenv = env.copy()
        if gap is not None:
            if name != 'off':
                runenv.update(extra)
            runenv.update(DS4_METAL_V41_RESIDENT_ROUTE_GAP=gap,
                          DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS=a.mass)
        args = [str(baseline if gap is None else candidate), str(a.model.resolve()),
                str(replay), str(a.output/(name+'.csv')), str(ref),
                'record' if record else 'compare']
        with (a.output/(name+'.json')).open('w') as out, (a.output/(name+'.log')).open('w') as err:
            subprocess.run(args, cwd=ROOT, env=runenv, stdout=out, stderr=err, check=True)
        result = json.loads((a.output/(name+'.json')).read_text())
        result.update(name=name, **summarize(a.output/(name+'.csv')))
        results.append(result)
        (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='per_prompt'}), flush=True)
    if any(r['tokens'] != r['bit_equal_rows'] for r in results
           if r['name'] in ('off', 'baseline-repeat')):
        raise RuntimeError('disabled or repeated baseline differs from reference')


if __name__ == '__main__':
    main()
