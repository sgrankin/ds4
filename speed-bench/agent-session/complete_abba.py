#!/usr/bin/env python3
"""Join the last matching control with an immediately following BBA continuation.

The shared control is one observation, not a new independent repeat. Call only
when runs were consecutive, with no intervening GPU workload or rebuild.
"""
import json
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(path.read_text())


def complete(previous, following, output):
    previous, following, output = map(Path, (previous, following, output))
    pm, fm = [read(p/'manifest.json') for p in (previous, following)]
    keys = ('replay_sha256', 'binary_sha256', 'model', 'routes_sha256', 'cache_gb')
    if any(pm.get(k) != fm.get(k) for k in keys):
        raise ValueError('incompatible executable/model/recording/cache identity')
    shaders = [read(p/'metal/sha256.json') for p in (previous, following)]
    if shaders[0] != shaders[1]:
        raise ValueError('different shaders')
    ignored = set(shaders[0])
    if pm.get('routes_sha256'):
        ignored.add('DS4_V41_ROUTE_ORACLE')
    normalize = lambda m: {k:v for k,v in m['env'].items() if k not in ignored}
    if normalize(pm) != normalize(fm):
        raise ValueError('different control environment')
    prior, tail = [read(p/'summary.json') for p in (previous, following)]
    if prior[-1]['variant'] != 'A' or [r['variant'] for r in tail] != list('BBA'):
        raise ValueError('requires last A followed by completed BBA')
    rows = [prior[-1]] + tail
    for key in ('logits_sha256', 'snapshot_sha256'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError('numerical mismatch')
    comparison = {}
    for variant in ('A', 'B'):
        group = [r for r in rows if r['variant'] == variant]
        comparison[variant] = {k:statistics.mean(r[k] for r in group) for k in
                              ('startup_prefill_ms', 'append_prefill_ms', 'decode_ms', 'model_ms')}
        comparison[variant]['turn_model_ms'] = sum(comparison[variant][k] for k in
                                                   ('append_prefill_ms', 'decode_ms'))
    comparison['candidate_change_percent'] = {k:100*(comparison['B'][k]/comparison['A'][k]-1)
                                              for k in ('append_prefill_ms', 'decode_ms', 'turn_model_ms')}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('summary', rows), ('comparison', comparison),
                        ('provenance', dict(previous=str(previous.resolve()),
                                            following=str(following.resolve()),
                                            shared_control_index=len(prior)-1,
                                            manifests=[pm, fm], shader_sha256=shaders[0]))]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    return comparison


if __name__ == '__main__':
    print(json.dumps(complete(*sys.argv[1:]), indent=2))
