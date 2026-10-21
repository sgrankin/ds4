#!/usr/bin/env python3
"""Compare fixed teacher logits across layouts, separately from stale suffix error."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

VOCAB = 129280


def read(path):
    data = path.read_bytes()
    magic, vocab, moves, steps = struct.unpack_from('<4I', data)
    assert magic == 0x31525053 and vocab == VOCAB and steps == 4
    targets = list(struct.unpack_from('<4i', data, 16))
    data = data[32:]
    size = VOCAB*4
    assert len(data) == moves*2*steps*size
    rows = [data[i:i+size] for i in range(0,len(data),size)]
    return targets, moves, [[rows[m*8+k*4+s] for m in range(moves) for s in range(4)] for k in range(2)]


def compare(p, q, targets):
    assert len(p) == len(q) == len(targets)
    dnlls, kls, flips = [], [], 0
    equal = p == q
    for pb, qb, target in zip(p,q,targets):
        if pb == qb:
            dnlls.append(0.0); kls.append(0.0)
            continue
        pv, qv = struct.unpack(f'<{VOCAB}f', pb), struct.unpack(f'<{VOCAB}f', qb)
        pm, qm = max(pv), max(qv)
        pz = pm+math.log(math.fsum(math.exp(x-pm) for x in pv))
        qz = qm+math.log(math.fsum(math.exp(x-qm) for x in qv))
        dnlls.append(qz-qv[target]-pz+pv[target])
        kls.append(math.fsum(math.exp(x-pz)*(x-pz-y+qz) for x,y in zip(pv,qv)))
        flips += pv.index(pm) != qv.index(qm)
    return {'byte_equal':equal, 'rows':len(p),'mean_delta_nll':sum(dnlls)/len(p),
            'mean_kl':sum(kls)/len(p),'max_kl':max(kls),'argmax_flips':flips}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    results = json.loads((a.input/'result.json').read_text())
    assert len(results) == 6 and all(x['returncode']==0 for x in results)
    target, _, baseline = read(a.input/'01-f32/logits.bin')
    summary = {'scope':'fixed teacher histories; layout effect separate from existing stale suffix approximation',
               'layout_comparisons':{}, 'stale_suffix_metrics':{}, 'timing':{}}
    for result in results:
        name = result['arm']
        ids, moves, rows = read(a.input/name/'logits.bin')
        assert ids == target
        if moves == 4:
            summary['layout_comparisons'][name] = {
                'native':compare(baseline[0],rows[0],target*4),
                'grafted':compare(baseline[1],rows[1],target*4),
                'raw_logits_sha256':hashlib.sha256((a.input/name/'logits.bin').read_bytes()).hexdigest()}
        with (a.input/name/'rows.csv').open() as f:
            metrics = list(csv.DictReader(f))
        summary['stale_suffix_metrics'][name] = {
            'rows':len(metrics),'mean_delta_nll':sum(float(r['delta_nll']) for r in metrics)/len(metrics),
            'mean_kl':sum(float(r['kl']) for r in metrics)/len(metrics),
            'max_kl':max(float(r['kl']) for r in metrics),
            'argmax_flips':sum(int(r['argmax_flip']) for r in metrics)}
        with (a.input/name/'moves.csv').open() as f:
            timing = list(csv.DictReader(f))
        summary['timing'][name] = {k:sum(float(r['total_edit_ms']) for r in timing if r['arm']==k)/1000
                                    for k in ['native','reuse']}
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
