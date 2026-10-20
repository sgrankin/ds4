#!/usr/bin/env python3
"""Compare diagnostic F32 stage traces; exit 1 on any non-identical vector."""
import argparse
import array
import json
import math
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('control', type=Path)
p.add_argument('candidate', type=Path)
a = p.parse_args()
failed = False
for source in sorted(a.control.glob('*.bin')):
    other = a.candidate / source.name
    if not other.exists():
        print(json.dumps({'stage': source.name, 'missing': True})); failed = True; continue
    x, y = array.array('f'), array.array('f')
    xb, yb = source.read_bytes(), other.read_bytes()
    if len(xb) != len(yb):
        print(json.dumps({'stage': source.name, 'size_mismatch': True})); failed = True; continue
    x.frombytes(xb); y.frombytes(yb)
    delta = [u-v for u,v in zip(x,y)]
    different = sum(u != v for u,v in zip(x,y))
    failed |= xb != yb
    print(json.dumps({'stage': source.name, 'elements': len(x), 'different': different,
                      'bit_identical': xb == yb,
                      'max_abs': max(map(abs,delta), default=0),
                      'rmse': math.sqrt(sum(d*d for d in delta)/len(delta)) if delta else 0,
                      'first': next(((i,u,v) for i,(u,v) in enumerate(zip(x,y)) if u != v), None)}))
raise SystemExit(int(failed))
