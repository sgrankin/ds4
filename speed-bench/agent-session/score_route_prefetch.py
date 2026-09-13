#!/usr/bin/env python3
"""Offline cache-aware prediction screen, with native cache evolution held fixed.

No simulated eviction, contention, or deadlines: optimistic coverage, not a
runtime speed estimate. Thresholds use uncalibrated 6*softmax selection scores.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from train_route_predictor import dataset, inputs, digest, accuracy


def cache_data(path, labels):
    header = np.fromfile(path, dtype='<u4', count=4)
    if header.tolist() != [0x44534331, 1, 40, 384]: raise ValueError('invalid cache header')
    dt = np.dtype([('meta', '<u4', (3,)), ('resident', 'u1', (384,))])
    if Path(path).stat().st_size != 16 + len(labels)*dt.itemsize:
        raise ValueError('cache/route length mismatch')
    data = np.memmap(path, dtype=dt, mode='r', offset=16, shape=(len(labels),))
    if not np.array_equal(data['meta'], labels[:, :3]): raise ValueError('cache/route alignment mismatch')
    if np.any(data['resident'] > 1): raise ValueError('invalid residency byte')
    return data


def score(pred, allow, actual, resident):
    rows = np.arange(len(actual))[:, None]
    missing = ~resident[rows, actual].astype(bool)
    issued = allow & ~resident[rows, pred].astype(bool)
    matched = (pred[:, :, None] == actual[:, None, :]) & issued[:, :, None]
    covered = matched.any(axis=1) & missing
    demand = int(missing.sum()); reads = int(issued.sum()); useful = int(covered.sum())
    per_row = missing.sum(axis=1)
    return dict(rows=len(actual), demand_misses=demand, predicted_reads=reads,
        useful_reads=useful, unnecessary_reads=reads-useful,
        miss_recall=useful/demand if demand else None,
        read_precision=useful/reads if reads else None,
        all_demand_misses_covered=float((covered.sum(axis=1)[per_row>0] == per_row[per_row>0]).mean()) if demand else None,
        reads_on_native_all_hit_layers=int(issued[per_row == 0].sum()),
        extra_reads_over_native_fraction=(reads-useful)/demand if demand else None,
        limits='Native residency frozen; no eviction, I/O contention, predictor overhead or readiness deadline.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'routes', 'cache', 'weights', 'output'): p.add_argument(name, type=Path)
    p.add_argument('--split', choices=['test','validation','all'], default='test')
    a = p.parse_args()
    d, labels = dataset(a.features, a.routes)
    cache = cache_data(a.cache, labels)
    n = len(d)//40
    lo, hi = {'test':(int(n*.8)*40,len(d)), 'validation':(int(n*.6)*40,int(n*.8)*40), 'all':(0,len(d))}[a.split]
    import mlx.core as mx
    w = mx.load(str(a.weights))
    if set(w) != {'stem.weight','stem.bias','heads','bias'}: raise ValueError('unexpected weights')
    pred, confidence = [], []
    for start in range(lo, hi, 512):
        ix = np.arange(start, min(start+512,hi))
        z = mx.tanh(mx.array(inputs(d, ix)) @ w['stem.weight'].T + w['stem.bias'])
        logits = mx.einsum('bi,bij->bj', z, w['heads'][mx.array(d['meta'][ix,2])]) + w['bias'][mx.array(d['meta'][ix,2])]
        probs = np.array(mx.softmax(logits, axis=-1))
        top = np.argsort(-probs, axis=1, kind='stable')[:, :12]
        pred.append(top); confidence.append(6*np.take_along_axis(probs, top, axis=1))
    pred = np.concatenate(pred); confidence = np.concatenate(confidence)
    actual = labels[lo:hi,3:]; resident = cache['resident'][lo:hi]
    result = dict(split=a.split, rows=hi-lo, row_range=[lo,hi],
        features_sha256=digest(a.features), routes_sha256=digest(a.routes),
        cache_sha256=digest(a.cache), weights_sha256=digest(a.weights),
        script_sha256=digest(__file__), all_route_top6=accuracy(pred[:,:6], actual),
        early_top6=score(d['early'][lo:hi], np.ones((hi-lo,6),bool), actual, resident), policies={})
    for k in (6,12):
        for threshold in (0,.25,.5,.75,.9):
            result['policies'][f'top{k}_score{threshold}'] = score(pred[:,:k], confidence[:,:k]>=threshold, actual, resident)
    with a.output.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)

if __name__ == '__main__': main()
