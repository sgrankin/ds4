#!/usr/bin/env python3
"""Train a small pre-attention route predictor; accuracy study, not runtime speed.

Requires numpy and MLX. Never run alongside inference benchmarks. Tokens, not
individual layer rows, are split chronologically 60/20/20. Validation chooses
an epoch; test is evaluated once afterward. This split does NOT establish
cross-task generalization. Feature capture is intrusive and contains no cache
miss/deadline labels, so recall alone cannot justify speculative SSD reads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024): h.update(b)
    return h.hexdigest()


def dataset(features, routes):
    header = np.fromfile(features, dtype='<u4', count=8)
    if header.tolist() != [0x44534631, 1, 40, 5120, 384, 6, 2, 0]:
        raise ValueError('invalid feature header')
    rh = np.fromfile(routes, dtype='<u4', count=4)
    if rh.tolist() != [0x44535231, 40, 384, 6]:
        raise ValueError('invalid route header')
    dt = np.dtype([('meta', '<u4', (3,)), ('x', '<u2', (5120,)), ('early', '<i4', (6,))])
    size = Path(features).stat().st_size - 32
    if size <= 0 or size % (dt.itemsize * 40):
        raise ValueError('incomplete feature token')
    n = size // dt.itemsize
    if Path(routes).stat().st_size != 16 + n * 36:
        raise ValueError('route/feature row count mismatch')
    data = np.memmap(features, dtype=dt, mode='r', offset=32, shape=(n,))
    labels = np.memmap(routes, dtype='<u4', mode='r', offset=16, shape=(n, 9))
    if not np.array_equal(data['meta'], labels[:, :3]):
        raise ValueError('route/feature alignment mismatch')
    meta = data['meta'].reshape(-1, 40, 3)
    if not np.all(meta[:, :, 2] == np.arange(40)) or not np.all(meta[:, :, :2] == meta[:, :1, :2]):
        raise ValueError('invalid layer or token ordering')
    for ids in (labels[:, 3:], data['early']):
        if np.any(ids < 0) or np.any(ids >= 384) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError('invalid expert IDs')
    return data, labels


def inputs(data, indices):
    x = (data['x'][indices].astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(x).all(): raise ValueError('nonfinite features')
    return x


def accuracy(pred, actual):
    hits = (pred[:, :, None] == actual[:, None, :]).any(axis=1).sum(axis=1)
    return dict(selected_recall=float(hits.mean() / 6),
                all_six_coverage=float((hits == 6).mean()), rows=len(hits))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('features', type=Path); p.add_argument('routes', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--holdout-features', type=Path)
    p.add_argument('--holdout-routes', type=Path)
    a = p.parse_args()
    if a.epochs < 1 or a.width < 1 or a.batch < 1: p.error('positive training dimensions required')
    if bool(a.holdout_features) != bool(a.holdout_routes): p.error('both holdout paths required')
    data, labels = dataset(a.features, a.routes)
    tokens = len(data) // 40
    cut1, cut2 = int(tokens * .6) * 40, int(tokens * .8) * 40
    if cut1 < 40 or cut2 == cut1 or cut2 == len(data): raise ValueError('too few tokens')
    a.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(5120, a.width)
            self.heads = mx.random.normal((40, a.width, 384)) * (a.width ** -.5)
            self.bias = mx.zeros((40, 384))
        def __call__(self, x, layer):
            z = mx.tanh(self.stem(x))
            return mx.einsum('bi,bij->bj', z, self.heads[layer]) + self.bias[layer]

    model = Predictor()
    optimizer = optim.Adam(learning_rate=.001)
    def loss_fn(model, x, layer, target):
        scores = model(x, layer)
        logprob = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        return -mx.mean(mx.take_along_axis(logprob, target, axis=1))
    value_grad = nn.value_and_grad(model, loss_fn)

    def evaluate(d, lab, start, stop):
        predicted = []
        for i in range(start, stop, a.batch):
            ix = np.arange(i, min(i + a.batch, stop))
            scores = np.array(model(mx.array(inputs(d, ix)), mx.array(d['meta'][ix, 2])))
            predicted.append(np.argsort(-scores, axis=1, kind='stable')[:, :12])
        pred = np.concatenate(predicted)
        actual = lab[start:stop, 3:]
        result = {f'top{k}': accuracy(pred[:, :k], actual) for k in (6, 12)}
        result['early_top6'] = accuracy(d['early'][start:stop], actual)
        result['layer_top6'] = [accuracy(pred[np.arange(start,stop)%40 == il, :6], actual[np.arange(start,stop)%40 == il]) for il in range(40)]
        return result

    history, best, best_epoch = [], -1, None
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(cut1)
        total = 0.
        for off in range(0, cut1, a.batch):
            ix = order[off:off+a.batch]
            loss, grads = value_grad(model, mx.array(inputs(data, ix)),
                                    mx.array(data['meta'][ix, 2]), mx.array(labels[ix, 3:]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total += float(loss) * len(ix)
        val = evaluate(data, labels, cut1, cut2)
        score = val['top6']['selected_recall']
        row = dict(epoch=epoch, train_loss=total/cut1, validation=val)
        history.append(row)
        print(json.dumps(dict(epoch=epoch, train_loss=total/cut1, validation_recall=score)), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            model.save_weights(str(a.output/'best.safetensors'))
    model.load_weights(str(a.output/'best.safetensors'))
    result = dict(architecture=f'5120->{a.width} tanh shared stem, 40x{a.width}->384 heads',
        parameters=5120*a.width+a.width+40*a.width*384+40*384,
        seed=a.seed, epochs=a.epochs, batch=a.batch, best_epoch=best_epoch,
        split_tokens=[cut1//40,(cut2-cut1)//40,(len(data)-cut2)//40],
        features_sha256=digest(a.features), routes_sha256=digest(a.routes),
        weights_sha256=digest(a.output/'best.safetensors'),
        test=evaluate(data, labels, cut2, len(data)), history=history,
        limits='Chronological within-session test; no miss/deadline labels or deployment timings.')
    if a.holdout_features:
        hd, hl = dataset(a.holdout_features, a.holdout_routes)
        result['independent_holdout'] = evaluate(hd, hl, 0, len(hd))
        result['holdout_features_sha256'] = digest(a.holdout_features)
        result['holdout_routes_sha256'] = digest(a.holdout_routes)
    (a.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(best_epoch=best_epoch, test=result['test']['top6'], early=result['test']['early_top6'])), flush=True)

if __name__ == '__main__': main()
