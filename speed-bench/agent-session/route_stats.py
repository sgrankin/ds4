#!/usr/bin/env python3
"""Inspect local oracle routes and screen cheap prediction baselines offline.

Chronological holdout within ONE recording is exploratory, not evidence of
cross-task generalization. These are all-route metrics, not miss/deadline metrics.
"""
from array import array
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys


def route_bytes(path):
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if Path(path).suffix == '.gz' else raw


def read_routes(path):
    raw = route_bytes(path)
    if len(raw) % 4:
        raise ValueError('truncated word')
    data = array('I')
    if data.itemsize != 4:
        raise ValueError('requires native uint32')
    data.frombytes(raw)
    if len(data) < 4:
        raise ValueError('missing header')
    magic, layers, experts, used = data[:4]
    if magic != 0x44535231 or not (1 <= layers <= 79 and 1 <= used <= experts <= 1024):
        raise ValueError('invalid header')
    stride = 3 + used
    if len(data) == 4 or (len(data) - 4) % (stride * layers):
        raise ValueError('incomplete scalar token')
    tokens = []
    for base in range(4, len(data), stride * layers):
        pos, token = data[base:base+2]
        routes = []
        for layer in range(layers):
            off = base + layer * stride
            if list(data[off:off+3]) != [pos, token, layer]:
                raise ValueError('position/token/layer mismatch')
            ids = list(data[off+3:off+stride])
            if len(set(ids)) != used or any(e >= experts for e in ids):
                raise ValueError('invalid expert IDs')
            routes.append(ids)
        tokens.append((pos, token, routes))
    return layers, experts, used, tokens


def summarize(path):
    layers, experts, used, tokens = read_routes(path)
    split = int(len(tokens) * .6)
    if split < 2 or len(tokens) - split < 2:
        raise ValueError('too few tokens for split')
    hot = [Counter() for _ in range(layers)]
    transitions = [defaultdict(Counter) for _ in range(layers)]
    for _, _, routes in tokens[:split]:
        for layer, ids in enumerate(routes):
            hot[layer].update(ids)
            if layer:
                for source in routes[layer-1]:
                    transitions[layer][source].update(ids)
    metrics = {}
    for budget in (used, used * 2):
        scores = defaultdict(list)
        for i in range(split, len(tokens)):
            pos, _, routes = tokens[i]
            for layer, actual in enumerate(routes):
                top = lambda counts: sorted(range(experts), key=lambda e: (-counts.get(e, 0), e))[:budget]
                predictions = {'frequency': top(hot[layer])}
                if tokens[i-1][0] + 1 == pos:
                    # This baseline only supplies the previous six, even for K=12.
                    predictions['previous_token'] = tokens[i-1][2][layer]
                if layer:
                    votes = Counter()
                    for source in routes[layer-1]:
                        counts = transitions[layer][source]
                        total = sum(counts.values())
                        if total:
                            for dest, count in counts.items():
                                votes[dest] += count / total
                    predictions['previous_layer_transition'] = top(votes)
                for name, predicted in predictions.items():
                    scores[name].append(len(set(actual) & set(predicted)))
        metrics[str(budget)] = {
            name: dict(layer_events=len(values), recall=sum(values)/len(values)/used,
                       all_selected_fraction=sum(v == used for v in values)/len(values))
            for name, values in scores.items()
        }
    return dict(routes_sha256=hashlib.sha256(route_bytes(path)).hexdigest(),
                layers=layers, experts=experts, selected=used, tokens=len(tokens),
                train_tokens=split, test_tokens=len(tokens)-split,
                warning='Within-session chronological screen; not held-out tasks, cache misses, deadlines or speed.',
                prediction_budget=metrics)


if __name__ == '__main__':
    print(json.dumps(summarize(sys.argv[1]), indent=2))
