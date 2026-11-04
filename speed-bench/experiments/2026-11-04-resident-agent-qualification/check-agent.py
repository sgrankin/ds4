"""Check the two generated tasks independently; no model execution or timing."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys


def interval_cases():
    rng = random.Random(0)
    cases = [[], [(1, 2), (3, 4)], [(3, 4), (1, 2)], [(1, 10), (2, 3)],
             [(-3, -1), (0, 2)], [(4, 4), (4, 4)], [[5, 7], [1, 2]]]
    for _ in range(200):
        cases.append([sorted((rng.randrange(-20, 21), rng.randrange(-20, 21)))
                      for _ in range(rng.randrange(12))])
    return cases


def interval_oracle(ranges):
    points = sorted({v for a, b in ranges for v in range(a, b + 1)})
    groups = itertools.groupby(enumerate(points), lambda p: p[1] - p[0])
    return [(values[0], values[-1]) for _, group in groups
            for values in [[v for _, v in group]]]


def layer_cases():
    cases = [([], []), ([4], []), ([7, -2, 0], []),
             ([3, 1, 2], [[1, 3], [1, 3], [2, 3]]),
             ([9, -3, 4, 1], [(-3, 1), (1, 4), (4, 9), (-3, 9)]),
             ([1, 2], [(1, 2), (2, 1)]),
             ([0, 1, 2, 3], [(0, 1), (2, 3), (3, 2)]), ([5], [(5, 5)])]
    rng = random.Random(1)
    for _ in range(40):
        nodes = rng.sample(range(-8, 9), rng.randrange(8))
        edges = [[u, v] for i, u in enumerate(nodes) for v in nodes[i + 1:]
                 if rng.random() < .25]
        if edges:
            edges.append(edges[0][:])
        cases.append((nodes, edges))
    for _ in range(10):
        nodes = rng.sample(range(-8, 9), rng.randrange(2, 8))
        cases.append((nodes, list(zip(nodes, nodes[1:] + nodes[:1]))))
    return cases


def layer_oracle(nodes, edges):
    # Independent of indegree-removal implementations: compute predecessor
    # depths recursively, detecting cycles even in disconnected components.
    parents = {node: set() for node in nodes}
    for u, v in edges:
        parents[v].add(u)
    depths, visiting = {}, set()

    def depth(node):
        if node in visiting:
            raise ValueError('cycle')
        if node not in depths:
            visiting.add(node)
            depths[node] = 1 + max((depth(p) for p in parents[node]), default=-1)
            visiting.remove(node)
        return depths[node]

    try:
        for node in nodes:
            depth(node)
    except ValueError:
        return None
    return [[node for node in sorted(nodes) if depths[node] == level]
            for level in range(1 + max(depths.values(), default=-1))]


def check_task(module, task):
    cases = interval_cases() if task == 'intervals' else layer_cases()
    failures = []
    for number, case in enumerate(cases):
        expected = interval_oracle(case) if task == 'intervals' else layer_oracle(*case)
        for form in ['list', 'iterator', 'generator']:
            inputs = copy.deepcopy([case] if task == 'intervals' else list(case))
            before = copy.deepcopy(inputs)
            arguments = inputs if form == 'list' else [iter(x) for x in inputs]
            if form == 'generator':
                arguments = [(item for item in x) for x in inputs]
            try:
                function = module.merge_ranges if task == 'intervals' else module.topological_layers
                raised_cycle = False
                try:
                    actual = function(*arguments)
                except ValueError:
                    if expected is not None:
                        raise
                    raised_cycle = True
                    actual = None
                assert inputs == before, 'mutated input'
                assert actual == expected, f'expected {expected!r}, got {actual!r}'
                if expected is not None:
                    item_type = tuple if task == 'intervals' else list
                    assert type(actual) is list and all(type(x) is item_type for x in actual), 'wrong container types'
                    assert all(type(v) is int for row in actual for v in row), 'non-integer output'
                else:
                    assert raised_cycle, 'cycle did not raise ValueError'
            except Exception as error:
                failures.append({'case': number, 'form': form, 'input': case,
                                 'error': f'{type(error).__name__}: {error}'})
    return {'cases': len(cases), 'input_forms': ['list', 'iterator', 'generator'],
            'evaluations': 3 * len(cases), 'failed_evaluations': len(failures), 'failures': failures}


def main(root):
    results = {}
    for arm in ['baseline', 'candidate']:
        results[arm] = {}
        for task in ['intervals', 'layers']:
            dest = root / arm / task
            try:
                name = f'{arm}_{task}'
                spec = importlib.util.spec_from_file_location(name, dest / f'{task}.py')
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                result = check_task(module, task)
                result['agent_exit_code'] = json.loads((dest / 'exit.json').read_text())['exit_code']
            except Exception as error:
                result = {'error': f'{type(error).__name__}: {error}'}
            results[arm][task] = result
    (root / 'oracle-result.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return int(any(r.get('error') or r.get('failed_evaluations') or r.get('agent_exit_code')
                   for tasks in results.values() for r in tasks.values()))


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
