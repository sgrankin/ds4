"""Freeze fresh tasks and golds before the first model attempt."""
import copy
import importlib.util
import json
from pathlib import Path
import random
from gold_oracles import interval_gold, dependency_gold

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / 'fresh-clm-coding-hour32-20261004/practical/spec.json'


def intervals(pairs):
    return [{'start': a, 'end': b} for a, b in pairs]


def tasks(rows):
    return [{'id': name, 'deps': deps} for name, deps in rows]


def main():
    sample_i = intervals([(8, 10), (0, 2), (3, 5), (1, 4), (12, 13)])
    fixed_i = [([], 0), ([(1, 2)], 0), ([(0, 2), (2, 4)], 0),
        ([(0, 2), (3, 5)], 0), ([(0, 2), (3, 5)], 1),
        ([(0, 2), (4, 6)], 1), ([(0, 10), (2, 3), (9, 12)], 0),
        ([(4, 5), (0, 2), (2, 4)], 0), ([(0, 1), (2, 3), (4, 5)], 1),
        ([(0, 1), (3, 4), (5, 6)], 1), ([(2, 5), (2, 5), (2, 5)], 0),
        ([(-10, -8), (-7, -4), (-5, 2)], 1), ([(0, 100), (20, 21), (3, 4)], 0),
        ([(10**18, 10**18 + 2), (10**18 + 3, 10**18 + 4)], 1),
        ([(7, 8), (3, 4), (1, 2)], 0), ([(0, 3), (5, 7), (9, 11)], 2)]
    inputs_i = [(sample_i, {'gap': 1})] + [(intervals(pairs), {'gap': gap}) for pairs, gap in fixed_i]
    rng = random.Random(20261008401)
    for case in range(144):
        data = []
        for i in range(rng.randrange(25)):
            start = rng.randrange(-20, 41)
            data.append({'start': start, 'end': start + rng.randrange(1, 13)})
        if case % 9 == 0 and data:
            data.append(dict(rng.choice(data)))
        rng.shuffle(data)
        inputs_i.append((data, {'gap': rng.choice([0, 0, 1, 2, 5, 10])}))
    sample_d = tasks([('build', ['fetch']), ('publish', ['test']), ('test', ['build']), ('fetch', []), ('docs', [])])
    fixed_d = [([], 'input'), ([('a', [])], 'lex'),
        ([('b', []), ('a', [])], 'input'), ([('b', []), ('a', [])], 'lex'),
        ([('c', ['b']), ('b', ['a']), ('a', [])], 'input'),
        ([('a', []), ('c', ['a']), ('b', ['a']), ('d', ['b', 'c'])], 'lex'),
        ([('a', ['a'])], 'input'), ([('a', ['b']), ('b', ['a'])], 'lex'),
        ([('c', ['a']), ('b', ['a']), ('a', ['b']), ('free', [])], 'input'),
        ([('a', ['missing']), ('b', ['a']), ('c', [])], 'lex'),
        ([('a', []), ('b', ['a', 'a', 'a'])], 'input'),
        ([('', []), ('é', ['']), ('é', ['']), ('A', []), ('a', [])], 'lex'),
        ([('z', ['unknown']), ('b', ['z']), ('a', ['b']), ('c', [])], 'input'),
        ([('d', ['b', 'c']), ('c', ['a']), ('b', ['a']), ('a', [])], 'input'),
        ([('a', []), ('c', ['a']), ('b', ['a']), ('x', ['lost'])], 'input'),
        ([('a', []), ('b', []), ('c', ['a', 'b']), ('d', ['c', 'a'])], 'lex')]
    inputs_d = [(sample_d, {'order': 'lex'})] + [(tasks(rows), {'order': order}) for rows, order in fixed_d]
    rng = random.Random(20261008402)
    for case in range(144):
        n = rng.randrange(20)
        names = ['task' + str(i) for i in range(n)]
        if n > 3 and case % 7 == 0:
            names[0:3] = ['', 'é', 'é']
        data = []
        for i, name in enumerate(names):
            pool = names[:i] if case % 3 == 0 else names
            deps = [rng.choice(pool) for j in range(rng.randrange(5))] if pool else []
            if case % 11 == 0 and i % 4 == 0:
                deps.append('missing-task')
            data.append({'id': name, 'deps': deps})
        rng.shuffle(data)
        inputs_d.append((data, {'order': rng.choice(['input', 'lex'])}))
    inputs = {'interval_groups': inputs_i, 'dependency_layers': inputs_d}
    golds = {'interval_groups': interval_gold, 'dependency_layers': dependency_gold}
    rows = {key: [{'payload': data, 'config': config, 'gold': golds[key](data, config)}
                  for data, config in cases] for key, cases in inputs.items()}
    assert all(len(value) == 161 for value in rows.values())
    prior = json.loads(PRIOR.read_text())
    contract = 'The project contains config.json' + prior['tasks'][0]['prompt'].split('The project contains config.json', 1)[1]
    prompts = {
        'interval_groups': ('Task: Interval groups with index provenance.\nCreate solution.py containing def solve(payload, config). '
            'payload is an arbitrary-order list of intervals with integer start/end and start < end; each denotes [start,end). '
            'config gap is a nonnegative integer. Sort conceptually by (start,end,original index). '
            'Start a group from the first interval. A following interval joins the current group if its start <= the current group maximum end + gap; '
            'joining updates the maximum end. Otherwise start a new group. Thus touching intervals DO join at gap0 despite half-open notation, '
            'and equality with the gap boundary joins. This is transitive grouping, not comparing only with the immediately previous interval end. '
            'Return {groups}, where groups is the list of {start,end,indices} in increasing group start order. '
            'start/end are the minimum start and maximum end of all original intervals in that group; indices are original zero-based payload positions '
            'sorted numerically within each group. Duplicate intervals keep their distinct indices. Empty input returns {groups:[]}. '
            'Use exact integer arithmetic, permit negative endpoints, and do not mutate payload or config.\n'),
        'dependency_layers': ('Task: Deterministic dependency layers.\nCreate solution.py containing def solve(payload, config). '
            'payload is an arbitrary-order list of task records {id,deps}. id values are unique strings, including possible empty/Unicode strings; '
            'deps is a list of exact string IDs and may contain duplicates, self-dependencies, cycles, or IDs absent from payload. '
            'config order is either input or lex. Start with no completed tasks. In EACH round select ALL remaining tasks whose every dependency '
            'was completed in an EARLIER round. Emit their IDs as one layer, then mark the entire layer complete. Never unlock another task within '
            'the same round. With order=input, list IDs in original payload order within each layer; with order=lex, use Python string ascending order. '
            'Stop when no tasks are ready. Return {layers,unresolved}; unresolved contains all remaining task IDs in the configured order. '
            'An absent dependency is never completed, so it blocks its task and dependents; cycles/self-dependencies also remain unresolved. '
            'Duplicate dependencies impose no extra work. Empty payload returns {layers:[],unresolved:[]}. '
            'Compare IDs exactly without normalization, and do not mutate payload or config.\n')}
    spec = {'task_order': list(inputs), 'tests_per_task': 161,
            'scope': 'Fresh interval-connectivity and dependency-layer families, fixed before any model attempt; prior task keys join/window_argmax/ledger/lru/dedupe/bucket reviewed.',
            'tasks': []}
    for key in inputs:
        sample, config = inputs[key][0]
        spec['tasks'].append({'key': key, 'prompt': prompts[key] + contract,
                             'system': prior['tasks'][0]['system'], 'config': dict(config, sample=sample)})
    (HERE / 'spec.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    (HERE / 'cases.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    assert interval_gold(sample_i, {'gap': 1}) == {'groups': [
        {'start': 0, 'end': 5, 'indices': [1, 2, 3]}, {'start': 8, 'end': 10, 'indices': [0]}, {'start': 12, 'end': 13, 'indices': [4]}]}
    assert dependency_gold(sample_d, {'order': 'lex'}) == {'layers': [['docs', 'fetch'], ['build'], ['test'], ['publish']], 'unresolved': []}
    assert dependency_gold(tasks([('a', ['x']), ('b', ['a']), ('c', [])]), {'order': 'input'}) == {'layers': [['c']], 'unresolved': ['a', 'b']}
    for key, cases in rows.items():
        module_spec = importlib.util.spec_from_file_location('reference_' + key, HERE / ('reference-' + key + '.py'))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for row in cases:
            payload, config = copy.deepcopy(row['payload']), copy.deepcopy(row['config'])
            assert module.solve(payload, config) == row['gold']
            assert payload == row['payload'] and config == row['config']
    result = {'status': 'GOLD_SANITY_PASS', 'cases': {key: len(cases) for key, cases in rows.items()},
              'independent_checks': 'interval all-pairs connectivity vs sorted sweep; recursive dependency depth vs simultaneous ready rounds',
              'hand_checks': 3, 'seeds': [20261008401, 20261008402], 'model_executed': False}
    (HERE / 'check-results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
