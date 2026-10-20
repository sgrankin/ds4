"""Private fixed inputs and two independent gold methods; never place in project."""
import copy, json, random
from pathlib import Path

KEYS = ['', 'a', 'b', 'é', 'e\u0301', '🌿', '東京', 'missing']
VALUES = [None, False, 0, -7, '', '🌿', [1, None], {'nested': False}]

def reference(key, payload, config):
    if key == 'join':
        buckets = {}
        for row in payload['right']:
            buckets.setdefault(row['key'], []).append(row['value'])
        out = []
        for row in payload['left']:
            matches = buckets.get(row['key'], [])
            if matches:
                out.extend({'key': row['key'], 'left': row['value'], 'right': value} for value in matches)
            elif config['include_unmatched']:
                out.append({'key': row['key'], 'left': row['value'], 'right': None})
        return out
    if key == 'window_argmax':
        width = config['window']
        if width < 1 or width > len(payload): return []
        return [start + payload[start:start+width].index(max(payload[start:start+width]))
                for start in range(len(payload)-width+1)]
    raise ValueError(key)

def oracle(key, payload, config):
    if key == 'join':
        out = []
        for left in payload['left']:
            matched = False
            for right in payload['right']:
                if left['key'] == right['key']:
                    out.append({'key': left['key'], 'left': left['value'], 'right': right['value']})
                    matched = True
            if not matched and config['include_unmatched']:
                out.append({'key': left['key'], 'left': left['value'], 'right': None})
        return out
    if key == 'window_argmax':
        width = config['window']
        if width < 1 or width > len(payload): return []
        queue = []
        head = 0
        out = []
        for index, value in enumerate(payload):
            while head < len(queue) and queue[head] <= index-width: head += 1
            while len(queue) > head and payload[queue[-1]] < value: queue.pop()
            queue.append(index)
            if index >= width-1: out.append(queue[head])
        return out
    raise ValueError(key)

def fixtures(key):
    tasks = json.loads((Path(__file__).with_name('spec.json')).read_text())['tasks']
    task = next(task for task in tasks if task['key'] == key)
    config = copy.deepcopy(task['config'])
    sample = config.pop('sample')
    cases = [(sample, config)]
    rng = random.Random(20261003 + (1 if key == 'join' else 2))
    if key == 'join':
        edges = [([], []), ([{'key': '', 'value': None}], []),
            ([], [{'key': '', 'value': False}]),
            ([{'key': 'a', 'value': 0}], [{'key': 'b', 'value': 0}]),
            ([{'key': 'é', 'value': 1}], [{'key': 'e\u0301', 'value': 2}]),
            ([{'key': '🌿', 'value': None}], [{'key': '🌿', 'value': False}]),
            ([{'key': 'a', 'value': 1}, {'key': 'a', 'value': 2}],
             [{'key': 'a', 'value': 3}, {'key': 'a', 'value': 4}]),
            ([{'key': '東京', 'value': [1]}, {'key': '', 'value': {'x': None}}],
             [{'key': '', 'value': None}, {'key': '東京', 'value': [2]}])]
        for left, right in edges:
            for include in (False, True):
                cases.append(({'left': copy.deepcopy(left), 'right': copy.deepcopy(right)},
                              {'include_unmatched': include}))
        for index in range(144):
            left = [{'key': rng.choice(KEYS), 'value': copy.deepcopy(rng.choice(VALUES))}
                    for _ in range(rng.randrange(19))]
            right = [{'key': rng.choice(KEYS), 'value': copy.deepcopy(rng.choice(VALUES))}
                     for _ in range(rng.randrange(19))]
            cases.append(({'left': left, 'right': right}, {'include_unmatched': bool(index % 2)}))
    elif key == 'window_argmax':
        edges = [([], 0), ([], 1), ([1], 1), ([1], 2), ([1], -1),
                 ([3, 3, 3], 2), ([5, 1, 5, 1], 3), ([1, 5, 5, 1], 2),
                 ([5, 4, 3, 2], 2), ([1, 2, 3, 4], 2), ([-4, -1, -1, -2], 3),
                 ([0, 0, -1, 0], 4), ([2, 1], 0), ([2, 1], 1),
                 ([9, 1, 8, 8, 8, 2], 3), ([7, 7, 6, 7, 7], 3)]
        cases.extend((values, {'window': width}) for values, width in edges)
        for index in range(144):
            values = [rng.randrange(-9, 10) for _ in range(rng.randrange(91))]
            width = rng.randrange(-2, len(values)+4)
            cases.append((values, {'window': width}))
    else: raise ValueError(key)
    if len(cases) != 161: raise AssertionError('fixed 161 inputs')
    checked = []
    for payload, config in cases:
        expected = reference(key, payload, config)
        if expected != oracle(key, payload, config): raise AssertionError('independent gold disagreement')
        checked.append((payload, config, expected))
    return checked
