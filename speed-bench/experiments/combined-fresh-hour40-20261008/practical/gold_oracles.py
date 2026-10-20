"""Independent small-input golds: connectivity and recursive dependency depth."""
def interval_gold(payload, config):
    neighbors = [set() for item in payload]
    for i, left in enumerate(payload):
        for j, right in enumerate(payload):
            distance = max(left['start'], right['start']) - min(left['end'], right['end'])
            if distance <= config['gap']:
                neighbors[i].add(j)
    unseen = set(range(len(payload)))
    groups = []
    while unseen:
        start = min(unseen)
        component = {start}
        work = [start]
        while work:
            i = work.pop()
            for j in neighbors[i] - component:
                component.add(j)
                work.append(j)
        unseen -= component
        groups.append({'start': min(payload[i]['start'] for i in component),
                       'end': max(payload[i]['end'] for i in component),
                       'indices': sorted(component)})
    groups.sort(key=lambda row: (row['start'], row['end']))
    return {'groups': groups}


def dependency_gold(payload, config):
    tasks = {row['id']: row for row in payload}
    memo = {}
    active = set()
    def depth(name):
        if name not in tasks or name in active:
            return None
        if name in memo:
            return memo[name]
        active.add(name)
        levels = [depth(dep) for dep in set(tasks[name]['deps'])]
        active.remove(name)
        value = None if any(level is None for level in levels) else (max(levels) + 1 if levels else 0)
        memo[name] = value
        return value
    levels = [depth(row['id']) for row in payload]
    finite = [value for value in levels if value is not None]
    layers = []
    for level in range(max(finite) + 1 if finite else 0):
        names = [row['id'] for row, value in zip(payload, levels) if value == level]
        layers.append(sorted(names) if config['order'] == 'lex' else names)
    unresolved = [row['id'] for row, value in zip(payload, levels) if value is None]
    if config['order'] == 'lex':
        unresolved.sort()
    return {'layers': layers, 'unresolved': unresolved}
