def solve(payload, config):
    done = set()
    pending = list(range(len(payload)))
    layers = []
    while pending:
        ready = [i for i in pending if all(dep in done for dep in payload[i]['deps'])]
        if not ready:
            break
        if config['order'] == 'lex':
            ready.sort(key=lambda i: payload[i]['id'])
        layers.append([payload[i]['id'] for i in ready])
        done.update(payload[i]['id'] for i in ready)
        readyset = set(ready)
        pending = [i for i in pending if i not in readyset]
    unresolved = [payload[i]['id'] for i in pending]
    if config['order'] == 'lex':
        unresolved.sort()
    return {'layers': layers, 'unresolved': unresolved}
