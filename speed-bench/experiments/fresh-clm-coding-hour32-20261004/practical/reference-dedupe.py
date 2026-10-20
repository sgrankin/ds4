def solve(payload, config):
    last = {}
    accepted = []
    rejected = []
    for index, event in enumerate(payload):
        key = event['key']
        time = event['t']
        if key not in last or time - last[key] >= config['ttl']:
            accepted.append(index)
            last[key] = time
        else:
            rejected.append(index)
    return {'accepted':accepted,'rejected':rejected}
