def solve(payload, config):
    capacity = config['capacity']
    rate = config['rate']
    tokens = capacity
    previous = 0
    accepted = []
    rejected = []
    remaining = []
    for index, event in enumerate(payload):
        time = event['t']
        tokens = min(capacity, tokens + (time - previous) * rate)
        previous = time
        if event['cost'] <= tokens:
            tokens -= event['cost']
            accepted.append(index)
        else:
            rejected.append(index)
        remaining.append(tokens)
    return {'accepted':accepted,'rejected':rejected,'remaining':remaining,'tokens':tokens}
