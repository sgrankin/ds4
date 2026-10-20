"""Private fixture golds: accepted-history scan and event arithmetic."""


def dedupe_gold(events, config):
    accepted, rejected = [], []
    for index, event in enumerate(events):
        previous = [events[j]['t'] for j in accepted if events[j]['key'] == event['key']]
        if previous and event['t'] - previous[-1] < config['ttl']:
            rejected.append(index)
        else:
            accepted.append(index)
    return {'accepted':accepted,'rejected':rejected}


def bucket_gold(events, config):
    tokens, previous = config['capacity'], 0
    accepted, rejected, remaining = [], [], []
    for index, event in enumerate(events):
        available = min(config['capacity'], tokens + (event['t']-previous)*config['rate'])
        if event['cost'] <= available:
            accepted.append(index)
            available -= event['cost']
        else:
            rejected.append(index)
        tokens, previous = available, event['t']
        remaining.append(tokens)
    return {'accepted':accepted,'rejected':rejected,'remaining':remaining,'tokens':tokens}
