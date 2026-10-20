def solve(payload, config):
    winners = {}
    for event in payload:
        key = event['key']
        if key not in winners or event['revision'] >= winners[key]['revision']:
            winners[key] = event
    live = []
    deleted = []
    for key in sorted(winners):
        event = winners[key]
        if event['op'] == 'put':
            live.append({'key':key, 'revision':event['revision'], 'value':event['value']})
        elif config['include_deleted']:
            deleted.append({'key':key, 'revision':event['revision']})
    return {'live':live, 'deleted':deleted}
