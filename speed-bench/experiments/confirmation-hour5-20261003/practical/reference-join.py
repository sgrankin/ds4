def solve(payload, config):
    lookup = {}
    for row in payload['right']:
        lookup.setdefault(row['key'], []).append(row['value'])
    output = []
    for row in payload['left']:
        matches = lookup.get(row['key'], [])
        if matches:
            for value in matches:
                output.append({'key': row['key'], 'left': row['value'], 'right': value})
        elif config['include_unmatched']:
            output.append({'key': row['key'], 'left': row['value'], 'right': None})
    return output
