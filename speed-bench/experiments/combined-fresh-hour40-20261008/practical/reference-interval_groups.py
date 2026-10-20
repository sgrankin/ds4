def solve(payload, config):
    ordered = sorted(range(len(payload)), key=lambda i: (payload[i]['start'], payload[i]['end'], i))
    groups = []
    for index in ordered:
        item = payload[index]
        if groups and item['start'] <= groups[-1]['end'] + config['gap']:
            group = groups[-1]
            group['end'] = max(group['end'], item['end'])
            group['indices'].append(index)
        else:
            groups.append({'start': item['start'], 'end': item['end'], 'indices': [index]})
    for group in groups:
        group['indices'].sort()
    return {'groups': groups}
