def solve(payload, config):
    width = config['window']
    if width < 1 or width > len(payload):
        return []
    output = []
    for start in range(len(payload) - width + 1):
        best = start
        for index in range(start + 1, start + width):
            if payload[index] > payload[best]:
                best = index
        output.append(best)
    return output
