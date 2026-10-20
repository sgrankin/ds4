def solve(payload, config):
    window = config["window"]
    length = len(payload)
    if window < 1 or window > length:
        return []
    result = []
    start = 0
    while start + window <= length:
        best_index = start
        best_value = payload[start]
        offset = 1
        while offset < window:
            index = start + offset
            value = payload[index]
            if value > best_value:
                best_value = value
                best_index = index
            offset += 1
        result.append(best_index)
        start += 1
    return result
