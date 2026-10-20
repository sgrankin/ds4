def solve(payload, config):
    window = config["window"]
    length = len(payload)
    if window < 1 or window > length:
        return []
    result = []
    for start in range(0, length - window + 1):
        best_index = start
        best_value = payload[start]
        for offset in range(1, window):
            index = start + offset
            value = payload[index]
            if value > best_value:
                best_value = value
                best_index = index
        result.append(best_index)
    return result
