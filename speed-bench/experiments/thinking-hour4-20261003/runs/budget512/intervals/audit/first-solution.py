def solve(payload, config):
    merge_touching = False
    if isinstance(config, dict):
        merge_touching = config.get("merge_touching", False)
    elif isinstance(config, bool):
        merge_touching = config

    intervals = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start = item[0]
        end = item[1]
        if start >= end:
            continue
        intervals.append((start, end))

    intervals.sort()

    result = []
    for start, end in intervals:
        if not result:
            result.append([start, end])
            continue
        last_start, last_end = result[-1]
        if start < last_end or (merge_touching and start == last_end):
            if end > last_end:
                result[-1][1] = end
        else:
            result.append([start, end])

    return result
