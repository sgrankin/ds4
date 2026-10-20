def solve(payload, config):
    merge_touching = False
    if isinstance(config, dict):
        if "merge_touching" in config:
            merge_touching = config["merge_touching"]
    elif isinstance(config, bool):
        merge_touching = config

    intervals = []
    for item in payload:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start = item[0]
            end = item[1]
            if start < end:
                intervals.append((start, end))

    intervals.sort()

    result = []
    for start, end in intervals:
        if result:
            last = result[-1]
            last_start = last[0]
            last_end = last[1]
            if start < last_end or (merge_touching and start == last_end):
                if end > last_end:
                    result[-1] = [last_start, end]
            else:
                result.append([start, end])
        else:
            result.append([start, end])

    return result
