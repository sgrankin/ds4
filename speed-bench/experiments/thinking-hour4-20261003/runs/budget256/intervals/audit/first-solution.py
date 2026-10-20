def solve(payload, config):
    intervals = []
    for item in payload:
        if type(item) is not list and type(item) is not tuple:
            continue
        if len(item) != 2:
            continue
        start = item[0]
        end = item[1]
        if type(start) is not int or type(end) is not int:
            continue
        if start >= end:
            continue
        intervals.append([start, end])
    intervals.sort()
    merge_touching = False
    if type(config) is dict:
        value = config.get("merge_touching")
        if value is True:
            merge_touching = True
        elif value is False:
            merge_touching = False
    elif type(config) is bool:
        merge_touching = config
    result = []
    for interval in intervals:
        if result:
            last = result[-1]
            if interval[0] < last[1] or (merge_touching and interval[0] == last[1]):
                if interval[1] > last[1]:
                    last[1] = interval[1]
                continue
        result.append([interval[0], interval[1]])
    return result
