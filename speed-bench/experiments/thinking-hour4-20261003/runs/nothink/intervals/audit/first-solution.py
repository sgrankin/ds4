def solve(payload, config):
    merge_touching = False
    if config is not None:
        value = config.get("merge_touching")
        if value is True:
            merge_touching = True

    intervals = []
    if payload is not None:
        for item in payload:
            if item is None:
                continue
            start = item[0]
            end = item[1]
            if start > end:
                continue
            intervals.append([start, end])

    intervals.sort()
    result = []
    for interval in intervals:
        start = interval[0]
        end = interval[1]
        if result:
            previous = result[-1]
            previous_end = previous[1]
            if merge_touching:
                joins = start <= previous_end
            else:
                joins = start <= previous_end and start > previous[0]
                if start == previous[0]:
                    joins = True
            if joins:
                if end > previous_end:
                    previous[1] = end
                continue
        result.append([start, end])

    return result
