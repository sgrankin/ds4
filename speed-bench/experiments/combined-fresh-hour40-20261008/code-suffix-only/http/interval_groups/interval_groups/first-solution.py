def solve(payload, config):
    gap = config["gap"]
    items = []
    for index in range(len(payload)):
        interval = payload[index]
        items.append((interval["start"], interval["end"], index))
    items.sort()
    groups = []
    current_start = None
    current_end = None
    current_indices = []
    for item in items:
        start = item[0]
        end = item[1]
        index = item[2]
        if current_start is None:
            current_start = start
            current_end = end
            current_indices = [index]
        elif start <= current_end + gap:
            if end > current_end:
                current_end = end
            current_indices.append(index)
        else:
            current_indices.sort()
            groups.append({
                "start": current_start,
                "end": current_end,
                "indices": current_indices
            })
            current_start = start
            current_end = end
            current_indices = [index]
    if current_start is not None:
        current_indices.sort()
        groups.append({
            "start": current_start,
            "end": current_end,
            "indices": current_indices
        })
    return {"groups": groups}
