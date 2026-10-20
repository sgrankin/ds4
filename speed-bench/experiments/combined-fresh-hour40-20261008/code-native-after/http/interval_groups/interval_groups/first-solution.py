def solve(payload, config):
    gap = config["gap"]
    ordered = []
    position = 0
    while position < len(payload):
        interval = payload[position]
        ordered.append((interval["start"], interval["end"], position))
        position = position + 1
    ordered.sort()

    groups = []
    current_start = None
    current_end = None
    current_indices = []
    cursor = 0
    while cursor < len(ordered):
        start, end, index = ordered[cursor]
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
            groups.append({"start": current_start, "end": current_end, "indices": current_indices})
            current_start = start
            current_end = end
            current_indices = [index]
        cursor = cursor + 1

    if current_start is not None:
        current_indices.sort()
        groups.append({"start": current_start, "end": current_end, "indices": current_indices})

    return {"groups": groups}
