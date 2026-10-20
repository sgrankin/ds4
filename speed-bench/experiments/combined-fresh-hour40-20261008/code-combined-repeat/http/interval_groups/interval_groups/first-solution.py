def solve(payload, config):
    gap = config["gap"]
    ordered = []
    for index in range(len(payload)):
        interval = payload[index]
        ordered.append((interval["start"], interval["end"], index))
    ordered.sort()
    groups = []
    for start, end, index in ordered:
        if groups:
            current = groups[-1]
            if start <= current["end"] + gap:
                if end > current["end"]:
                    current["end"] = end
                current["indices"].append(index)
                continue
        groups.append({"start": start, "end": end, "indices": [index]})
    for group in groups:
        group["indices"].sort()
    return {"groups": groups}
