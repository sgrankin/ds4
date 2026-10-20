def solve(payload, config):
    winners = {}
    for index in range(len(payload)):
        event = payload[index]
        key = event["key"]
        revision = event["revision"]
        if key not in winners:
            winners[key] = index
        else:
            current_index = winners[key]
            current = payload[current_index]
            if revision > current["revision"]:
                winners[key] = index
            elif revision == current["revision"]:
                winners[key] = index

    live = []
    deleted = []
    include_deleted = config["include_deleted"]

    for key in winners:
        event = payload[winners[key]]
        if event["op"] == "put":
            live.append({
                "key": key,
                "revision": event["revision"],
                "value": event["value"]
            })
        elif include_deleted:
            deleted.append({
                "key": key,
                "revision": event["revision"]
            })

    live.sort(key=lambda item: item["key"])
    deleted.sort(key=lambda item: item["key"])

    return {
        "live": live,
        "deleted": deleted
    }
