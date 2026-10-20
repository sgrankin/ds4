def solve(payload, config):
    winners = {}
    order = []
    for index in range(len(payload)):
        event = payload[index]
        key = event["key"]
        revision = event["revision"]
        if key not in winners:
            winners[key] = index
            order.append(key)
        else:
            current = winners[key]
            current_event = payload[current]
            if revision > current_event["revision"]:
                winners[key] = index
            elif revision == current_event["revision"]:
                winners[key] = index
    live = []
    deleted = []
    include_deleted = False
    if config is not None:
        include_deleted = config.get("include_deleted", False)
    for key in order:
        event = payload[winners[key]]
        if event["op"] == "put":
            live.append({"key": key, "revision": event["revision"], "value": event["value"]})
        else:
            if include_deleted:
                deleted.append({"key": key, "revision": event["revision"]})
    live.sort(key=lambda item: item["key"])
    deleted.sort(key=lambda item: item["key"])
    return {"live": live, "deleted": deleted}
