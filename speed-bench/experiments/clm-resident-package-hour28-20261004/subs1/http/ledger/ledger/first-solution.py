def solve(payload, config):
    winners = {}
    order = []
    for event in payload:
        key = event["key"]
        revision = event["revision"]
        if key not in winners:
            winners[key] = event
            order.append(key)
        else:
            current = winners[key]
            if revision >= current["revision"]:
                winners[key] = event
    live = []
    deleted = []
    include_deleted = False
    if config is not None:
        include_deleted = config.get("include_deleted", False)
    for key in order:
        event = winners[key]
        if event["op"] == "put":
            live.append({"key": key, "revision": event["revision"], "value": event["value"]})
        else:
            if include_deleted:
                deleted.append({"key": key, "revision": event["revision"]})
    live.sort(key=lambda item: item["key"])
    deleted.sort(key=lambda item: item["key"])
    return {"live": live, "deleted": deleted}
