def solve(payload, config):
    winners = {}
    for index in range(len(payload)):
        event = payload[index]
        key = event["key"]
        revision = event["revision"]
        if key not in winners:
            winners[key] = (revision, index, event)
        else:
            current_revision, current_index, current_event = winners[key]
            if revision > current_revision or (revision == current_revision and index > current_index):
                winners[key] = (revision, index, event)

    live = []
    deleted = []
    include_deleted = config["include_deleted"]

    for key in winners:
        revision, index, event = winners[key]
        if event["op"] == "put":
            live.append({"key": key, "revision": revision, "value": event["value"]})
        elif include_deleted:
            deleted.append({"key": key, "revision": revision})

    live.sort(key=lambda item: item["key"])
    deleted.sort(key=lambda item: item["key"])

    return {"live": live, "deleted": deleted}
