def solve(payload, config):
    order = config["order"]
    ids = []
    deps_by_id = {}
    for record in payload:
        task_id = record["id"]
        ids.append(task_id)
        deps_by_id[task_id] = record["deps"]

    completed = set()
    layers = []
    remaining = list(ids)

    while True:
        ready = []
        for task_id in remaining:
            blocked = False
            for dep in deps_by_id[task_id]:
                if dep not in completed:
                    blocked = True
                    break
            if not blocked:
                ready.append(task_id)

        if not ready:
            break

        if order == "lex":
            ready.sort()

        layers.append(ready)
        for task_id in ready:
            completed.add(task_id)

        still_remaining = []
        for task_id in remaining:
            if task_id not in completed:
                still_remaining.append(task_id)
        remaining = still_remaining

    if order == "lex":
        remaining.sort()

    return {"layers": layers, "unresolved": remaining}
