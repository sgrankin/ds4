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
    unresolved = []

    remaining = []
    for task_id in ids:
        remaining.append(task_id)

    while True:
        ready = []
        for task_id in remaining:
            all_done = True
            for dep in deps_by_id[task_id]:
                if dep not in completed:
                    all_done = False
                    break
            if all_done:
                ready.append(task_id)

        if not ready:
            break

        if order == "lex":
            ready.sort()
        layers.append(ready)

        for task_id in ready:
            completed.add(task_id)

        next_remaining = []
        for task_id in remaining:
            if task_id not in completed:
                next_remaining.append(task_id)
        remaining = next_remaining

    if order == "lex":
        remaining.sort()
    unresolved = remaining

    return {"layers": layers, "unresolved": unresolved}
