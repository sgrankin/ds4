def solve(payload, config):
    capacity = config["capacity"]
    rate = config["rate"]
    tokens = capacity
    previous = 0
    accepted = []
    rejected = []
    remaining = []
    index = 0
    for event in payload:
        t = event["t"]
        cost = event["cost"]
        tokens = tokens + (t - previous) * rate
        if tokens > capacity:
            tokens = capacity
        previous = t
        if cost <= tokens:
            accepted.append(index)
            tokens = tokens - cost
        else:
            rejected.append(index)
        remaining.append(tokens)
        index = index + 1
    return {
        "accepted": accepted,
        "rejected": rejected,
        "remaining": remaining,
        "tokens": tokens
    }