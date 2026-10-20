def solve(payload, config):
    capacity = config["capacity"]
    rate = config["rate"]
    tokens = capacity
    previous_t = 0
    accepted = []
    rejected = []
    remaining = []
    index = 0
    for event in payload:
        t = event["t"]
        cost = event["cost"]
        elapsed = t - previous_t
        tokens = tokens + elapsed * rate
        if tokens > capacity:
            tokens = capacity
        previous_t = t
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
        "tokens": tokens,
    }
