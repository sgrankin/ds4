def solve(payload, config):
    capacity = config["capacity"]
    rate = config["rate"]
    tokens = capacity
    previous = 0
    accepted = []
    rejected = []
    remaining = []
    for index in range(len(payload)):
        event = payload[index]
        t = event["t"]
        cost = event["cost"]
        elapsed = t - previous
        tokens = tokens + elapsed * rate
        if tokens > capacity:
            tokens = capacity
        previous = t
        if cost <= tokens:
            tokens = tokens - cost
            accepted.append(index)
        else:
            rejected.append(index)
        remaining.append(tokens)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "remaining": remaining,
        "tokens": tokens,
    }
