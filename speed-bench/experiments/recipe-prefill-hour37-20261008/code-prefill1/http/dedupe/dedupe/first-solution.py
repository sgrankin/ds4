def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last = {}
    index = 0
    while index < len(payload):
        event = payload[index]
        key = event["key"]
        t = event["t"]
        if key in last:
            if t - last[key] < ttl:
                rejected.append(index)
            else:
                accepted.append(index)
                last[key] = t
        else:
            accepted.append(index)
            last[key] = t
        index = index + 1
    return {"accepted": accepted, "rejected": rejected}
