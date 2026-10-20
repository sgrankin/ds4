def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last = {}
    index = 0
    for event in payload:
        key = event["key"]
        t = event["t"]
        if key not in last:
            accepted.append(index)
            last[key] = t
        else:
            if t - last[key] >= ttl:
                accepted.append(index)
                last[key] = t
            else:
                rejected.append(index)
        index = index + 1
    return {"accepted": accepted, "rejected": rejected}
