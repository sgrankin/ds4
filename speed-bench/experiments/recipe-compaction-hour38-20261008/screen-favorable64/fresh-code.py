def solve(payload, config):
    accepted = []
    rejected = []
    last_accepted = {}
    ttl = config["ttl"]
    index = 0
    for event in payload:
        key = event["key"]
        t = event["t"]
        if key not in last_accepted:
            accepted.append(index)
            last_accepted[key] = t
        else:
            if t - last_accepted[key] >= ttl:
                accepted.append(index)
                last_accepted[key] = t
            else:
                rejected.append(index)
        index = index + 1
    return {"accepted": accepted, "rejected": rejected}