def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last = {}
    i = 0
    for event in payload:
        key = event["key"]
        t = event["t"]
        if key in last:
            if t - last[key] >= ttl:
                accepted.append(i)
                last[key] = t
            else:
                rejected.append(i)
        else:
            accepted.append(i)
            last[key] = t
        i = i + 1
    return {"accepted": accepted, "rejected": rejected}
