def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last = {}
    i = 0
    while i < len(payload):
        event = payload[i]
        t = event["t"]
        key = event["key"]
        if key not in last:
            accepted.append(i)
            last[key] = t
        else:
            if t - last[key] >= ttl:
                accepted.append(i)
                last[key] = t
            else:
                rejected.append(i)
        i = i + 1
    return {"accepted": accepted, "rejected": rejected}
