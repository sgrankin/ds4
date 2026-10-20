def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last = {}
    for i in range(len(payload)):
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
    return {"accepted": accepted, "rejected": rejected}
