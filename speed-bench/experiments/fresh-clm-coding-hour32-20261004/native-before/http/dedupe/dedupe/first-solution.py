def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last = {}
    for index in range(len(payload)):
        event = payload[index]
        t = event["t"]
        key = event["key"]
        if key not in last:
            accepted.append(index)
            last[key] = t
        else:
            if t - last[key] >= ttl:
                accepted.append(index)
                last[key] = t
            else:
                rejected.append(index)
    return {"accepted": accepted, "rejected": rejected}
