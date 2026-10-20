def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last_accepted = {}
    for index in range(len(payload)):
        event = payload[index]
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
    return {"accepted": accepted, "rejected": rejected}
