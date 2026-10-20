def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last_accepted = {}
    for index, event in enumerate(payload):
        key = event["key"]
        t = event["t"]
        if key not in last_accepted:
            accepted.append(index)
            last_accepted[key] = t
        else:
            if t - last_accepted[key] < ttl:
                rejected.append(index)
            else:
                accepted.append(index)
                last_accepted[key] = t
    return {"accepted": accepted, "rejected": rejected}