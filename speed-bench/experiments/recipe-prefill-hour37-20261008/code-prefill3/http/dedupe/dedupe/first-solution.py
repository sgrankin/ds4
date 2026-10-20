def solve(payload, config):
    ttl = config["ttl"]
    accepted = []
    rejected = []
    last_accepted = {}
    for index in range(len(payload)):
        event = payload[index]
        t = event["t"]
        key = event["key"]
        if key not in last_accepted:
            accepted.append(index)
            last_accepted[key] = t
        else:
            previous = last_accepted[key]
            if t - previous >= ttl:
                accepted.append(index)
                last_accepted[key] = t
            else:
                rejected.append(index)
    return {"accepted": accepted, "rejected": rejected}
