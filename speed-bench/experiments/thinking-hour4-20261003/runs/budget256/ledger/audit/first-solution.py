def solve(payload, config):
    initial = config["initial"]
    balances = {}
    for account in initial:
        balances[account] = initial[account]
    seen = {}
    for event in payload:
        event_id = event["id"]
        if event_id in seen:
            continue
        seen[event_id] = True
        account = event["account"]
        delta = event["delta"]
        if account in balances:
            balances[account] = balances[account] + delta
        else:
            balances[account] = delta
    return balances
