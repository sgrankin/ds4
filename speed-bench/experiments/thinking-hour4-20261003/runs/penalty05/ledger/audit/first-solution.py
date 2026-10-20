def solve(payload, config):
    initial = config["initial"]
    balances = {}
    for account in initial:
        balances[account] = initial[account]
    seen = {}
    for event in payload:
        eid = event["id"]
        if eid in seen:
            continue
        seen[eid] = True
        account = event["account"]
        delta = event["delta"]
        if account in balances:
            balances[account] = balances[account] + delta
        else:
            balances[account] = delta
    return balances
