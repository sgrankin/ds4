def solve(payload, config):
    balances = {}
    initial = config.get("initial", {})
    for account in initial:
        balances[account] = initial[account]
    seen = {}
    for event in payload:
        event_id = event.get("id")
        if event_id in seen:
            continue
        seen[event_id] = True
        account = event.get("account")
        if account not in balances:
            balances[account] = 0
        balances[account] = balances[account] + event.get("delta", 0)
    return balances
