from .money import cents


def summarize(rows):
    totals = {}
    seen = set()
    for row in rows:
        event_id = row['event_id']
        if event_id in seen:
            continue
        seen.add(event_id)
        kind = row['kind']
        if kind == 'void':
            continue
        account = row['account']
        amount = cents(row['amount'])
        if kind == 'refund':
            amount = -amount
        totals[account] = totals.get(account, 0) + amount
    return totals
