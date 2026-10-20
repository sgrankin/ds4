from .money import cents

KINDS = {'charge': 1, 'refund': -1}


def summarize(rows):
    totals = {}
    seen = set()
    for row in rows:
        event_id = row['event_id']
        if event_id in seen:
            continue
        seen.add(event_id)
        kind = row['kind']
        if kind not in KINDS:
            continue
        account = row['account']
        totals[account] = totals.get(account, 0) + KINDS[kind] * cents(row['amount'])
    return totals
