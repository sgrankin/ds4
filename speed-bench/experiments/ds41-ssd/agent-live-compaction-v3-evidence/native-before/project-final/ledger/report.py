from .money import cents

_EFFECT = {'charge': 1, 'refund': -1}


def summarize(rows, *, since=None):
    totals = {}
    seen = set()
    for row in rows:
        event_id = row['event_id']
        if event_id in seen:
            continue
        seen.add(event_id)
        if since is not None and row['date'] < since:
            continue
        kind = row['kind']
        if kind not in _EFFECT:
            continue
        account = row['account']
        amount = cents(row['amount']) * _EFFECT[kind]
        totals[account] = totals.get(account, 0) + amount
    return totals
