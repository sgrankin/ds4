from .money import cents


def summarize(rows):
    totals = {}
    for row in rows:
        account = row['account']
        amount = cents(row['amount'])
        totals[account] = totals.get(account, 0) + amount
    return totals
