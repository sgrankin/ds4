# Ledger reconciliation

A small standard-library Python package reconciles payment processor exports.
Run `python3 -m unittest discover -s tests -v` from this directory.

Each CSV record has an event_id, account, kind, amount and date. Amount is a
nonnegative decimal amount in dollars with at most two fractional digits.
Kinds are charge, refund and void. Charges increase net receipts; refunds
subtract; void events have no financial effect. Processor retries repeat the
same event_id: count each event exactly once, using its first occurrence.
Return integer cents, never floating-point money. Include accounts whose valid
charge/refund events net to zero; omit accounts having only void events.

The public API is `ledger.report.summarize(rows)`, returning a dictionary from
account name to integer net cents. `ledger.io.read_rows(path)` loads a CSV.
`python3 -m ledger.cli data.csv` prints the report as sorted JSON.
The current implementation has reconciliation bugs. Keep public APIs compatible.
