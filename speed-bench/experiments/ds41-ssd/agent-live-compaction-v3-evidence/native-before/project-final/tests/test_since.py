import unittest
from ledger.report import summarize


def event(id, account, kind, amount, date):
    return dict(event_id=id, account=account, kind=kind, amount=amount,
                date=date)


class SinceTests(unittest.TestCase):
    def test_inclusive_boundary(self):
        rows = [event('a', 'north', 'charge', '1.00', '2026-08-01'),
                event('b', 'north', 'charge', '2.00', '2026-08-02')]
        self.assertEqual(summarize(rows, since='2026-08-02'), {'north': 200})

    def test_duplicate_straddling_cutoff(self):
        first = event('a', 'north', 'charge', '4.00', '2026-07-31')
        retry = event('a', 'north', 'charge', '4.00', '2026-08-05')
        self.assertEqual(summarize([first, retry], since='2026-08-01'), {})

    def test_default_unchanged(self):
        rows = [event('a', 'north', 'charge', '1.00', '2026-08-01')]
        self.assertEqual(summarize(rows), {'north': 100})


if __name__ == '__main__':
    unittest.main()
