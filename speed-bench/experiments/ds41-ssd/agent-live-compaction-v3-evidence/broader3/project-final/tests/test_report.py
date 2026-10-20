import unittest
from ledger.money import cents
from ledger.report import summarize


def event(id, account, kind, amount):
    return dict(event_id=id, account=account, kind=kind, amount=amount,
                date='2026-08-01')


class ReportTests(unittest.TestCase):
    def test_decimal_money(self):
        self.assertEqual(cents('0.29'), 29)
        self.assertEqual(cents('19.99'), 1999)

    def test_charge_and_refund(self):
        rows = [event('a', 'north', 'charge', '12.00'),
                event('b', 'north', 'refund', '2.50')]
        self.assertEqual(summarize(rows), {'north': 950})

    def test_retry(self):
        row = event('a', 'north', 'charge', '4.00')
        self.assertEqual(summarize([row, dict(row)]), {'north': 400})

    def test_void(self):
        self.assertEqual(summarize([event('v', 'west', 'void', '9.00')]), {})

    def test_zero_balance(self):
        self.assertEqual(summarize([event('a', 'east', 'charge', '1.00'),
                                    event('b', 'east', 'refund', '1.00')]), {'east': 0})
