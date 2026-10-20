from decimal import Decimal


def cents(text):
    """Convert a dollar amount to integer cents."""
    return int(Decimal(text) * 100)
