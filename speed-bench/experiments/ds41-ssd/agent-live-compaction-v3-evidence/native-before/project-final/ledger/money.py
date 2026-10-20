def cents(text):
    """Convert a dollar amount to integer cents."""
    text = str(text).strip()
    sign = 1
    if text.startswith('-'):
        sign = -1
        text = text[1:]
    elif text.startswith('+'):
        text = text[1:]
    if '.' in text:
        whole, frac = text.split('.', 1)
    else:
        whole, frac = text, ''
    frac = (frac + '00')[:2]
    return sign * (int(whole or '0') * 100 + int(frac or '0'))
