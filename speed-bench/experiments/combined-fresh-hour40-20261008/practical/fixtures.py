"""Load the authored and precomputed 161-case JSON packet."""
import json
from pathlib import Path


def fixtures(key):
    rows = json.loads((Path(__file__).parent/'cases.json').read_text())[key]
    return [(row['payload'],row['config'],row['gold']) for row in rows]
