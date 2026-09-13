import json
import sys
from .io import read_rows
from .report import summarize

if __name__ == '__main__':
    print(json.dumps(summarize(read_rows(sys.argv[1])), sort_keys=True))
