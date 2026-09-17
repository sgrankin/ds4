#!/usr/bin/env python3
"""Capture a separate, checked tool-using task for predictor evaluation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from trace_session import extract
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'experiments/ds41-ssd'))
from snapshot_shaders import snapshot_shaders

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
a = p.parse_args()
out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
project = out/'project'; project.mkdir()
source = '''class TTLCache:
    def __init__(self, clock):
        self.clock = clock
        self.entries = {}

    def put(self, key, value, ttl):
        self.entries[key] = (value, self.clock() + ttl)

    def get(self, key, default=None):
        item = self.entries.get(key)
        if item is None:
            return default
        value, deadline = item
        if self.clock() > deadline:
            return default
        return value
'''
(project/'ttl_cache.py').write_text(source)
(project/'README.md').write_text('''A tiny cache with an injected monotonic clock. Contract: get returns default
at or after the expiry deadline and removes expired entries. A nonpositive TTL
must remove an existing key immediately. None is a valid cached value.
''')
prompt = ('Inspect this small TTL cache project. Fix the implementation to satisfy '
          'the README contract. Add unittest coverage using a fake clock, including '
          'expiry exactly at the boundary, deleting expired entries, replacing a '
          'key with a nonpositive TTL, and caching None. Run the tests and briefly '
          'explain the changes. Work only in this project directory.')
(out/'prompt.txt').write_text(prompt)
binary = out/'ds4-agent'; shutil.copy2('ds4-agent', binary)
cache = out/'kv'; cache.mkdir()
env = os.environ.copy(); env.update(snapshot_shaders(out))
env.update(DS4_AGENT_CACHE_DIR=str(cache), DS4_AGENT_TEST_TIME='1789319891',
           TZ='America/New_York', DS4_V41_ROUTE_RECORD=str(out/'routes.bin'),
           DS4_V41_ROUTE_FEATURES=str(out/'features.bin'))
cmd = [str(binary), '--ssd-streaming', '--non-interactive', '--seed', '1234',
       '--temp', '0', '-n', '4096', '--chdir', str(project), '--trace', str(out/'trace.log'), '-p', prompt]
(out/'manifest.json').write_text(json.dumps(dict(argv=cmd, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    original_source=source, env={k:v for k,v in env.items() if k.startswith('DS4_') or k=='TZ'}), indent=2)+'\n')
with (out/'stdout.log').open('wb') as log:
    subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
# Independent checks run outside the project and are not supplied to the agent.
check = '''import importlib.util, sys
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t = [100.0]; c = m.TTLCache(lambda: t[0])
c.put('a', 3, 5); t[0] = 104.999; assert c.get('a') == 3
t[0] = 105; assert c.get('a', 'missing') == 'missing' and 'a' not in c.entries
c.put('a', 4, 20); c.put('a', 5, 0); assert 'a' not in c.entries
c.put('a', 4, 20); c.put('a', 5, -1); assert 'a' not in c.entries
c.put('b', None, 2); assert c.get('b', 'missing') is None
t[0] = 200; assert c.get('b', 'missing') == 'missing' and 'b' not in c.entries
'''
checked = subprocess.run([sys.executable, '-c', check, str(project/'ttl_cache.py')], capture_output=True, text=True)
summary, replay = extract(out/'trace.log')
summary.update(held_out_rc=checked.returncode, held_out_output=checked.stdout+checked.stderr,
               purpose='Independent task evaluation; do not use for training or epoch selection. Intrusive capture, not a speed measurement.')
(out/'replay.txt').write_text(replay)
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary), flush=True)
if checked.returncode: raise SystemExit('independent task checks failed')
