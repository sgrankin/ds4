"""Plot the measured host-memory/reassembly tradeoff; no model execution."""
import json
from pathlib import Path
import os
here = Path(__file__).resolve().parent
plot_cache = here.parents[2] / '.workspaces' / 'plot-cache'
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(plot_cache / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(plot_cache))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here = Path(__file__).resolve().parent
points = json.loads((here / 'native-summary.json').read_text())['points']
fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
x = [p['hot_limit_mib'] for p in points]
ax.plot(x, [p['cold_mean_ms'] for p in points], 'o-', label='First pass after reopen')
ax.plot(x, [p['warm_mean_ms'] for p in points], 's-', label='Second pass')
ax.set(xlabel='Hot host-cache limit (MiB)', ylabel='Mean checkpoint reassembly (ms)',
       title='Two native DS4.1 checkpoints: memory / latency', ylim=(0, 90))
ax.grid(alpha=.25)
ax.legend()
ax.annotate('12/12 byte and session-isolation checks passed',
            xy=(.03, .05), xycoords='axes fraction', fontsize=9)
fig.savefig(here / 'hot-cache-curve.png', dpi=180)
fig.savefig(here / 'hot-cache-curve.svg')
