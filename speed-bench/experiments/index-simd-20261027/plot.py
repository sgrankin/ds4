#!/usr/bin/env python3
"""Plot measured index-buffer capacity against dense score GPU time."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

with Path(sys.argv[1]).open(newline="") as source:
    rows = [row for row in csv.DictReader(source)
            if row["selection"] == "dense" and row["scope"] == "score"]
shapes = sorted({int(row["rows"]) for row in rows})
fig, axes = plt.subplots(1, len(shapes), figsize=(4 * len(shapes), 3.8), squeeze=False)
for ax, shape in zip(axes[0], shapes):
    for row in rows:
        if int(row["rows"]) != shape:
            continue
        packed = row["dtype"] == "fp4"
        simd = row["implementation"] == "simd"
        size = shape * (68 if packed else 512) / 2**20
        ax.scatter(size, float(row["gpu_ms"]), s=70,
                   marker="o" if simd else "x", color="#0072B2" if packed else "#D55E00",
                   label=f'{"Register" if simd else "Shared"} {row["dtype"].upper()}')
    ax.set(title=f"{shape:,} index keys", xlabel="Key buffer (MiB)", ylabel="GPU scoring time (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Measured buffer-size / scoring-time choices; lower and left is better")
fig.tight_layout(rect=(0, .10, 1, .94))
fig.savefig(sys.argv[2], dpi=160)
