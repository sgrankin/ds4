#!/usr/bin/env python3
"""Plot measured SWA checkpoint tradeoffs; no interpolation or fitted frontier."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

cache = Path(__file__).resolve().parents[3] / ".workspaces/plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("summary", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
arms = summary["arms"]
assert all(not arm["verify"] for arm in arms), "Plot measurement arms, not poison diagnostics"
names = [case["case"] for case in arms[0]["cases"]]
fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False, layout="constrained")
styles = [("exact", "Exact restore", "o", "#2563eb"),
          ("recover", "128-row recovery", "D", "#d97706"),
          ("fresh", "Full recomputation", "s", "#15803d")]
for ax, name in zip(axes[0], names):
    for mode, label, marker, color in styles:
        cases = [case for arm in arms if arm["mode"] == mode for case in arm["cases"] if case["case"] == name]
        if not cases:
            continue
        sizes = [0 if mode == "fresh" else case["blocks_bytes" if mode == "recover" else "full_bytes"] for case in cases]
        times = [case["practical_ms"] / 1000 for case in cases]
        x, y = statistics.mean(sizes) / 2**20, statistics.mean(times)
        dnll = statistics.mean(case["mean_delta_nll"] for case in cases)
        grade = sum(case["answer"]["exact"] and case["answer"]["stopped"] for case in cases)
        text = f"{label}\nΔNLL {dnll:+.4f}; PPL {100 * math.expm1(dnll):+.2f}%\n{grade}/{len(cases)} answers passed"
        ax.errorbar(x, y, yerr=[[y - min(times)], [max(times) - y]], fmt=marker, color=color, capsize=4, markersize=7)
        ax.annotate(text, (x, y), xytext=(7 if mode == "fresh" else -7, 10 if mode == "recover" else -12),
                    textcoords="offset points", ha="left" if mode == "fresh" else "right",
                    va="bottom" if mode == "recover" else "top", fontsize=8, color=color)
    ax.set(title="Adverse correction" if name == "changed" else name.capitalize(),
           xlabel="Stored checkpoint (MiB); fresh = no entry", ylabel="Restore/recompute + first answer (seconds)")
    ax.margins(x=0.22, y=0.28)
    ax.grid(alpha=0.2)
teacher_label = "Baseline reasoning/answer trace" if summary.get("teacher_trace") else "Short answer-only forced teachers"
fig.suptitle(f"Measured SWA recovery tradeoffs — {teacher_label}\nExact controls averaged; bars show their observed range")
args.output.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output / "swa-tradeoff.svg")
fig.savefig(args.output / "swa-tradeoff.png", dpi=160)
plt.close(fig)
