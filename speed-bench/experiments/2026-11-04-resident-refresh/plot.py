#!/usr/bin/env python3
"""Plot question 12 summaries; pass .workspaces/hillclimb-20261104 as root."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path, help="Directory containing experiment 11 and 12 evidence")
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pareto.svg"))
args = parser.parse_args()
current = args.root / "12-resident-refresh"
timing_roots = {
    4: current / "timing32k-r4",
    8: args.root / "11-resident-parallel" / "combined32k",
    16: current / "timing32k-r16",
}
timings = {r: json.loads((path / "timing-summary.json").read_text())
           for r, path in timing_roots.items()}
qualities = {r: json.loads((current / f"quality32k-r{r}" / "quality-summary.json").read_text())
             for r in timing_roots}

plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.6), sharex=True, sharey=True)
colors = {4: "#0072B2", 8: "#D55E00", 16: "#009E73"}
print("phase,refresh,rows,perplexity_change_pct,delta_nll_mean,delta_nll_p95,"
      "delta_nll_p99,delta_nll_max,kl_mean,kl_p95,kl_p99,kl_max,top1_mismatches")
for ax, phase, frontier, title in zip(
        axes, ["cold", "warm"], ["32768", "33280"],
        ["Fresh 32K context", "Warm context after 512-token append"]):
    points = [(0.0, 0.0)]
    ax.scatter(0, 0, color="#333333", marker="s", s=38, zorder=4)
    ax.annotate("Ordinary control", (0, 0), xytext=(7, 16), textcoords="offset points")
    for refresh in timing_roots:
        timing = timings[refresh]
        paired = timing["paired"][f"{phase}_decode_steady_s"]["percent"]
        quality = qualities[refresh]["frontiers"][frontier]
        if paired["n"] != 5 or quality["rows"] != 512:
            raise ValueError(f"refresh {refresh}, {phase}: expected five timing pairs and 512 quality rows")
        for arm in ["baseline", "candidate"]:
            if timing["arms"][arm][f"{phase}_decode_tokens"]["samples"] != [128] * 5:
                raise ValueError(f"refresh {refresh}, {phase}, {arm}: expected 128-token timing windows")
        x = math.expm1(quality["delta_nll"]["mean"]) * 100
        y = -paired["mean"]
        points.append((x, y))
        ax.errorbar(x, y, yerr=paired["sd"], fmt="o", color=colors[refresh],
                    markersize=7, capsize=4, linewidth=1.5, zorder=3)
        offset = {("cold", 8): (7, -17), ("cold", 16): (-7, 9),
                  ("warm", 4): (7, -17)}.get((phase, refresh), (7, 7))
        ax.annotate(f"Refresh {refresh}", (x, y), xytext=offset,
                    textcoords="offset points", color=colors[refresh],
                    ha="right" if offset[0] < 0 else "left")
        values = [phase, refresh, quality["rows"], f"{x:.6g}"]
        values += [f"{quality[metric][stat]:.6g}" for metric in ["delta_nll", "kl"]
                   for stat in ["mean", "p95", "p99", "max"]]
        print(",".join(map(str, values + [quality["top1_mismatches"]])))

    # Lower perplexity change and greater time saving are preferable. This uses
    # point estimates only; it does not assert statistical dominance.
    frontier_points = sorted(p for p in points if not any(
        q[0] <= p[0] and q[1] >= p[1] and q != p for q in points))
    if len(frontier_points) > 1:
        ax.plot(*zip(*frontier_points), "--", color="#777777", linewidth=1, zorder=1)
    ax.axhline(0, color="#aaaaaa", linewidth=.8)
    ax.axvline(0, color="#aaaaaa", linewidth=.8)
    ax.set(title=title, xlabel="Teacher perplexity change (%) · lower is better")
    ax.grid(alpha=.18)
    ax.margins(x=.25, y=.22)
axes[0].set_ylabel("Steady decode time saved (%) · higher is better")
fig.suptitle("Resident refresh: speed and fixed-history quality", y=.97, fontsize=15)
fig.text(.5, .917, "Runtime 1b692d25 · GPU routing, address reuse and parallel FFN · fixed 64 GB cache",
         ha="center", fontsize=10)
fig.legend(handles=[Line2D([], [], color="#333333", marker="s", linestyle="none", label="Ordinary CPU cap-five control"),
                    Line2D([], [], color="#777777", linestyle="--", label="Pareto frontier of point estimates")],
           loc="lower center", bbox_to_anchor=(.5, .205), ncol=2, frameon=False)
notes = [
    "Y: mean paired time saving ± 1 sample SD, five alternating pairs; steady decode excludes token 1 of each 128-token window.",
    "X: 100 × [exp(mean delta-NLL) − 1], 512 teacher tokens per phase; dependent rows have no plotted uncertainty. Tails are reported separately.",
    "Quality and timing use separate 512- and 128-token windows, so warm expert-cache histories differ; points compare configurations, not identical windows.",
    "Refresh 8 timing reuses question 11 combined32k at identical settings; refresh 4/16 timing and all longer quality checks are from question 12.",
    "Upper left is preferable. The frontier implies neither statistical dominance nor dominance on KL or top-choice changes; perplexity is not task accuracy.",
]
for y, note in zip([.164, .132, .100, .068, .036], notes):
    fig.text(.06, y, note, fontsize=8.5)
fig.subplots_adjust(left=.08, right=.97, bottom=.33, top=.84, wspace=.10)
fig.savefig(args.output)
print(f"Wrote {args.output}")
