#!/usr/bin/env python3
"""Plot individual fixed-screen requests, without fitting a tradeoff curve."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs=2, action="append", required=True, metavar=("BUDGET", "SCENARIOS_JSON"))
    parser.add_argument("--output", type=Path, default=Path(".workspaces/reference-output-20261026/tradeoff"),
                        help="Output stem; writes .svg and .png")
    parser.add_argument("--caption", default="Individual requests; no fitted curve")
    args = parser.parse_args()
    rows = []
    for label, filename in args.sample:
        budget = 0 if label == "off" else int(label)
        source = json.loads(Path(filename).read_text())
        if source.get("inline_bytes") not in (None, budget):
            raise ValueError(f"Budget {budget} disagrees with {filename}")
        for row in source["measurements"]:
            rows.append(row | {"budget": budget, "call_cap": source.get("max_tool_calls", 8)})
    if not rows:
        raise ValueError("No measured requests to plot")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / "matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = list(dict.fromkeys(row["case"] for row in rows))
    budgets = list(dict.fromkeys(row["budget"] for row in rows))
    mixed_caps = len({row["call_cap"] for row in rows}) > 1
    colors = {budget: plt.get_cmap("tab10")(index % 10) for index, budget in enumerate(budgets)}
    figure, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.4), squeeze=False)
    for axis, case in zip(axes[0], cases):
        points = [row for row in rows if row["case"] == case]
        for index, row in enumerate(points):
            passed = row["correct"] and row["valid_json"] and row["status"] == "completed"
            color = colors[row["budget"]] if passed else "#c62828"
            axis.scatter(row["program_result_bytes"], row["wall_seconds"],
                         color=color, marker="o" if passed else "X", s=70, zorder=3, clip_on=False)
            label = "off" if row["budget"] == 0 else str(row["budget"])
            if mixed_caps:
                label += f" / {row['call_cap']} calls"
            axis.annotate(label, (row["program_result_bytes"], row["wall_seconds"]),
                          xytext=(7, 7 + 8 * (index % 2)), textcoords="offset points", color=color)
        axis.set_title(f"{case.capitalize()} ({len(points)} requests)")
        axis.set_xlabel("Program result bytes sent to main model")
        axis.set_ylabel("Whole-response time (seconds)")
        axis.set_xlim(0, max(1, max(row["program_result_bytes"] for row in points) * 1.25))
        axis.set_ylim(0, max(1, max(row["wall_seconds"] for row in points) * 1.3))
        axis.grid(alpha=0.25)
    figure.suptitle("Inline output budget: emitted bytes versus latency\n"
                    + args.caption, fontsize=12)
    figure.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666",
               label="Numeric + JSON pass"),
        Line2D([], [], marker="X", color="none", markerfacecolor="#c62828", markeredgecolor="#c62828",
               label="Numeric/JSON or response failure")],
        loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.09, 1, 0.84))
    for suffix in (".svg", ".png"):
        target = args.output.with_suffix(suffix)
        figure.savefig(target, dpi=160)
        print(target)
    plt.close(figure)


if __name__ == "__main__":
    main()
