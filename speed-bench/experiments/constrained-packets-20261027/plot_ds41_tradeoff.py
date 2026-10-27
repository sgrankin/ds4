#!/usr/bin/env python3
"""Plot three measured DS41 packet limits; no fit or optimality claim."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        default=Path(".workspaces/constrained-packets-20261027"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("ds41-tradeoff.svg"))
    args = parser.parse_args()
    cache = args.workspace / "matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    os.environ["XDG_CACHE_HOME"] = str(cache.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "ds41-constrained-packets-20261027"

    cases = {"strict-tool": "Strict tool (169 tokens)",
             "schema-answer": "Schema answer (183 tokens)"}
    points = {case: [] for case in cases}
    for limit, arm in [(1, "disabled-deepseek"), (3, "cap2-deepseek"),
                       (6, "enabled-deepseek")]:
        source = json.loads((args.workspace / arm / "summary.json").read_text())
        assert source["clean_shutdown"]
        for index, row in enumerate(source["requests"]):
            if row["warmup"] or row["temperature"] != 0 or row["case"] not in cases:
                continue
            phase = source["generation_phases"][index]
            assert row["quality"]
            assert phase["generated_tokens"] == row["usage"]["output_tokens"]
            points[row["case"]].append((limit, phase["decode_tps"], row["seconds"]))
    assert all(len(rows) == 3 for rows in points.values())

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.1))
    colors = ["#245d94", "#b45122"]
    for (case, rows), color in zip(points.items(), colors):
        for metric, axis in enumerate(axes, start=1):
            x = [row[0] for row in rows]
            y = [row[metric] for row in rows]
            axis.plot(x, y, marker="o", color=color, linewidth=1,
                      label=cases[case])
            for px, py in zip(x, y):
                axis.annotate(f"{py:.2f}", (px, py), xytext=(0, 8),
                              textcoords="offset points", ha="center",
                              fontsize=8, color=color)
            axis.set_xticks([1, 3, 6], ["1 (scalar)", "3 (cap2)", "6 (default)"])
            axis.set_xlim(0.6, 6.4)
            axis.grid(alpha=0.2)
            axis.set_xlabel("Configured maximum tokens per packet")
    axes[0].set_ylabel("Decode tokens/second (higher is faster)")
    axes[0].set_ylim(15.0, 16.6)
    axes[1].set_ylabel("Client request seconds (lower is faster)")
    axes[1].set_ylim(10.1, 12.1)
    figure.suptitle("DS41 constrained packet limit: three measured settings", fontsize=12)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                  bbox_to_anchor=(0.5, 0.075), ncol=2, frameon=False, fontsize=9)
    figure.text(0.5, 0.025,
                "Single T0 requests after cache restore; narrow y ranges. "
                "Lines connect observations only; no fit.\n"
                "Near-neutral/noisy results; all semantic checks pass. "
                "The first 128 generated tokens run scalar.",
                ha="center", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.16, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={"Date": None})
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
