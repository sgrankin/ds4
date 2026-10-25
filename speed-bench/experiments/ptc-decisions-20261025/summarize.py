#!/usr/bin/env python3
"""Summarize completed scenario files without changing the recorded baseline.

Optional --ordinary-clean replaces ordinary repeat0 with the measured clean
replacement; repeat1 stays from the original reverse-order pass. --plot uses
Matplotlib only for offline analysis after model timing has stopped.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


ARMS = ("main_only", "hosted", "ptc")
LABELS = {"main_only": "Main alone", "hosted": "Hosted helper", "ptc": "Code + helper"}


def correct_fields(quality):
    answer, expected = quality.get("answer"), quality["expected"]
    if not isinstance(answer, dict):
        return False
    ids = answer.get("ids")
    return (isinstance(ids, list) and all(isinstance(item, str) for item in ids)
        and sorted(ids) == sorted(expected["ids"])
        and answer.get("count") == expected["count"]
        and answer.get("max_charge_cents") == expected["max_charge_cents"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--ordinary-clean", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    raw = json.loads(args.results.read_text())
    rows = list(raw["measurements"])
    quality = {item["name"]: item for item in raw["quality"]}
    if args.ordinary_clean:
        clean = json.loads(args.ordinary_clean.read_text())
        rows = [row for row in rows if not (row["case"] == "ordinary" and row["repeat"] == 0)]
        rows += [row for row in clean["measurements"] if row["case"] == "ordinary"]
        quality.update({item["name"]: item for item in clean["quality"]})
    for row in rows:
        row["source_correct"] = row["correct"]
        row["correct"] = correct_fields(quality[row["name"]])
    summary = []
    for case in ("ordinary", "favorable"):
        for arm in ARMS:
            group = [row for row in rows if row["case"] == case and row["arm"] == arm]
            if not group:
                continue
            summary.append({"case": case, "arm": arm, "samples": len(group),
                "median_wall_seconds": statistics.median(row["wall_seconds"] for row in group),
                "min_wall_seconds": min(row["wall_seconds"] for row in group),
                "max_wall_seconds": max(row["wall_seconds"] for row in group),
                "median_main_input_tokens": statistics.median(row["main_input_tokens"] for row in group),
                "median_main_output_tokens": statistics.median(row["main_output_tokens"] for row in group),
                "complete_answers": sum(row["status"] == "completed" for row in group),
                "correct_fields": sum(row["correct"] for row in group),
                "valid_json": sum(row["valid_json"] for row in group),
                "program_calls": sum(row["program_calls"] for row in group),
                "native_calls": sum(row["native_calls"] for row in group),
                "native_successes": sum(row["native_successes"] for row in group),
                "program_errors": sum(row["program_errors"] for row in group)})
    output = args.output or args.results.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"source": str(args.results),
        "ordinary_clean": str(args.ordinary_clean) if args.ordinary_clean else None,
        "groups": summary, "samples": rows, "source_errors": raw["errors"]}, indent=2) + "\n")
    if summary:
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    for group in summary:
        print(f"{group['case']:10} {group['arm']:10} n={group['samples']} "
            f"median={group['median_wall_seconds']:.3f}s "
            f"input/output={group['median_main_input_tokens']:g}/{group['median_main_output_tokens']:g} "
            f"fields={group['correct_fields']}/{group['samples']} json={group['valid_json']}/{group['samples']} "
            f"program/native/success={group['program_calls']}/{group['native_calls']}/{group['native_successes']}")
    if args.plot:
        plot(summary, output)


def plot(groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cases = [case for case in ("ordinary", "favorable") if any(group["case"] == case for group in groups)]
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.5), squeeze=False)
    for axis, case in zip(axes[0], cases):
        subset = [group for group in groups if group["case"] == case]
        for position, group in enumerate(subset):
            correct = group["correct_fields"] == group["samples"]
            clean = group["valid_json"] == group["samples"]
            color = "#348b66" if correct and clean else "#d6a13c" if correct else "#b95c57"
            middle = group["median_wall_seconds"]
            axis.barh(position, middle, color=color, height=.62,
                xerr=[[middle - group["min_wall_seconds"]], [group["max_wall_seconds"] - middle]],
                error_kw={"capsize": 3, "linewidth": 1})
            status = "incomplete" if group["complete_answers"] < group["samples"] else f"fields {group['correct_fields']}/{group['samples']}"
            axis.text(middle + .3, position, f"{middle:.2f}s · {status}", va="center", fontsize=9)
        axis.set_yticks(range(len(subset)), [LABELS[group["arm"]] for group in subset])
        axis.invert_yaxis()
        axis.set_xlim(0, max(group["max_wall_seconds"] for group in subset) * 1.45)
        axis.set_xlabel("Whole Response wall time (seconds)")
        axis.set_title("Four visible records" if case == "ordinary" else "24 records; helper/code notes bound")
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Natural support triage: latency and final-answer fields", fontsize=14)
    fig.legend(handles=[Patch(color="#348b66", label="Correct fields + valid JSON"),
        Patch(color="#d6a13c", label="Correct fields + malformed JSON"),
        Patch(color="#b95c57", label="No correct final fields")], loc="lower center", ncol=3, fontsize=9)
    fig.text(.5, .075, "Whiskers show the observed sample range. Incomplete responses are not successful-task speedups.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .91))
    fig.savefig(output / "natural-scenarios.png", dpi=180)
    fig.savefig(output / "natural-scenarios.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
