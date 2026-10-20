#!/usr/bin/env python3
"""Analyze small HTTP result JSON. Plot only after ROOT releases timings."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
GROUPS = {
    "source_warmup": ("A-source", "B-source"),
    "ordinary": ("A-edit1", "B-edit1"),
    "stress": ("A-edit2", "B-edit2"),
    "all_edited": ("A-edit1", "B-edit1", "A-edit2", "B-edit2"),
    "all_task": ("A-source", "B-source", "A-edit1", "B-edit1", "C-turnover", "A-edit2", "B-edit2"),
    "excluding_source": ("A-edit1", "B-edit1", "C-turnover", "A-edit2", "B-edit2"),
}


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def seconds(step):
    return step.get("seconds_including_edit", step.get("seconds"))


def inspect_arm(root, name, launch):
    directory = root / name
    phases = {p: load(directory / "http" / p / "result.json") for p in ("before", "after")}
    steps = [step for phase in phases.values() if phase for step in phase.get("steps", [])]
    by_name = {step["step"]: step for step in steps}
    groups = {}
    for group, names in GROUPS.items():
        selected = [by_name[n] for n in names if n in by_name]
        values = [seconds(step) for step in selected]
        available = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        groups[group] = {
            "seconds": math.fsum(values) if len(selected) == len(names) and available else None,
            "observed_seconds": math.fsum(values) if available else None,
            "recorded": len(selected), "expected": len(names),
            "correct": sum(step.get("correct") is True for step in selected),
            "passed": sum(step.get("pass") is True for step in selected),
            "grafted_steps": sum(any("suffix APPROXIMATE slot=" in line
                for line in step.get("suffix_log_lines", [])) for step in selected)}
    detailed = []
    for step in steps:
        lines = step.get("suffix_log_lines", [])
        grafts = []
        for line in lines:
            if "suffix APPROXIMATE slot=" not in line:
                continue
            fields = {k: int(v) for k, v in re.findall(
                r"\b(slot|reused|fresh|graft_fresh|source_ssd|source_approximate)=(\d+)", line)}
            grafts.append(fields | {"line": line})
        detailed.append({"step": step["step"], "seconds": seconds(step),
            "correct": step.get("correct"), "pass": step.get("pass"),
            "fresh_read": step.get("fresh_read"), "complete": step.get("complete"),
            "answer": step.get("answer"), "text": step.get("text"), "gold": step.get("gold"),
            "errors": step.get("errors"), "grafts": grafts, "suffix_log_lines": lines,
            "exact_hit_log_lines": step.get("exact_hit_log_lines", []),
            "slot_eviction_log_lines": step.get("slot_eviction_log_lines", [])})
    startups = [load(directory / (p + "-startup.json")) for p in ("before", "after")]
    startup_values = [b["seconds"] for b in startups if b]
    executions = {p: load(directory / (p + "-execution.json")) for p in ("before", "after")}
    complete = (all(phases.values()) and set(by_name) == set(GROUPS["all_task"])
                and launch is not None and load(directory / "result.json") is not None)
    return {"arm": name, "complete": complete, "launcher": launch,
        "groups": groups, "steps": detailed, "server_execution": executions,
        "server_startups": startups,
        "startup_restart_seconds": math.fsum(startup_values) if len(startup_values) == 2 else None,
        "driver_phase_seconds": math.fsum(b["seconds"] for b in phases.values() if b),
        "graft_count": sum(len(step["grafts"]) for step in detailed),
        "approximate_source_grafts": sum(g.get("source_approximate") == 1
            for step in detailed for g in step["grafts"]),
        "fresh_reads": sum(step["fresh_read"] is True for step in detailed)}


def plot(summary, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = summary["arms"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    for axis, group, title in zip(axes, ("ordinary", "stress", "all_task"),
            ("Ordinary edited turns", "Changed files after restart", "All tasks, cold sources included")):
        for i, arm in enumerate(arms):
            point = arm["groups"][group]
            value = point["seconds"] if point["seconds"] is not None else point["observed_seconds"]
            bar = axis.bar(i, value or 0, color="#4477aa" if arm["complete"] else "#aaaaaa")
            if point["seconds"] is None:
                bar[0].set_hatch("//")
            label = (f'{point["passed"]}/{point["expected"]} pass\n'
                     f'graft {point["grafted_steps"]}/{point["expected"]}')
            if not arm["complete"]:
                label += "\nincomplete"
            axis.annotate(label, (i, value or 0), xytext=(0, 4), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(len(arms)), [a["arm"] for a in arms], rotation=25)
        axis.set_ylabel("Seconds (failed/fallback points retained)")
        axis.margins(y=.35)
    fig.suptitle("HTTP task latency: generated histories differ; no HTTP NLL/KL claim")
    fig.tight_layout()
    fig.savefig(output / "http-latency.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE / "http")
    p.add_argument("--output", type=Path, default=HERE / "http-analysis")
    p.add_argument("--plot", action="store_true", help="Use the existing plotting venv after final timings.")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    execution = load(a.input / "sweep-execution.json") or {}
    order = execution.get("order", ["tail64", "tail256", "tail128", "native"])
    launches = {row["arm"]: row for row in (load(a.input / "sweep-result.json") or [])}
    arms = [inspect_arm(a.input, name, launches.get(name)) for name in order]
    complete = all(arm["complete"] for arm in arms)
    native = next((arm for arm in arms if arm["arm"] == "native"), None)
    rows = []
    for arm in arms:
        row = {"arm": arm["arm"], "complete": arm["complete"],
            "pass_count": arm["groups"]["all_task"]["passed"],
            "correct_count": arm["groups"]["all_task"]["correct"],
            "fresh_reads": arm["fresh_reads"], "grafts": arm["graft_count"],
            "approximate_source_grafts": arm["approximate_source_grafts"],
            "startup_restart_seconds": arm["startup_restart_seconds"],
            "launcher_seconds": (arm["launcher"] or {}).get("seconds"),
            "launcher_returncode": (arm["launcher"] or {}).get("returncode")}
        for group, point in arm["groups"].items():
            row[group + "_seconds"] = point["seconds"]
            baseline = native["groups"][group]["seconds"] if native else None
            ratio = point["seconds"] / baseline if complete and baseline and point["seconds"] is not None else None
            row[group + "_native_ratio"] = ratio
            row[group + "_saving_percent"] = (1-ratio)*100 if ratio is not None else None
        rows.append(row)
    summary = {"complete": complete, "execution": execution, "arms": arms,
        "interpretation": "One reverse-order HTTP sweep, separate from core NLL curve. Ratios appear only after all planned arms complete. Failures and fallback remain included. Edited sums include file/context edits; step sums exclude server startup/restart; launcher wall includes them. Any sent Responses seed is ignored and is not a reproducibility guarantee."}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if a.plot:
        plot(summary, a.output)
    print(json.dumps({"complete": complete, "arms": len(arms), "output": str(a.output)}))


if __name__ == "__main__":
    main()
