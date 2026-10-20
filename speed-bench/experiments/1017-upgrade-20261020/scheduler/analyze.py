"""Recompute the four-arm scheduler screen and draw its observed tradeoff."""
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[3] / ".workspaces/upgrade-20261020/scheduler-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.loads((HERE / "result.json").read_text())
fixture = json.loads((HERE / "fixture.json").read_text())
rows = []
for arm in data["arms"]:
    a, b = arm["requests"]["A"], arm["requests"]["B"]
    times = [event["received_s"] for event in b["content_events"]]
    gap = max(right - left for left, right in zip(times, times[1:]) if right >= a["started_s"])
    assert abs(gap - arm["B_max_content_gap_after_A_start_seconds"]) < 1e-9
    rows.append({"arm": arm["arm"], "quantum": arm["mixed_prefill_quantum"],
                 "A_seconds": a["seconds"], "A_ttft_seconds": a["ttft_seconds"],
                 "B_seconds": b["seconds"], "B_ttft_seconds": b["ttft_seconds"],
                 "B_max_content_gap_seconds": gap, "arm_seconds": arm["arm_seconds"],
                 "A_correct": a["content"].strip() == fixture["expected_A"],
                 "B_correct": json.loads(b["content"]) == fixture["expected_B"],
                 "both_completed_stop": all(r["done"] and r["finish_reason"] == "stop"
                                             and r["error"] is None for r in [a, b])})
opening, closing = rows[0], rows[-1]
baseline = (opening["A_seconds"] + closing["A_seconds"]) / 2
gap_baseline = (opening["B_max_content_gap_seconds"] + closing["B_max_content_gap_seconds"]) / 2
for row in rows:
    row["A_saving_vs_128_mean_percent"] = 100 * (1 - row["A_seconds"] / baseline)
    row["B_gap_change_vs_128_mean_percent"] = 100 * (row["B_max_content_gap_seconds"] / gap_baseline - 1)
summary = {"rows": rows, "A_128_bracket_mean_seconds": baseline,
           "A_128_closing_drift_percent": 100 * (closing["A_seconds"] / opening["A_seconds"] - 1),
           "B_128_gap_mean_seconds": gap_baseline,
           "B_128_gap_closing_drift_percent": 100 * (closing["B_max_content_gap_seconds"] / opening["B_max_content_gap_seconds"] - 1),
           "answers_passed": sum(r["A_correct"] + r["B_correct"] for r in rows),
           "answers_total": 8, "all_completed_stop": all(r["both_completed_stop"] for r in rows),
           "dominated_quantum": 512, "interactive_default_unchanged": 128,
           "scope": "Ordered one-fixture mixed-prefill tradeoff; no general throughput claim. SSE gaps are client content-event gaps, not device token latency."}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (HERE / "curve.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
for row, label, color, offset in zip(rows, ["128 opening", "512 (dominated)", "2048", "128 closing"],
                                    ["#2674ae", "#858585", "#d17719", "#2674ae"],
                                    [(8, 8), (-100, -20), (9, -19), (-70, -21)]):
    ax.scatter(row["A_seconds"], row["B_max_content_gap_seconds"], s=78, color=color,
               marker="X" if row["quantum"] == 512 else "o", zorder=3)
    ax.annotate(label, (row["A_seconds"], row["B_max_content_gap_seconds"]),
                xytext=offset, textcoords="offset points", fontsize=10)
ax.set(xlabel="Long A: request completion (s; lower is better)",
       ylabel="Active B: longest content gap (s; lower is better)",
       title="Mixed-prefill quantum: observed latency tradeoff\nAll 8 answers pass; order 128 → 512 → 2048 → 128")
ax.grid(alpha=.22); ax.margins(x=.16, y=.19)
fig.savefig(HERE / "pareto.png", dpi=160)
fig.savefig(HERE / "pareto.svg")
plt.close(fig)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
