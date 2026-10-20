#!/usr/bin/env python3
"""Grade completed generation logs; no model, tokenizer, or grader execution."""
import csv
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "generation"
FOOTER = re.compile(
    r"GENERATION_SUMMARY tokens=(\d+) cap=(\d+) stop=(cap|interrupt|stop_token) "
    r"prefill_ms=([0-9.]+) decode_ms=([0-9.]+)"
)
RATE = re.compile(r"prefill: ([0-9.]+) (?:t/s|tok/s), generation: ([0-9.]+) (?:t/s|tok/s)")


def grade(path):
    stem = path.stem
    arm, case = stem.split("-confirm_g", 1)
    case = "confirm_g" + case
    row = dict(arm=arm, case=case, **{"pass": False}, fields_correct=False,
               failure_reason="", seconds=None, returncode=None, generated_tokens=None,
               token_cap=None, cap_status="unknown", completion_status="unknown",
               stop_reason=None, prefill_ms=None, decode_ms=None,
               prefill_tps=None, generation_tps=None)
    reasons = []
    try:
        execution = json.loads((OUT / (stem + ".execution.json")).read_text())
        row["returncode"] = execution["returncode"]
        row["seconds"] = execution["seconds"]
        stderr = (OUT / (stem + ".stderr")).read_text(errors="replace")
        rates = RATE.findall(stderr)
        if rates:
            row["prefill_tps"], row["generation_tps"] = map(float, rates[-1])
        summaries = FOOTER.findall(stderr)
        if summaries:
            tokens, cap, stop, prefill, decode = summaries[-1]
            row.update(generated_tokens=int(tokens), token_cap=int(cap), stop_reason=stop,
                       prefill_ms=float(prefill), decode_ms=float(decode))
            capped = stop == "cap" or int(tokens) >= int(cap)
            row["cap_status"] = "reached" if capped else "not_reached"
            row["completion_status"] = "complete" if stop == "stop_token" and not capped else "incomplete"
            if capped:
                reasons.append("generation cap reached")
            elif stop == "interrupt":
                reasons.append("generation interrupted")
        if row["returncode"] != 0:
            row["completion_status"] = "incomplete"
            reasons.append("process returncode " + str(row["returncode"]))
        text = path.read_text()
        start = text.find("{")
        if start < 0:
            raise ValueError("no JSON object in stdout")
        answer, end = json.JSONDecoder().raw_decode(text, start)
        if not isinstance(answer, dict):
            raise ValueError("first JSON value is not an object")
        gold = json.loads((HERE / "fixtures" / (case + ".gold.json")).read_text())["gold"]
        if case == "functional24_24_literal_unicode_note":
            paragraph = answer.get("paragraph", text[end:].strip())
            row["fields_correct"] = (
                answer.get("label") == gold["label"] and answer.get("motto") == gold["motto"]
                and isinstance(paragraph, str) and bool(paragraph.strip())
                and gold["label"] in paragraph and gold["motto"] in paragraph
            )
        elif case == "functional24_01_cursor_pages":
            # The prompt asks for a stop reason without prescribing an enum.
            # The native answer uses this equivalent plain-language spelling.
            row["fields_correct"] = (answer.get("ids") == gold["ids"]
                and answer.get("requests") == gold["requests"]
                and answer.get("stop_reason") in (gold["stop_reason"], "next_cursor is null"))
        else:
            row["fields_correct"] = all(answer.get(key) == value for key, value in gold.items())
        if not row["fields_correct"]:
            reasons.append("required factual fields differ from gold")
        row["pass"] = row["fields_correct"] and row["returncode"] == 0 and row["completion_status"] == "complete"
        if row["completion_status"] == "unknown":
            reasons.append("missing generation summary")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reasons.append(str(exc))
    row["failure_reason"] = "; ".join(reasons)
    return row


def main():
    rows = [grade(path) for path in sorted(OUT.glob("*.stdout"))]
    summary = {}
    for row in rows:
        totals = summary.setdefault(row["arm"], dict(cases=0, passes=0, factual_correct=0, incomplete=0, unknown=0))
        totals["cases"] += 1
        totals["passes"] += row["pass"]
        totals["factual_correct"] += row["fields_correct"]
        totals["incomplete"] += row["completion_status"] == "incomplete"
        totals["unknown"] += row["completion_status"] == "unknown"
        totals["wall_seconds"] = totals.get("wall_seconds", 0) + (row["seconds"] or 0)
        totals["generated_tokens"] = totals.get("generated_tokens", 0) + (row["generated_tokens"] or 0)
        totals["caps"] = totals.get("caps", 0) + (row["cap_status"] == "reached")
    OUT.mkdir(exist_ok=True)
    result = dict(scope="First JSON factual grading; unknown completion does not prove a stop token.",
                  rows=rows, by_arm=summary)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "results.csv").open("w", newline="") as fp:
        if rows:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if summary:
        try:
            os.environ['MPLCONFIGDIR'] = str(OUT / '.mplcache')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.6, 4.6))
            colors = {'native': '#777777', 'affinity': '#247ba0', 'deadline': '#df7c26'}
            for i, (arm, totals) in enumerate(sorted(summary.items())):
                kind = 'native' if arm.startswith('native') else 'deadline' if arm.startswith('deadline') else 'affinity'
                x, y = totals['wall_seconds'], totals['passes']
                ax.scatter(x, y, color=colors[kind], s=55)
                label = f"{arm}: {totals['passes']}/{totals['cases']}, caps {totals['caps']}"
                ax.annotate(label, (x,y), xytext=(8,15 if i%2 == 0 else -25),
                            textcoords='offset points', fontsize=9,
                            arrowprops={'arrowstyle':'-', 'color':'#999999', 'linewidth':.5})
            ax.set_xlabel('Summed generation process wall time (seconds)')
            ax.set_ylabel('Completed correct cases (count)')
            ax.set_title('Four new generation tasks')
            ax.set_yticks(range(max(t['cases'] for t in summary.values())+1))
            ax.set_ylim(-.5, max(t['cases'] for t in summary.values())+.5)
            ax.grid(alpha=.15)
            ax.margins(x=.25)
            fig.tight_layout()
            fig.savefig(OUT/'task_quality.png', dpi=170)
            fig.savefig(OUT/'task_quality.svg')
            plt.close(fig)
        except ImportError:
            print('Plots skipped: matplotlib is unavailable.')


if __name__ == "__main__":
    main()
