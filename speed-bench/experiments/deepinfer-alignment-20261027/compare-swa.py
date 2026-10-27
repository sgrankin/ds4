#!/usr/bin/env python3
"""Reuse the fixed-teacher comparison; keep checkpoint mechanics separate."""
import importlib.util
import csv
import json
from pathlib import Path
import statistics
import sys

helper = Path(__file__).resolve().parents[1] / "decoder-replay-hour47-20261020/analyze.py"
spec = importlib.util.spec_from_file_location("decoder_analysis", helper)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
root = Path(sys.argv[1])
records = json.loads((root / "result.json").read_text())
arms = [analysis.read_arm(root / row["directory"]) for row in records]
wall = lambda arm: sum(float(t["greedy_sync_ms"]) + float(t["greedy_ms"]) for t in arm["timing"].values())
controls = [arm for record, arm in zip(records, arms) if record["mode"] == "exact"]
bracket = statistics.mean(wall(arm) for arm in controls) if len(controls) == 2 else None
summary = []
for record, arm in zip(records, arms):
    rows = analysis.paired(arms[0], arm)
    formats = {row["case"]: row for row in csv.DictReader((arm["path"] / "formats.csv").open())}
    cases = []
    for name, timing in arm["timing"].items():
        paired = [row for row in rows if row["case"] == name]
        stored = formats[name]
        elapsed = float(timing["greedy_sync_ms"]) + float(timing["greedy_ms"])
        native = statistics.mean(float(control["timing"][name]["greedy_sync_ms"]) +
                                 float(control["timing"][name]["greedy_ms"]) for control in controls) if bracket is not None else None
        cases.append(dict(case=name, practical_ms=elapsed, sync_ms=float(timing["greedy_sync_ms"]),
                          decode_ms=float(timing["greedy_ms"]), bracket_mean_ms=native,
                          relative_to_bracket=elapsed / native - 1 if native is not None else None,
                          mean_delta_nll=statistics.mean(row["delta_nll"] for row in paired),
                          mean_kl=statistics.mean(row["kl"] for row in paired),
                          full_bytes=int(stored["full_bytes"]), blocks_bytes=int(stored["blocks_bytes"]),
                          omitted_bytes=int(stored["full_bytes"]) - int(stored["blocks_bytes"]),
                          prefix_tokens=int(stored["prefix_tokens"]),
                          answer=next(answer for answer in arm["answers"] if answer["case"] == name)))
    equal = None
    if all(row["mode"] == "recover" for row in records):
        equal = all((arms[0]["path"] / f'{case["name"]}-recovered-frontier.bin').read_bytes() ==
                    (arm["path"] / f'{case["name"]}-recovered-frontier.bin').read_bytes() for case in arm["meta"]["cases"])
        assert equal and all(row["byte_equal"] for row in rows), "Poisoned and ordinary recovery logits differ"
    summary.append(dict(directory=record["directory"], mode=record["mode"], verify=record["verify"],
                        practical_ms=wall(arm), relative_to_opening=wall(arm) / wall(arms[0]) - 1,
                        bracket_mean_ms=bracket, relative_to_bracket=wall(arm) / bracket - 1 if bracket is not None else None,
                        mean_delta_nll=statistics.mean(row["delta_nll"] for row in rows),
                        mean_kl=statistics.mean(row["kl"] for row in rows),
                        byte_equal_rows=sum(row["byte_equal"] for row in rows),
                        correct=sum(answer["exact"] and answer["stopped"] for answer in arm["answers"]),
                        case_count=len(cases), cases=cases, poisoned_frontier_equal=equal))
timing = "diagnostic only" if all(row["mode"] == "recover" for row in records) else "greedy_sync_ms + greedy_ms; verify excluded"
teacher_trace = all(arm["meta"].get("teacher_trace", False) for arm in arms)
print(json.dumps(dict(reference=records[0]["directory"], timing=timing,
                      teacher_trace=teacher_trace,
                      teacher="baseline reasoning/answer tokens plus EOS" if teacher_trace else "short answer-only forced tokens at reasoning frontier",
                      arms=summary), indent=2))
