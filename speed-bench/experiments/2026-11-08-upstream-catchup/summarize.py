#!/usr/bin/env python3
"""Summarize the five alternating pairs produced by run.nu; retain every sample."""
import csv
import json
from pathlib import Path
import statistics
import sys


def samples(directory, arm):
    result = []
    for repeat in range(1, 6):
        run = directory / f"{repeat}-{arm}"
        with (run / "timings.csv").open() as stream:
            fresh, append = [{key: float(value) for key, value in row.items()}
                             for row in csv.DictReader(stream)]
        assert append["ctx_tokens"] == fresh["ctx_tokens"] + 512
        row = {
            "fresh_prefill_s": fresh["prefill_tokens"] / fresh["prefill_tps"],
            "append_s": append["prefill_tokens"] / append["prefill_tps"],
            "fresh_decode_s": fresh["gen_tokens"] / fresh["gen_tps"],
            "append_decode_s": append["gen_tokens"] / append["gen_tps"],
            "fresh_prefill_tps": fresh["prefill_tps"],
            "fresh_steady_tps": fresh["gen_steady_tps"],
            "append_steady_tps": append["gen_steady_tps"],
            "fresh_first_ms": fresh["gen_first_ms"],
            "append_first_ms": append["gen_first_ms"],
            "wall_s": json.loads((run / "wall.json").read_text())["wall_s"],
        }
        row["inference_s"] = sum(row[key] for key in
                                 ("fresh_prefill_s", "append_s", "fresh_decode_s", "append_decode_s"))
        result.append(row)
    return result


def describe(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "min": min(values), "max": max(values)}


directory = Path(sys.argv[1])
before, after = samples(directory, "baseline"), samples(directory, "candidate")
summary = {}
for key in before[0]:
    a, b = [row[key] for row in before], [row[key] for row in after]
    summary[key] = {"baseline": describe(a), "candidate": describe(b),
                    "mean_change_percent": 100 * (statistics.mean(b) / statistics.mean(a) - 1),
                    "paired_change_percent": [100 * (y / x - 1) for x, y in zip(a, b)]}
print(json.dumps({"directory": str(directory), "baseline": before, "candidate": after,
                  "summary": summary}, indent=2))
