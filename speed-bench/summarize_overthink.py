#!/usr/bin/env python3
"""Summarize recorded final-answer correctness, caps, token counts, and pressure."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def summarize(directory):
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line]
    arms = collections.defaultdict(list)
    cases = collections.defaultdict(dict)
    for row in rows:
        arms[str(row["lambda"])].append(row)
        if row["lambda"] in cases[row["case"]]:
            raise ValueError("duplicate case/penalty record")
        cases[row["case"]][row["lambda"]] = row
    result = dict(directory=str(directory.resolve()), records=len(rows), arms={}, cases=[])
    for key, group in arms.items():
        result["arms"][key] = dict(cases=len(group),
            correct=sum(r["status"] == "CORRECT" for r in group),
            incorrect=sum(r["status"] == "INCORRECT" for r in group),
            invalid=sum(r["status"] == "INVALID" for r in group),
            cap_hits=sum(r["cap_hit"] for r in group),
            tokens=sum(r["generated_tokens"] for r in group),
            reasoning_tokens=sum(r["reasoning_tokens"] for r in group),
            decode_seconds=sum(r["decode_seconds"] for r in group))
    regressions, improvements = [], []
    for name, group in cases.items():
        entry = dict(case=name, arms={})
        for penalty, row in group.items():
            raw = directory / f"{name}-lambda{penalty:.1f}.txt"
            thinking = row["reasoning_tokens"] - int(row["think_closed"])
            after = row.get("overthink_after", 0)
            entry["arms"][str(penalty)] = dict(status=row["status"],
                parsed_answer=row["parsed_answer"], expected=row["expected"],
                tokens=row["generated_tokens"], reasoning_tokens=row["reasoning_tokens"],
                ordinary_generated_thinking_tokens=thinking,
                reached_late_threshold=thinking >= after,
                estimated_policy_sampling_steps=max(0, row["reasoning_tokens"] - after) if penalty else 0,
                cap_hit=row["cap_hit"], parse_reason=row["parse_reason"],
                trace_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        baseline = group.get(0.0)
        if baseline:
            for penalty, row in group.items():
                if not penalty:
                    continue
                if baseline["status"] == "CORRECT" and row["status"] != "CORRECT":
                    regressions.append(dict(case=name, penalty=penalty, status=row["status"]))
                if baseline["status"] != "CORRECT" and row["status"] == "CORRECT":
                    improvements.append(dict(case=name, penalty=penalty))
        result["cases"].append(entry)
    result.update(correctness_regressions=regressions, correctness_improvements=improvements,
                  all_records_correct=bool(rows) and all(r["status"] == "CORRECT" for r in rows),
                  timing_qualification="exploratory arm timings; cache reuse prevents a speedup claim")
    manifest_path = directory.parent / (directory.name + "-manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_arms = 2 if manifest["candidate"] is not None else 3
        result["expected_records"] = manifest["case_limit"] * expected_arms
        result["complete"] = len(rows) == result["expected_records"]
        result["configuration"] = {key: manifest[key] for key in (
            "revision", "cache_gib_requested", "max_tokens", "think_level", "suite",
            "candidate", "case_limit", "overthink_after")}
    pressure_dir = directory.parent / (directory.name + "-pressure")
    if (pressure_dir / "result.json").exists():
        result["guard"] = json.loads((pressure_dir / "result.json").read_text())
    if (pressure_dir / "pressure.jsonl").exists():
        pressure = [json.loads(line) for line in (pressure_dir / "pressure.jsonl").read_text().splitlines() if line]
        if pressure:
            result["host_pressure"] = dict(samples=len(pressure),
                minimum_free_gib=min(r["free_gb"] for r in pressure),
                maximum_wired_gib=max(r["wired_gb"] for r in pressure),
                maximum_compressor_gib=max(r["compressor_gb"] for r in pressure),
                maximum_swap_gib=max(r["swap_gb"] for r in pressure),
                maximum_kernel_level=max(r["level"] for r in pressure))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps([summarize(directory) for directory in args.directories], indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
