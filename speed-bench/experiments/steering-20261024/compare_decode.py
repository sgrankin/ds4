#!/usr/bin/env python3
"""Compare teacher-forced decoder dumps; NumPy handles the numerical work."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def distribution(logits):
    values = logits.astype(np.float64)
    maximum = values.max()
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def summary(rows):
    if not rows:
        return None
    return {
        "positions": len(rows),
        "mean_delta_nll_nats": float(np.mean([r["delta_nll_nats"] for r in rows])),
        "mean_forward_kl_nats": float(np.mean([r["forward_kl_nats"] for r in rows])),
        "max_forward_kl_nats": max(r["forward_kl_nats"] for r in rows),
        "mean_total_variation": float(np.mean([r["total_variation"] for r in rows])),
        "top1_agreement": float(np.mean([r["top1_equal"] for r in rows])),
        "max_abs_logit_delta": max(r["max_abs_logit_delta"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference decode_FRONTIER.json")
    parser.add_argument("candidate", type=Path, help="Candidate decode_FRONTIER.json")
    parser.add_argument("out", type=Path, help="Output prefix for .json and .csv")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    for key in ("model", "ctx", "vocab", "frontier_tokens", "prompt_token_ids", "steps"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Different {key}; compare identical teacher-forced inputs")
    for metadata in (reference, candidate):
        if metadata["dtype"] != "float32" or metadata["byte_order"] != "little":
            raise ValueError("Expected little-endian float32 logits")
    rows = []
    for step in reference["steps"]:
        filename = step["logits_file"]
        p = np.fromfile(args.reference.parent / filename, dtype="<f4")
        q = np.fromfile(args.candidate.parent / filename, dtype="<f4")
        if p.size != reference["vocab"] or q.size != p.size:
            raise ValueError(f"Wrong full-vocabulary size: {filename}")
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError(f"Nonfinite logits: {filename}")
        lp, lq = distribution(p), distribution(q)
        pp, pq = np.exp(lp), np.exp(lq)
        target = step["target"]
        rows.append({
            "step": step["step"], "position": step["position"], "target": target,
            "reference_nll_nats": float(-lp[target]),
            "candidate_nll_nats": float(-lq[target]),
            "delta_nll_nats": float(lp[target] - lq[target]),
            "forward_kl_nats": max(0.0, float(np.sum(pp * (lp - lq)))),
            "total_variation": float(np.abs(pp - pq).sum() / 2),
            "top1_equal": bool(p.argmax() == q.argmax()),
            "logits_exact": bool(np.array_equal(p, q)),
            "max_abs_logit_delta": float(np.abs(p.astype(np.float64) - q).max()),
        })
    result = {
        "reference": str(args.reference), "candidate": str(args.candidate),
        "all_positions": summary(rows),
        "prefill_control": summary([r for r in rows if r["step"] == 0]),
        "decoder_positions": summary([r for r in rows if r["step"] > 0]),
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
