#!/usr/bin/env python3
"""Compare teacher-forced FP32 vocabulary dumps without loading a model.

python3 tools/flint-quality.py SUITE.json REFERENCE_PREFIX NATIVE_PREFIX
Each prefix has one PREFIX-ID.f32 file per {id, token_ids, target_token} case.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.suite.read_text())
    reference = np.stack([np.fromfile(f"{args.reference}-{case['id']}.f32", dtype="<f4") for case in cases]).astype(np.float64)
    native = np.stack([np.fromfile(f"{args.native}-{case['id']}.f32", dtype="<f4") for case in cases]).astype(np.float64)
    if reference.shape != (len(cases), 150144) or native.shape != reference.shape:
        raise ValueError("Vocabulary logits must have 150144 FP32 values per case")
    if not np.isfinite(reference).all() or not np.isfinite(native).all():
        raise ValueError("Non-finite vocabulary logits")
    logp = reference - np.logaddexp.reduce(reference, axis=1, keepdims=True)
    logq = native - np.logaddexp.reduce(native, axis=1, keepdims=True)
    rows = []
    for case, first, second, p, q in zip(cases, reference, native, logp, logq):
        target = case["target_token"]
        row = {
            "case": case["id"], "prompt_tokens": len(case["token_ids"]),
            "target_token": target,
            "reference_nll": float(-p[target]),
            "native_nll": float(-q[target]),
            "delta_nll": float(p[target] - q[target]),
            "kl_reference_to_native": float(np.sum(np.exp(p) * (p - q))),
            "top1_match": bool(np.argmax(first) == np.argmax(second)),
            "reference_top1": int(np.argmax(first)), "native_top1": int(np.argmax(second)),
            "max_logit_difference": float(np.max(np.abs(first - second))),
        }
        rows.append(row)
        print(json.dumps(row))
    summary = {"samples": len(rows), "top1_matches": sum(row["top1_match"] for row in rows)}
    for key in ("delta_nll", "kl_reference_to_native", "reference_nll", "native_nll"):
        summary[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    nll = summary["mean_reference_nll"]
    summary["mean_nll_relative_change"] = summary["mean_delta_nll"] / nll if nll else None
    summary["perplexity_ratio"] = float(np.exp(summary["mean_delta_nll"]))
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
