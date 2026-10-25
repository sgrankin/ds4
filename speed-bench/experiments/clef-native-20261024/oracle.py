#!/usr/bin/env python3
"""Offline numerical/latency oracle using the checkpoint's pinned MLX reference.

This is qualification tooling, never part of native serving. Run under the
existing pressure guard, one model job at a time. Raw tensors stay local.
"""
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--omni", action="store_true")
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np

    spec = importlib.util.spec_from_file_location("clef_mlx_reference", args.model / "clef_mlx.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = ref.load(args.model, head_dtype=mx.float32) if args.omni else ref.load(args.model, head_dtype=mx.float32, backend="lm")
    def forward(enc):
        return model._hidden(enc) if args.omni else model._hidden(mx.array(enc.input_ids, dtype=mx.int32), enc.media)
    load_seconds = time.perf_counter() - start
    first_record = json.loads(next(line for line in args.requests.read_text().splitlines() if line.strip()))
    arbitrary = dataclasses.replace(ref.encode_record(model.tokenizer, first_record, truncate=False), input_ids=tuple(range(1, 17)), questions=())
    hidden = forward(arbitrary)
    mx.eval(hidden)
    np.asarray(hidden.astype(mx.float32)).tofile(args.output / "arbitrary-16-hidden.f32")
    save_json(args.output / "arbitrary-16-ids.json", list(range(1, 17)))
    rows = []
    for index, line in enumerate(args.requests.read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        directory = args.output / f"case-{index:02d}"
        directory.mkdir(exist_ok=True)
        save_json(directory / "record.json", record)
        enc = ref.encode_record(model.tokenizer, record, truncate=False)
        encoding = dataclasses.asdict(enc)
        encoding.pop("media", None)
        save_json(directory / "encoding.json", encoding)
        save_json(directory / "ids.json", enc.input_ids)
        np.asarray(enc.input_ids, dtype=np.uint32).tofile(directory / "ids.u32")
        ids = mx.array(enc.input_ids, dtype=mx.int32)
        hidden = forward(enc)
        logits = model.head(hidden, ids, enc, model._lexical)
        mx.eval(hidden, logits)
        np.asarray(hidden.astype(mx.float32)).tofile(directory / "hidden.f32")
        lexical_ids = [token for q in enc.questions for a, b in q.option_spans for token in enc.input_ids[a:b]]
        lexical = model._lexical(mx.array(lexical_ids, dtype=mx.int32))
        mx.eval(lexical)
        np.asarray(lexical.astype(mx.float32)).tofile(directory / "lexical.f32")
        save_json(directory / "lexical.ids.json", lexical_ids)
        output = {q.question_id: dict(zip(q.option_ids, mx.softmax(lg.astype(mx.float32)).tolist()))
                  for q, lg in zip(enc.questions, logits)}
        save_json(directory / "probabilities.json", output)
        save_json(directory / "logits.json", [lg.tolist() for lg in logits])
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            h = forward(enc)
            result = model.head(h, ids, enc, model._lexical)
            mx.eval(result)
            timings.append(time.perf_counter() - start)
        # Separate synchronized passes expose stages without changing the
        # combined warm loop above or relying on a new profiling framework.
        backbone_seconds, head_seconds = [], []
        for _ in range(4):
            start = time.perf_counter()
            h = forward(enc)
            mx.eval(h)
            backbone_end = time.perf_counter()
            result = model.head(h, ids, enc, model._lexical)
            mx.eval(result)
            end = time.perf_counter()
            backbone_seconds.append(backbone_end - start)
            head_seconds.append(end - backbone_end)
        row = {"case": index, "tokens": len(enc.input_ids), "hidden_shape": list(hidden.shape),
               "question_count": len(enc.questions), "median_seconds": statistics.median(timings),
               "seconds": timings, "probabilities": output,
               "median_backbone_seconds": statistics.median(backbone_seconds),
               "median_head_seconds": statistics.median(head_seconds),
               "backbone_seconds": backbone_seconds, "head_seconds": head_seconds}
        rows.append(row)
        save_json(args.output / "summary.json", {"load_seconds": load_seconds,
                  "head_dtype": "float32", "backend": "mlx-vlm" if args.omni else "mlx-lm", "cases": rows})
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
