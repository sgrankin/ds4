#!/usr/bin/env python3
"""Measure the original Flint harness and optionally capture eight teacher rows.

Run with Flint's MLX-enabled Python and rec_mlx importable. Output is local raw
evidence: two prompt fixtures, three fresh-prefix repeats in greedy/default
modes, and optional full-vocabulary FP32 teacher logits. No native jobs launch.
"""
import argparse
import json
import os
from pathlib import Path
import sys


REPEATS = 3
GENERATED_TOKENS = 128
CONTEXT_LIMIT = 262144
TEACHER_POSITIONS = (16, 32, 64, 96)

REQUESTS = (
    (
        "short",
        [{
            "role": "user",
            "content": "Write a Python function that merges overlapping integer intervals. Explain the edge cases and give three tests.",
        }],
    ),
    (
        "context",
        [
            {
                "role": "system",
                "content": "You help maintain a local software project. Use the notes when answering.",
            },
            {
                "role": "user",
                "content": (
                    "Project notes: store durable conversation history separately from inference state. "
                    "Keep model jobs serialized and measure prefill and decode separately. "
                    "Prefer simple native components and explicit ownership.\n"
                ) * 24 + "Propose a concise implementation plan for adding a second model backend.",
            },
        ],
    ),
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def measure(engine, sampling_type, output):
    prompts = {}
    continuations = {}
    for name, messages in REQUESTS:
        ids = engine.prompt(messages)
        prompts[name] = ids
        write_json(output / f"{name}-prompt.json", {"messages": messages, "token_ids": ids})
        modes = (
            ("greedy", sampling_type(temperature=0, max_tokens=GENERATED_TOKENS), None),
            ("default", sampling_type(max_tokens=GENERATED_TOKENS), engine.constraints.tools([])),
        )
        for mode, sampling, grammar in modes:
            for repeat in range(REPEATS):
                # Fresh prefix every time; keep compiled kernels/weight banks.
                engine._prompt_cache = None
                engine._prefix_preparation = None
                events = list(engine.generate_tokens(ids, sampling, grammar=grammar))
                metrics = events[-1]["metrics"]
                write_json(output / f"{name}-{mode}-{repeat}.json", {
                    "case": name, "mode": mode, "repeat": repeat, "events": events,
                })
                print(json.dumps({"case": name, "mode": mode, "repeat": repeat, **metrics}), flush=True)
                if mode == "greedy" and repeat == 0:
                    continuations[name] = metrics["token_ids"]
    return prompts, continuations


def teacher_rows(engine, prompts, continuations, output):
    import numpy as np

    suite = []
    for name, _ in REQUESTS:
        generated = continuations[name]
        if len(generated) <= max(TEACHER_POSITIONS):
            raise ValueError(f"{name}: quality capture needs at least 97 generated tokens")
        for position in TEACHER_POSITIONS:
            suite.append({
                "id": f"{name}-{position}",
                "token_ids": prompts[name] + generated[:position],
                "target_token": generated[position],
            })
    write_json(output / "teacher-suite.json", suite)
    engine._prompt_cache = None
    engine._prefix_preparation = None
    mx = engine.mx
    mx.synchronize()
    mx.clear_cache()
    for case in suite:
        session = engine.runtime.make_session(max_new_tokens=1)
        logits = session.prefill(case["token_ids"])
        mx.eval(logits)
        row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
        if row.size != 150144 or not np.isfinite(row).all():
            raise ValueError(f"{case['id']}: expected 150144 finite vocabulary logits")
        row.tofile(output / f"teacher-reference-{case['id']}.f32")
        print(json.dumps({
            "case": case["id"], "tokens": len(case["token_ids"]),
            "argmax": int(row.argmax()), "target_token": case["target_token"],
        }), flush=True)
        del session, logits, row
        mx.synchronize()
        mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Original Flint INT4 model directory")
    parser.add_argument("--output", required=True, type=Path, help="Local raw evidence directory")
    parser.add_argument("--quality", action="store_true", help="Also capture eight teacher-forced vocabulary rows")
    args = parser.parse_args()

    # This must be set before rec_mlx imports MLX; it affects host/GPU handoff.
    os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    from rec_mlx.engine import Engine, Sampling

    args.output.mkdir(parents=True, exist_ok=True)
    engine = Engine(args.model, args.model / "flint-int4.json", context_limit=CONTEXT_LIMIT)
    write_json(args.output / "run.json", {
        "command": sys.argv,
        "model": str(args.model.resolve()),
        "repeats": REPEATS,
        "max_generated_tokens": GENERATED_TOKENS,
        "context_limit": CONTEXT_LIMIT,
        "MLX_METAL_FAST_SYNCH": os.environ["MLX_METAL_FAST_SYNCH"],
        "startup": engine.startup,
    })
    print(json.dumps({"startup": engine.startup}), flush=True)
    prompts, continuations = measure(engine, Sampling, args.output)
    if args.quality:
        teacher_rows(engine, prompts, continuations, args.output)


if __name__ == "__main__":
    main()
