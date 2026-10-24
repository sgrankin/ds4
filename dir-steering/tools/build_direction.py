#!/usr/bin/env python3
"""Build a residual-stream steering bank from paired target/control prompts.

Project mode writes unit mean-difference directions. Add mode retains their
learned magnitudes. Zero rows select no layer; runtime add strength is divided
by the number of nonzero rows, while project strength remains per layer.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PROFILES = {
    "deepseek-v4.1-flash": (40, 5120),
    "flint": (49, 2048),
    "glm-5.3-flash": (45, 4096),
    "qwen3.8-flash-next": (48, 2560),
}


def read_prompt_file(path: Path) -> list[str]:
    """Read one prompt per non-empty line, ignoring shell-style comments."""
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def normalize(v: list[float]) -> list[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0.0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]


def selected_layers(text: str, count: int) -> set[int]:
    """Zero-based indices and inclusive ranges, e.g. 8-15,24,32-39."""
    if text == "all":
        return set(range(count))
    selected = set()
    for item in text.split(","):
        first, separator, last = item.partition("-")
        start = int(first)
        end = int(last) if separator else start
        if start < 0 or end < start or end >= count:
            raise ValueError(f"layer selection {item!r} is outside 0..{count - 1}")
        selected.update(range(start, end + 1))
    return selected


def run_capture(
    ds4: Path,
    model: Path,
    prompt: str,
    system: str,
    think: bool,
    ctx: int,
    component: str,
    n_layer: int,
    n_embd: int,
    work: Path,
    *,
    profile: str = "qwen3.8-flash-next",
) -> list[list[float]]:
    """Run ds4 once and return the last prompt-row dump for every layer."""
    prompt_path = work / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    dump_prefix = work / "dump"

    env = os.environ.copy()
    env["DS4_METAL_GRAPH_DUMP_PREFIX"] = str(dump_prefix)
    env["DS4_METAL_GRAPH_DUMP_NAME"] = component
    env["DS4_METAL_GRAPH_DUMP_POS"] = "0"
    env["DS4_METAL_GRAPH_DUMP_LAYER"] = "all"
    if profile == "qwen3.8-flash-next":
        env["DS4_QWEN4_PREFILL_CHUNK"] = str(max(ctx, 1024))

    cmd = [
        str(ds4),
        "-m", str(model),
        "--ctx", str(ctx),
        "--prompt-file", str(prompt_path),
        "-n", "1",
        "--dump-logits", str(work / "prefill-logits.json"),
    ]
    # GLM rejects the public chunk override. Its native 2048-row default covers
    # the short prompt captures; the other profiles accept this CLI option.
    if profile != "glm-5.3-flash":
        cmd += ["--prefill-chunk", str(max(ctx, 1024))]
    # Match the preceding qualification's memory/cache settings. Capturing a
    # direction must not accidentally request a fully resident huge model.
    if profile == "deepseek-v4.1-flash":
        cmd += ["--ssd-streaming", "--ssd-streaming-cache-experts", "32GB"]
        env.update(DS4_V41_DECODER_REPLAY_ROWS="0",
                   DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
                   DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
                   DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0")
    elif profile == "glm-5.3-flash":
        cmd += ["--ssd-streaming", "--ssd-streaming-cache-experts", "24GB"]
        env["DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP"] = "1"
    if system:
        cmd += ["--system", system]
    # Flint's canonical template chooses when to reason and rejects forced
    # thinking controls. The other profiles retain their explicit prompt mode.
    if profile != "flint":
        cmd.append("--think" if think else "--nothink")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    (work / "stderr.log").write_bytes(result.stderr)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ds4 activation capture failed:\n{error}")

    rows: list[list[float]] = []
    for layer in range(n_layer):
        path = work / f"dump_{component}-{layer}_pos0.bin"
        data = array.array("f")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // 4)
        if len(data) < n_embd or len(data) % n_embd != 0:
            raise RuntimeError(f"bad dump shape for {path}: {len(data)} floats")
        row = list(data[-n_embd:])
        if not all(math.isfinite(value) for value in row):
            raise RuntimeError(f"nonfinite activation in {path}")
        rows.append(row)
    return rows


def add_rows(total: list[list[float]], rows: list[list[float]], n_layer: int) -> None:
    for layer in range(n_layer):
        dst = total[layer]
        src = rows[layer]
        for i, value in enumerate(src):
            dst[i] += value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4", help="path to the ds4 CLI")
    ap.add_argument("--model", default="ds4flash.gguf",
                    help="GGUF path, or Flint checkpoint directory")
    ap.add_argument("--profile", choices=MODEL_PROFILES,
                    default="glm-5.3-flash",
                    help="model shape and chat-template profile")
    ap.add_argument("--good-file", required=True,
                    help="desired/target prompts, one per line")
    ap.add_argument("--bad-file", required=True,
                    help="contrast/control prompts, one per line")
    ap.add_argument("--out", default="dir-steering/out/direction.json",
                    help="metadata JSON path; .f32 is written next to it")
    ap.add_argument("--source", default="",
                    help="optional corpus source or provenance note")
    ap.add_argument("--work-dir", default=str(REPO_ROOT / ".workspaces" / "steering-capture"),
                    help="parent for retained per-prompt activation captures")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--component", default="residual",
                    choices=("residual", "attn_residual"),
                    help="post-FFN residual (default), or post-attention residual")
    ap.add_argument("--mode", choices=("project", "add"), default="project",
                    help="unit projection direction, or raw mean-difference addition")
    ap.add_argument("--layers", default="all",
                    help="zero-based inclusive layer selection, e.g. 8-15,24,32-39")
    ap.add_argument("--think", action="store_true",
                    help="capture after <think>; default captures direct answers")
    args = ap.parse_args()
    if args.ctx < 2:
        ap.error("--ctx must be at least 2")
    if args.profile == "flint" and args.think:
        ap.error("Flint uses its automatic thinking template; omit --think")
    if args.profile == "glm-5.3-flash" and args.ctx > 2048:
        ap.error("GLM capture currently requires --ctx <= 2048 (one prefill chunk)")

    ds4 = Path(args.ds4).resolve()
    model_arg = Path(args.model)
    model = model_arg.resolve()
    n_layer, n_embd = MODEL_PROFILES[args.profile]
    try:
        selected = selected_layers(args.layers, n_layer)
    except ValueError as error:
        ap.error(str(error))
    good_prompts = read_prompt_file(Path(args.good_file))
    bad_prompts = read_prompt_file(Path(args.bad_file))
    n = min(len(good_prompts), len(bad_prompts))
    good_prompts = good_prompts[:n]
    bad_prompts = bad_prompts[:n]

    good_sum = [[0.0] * n_embd for _ in range(n_layer)]
    bad_sum = [[0.0] * n_embd for _ in range(n_layer)]

    work_parent = Path(args.work_dir).resolve()
    work_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{args.profile}-", dir=work_parent))
    print(f"capture workspace: {root}", flush=True)
    for i, (good, bad) in enumerate(zip(good_prompts, bad_prompts), 1):
        print(f"pair {i}/{n}", flush=True)
        gw = root / f"good-{i}"
        bw = root / f"bad-{i}"
        gw.mkdir()
        bw.mkdir()
        good_rows = run_capture(ds4, model, good, args.system, args.think,
                                args.ctx, args.component, n_layer, n_embd, gw,
                                profile=args.profile)
        bad_rows = run_capture(ds4, model, bad, args.system, args.think,
                               args.ctx, args.component, n_layer, n_embd, bw,
                               profile=args.profile)
        add_rows(good_sum, good_rows, n_layer)
        add_rows(bad_sum, bad_rows, n_layer)

    layers = []
    for layer in range(n_layer):
        direction = [(good - bad) / n for good, bad in zip(good_sum[layer], bad_sum[layer])]
        if layer not in selected:
            direction = [0.0] * n_embd
        elif args.mode == "project":
            direction = normalize(direction)
        layers.append(direction)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-residual-steering-v1",
        "shape": [n_layer, n_embd],
        "profile": args.profile,
        "component": args.component,
        "mode": args.mode,
        "normalization": "unit-l2" if args.mode == "project" else "none",
        "selected_layers": sorted(selected),
        "nonzero_layers": [i for i, row in enumerate(layers) if any(row)],
        "pairs": n,
        "thinking": bool(args.think),
        "good_file": str(Path(args.good_file)),
        "bad_file": str(Path(args.bad_file)),
        "model": str(model_arg),
        "source": args.source,
        "capture_workspace": str(root),
        "note": ("positive strength suppresses the represented direction; strength is per layer"
                 if args.mode == "project" else
                 "positive strength adds the target-minus-control residual; runtime divides strength by the number of nonzero layers"),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = array.array("f")
    for direction in layers:
        flat.extend(direction)
    f32_out = out.with_suffix(".f32")
    with f32_out.open("wb") as f:
        flat.tofile(f)
    print(f"wrote {out}")
    print(f"wrote {f32_out}")


if __name__ == "__main__":
    main()
