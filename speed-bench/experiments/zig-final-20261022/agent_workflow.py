from pathlib import Path
import argparse, json, subprocess, time

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(
    description="Repeat the existing six-turn C/native persistent-agent workflow."
)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument(
    "--binaries", type=Path, default=root / ".workspaces/zig-final-rerun"
)
parser.add_argument(
    "--out", type=Path, default=root / ".workspaces/zig-final-agent-rerun"
)
args = parser.parse_args()
base = args.binaries.resolve()
out = args.out.resolve()
model = args.model.resolve()
out.mkdir(parents=True, exist_ok=False)
rows = []
for arm in ["c", "native"]:
    command = list(
        map(
            str,
            [
                "python3",
                "-B",
                root / "speed-bench/experiments/zig-combined-20261022/run_agent.py",
                "--model",
                model,
                "--agent",
                base / arm / "bin/ds4-agent",
                "--output",
                out / arm / "run",
                "--shared",
                out / arm / "shared",
                "--mode",
                "suffix",
                "--prerope",
                "1",
                "--tail",
                "128",
                "--notes",
                "24",
                "--blob-units",
                "128",
                "--timeout",
                "600",
            ],
        )
    )
    start = time.monotonic()
    with (
        (out / (arm + ".stdout")).open("wb") as stdout,
        (out / (arm + ".stderr")).open("wb") as stderr,
    ):
        process = subprocess.run(
            command, cwd=root, stdout=stdout, stderr=stderr, timeout=1200
        )
    row = dict(
        label=arm,
        command=command,
        returncode=process.returncode,
        seconds=time.monotonic() - start,
    )
    rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    assert process.returncode == 0, arm
