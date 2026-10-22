from pathlib import Path
import json, subprocess, time

base = Path(__file__).resolve().parents[3] / ".workspaces/zig-gguf-input-20261022"
out = base / "qwen-results"
out.mkdir()
rows = []
for arm in ("c", "native", "c-debug", "native-debug"):
    command = [str(base / arm / "bin/test-qwen4-ngrams")]
    started = time.monotonic()
    process = subprocess.run(command, capture_output=True, timeout=60)
    (out / (arm + ".stdout")).write_bytes(process.stdout)
    (out / (arm + ".stderr")).write_bytes(process.stderr)
    row = dict(
        arm=arm,
        command=command,
        returncode=process.returncode,
        seconds=time.monotonic() - started,
    )
    rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
