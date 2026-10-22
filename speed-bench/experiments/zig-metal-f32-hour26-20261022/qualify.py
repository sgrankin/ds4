from pathlib import Path
import csv, hashlib, json, os, subprocess, time

root = Path.cwd()
base = root / ".workspaces/zig-metal-f32-20261022"
out = base / "real"
out.mkdir()
rows = []
env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4", "PT_"))}


def run(label, command, extra=None):
    start = time.monotonic()
    command = list(map(str, command))
    p = subprocess.run(
        command, env=env | dict(extra or {}), capture_output=True, timeout=300
    )
    (out / (label + ".stdout")).write_bytes(p.stdout)
    (out / (label + ".stderr")).write_bytes(p.stderr)
    row = dict(
        label=label,
        command=command,
        env=extra or {},
        seconds=time.monotonic() - start,
        returncode=p.returncode,
        binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
    )
    rows.append(row)
    save()
    print(
        json.dumps({k: row[k] for k in ["label", "returncode", "seconds"]}), flush=True
    )
    assert p.returncode == 0, label
    return row


def save():
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


for arm in ["c", "native"]:
    run(arm + "-router", [base / arm / "bin/test-metal-dense-mpp", "--router"])
    run(arm + "-ssd", [base / arm / "bin/test-metal-ssd-experts"])
reference = None
for i, arm in enumerate(["c", "native", "native", "c"]):
    label = f"{i}-{arm}"
    folder = out / label
    folder.mkdir()
    (folder / "logits").mkdir()
    binary = (
        root / ".workspaces/zig-gguf-input-20261022/native/bin/ds4-bench"
        if arm == "c"
        else base / "native/bin/ds4-bench"
    )
    row = run(
        label,
        [
            binary,
            "--model",
            root / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf",
            "--metal",
            "--ssd-streaming",
            "--ssd-streaming-cache-experts",
            "48GB",
            "--prompt-file",
            root / "speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt",
            "--ctx-start",
            "512",
            "--ctx-max",
            "512",
            "--ctx-alloc",
            "1024",
            "--prefill-chunk",
            "512",
            "--gen-tokens",
            "128",
            "--teacher-forced-decode",
            "--csv",
            folder / "metrics.csv",
            "--dump-frontier-logits-dir",
            folder / "logits",
        ],
    )
    logits = json.loads(next((folder / "logits").glob("*.json")).read_text())["logits"]
    if reference is None:
        reference = logits
    row["identical_logits"] = logits == reference
    row["metrics"] = list(csv.DictReader((folder / "metrics.csv").open()))
    save()
    assert row["identical_logits"], label
