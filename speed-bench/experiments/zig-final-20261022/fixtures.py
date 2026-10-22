from pathlib import Path
import argparse, hashlib, json, subprocess, time

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(
    description="Build and run the final C/native CPU fixtures."
)
parser.add_argument("--out", type=Path, default=root / ".workspaces/zig-final-rerun")
base = parser.parse_args().out.resolve()
base.mkdir(parents=True, exist_ok=False)
out = base / "fixtures"
out.mkdir()
rows = []
common = [
    "agent-test",
    "server-test",
    "state-cache-test",
    "session-state-test",
    "tokenizer-test",
    "pretokenizer-test",
    "gguf-test",
    "ssd-test",
    "prompt-prefix-test",
    "engram-test",
    "image-test",
    "vision-image-test",
    "sampling-test",
    "resident-route-test",
    "response-store-test",
]
extra = [
    "vocab-test",
    "native-f16-publication-test",
    "native-hc-projection-test",
    "native-norm-test",
    "native-activation-test",
]
executables = [
    "ds4_agent_test",
    "test-session-state",
    "test-tokenizer-piece",
    "test-pretokenizer",
    "test-gguf-cursor",
    "test-ssd-cache",
    "test-prompt-prefix",
    "test-engram",
    "test-image-decode",
    "test-deepseek4-vision-image",
    "test-sampling",
    "test-resident-route",
    "test-resident-three",
    "test-resident-drop",
]
extra_executables = [
    "test-vocab",
    "test-f16-publication",
    "test-hc-projection",
    "test-norm-cpu",
    "test-activation",
]


def run(label, command):
    command = list(map(str, command))
    start = time.monotonic()
    p = subprocess.run(command, cwd=root, capture_output=True, timeout=180)
    (out / (label + ".stdout")).write_bytes(p.stdout)
    (out / (label + ".stderr")).write_bytes(p.stderr)
    row = dict(
        label=label,
        command=command,
        returncode=p.returncode,
        seconds=time.monotonic() - start,
    )
    rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    assert p.returncode == 0, label


for arm in ["c", "native"]:
    for mode in ["ReleaseFast", "Debug"]:
        label = arm + ("-debug" if mode == "Debug" else "")
        prefix = base / label
        run(
            label + "-build",
            [
                root / "tools/zig-build",
                "-Dnative=" + str(arm == "native").lower(),
                "-Doptimize=" + mode,
                "--prefix",
                prefix,
                *(["install"] if mode == "ReleaseFast" else []),
                *common,
                *(extra if arm == "native" else []),
                *(["dense-mpp-test"] if arm == "native" and mode == "Debug" else []),
            ],
        )
        for exe in executables + (extra_executables if arm == "native" else []):
            run(label + "-" + exe, [prefix / "bin" / exe])
        run(label + "-server", [prefix / "bin/ds4_test", "--server"])
        run(
            label + "-cache",
            [prefix / "bin/test-state-cache", out / (label + "-cache-files")],
        )
        run(
            label + "-response-store",
            [prefix / "bin/test-response-store", out / (label + "-response-files")],
        )
identities = {}
for arm in ["c", "native"]:
    for program in ["ds4", "ds4-agent", "ds4-server", "ds4-bench", "ds4-eval"]:
        path = base / arm / "bin" / program
        identities[f"{arm}/{program}"] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "binaries.json").write_text(json.dumps(identities, indent=2) + "\n")
