"""Cross-restart C/native durable Responses qualification; root launches under guard."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def stop(process):
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return process.returncode


def request(base, output, name, method, path, payload=None, expected=200):
    save(
        output / (name + ".request.json"), dict(method=method, path=path, body=payload)
    )
    message = urllib.request.Request(
        base + path,
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(message, timeout=180) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    (output / (name + ".response.json")).write_bytes(raw)
    body = json.loads(raw)
    record = dict(
        name=name,
        status=status,
        expected=expected,
        seconds=time.monotonic() - started,
        pass_status=status == expected,
    )
    save(output / (name + ".result.json"), record)
    if status != expected:
        raise RuntimeError(f"{name}: HTTP{status}, expected{expected}")
    return body, raw


def files(directory, snapshot=None):
    if snapshot is not None:
        snapshot.mkdir()
        for path in sorted(directory.iterdir()):
            shutil.copyfile(path, snapshot / path.name)
    return dict(
        directory_mode=stat.S_IMODE(directory.stat().st_mode),
        entries=[
            dict(
                name=path.name,
                bytes=path.stat().st_size,
                mode=stat.S_IMODE(path.stat().st_mode),
            )
            for path in sorted(directory.iterdir())
        ],
    )


def text(response):
    return "".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--c-server",
        type=Path,
        default=ROOT / ".workspaces/zig-metal-f32-20261022/native/bin/ds4-server",
    )
    parser.add_argument(
        "--native-server",
        type=Path,
        default=ROOT / ".workspaces/zig-response-store-20261022/native/bin/ds4-server",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shaders = dict(
        re.findall(
            r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
            (ROOT / "ds4_metal.m").read_text(),
        )
    )
    results = []
    for name, writer, reader in [
        ("c-to-native", args.c_server, args.native_server),
        ("native-to-c", args.native_server, args.c_server),
    ]:
        direction = output / name
        direction.mkdir()
        store, cache = direction / "responses", direction / "cache"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("DS4", "PT_"))
        }
        env.update({key: str(ROOT / path) for key, path in shaders.items()})
        env.update(
            DS4_SERVER_RESPONSE_STORE_DIR=str(store),
            DS4_STATE_CACHE_DIR=str(cache),
            DS4_STATE_CACHE_HOT_MB="0",
            DS4_STATE_CACHE_SSD_MB="512",
            DS4_SERVER_CONTEXT_SUFFIX_REUSE="0",
            DS4_METAL_V41_RESIDENT_ROUTE_GAP="0",
            DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
            DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0",
        )
        row = dict(direction=name, phases=[], model_calls=[], checks={}, error=None)
        parent, parent_raw = None, None
        for phase, binary in [("before", writer), ("after", reader)]:
            current = direction / phase
            current.mkdir()
            binary = binary.resolve()
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [
                str(binary),
                "--model",
                str(MODEL),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--ssd-streaming",
                "--ssd-streaming-cache-experts",
                "48GB",
                "--ctx",
                "8192",
                "--prefill-chunk",
                "512",
                "--mtp-draft",
                "1",
                "--tokens",
                "128",
            ]
            execution = dict(
                command=command,
                cwd=str(ROOT),
                environment={
                    key: value for key, value in env.items() if key.startswith("DS4")
                },
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            )
            save(current / "execution.json", execution)
            process, started = None, time.monotonic()
            phase_row = dict(phase=phase, error=None)
            try:
                with (
                    (current / "server.stdout").open("wb") as stdout,
                    (current / "server.stderr").open("wb") as stderr,
                ):
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    phase_row["pid"] = process.pid
                    base = f"http://127.0.0.1:{port}"
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"startup exit{process.returncode}")
                        if time.monotonic() - started > 120:
                            raise TimeoutError("120-second startup deadline")
                        try:
                            with urllib.request.urlopen(
                                base + "/v1/models", timeout=1
                            ) as response:
                                model = json.load(response)["data"][0]["id"]
                            break
                        except (OSError, ValueError, KeyError, IndexError):
                            time.sleep(0.2)
                    phase_row["startup_seconds"] = time.monotonic() - started
                    common = dict(
                        model=model,
                        store=True,
                        stream=False,
                        temperature=0,
                        reasoning={"effort": "none"},
                        max_output_tokens=64,
                    )
                    if phase == "before":
                        parent, parent_raw = request(
                            base,
                            current,
                            "parent-create",
                            "POST",
                            "/v1/responses",
                            common
                            | {
                                "instructions": "Reply with OLD followed by the project codename and manifest count, separated by | characters. No spaces, explanation or tools.",
                                "input": "Remember these stable facts for later: the project codename is ORCHID42 and the manifest count is 37.",
                            },
                        )
                        result, raw = request(
                            base,
                            current,
                            "parent-get",
                            "GET",
                            "/v1/responses/" + parent["id"],
                        )
                        row["checks"]["parent_get_object_exact"] = result == parent
                        row["checks"]["parent_get_bytes_exact"] = raw == parent_raw
                        row["model_calls"].append(
                            dict(
                                role="parent",
                                expected="OLD|ORCHID42|37",
                                actual=text(parent),
                                quality_pass=text(parent) == "OLD|ORCHID42|37",
                                status=parent.get("status"),
                                usage=parent.get("usage"),
                                incomplete_details=parent.get("incomplete_details"),
                            )
                        )
                        row["checks"]["parent_file_permissions"] = all(
                            entry["mode"] == 0o600 for entry in files(store)["entries"]
                        )
                        row["store_before_restart"] = files(
                            store, current / "stored-records"
                        )
                    else:
                        if parent is None:
                            raise RuntimeError(
                                "parent creation did not finish before restart"
                            )
                        result, raw = request(
                            base,
                            current,
                            "parent-get",
                            "GET",
                            "/v1/responses/" + parent["id"],
                        )
                        row["checks"]["restart_parent_get_object_exact"] = (
                            result == parent
                        )
                        row["checks"]["restart_parent_get_bytes_exact"] = (
                            raw == parent_raw
                        )
                        child, child_raw = request(
                            base,
                            current,
                            "child-create",
                            "POST",
                            "/v1/responses",
                            common
                            | {
                                "previous_response_id": parent["id"],
                                "instructions": "Reply with NEW followed by the remembered project codename and manifest count, separated by | characters. No spaces, explanation or tools.",
                                "input": "What are the project codename and manifest count? Use the current response format.",
                            },
                        )
                        result, raw = request(
                            base,
                            current,
                            "child-get",
                            "GET",
                            "/v1/responses/" + child["id"],
                        )
                        row["checks"]["child_get_object_exact"] = result == child
                        row["checks"]["child_get_bytes_exact"] = raw == child_raw
                        row["model_calls"].append(
                            dict(
                                role="child",
                                expected="NEW|ORCHID42|37",
                                actual=text(child),
                                quality_pass=text(child) == "NEW|ORCHID42|37",
                                status=child.get("status"),
                                usage=child.get("usage"),
                                incomplete_details=child.get("incomplete_details"),
                            )
                        )
                        row["store_before_delete"] = files(
                            store, current / "stored-records"
                        )
                        row["checks"]["store_directory_permissions"] = (
                            row["store_before_delete"]["directory_mode"] == 0o700
                        )
                        row["checks"]["child_file_permissions"] = all(
                            entry["mode"] == 0o600
                            for entry in row["store_before_delete"]["entries"]
                        )
                        for role, response in [("child", child), ("parent", parent)]:
                            deleted, _ = request(
                                base,
                                current,
                                role + "-delete",
                                "DELETE",
                                "/v1/responses/" + response["id"],
                            )
                            row["checks"][role + "_deleted_object"] = (
                                deleted.get("id") == response["id"]
                                and deleted.get("deleted") is True
                            )
                            request(
                                base,
                                current,
                                role + "-missing",
                                "GET",
                                "/v1/responses/" + response["id"],
                                expected=404,
                            )
                        row["store_after_delete"] = files(store)
                        row["checks"]["store_empty_after_delete"] = not row[
                            "store_after_delete"
                        ]["entries"]
            except Exception as error:
                phase_row["error"] = f"{type(error).__name__}: {error}"
                if row["error"] is None:
                    row["error"] = phase_row["error"]
            finally:
                phase_row["server_returncode"] = stop(process)
                phase_row["seconds"] = time.monotonic() - started
                row["phases"].append(phase_row)
                save(current / "result.json", phase_row)
        row["checks"]["distinct_processes"] = len(row["phases"]) == 2 and row["phases"][
            0
        ].get("pid") != row["phases"][1].get("pid")
        row["protocol_pass"] = bool(
            not row["error"]
            and all(row["checks"].values())
            and len(row["model_calls"]) == 2
        )
        row["model_quality_pass"] = len(row["model_calls"]) == 2 and all(
            call["quality_pass"]
            and call["status"] == "completed"
            and (call["usage"] or {}).get("output_tokens", 0) < 64
            for call in row["model_calls"]
        )
        row["pass"] = row["protocol_pass"] and row["model_quality_pass"]
        save(direction / "result.json", row)
        results.append(row)
        save(output / "results.json", results)
        print(json.dumps(row), flush=True)
    return 0 if all(row["pass"] for row in results) else 3


if __name__ == "__main__":
    raise SystemExit(main())
