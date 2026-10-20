#!/usr/bin/env python3
"""Preflight one frozen counting job; --execute still requires a root GPU grant.

The GGUF pathname is an opaque command argument. No model metadata, tokenizer,
model help command, path resolution, or native executable is consulted here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

MAIN = "/Volumes/Models/ds4"
MODEL = MAIN + "/gguf/DeepSeek-V4.1-Flash-Q2.gguf"
GUARD = MAIN + "/speed-bench/ideas/pressure_run.py"
STUDY = "speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study"
QUEUE = MAIN + "/.workspaces/idea-quant-effort/" + STUDY + "/generation-queue-v1.json"
QUEUE_SHA256 = "2af91c43a8c701cdcec392ef9041607cbf068180ea32528521b9afc01cc495b6"
CORE_PROOF = MAIN + "/.workspaces/idea-quant-effort/" + STUDY + "/native-effort-prepared-v1/core-objects.json"
CORE_PROOF_SHA256 = "1b52970c2928122f79591befd59545b82aa314de461793384836b8bb448a3fe9"
LABELS = ("narrow-half-effort75-pair", "native-effort25", "native-effort50")
GOLDS = {"circular10": 123, "restricted_derangements8": 10860,
         "lattice8x6": 2002, "subset_mod7": 132}
SETTINGS = dict(cap=2048, temperature_float=0.6000000238418579, top_p=0.95,
                min_p=0, seed=20261002, cache_gib=64, context=8192,
                prefill_chunk=512, mtp_tokens=1, diagnostic_stride=32,
                marker_profile="hesitation", after=0)


class PreflightError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PreflightError(message)


def safe_input(path):
    """Check spelling only, deliberately without stat/resolve/symlink queries."""
    text = os.fspath(path)
    norm = os.path.normpath(text)
    require(norm.startswith(MAIN + "/"), "input outside frozen repository: " + text)
    require(not text.lower().endswith(".gguf") and norm != MODEL,
            "model path I/O is forbidden")
    return Path(text)


def fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with safe_input(path).open("rb") as stream:
        while True:
            block = stream.read(256 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def verified_json(path, expected, verified):
    # Read once so the parsed bytes are exactly the bytes whose hash was checked.
    data = safe_input(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    require(actual == expected, "hash mismatch: " + os.fspath(path))
    verified[os.fspath(path)] = dict(sha256=actual, bytes=len(data))
    return json.loads(data)


def verify_file(path, expected, verified, size=None):
    actual, actual_size = fingerprint(path)
    require(actual == expected, "hash mismatch: " + os.fspath(path))
    require(size is None or size == actual_size, "size mismatch: " + os.fspath(path))
    verified[os.fspath(path)] = dict(sha256=actual, bytes=actual_size)


def split_frozen_env(command):
    require(command and command[0] == "env", "missing frozen env prefix")
    assignments = {}
    cursor = 1
    while cursor < len(command) and "=" in command[cursor]:
        name, value = command[cursor].split("=", 1)
        require(re.fullmatch(r"DS4_[A-Z0-9_]+", name) is not None,
                "unexpected command environment assignment")
        require(name not in assignments, "duplicate environment assignment")
        assignments[name] = value
        cursor += 1
    return assignments, command[cursor:]


def preflight(label):
    require(label in LABELS, "only the three scheduled job labels are supported")
    verified = {}
    queue = verified_json(QUEUE, QUEUE_SHA256, verified)
    require(queue["golds"] == GOLDS, "gold fixture changed")
    require(queue["shared_settings"] == SETTINGS, "shared settings changed")
    jobs = queue["jobs"]
    require(tuple(j["label"] for j in jobs if j["scheduled_by_default"]) == LABELS,
            "scheduled jobs changed")
    manifests = {}
    for entry in jobs:
        manifest = verified_json(entry["manifest"], entry["manifest_sha256"], verified)
        require(manifest["argv"] == entry["command"], "queue/manifest command mismatch")
        require(manifest["cwd"] == entry["cwd"], "queue/manifest cwd mismatch")
        require(manifest["revision"] == entry["source_revision"], "source revision mismatch")
        require(manifest["binary"]["sha256"] == entry["binary_sha256"],
                "queue/manifest binary hash mismatch")
        require(manifest["model"] == entry["required_model"], "stored model metadata mismatch")
        # Only compare the stored dictionary. Never touch this pathname.
        require(manifest["model"]["path"] == MODEL, "unexpected model argument")
        manifests[entry["label"]] = manifest

    shaders = queue["runtime_metal_sha256"]
    require(len(shaders) == 52, "expected exactly 52 frozen runtime shaders")
    for path, sha in shaders.items():
        require(path.endswith(".metal") and "/metal/" in path, "unexpected shader input")
        verify_file(path, sha, verified)

    proof = verified_json(CORE_PROOF, CORE_PROOF_SHA256, verified)
    require(len(proof["objects"]) == 8, "expected eight proven core objects")
    # Existing proof records explicit object hashes shared by both helper builds.
    # Its historical runtime_sources_equal count is not a new source proof.
    for cwd in {entry["cwd"] for entry in jobs}:
        for name, info in proof["objects"].items():
            require("/" not in name and name.endswith(".o"), "unexpected object input")
            verify_file(cwd + "/" + name, info["sha256"], verified, info["bytes"])
        manifest = next(m for m in manifests.values() if m["cwd"] == cwd)
        for relative, sha in manifest["source_sha256"].items():
            require(not os.path.isabs(relative) and ".." not in Path(relative).parts,
                    "unexpected source path")
            verify_file(cwd + "/" + relative, sha, verified)
        verify_file(manifest["binary"]["path"], manifest["binary"]["sha256"], verified,
                    manifest["binary"]["bytes"])

    manifest = manifests[label]
    command = manifest["argv"]
    assignments, guarded = split_frozen_env(command)
    require(guarded[:2] == ["python3", GUARD], "command must invoke the MAIN guard")
    require(guarded[2:4] == [manifest["pressure_dir"], "--"], "guard argument mismatch")
    require(guarded[4:] == manifest["pilot_argv"], "native child must be direct to guard")
    require(manifest["guard"]["path"] == GUARD, "unexpected guard")
    verify_file(GUARD, manifest["guard"]["sha256"], verified)
    require(manifest["pilot_argv"][:2] == [manifest["binary"]["path"], MODEL],
            "unexpected native child/model argv")
    require(manifest["suite"] == "counting" and manifest["max_tokens"] == 2048
            and manifest["cache_gib_requested"] == 64.0 and manifest["case_limit"] == 4
            and manifest["case_offset"] == 0 and manifest["diagnostic_stride"] == 32,
            "job settings changed")
    for name, value in manifest["policy_environment"].items():
        require(assignments.get(name) == value, "frozen policy environment mismatch: " + name)
    fixture = verified_json(manifest["cwd"] + "/" + STUDY + "/cases-v1.json",
                            queue["fixture_sha256"], verified)
    require({case["id"]: case["answer"] for case in fixture["cases"]} == GOLDS,
            "fixture golds mismatch")
    return dict(status="PREFLIGHT_PASS_NO_MODEL_IO", job=label, command=command,
                cwd=manifest["cwd"], frozen_environment=assignments,
                queue_sha256=QUEUE_SHA256, core_proof_sha256=CORE_PROOF_SHA256,
                source_revision=manifest["revision"], verified_files=verified,
                runtime_shader_files=52, model_argument=MODEL,
                model_metadata_refreshed=False, model_payload_hashed=False,
                results_dir=manifest["results_dir"], pressure_dir=manifest["pressure_dir"],
                results_jsonl=manifest["results_dir"] + "/results.jsonl",
                expected_records=8 if label == LABELS[0] else 4,
                scheduled_jobs=1, execution_requires_new_root_gpu_grant=True,
                returncode_rule="Preserve guard rc, including 3 for INVALID final records; rc0 may include INCORRECT finals.")


def execute_one(plan, inherited_env):
    # --execute is an action switch, never authorization. The root grant is external.
    environment = {name: value for name, value in inherited_env.items()
                   if not name.startswith("DS4")}
    environment.update(plan["frozen_environment"])
    # Replace this process: the MAIN guard remains the signal/returncode owner,
    # with the frozen C helper as its direct child. No retries or chained jobs.
    os.chdir(plan["cwd"])
    return os.execvpe(plan["command"][0], plan["command"], environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=LABELS, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="execute this one job only, after a fresh root model grant")
    args = parser.parse_args(argv)
    try:
        plan = preflight(args.job)
    except (PreflightError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps(dict(status="PREFLIGHT_REJECTED", error=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    return execute_one(plan, os.environ)


if __name__ == "__main__":
    sys.exit(main())
