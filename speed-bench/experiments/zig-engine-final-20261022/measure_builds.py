#!/usr/bin/env python3
"""Measure five-program builds serially, with no model or compiler jobs active."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=ROOT / ".workspaces/zig-build-rerun")
OUT = parser.parse_args().out.resolve()
OUT.mkdir(parents=True, exist_ok=False)
RESULT = OUT / "build-performance.json"
JOBS = 12  # Same setting as tools/zig-build and the prior Make comparison.
TARGETS = ["ds4", "ds4-server", "ds4-agent", "ds4-bench", "ds4-eval"]
C_UNITS = [
    "ds4.c",
    "ds4_image.c",
    "ds4_distributed.c",
    "ds4_tp.c",
    "ds4_ssd.c",
    "ds4_layer_pack.c",
    "ds4_help.c",
    "ds4_prompt_prefix.c",
    "ds4_web.c",
    "ds4_kvstore.c",
    "ds4_eval_cases.c",
    "ds4_gpu_args.c",
    "linenoise.c",
    "rax.c",
    "ds4_engram.c",
    "ds4_metal.m",
    "ds4_cli.c",
    "ds4_server.c",
    "ds4_agent.c",
    "ds4_bench.c",
    "ds4_eval.c",
    "ds4_state_cache.c",
]
# These residual C/ObjC units are actually compiled in native mode. Superseded
# C bodies are not edited to manufacture a full rebuild of unused code.
NATIVE_C_UNITS = [
    "ds4.c",
    "ds4_image.c",
    "ds4_distributed.c",
    "ds4_tp.c",
    "ds4_layer_pack.c",
    "ds4_help.c",
    "ds4_web.c",
    "ds4_eval_cases.c",
    "ds4_gpu_args.c",
    "linenoise.c",
    "rax.c",
    "ds4_metal.m",
    "ds4_server.c",
    "ds4_agent.c",
    "ds4_eval.c",
]
NATIVE_ZIG_UNITS = [
    "zig/state_cache.zig",
    "zig/kvstore.zig",
    "zig/ssd.zig",
    "zig/prompt_prefix.zig",
    "zig/engram.zig",
    "zig/gguf.zig",
    "zig/cli_chat.zig",
    "zig/tokenizer.zig",
    "zig/pretokenizer.zig",
    "zig/chat.zig",
    "zig/vocab.zig",
    "zig/agent_tools.zig",
    "zig/agent_file.zig",
    "zig/sampling.zig",
    "zig/resident_route.zig",
    "zig/response_store.zig",
    "zig/image.zig",
    "zig/image_preprocess.zig",
    "zig/bench.zig",
    "zig/cli.zig",
    "zig/cli_options.zig",
    "zig/cli_diagnostics.zig",
    "zig/metal/f16_publication.zig",
    "zig/metal/hc_projection.zig",
    "zig/metal/norm.zig",
    "zig/metal/activation.zig",
    "zig/metal/matmul_f32.zig",
    "zig/metal/bridge.zig",
    "zig/agent_bash.zig",
    "zig/agent_bash_entry.zig",
    "zig/agent_compaction.zig",
    "zig/agent_compaction_entry.zig",
    "zig/agent_context.zig",
    "zig/agent_context_entry.zig",
    "zig/engine/config.zig",
    "zig/engine/deepseek41/attention.zig",
    "zig/engine/deepseek41/batch_attention.zig",
    "zig/engine/deepseek41/dspark.zig",
    "zig/engine/deepseek41/engram_prefetch.zig",
    "zig/engine/deepseek41/executor.zig",
    "zig/engine/deepseek41/expert_cache.zig",
    "zig/engine/deepseek41/layout.zig",
    "zig/engine/deepseek41/moe.zig",
    "zig/engine/deepseek41/ops.zig",
    "zig/engine/deepseek41/prefill.zig",
    "zig/engine/deepseek41/prefill_buffers.zig",
    "zig/engine/deepseek41/prefill_policy.zig",
    "zig/engine/deepseek41/resources.zig",
    "zig/engine/deepseek41/session_batch.zig",
    "zig/engine/deepseek41/speculative.zig",
    "zig/engine/deepseek41/state_payload.zig",
    "zig/engine/deepseek41/streaming.zig",
    "zig/engine/deepseek41/suffix.zig",
    "zig/engine/graph.zig",
    "zig/engine/input.zig",
    "zig/engine/model.zig",
    "zig/engine/vision.zig",
    "zig/engine/vision_entry.zig",
    "zig/engine/weights.zig",
    "zig/server/coordinator.zig",
    "zig/server/http.zig",
    "zig/server/input.zig",
    "zig/server/input_entry.zig",
    "zig/server/input_values.zig",
    "zig/server/message_values.zig",
    "zig/server/output.zig",
    "zig/server/output_entry.zig",
    "zig/server/responses_values.zig",
    "zig/server/tool_memory.zig",
    "zig/server/tool_memory_entry.zig",
    "zig/metal/resources.zig",
    "zig/metal/resources_entry.zig",
]
report = {
    "scope": "All five Metal programs: Make C versus selected handwritten Zig modules.",
    "jobs": JOBS,
    "runs": [],
    "source_writes": [],
    "limits": "Warm caches; one full content-forced rebuild, two no-ops and two single Metal source edits per variant. No binaries, tests or models are executed. Comment-only edits preserve runtime semantics.",
    "edited_units": {"make-c": "ds4_metal.m", "zig-native": "zig/metal/matmul_f32.zig"},
    "make_timestamp_barrier_seconds": 1.05,
}
originals = {}
env = os.environ.copy()
env["ZIG_GLOBAL_CACHE_DIR"] = str(ROOT / ".workspaces/zig-build/global-cache")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save():
    RESULT.write_text(json.dumps(report, indent=2) + "\n")


def snapshot(destination):
    if destination.exists():
        raise RuntimeError(
            f"{destination} already exists; preserve earlier evidence and use a fresh workspace"
        )
    sources = [
        path
        for path in ROOT.iterdir()
        if path.is_file()
        and (
            path.suffix in {".c", ".h", ".m", ".inc"}
            or path.name in {"Makefile", "build.zig", "build.zig.zon"}
        )
    ]
    # Only source files: no model, .jj, object, fixture-image or output trees.
    for subtree in ["metal", "zig", "third_party/iris", "tests"]:
        sources += [
            path
            for path in (ROOT / subtree).rglob("*")
            if path.is_file()
            and path.suffix in {".c", ".h", ".m", ".inc", ".zig", ".metal"}
        ]
    sources.append(ROOT / "tools/zig-build")
    destination.mkdir()
    originals[destination] = {}
    for source in sorted(sources):
        if source.is_symlink():
            raise RuntimeError(f"refusing source symlink: {source}")
        name = str(source.relative_to(ROOT))
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[destination][name] = target.read_bytes()
    return {name: digest(data) for name, data in originals[destination].items()}


def write_changed(source, contents, label, make):
    changed = {
        name: data
        for name, data in contents.items()
        if (source / name).read_bytes() != data
    }
    if not changed:
        return
    # Apple Make 3.81 ignores subsecond mtimes. Wait AFTER the preceding build
    # and BEFORE each edit/restore; this wait is outside every timed interval.
    if make:
        time.sleep(1.05)
    stamps = {}
    for name, data in changed.items():
        path = source / name
        path.write_bytes(data)
        stamps[name] = path.stat().st_mtime_ns
    report["source_writes"].append(dict(label=label, cwd=str(source), mtime_ns=stamps))
    save()


def edit(source, units, label, make):
    contents = {}
    for name in units:
        marker = f"\n// {label}\n" if name.endswith(".zig") else f"\n/* {label} */\n"
        contents[name] = originals[source][name] + marker.encode()
    write_changed(source, contents, label, make)


def restore(source, units, label, make):
    write_changed(
        source, {name: originals[source][name] for name in units}, label, make
    )


def run(variant, source, command, phase, sample=0, measured=False, expected_unit=None):
    label = f"{variant}-{phase}" + (f"-{sample}" if sample else "")
    stdout_path, stderr_path = (
        OUT / (label + ".stdout.log"),
        OUT / (label + ".stderr.log"),
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        started = time.perf_counter()
        process = subprocess.run(
            command, cwd=source, env=env, stdout=stdout, stderr=stderr
        )
        seconds = time.perf_counter() - started
    lines = (
        stdout_path.read_text(errors="replace")
        + "\n"
        + stderr_path.read_text(errors="replace")
    ).splitlines()
    c_commands = [line for line in lines if re.search(r"(^|\s)-c(\s|$)", line)]
    zig_invocations = [
        line for line in lines if re.search(r"(^|\s)build-obj(\s|$)", line)
    ]
    # --verbose prints build-obj even on cache hits. The summary distinguishes
    # completed compilation from cached objects and repeated graph references.
    zig_summary = [line for line in lines if "compile obj " in line]
    zig_compiled = [line for line in zig_summary if " success" in line]
    zig_cached = [line for line in zig_summary if " cached" in line or "reused" in line]
    row = dict(
        variant=variant,
        phase=phase,
        sample=sample,
        measured=measured,
        command=command,
        cwd=str(source),
        seconds=seconds,
        returncode=process.returncode,
        c_compile_commands=len(c_commands),
        zig_compile_invocations=len(zig_invocations),
        zig_compile_commands=len(zig_compiled),
        zig_cached_or_reused=len(zig_cached),
        zig_object_summary=zig_summary,
        compile_commands=c_commands + zig_compiled,
        valid=False,
    )
    report["runs"].append(row)
    save()
    print(
        json.dumps(
            {
                key: value
                for key, value in row.items()
                if key not in {"compile_commands", "zig_object_summary"}
            }
        ),
        flush=True,
    )
    if process.returncode:
        raise RuntimeError(f"{label} failed; logs retained")
    if expected_unit:
        compiled = (
            any("compile obj matmul_f32 " in line for line in zig_compiled)
            if variant == "zig-native"
            else any(expected_unit in line for line in c_commands)
        )
        if not compiled:
            raise AssertionError(
                f"{label} did not compile {expected_unit}; invalid edit timing"
            )
    if variant == "zig-native" and not zig_summary:
        raise AssertionError(f"{label} has no object build summary")
    if phase == "forced-rebuild" and (
        not c_commands or (variant == "zig-native" and not zig_compiled)
    ):
        raise AssertionError(f"{label} did not compile selected C/native objects")
    if phase == "noop" and row["compile_commands"]:
        raise AssertionError(
            f"{label} unexpectedly compiled; inspect retained command lines"
        )
    if (
        phase == "noop"
        and variant == "zig-native"
        and len(zig_cached) != len(zig_summary)
    ):
        raise AssertionError(f"{label} has an object not marked cached/reused")
    row["valid"] = True
    save()


def measure(variant, source, command, units, metal_unit):
    make = variant == "make-c"
    report.setdefault("selected_units", {})[variant] = units
    run(variant, source, command, "warm-prime")
    try:
        edit(source, units, f"final Metal build {variant} full", make)
        run(variant, source, command, "forced-rebuild", 1, True)
    finally:
        restore(source, units, f"{variant} restore-full", make)
    run(variant, source, command, "restore-full")
    for sample in range(1, 3):
        run(variant, source, command, "noop", sample, True)
    try:
        for sample in range(1, 3):
            edit(
                source, [metal_unit], f"final Metal build {variant} edit {sample}", make
            )
            run(variant, source, command, "metal-edit", sample, True, metal_unit)
    finally:
        restore(source, [metal_unit], f"{variant} restore-metal", make)
    run(variant, source, command, "restore-metal")


def main():
    try:
        make_source, zig_source = OUT / "make-source", OUT / "zig-source"
        make_manifest, zig_manifest = snapshot(make_source), snapshot(zig_source)
        if make_manifest != zig_manifest:
            raise RuntimeError(
                "checkout changed during source copies; do not measure differing inputs"
            )
        (OUT / "source-hashes.json").write_text(
            json.dumps(make_manifest, indent=2) + "\n"
        )
        report["build_zig_sha256"] = make_manifest["build.zig"]
        report["ds4_metal_sha256"] = make_manifest["ds4_metal.m"]
        report["matmul_f32_sha256"] = make_manifest["zig/metal/matmul_f32.zig"]
        measure(
            "make-c",
            make_source,
            ["make", f"-j{JOBS}", *TARGETS],
            C_UNITS,
            "ds4_metal.m",
        )
        zig = shutil.which("zig")
        if zig is None:
            raise RuntimeError("zig is not installed/in PATH")
        command = [
            zig,
            "build",
            f"-j{JOBS}",
            "--verbose",
            "--summary",
            "all",
            "--color",
            "off",
            "-Doptimize=ReleaseFast",
            "-Dnative=true",
            "--cache-dir",
            str(OUT / "native-local-cache"),
            "--prefix",
            str(OUT / "native"),
            *TARGETS,
        ]
        measure(
            "zig-native",
            zig_source,
            command,
            NATIVE_C_UNITS + NATIVE_ZIG_UNITS,
            "zig/metal/matmul_f32.zig",
        )
        report["status"] = "complete"
    except BaseException as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        errors = []
        for source, source_files in originals.items():
            try:
                restore(
                    source, source_files, "final restore", source.name == "make-source"
                )
            except Exception as error:
                errors.append(f"{source}: {error}")
        report["restoration_errors"] = errors
        report["all_source_bytes_restored"] = not errors and all(
            (source / name).read_bytes() == data
            for source, source_files in originals.items()
            for name, data in source_files.items()
        )
        report["restored_hashes"] = {
            str(source): {
                name: digest((source / name).read_bytes())
                for name in source_files
                if name in {"ds4_metal.m", "zig/metal/matmul_f32.zig"}
            }
            for source, source_files in originals.items()
        }
        report["summary_seconds"] = {}
        for variant in ["make-c", "zig-native"]:
            report["summary_seconds"][variant] = {}
            for phase in ["forced-rebuild", "noop", "metal-edit"]:
                values = [
                    row["seconds"]
                    for row in report["runs"]
                    if row["variant"] == variant
                    and row["phase"] == phase
                    and row["measured"]
                    and row["valid"]
                ]
                if values:
                    report["summary_seconds"][variant][phase] = dict(
                        values=values, mean=statistics.mean(values)
                    )
        save()


if __name__ == "__main__":
    main()
