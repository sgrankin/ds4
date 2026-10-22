#!/usr/bin/env python3
"""Run only after ROOT releases the compiler/model window; never edits the checkout."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
JOBS = 12
FORCED_SAMPLES = 1  # Set to 2 for a second full content-forced rebuild.
TARGETS = ['ds4', 'ds4-server', 'ds4-agent', 'ds4-bench', 'ds4-eval']

# These are the C/ObjC units actually used by all five Make/Zig-C programs.
C_UNITS = [
    'ds4.c', 'ds4_image.c', 'ds4_distributed.c', 'ds4_tp.c', 'ds4_ssd.c',
    'ds4_layer_pack.c', 'ds4_help.c', 'ds4_prompt_prefix.c', 'ds4_web.c',
    'ds4_kvstore.c', 'ds4_eval_cases.c', 'ds4_gpu_args.c', 'linenoise.c',
    'rax.c', 'ds4_engram.c', 'ds4_metal.m', 'ds4_cli.c', 'ds4_server.c',
    'ds4_agent.c', 'ds4_bench.c', 'ds4_eval.c', 'ds4_state_cache.c',
]
# Native mode still compiles these residual bodies. Do not force the superseded
# C CLI/bench/cache/SSD/prefix/Engram files, which its build does not compile.
NATIVE_C_UNITS = [
    'ds4.c', 'ds4_image.c', 'ds4_distributed.c', 'ds4_tp.c',
    'ds4_layer_pack.c', 'ds4_help.c', 'ds4_web.c', 'ds4_eval_cases.c',
    'ds4_gpu_args.c', 'linenoise.c', 'rax.c', 'ds4_metal.m',
    'ds4_server.c', 'ds4_agent.c', 'ds4_eval.c',
]
# Twenty-two selected handwritten implementation/import files, plus their small
# shared Metal binding adapter. Header translation stays warm and unchanged.
NATIVE_ZIG_UNITS = [
    'zig/state_cache.zig', 'zig/kvstore.zig', 'zig/ssd.zig',
    'zig/prompt_prefix.zig', 'zig/engram.zig', 'zig/gguf.zig',
    'zig/cli_chat.zig', 'zig/tokenizer.zig', 'zig/pretokenizer.zig',
    'zig/chat.zig', 'zig/vocab.zig', 'zig/agent_tools.zig',
    'zig/image.zig', 'zig/image_preprocess.zig', 'zig/bench.zig',
    'zig/cli.zig', 'zig/cli_options.zig', 'zig/cli_diagnostics.zig',
    'zig/metal/f16_publication.zig', 'zig/metal/hc_projection.zig',
    'zig/metal/norm.zig', 'zig/metal/activation.zig', 'zig/metal/bridge.zig',
]

report = {
    'scope': 'All five Metal programs; warm toolchain/local caches; content edits force selected compilation units.',
    'jobs': JOBS, 'forced_samples': FORCED_SAMPLES, 'runs': [],
    'limits': 'Sequential build variants, not cold toolchain installation; no binaries/models/tests are executed.',
}
originals = {}
env = os.environ.copy()
env['ZIG_GLOBAL_CACHE_DIR'] = str(ROOT / '.workspaces/zig-build/global-cache')


def save_report():
    (OUT / 'build-performance.json').write_text(json.dumps(report, indent=2) + '\n')


def snapshot(destination):
    if destination.exists():
        raise RuntimeError(f'{destination} already exists; preserve it and choose a fresh location before rerunning')
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in {'.c', '.h', '.m', '.inc'} or p.name in {'Makefile', 'build.zig', 'build.zig.zon'})]
    # Build scripts declare fixture paths even though only five program targets
    # are requested. Copy small test sources, never fixture data/artifact trees.
    for subtree in ['metal', 'zig', 'third_party/iris', 'tests']:
        files += [p for p in (ROOT / subtree).rglob('*') if p.is_file() and
                  p.suffix in {'.c', '.h', '.m', '.inc', '.zig', '.metal'}]
    destination.mkdir()
    originals[destination] = {}
    for source in sorted(files):
        if source.is_symlink():
            raise RuntimeError(f'refusing source symlink: {source}')
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[destination][str(relative)] = target.read_bytes()


def run(variant, phase, command, source, sample=0, measured=False):
    label = f'{variant}-{phase}' + (f'-{sample}' if sample else '')
    with (OUT / f'{label}.stdout.log').open('wb') as stdout, (OUT / f'{label}.stderr.log').open('wb') as stderr:
        started = time.perf_counter()
        process = subprocess.run(command, cwd=source, env=env, stdout=stdout, stderr=stderr)
        seconds = time.perf_counter() - started
    row = {'variant': variant, 'phase': phase, 'sample': sample, 'measured': measured,
           'command': command, 'cwd': str(source), 'seconds': seconds, 'returncode': process.returncode}
    report['runs'].append(row)
    save_report()
    print(json.dumps(row), flush=True)
    if process.returncode:
        raise RuntimeError(f'{label} failed; see its retained logs')


def edit(source, names, label):
    for name in names:
        comment = f'\n// {label}\n' if name.endswith('.zig') else f'\n/* {label} */\n'
        (source / name).write_bytes(originals[source][name] + comment.encode())


def restore(source, names):
    for name in names:
        data = originals[source][name]
        path = source / name
        if path.read_bytes() != data:
            path.write_bytes(data)


def measure(variant, source, command, units, cache_unit, cli_unit):
    report.setdefault('selected_units', {})[variant] = units
    run(variant, 'warm-prime', command, source)
    try:
        for sample in range(1, FORCED_SAMPLES + 1):
            edit(source, units, f'final build measurement {variant} forced {sample}')
            run(variant, 'forced-rebuild', command, source, sample, True)
    finally:
        restore(source, units)
    run(variant, 'restore-all', command, source)
    for sample in range(1, 3):
        run(variant, 'noop', command, source, sample, True)
    for phase, unit in [('cache-edit', cache_unit), ('cli-edit', cli_unit)]:
        try:
            for sample in range(1, 3):
                edit(source, [unit], f'final build measurement {variant} {phase} {sample}')
                run(variant, phase, command, source, sample, True)
        finally:
            restore(source, [unit])
        run(variant, 'restore-' + phase, command, source)


def main():
    make_source = OUT / 'make-source'
    zig_source = OUT / 'zig-source'
    try:
        snapshot(make_source)
        snapshot(zig_source)
        report['source_copies'] = [str(make_source), str(zig_source)]
        report['build_zig_sha256'] = hashlib.sha256((zig_source / 'build.zig').read_bytes()).hexdigest()
        measure('make-c', make_source, ['make', f'-j{JOBS}', *TARGETS], C_UNITS,
                'ds4_state_cache.c', 'ds4_cli.c')
        zig = shutil.which('zig')
        if zig is None:
            raise RuntimeError('zig is not installed/in PATH')
        for variant, native in [('zig-c', False), ('zig-native', True)]:
            command = [zig, 'build', f'-j{JOBS}', '-Doptimize=ReleaseFast',
                       '-Dnative=' + str(native).lower(), '--cache-dir', str(OUT / (variant + '-local-cache')),
                       '--prefix', str(OUT / variant), *TARGETS]
            units = NATIVE_C_UNITS + NATIVE_ZIG_UNITS if native else C_UNITS
            measure(variant, zig_source, command, units,
                    'zig/state_cache.zig' if native else 'ds4_state_cache.c',
                    'zig/cli.zig' if native else 'ds4_cli.c')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        restoration_errors = []
        for source, files in originals.items():
            try:
                restore(source, files)
            except Exception as error:
                restoration_errors.append(f'{source}: {error}')
        report['all_source_bytes_restored'] = not restoration_errors and all(
            (source / name).read_bytes() == data for source, files in originals.items() for name, data in files.items())
        report['restoration_errors'] = restoration_errors
        report['summary_seconds'] = {}
        for variant in ['make-c', 'zig-c', 'zig-native']:
            report['summary_seconds'][variant] = {}
            for phase in ['forced-rebuild', 'noop', 'cache-edit', 'cli-edit']:
                values = [row['seconds'] for row in report['runs'] if row['variant'] == variant and
                          row['phase'] == phase and row['returncode'] == 0 and row['measured']]
                if values:
                    report['summary_seconds'][variant][phase] = {'samples': len(values), 'mean': statistics.mean(values), 'values': values}
        save_report()


if __name__ == '__main__':
    main()
