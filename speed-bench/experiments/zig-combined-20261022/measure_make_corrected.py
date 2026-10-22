#!/usr/bin/env python3
"""Run AFTER measure_builds.py finishes and ROOT releases the compiler window."""
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = OUT / 'make-source'
RESULT = OUT / 'build-performance-make-corrected.json'
COMMAND = ['make', '-j12', 'ds4', 'ds4-server', 'ds4-agent', 'ds4-bench', 'ds4-eval']
FORCED_SAMPLES = 2  # One is sufficient if the remaining window is short.
UNITS = [
    'ds4.c', 'ds4_image.c', 'ds4_distributed.c', 'ds4_tp.c', 'ds4_ssd.c',
    'ds4_layer_pack.c', 'ds4_help.c', 'ds4_prompt_prefix.c', 'ds4_web.c',
    'ds4_kvstore.c', 'ds4_eval_cases.c', 'ds4_gpu_args.c', 'linenoise.c',
    'rax.c', 'ds4_engram.c', 'ds4_metal.m', 'ds4_cli.c', 'ds4_server.c',
    'ds4_agent.c', 'ds4_bench.c', 'ds4_eval.c', 'ds4_state_cache.c',
]
report = {
    'scope': 'Corrected Make C rerun: Apple Make 3.81 whole-second source timestamps.',
    'invalid_prior_evidence': 'Make edit timings in build-performance.json missed compilation; retain those original logs, do not use them as valid Make build timings.',
    'command': COMMAND, 'cwd': str(SOURCE), 'forced_samples': FORCED_SAMPLES,
    'timestamp_wait_seconds': 1.05, 'runs': [], 'source_writes': [],
}
originals = {}


def save():
    RESULT.write_text(json.dumps(report, indent=2) + '\n')


def write_changed(contents, label):
    changed = {name: data for name, data in contents.items() if (SOURCE / name).read_bytes() != data}
    if not changed:
        return
    # Make 3.81 ignores subsecond mtimes. Wait AFTER the previous build and
    # BEFORE source writes; this time is deliberately outside the build timer.
    time.sleep(1.05)
    timestamps = {}
    for name, data in changed.items():
        path = SOURCE / name
        path.write_bytes(data)
        timestamps[name] = path.stat().st_mtime_ns
    report['source_writes'].append({'label': label, 'wall_time_ns': time.time_ns(), 'mtime_ns': timestamps})
    save()


def edit(names, label):
    marker = ('\n/* corrected Make timing ' + label + ' */\n').encode()
    write_changed({name: originals[name] + marker for name in names}, label)


def restore(names, label):
    write_changed({name: originals[name] for name in names}, label)


def run(phase, sample=0, measured=False, expect_compile=False):
    label = 'make-c-corrected-' + phase + (f'-{sample}' if sample else '')
    stdout_path = OUT / (label + '.stdout.log')
    with stdout_path.open('wb') as stdout, (OUT / (label + '.stderr.log')).open('wb') as stderr:
        started = time.perf_counter()
        process = subprocess.run(COMMAND, cwd=SOURCE, stdout=stdout, stderr=stderr)
        seconds = time.perf_counter() - started
    compile_commands = stdout_path.read_bytes().count(b' -c ')
    row = {'phase': phase, 'sample': sample, 'measured': measured,
           'command': COMMAND, 'cwd': str(SOURCE), 'seconds': seconds,
           'returncode': process.returncode, 'compile_commands': compile_commands,
           'wall_finished_ns': time.time_ns()}
    report['runs'].append(row)
    save()
    print(json.dumps(row), flush=True)
    if process.returncode:
        raise RuntimeError(f'{label} failed; retained logs identify the failure')
    if expect_compile and compile_commands == 0:
        raise AssertionError(f'{label} did not compile: no " -c " command in stdout')


def main():
    try:
        # The earlier runner owns this copy until it finishes and restores it.
        # Never reconstruct/reset it here, and never edit the real checkout.
        for name in UNITS:
            originals[name] = (SOURCE / name).read_bytes()
            if originals[name] != (ROOT / name).read_bytes():
                raise RuntimeError(f'{name} differs from checkout: wait for the original runner to finish restoration')
        run('warm-prime')
        try:
            for sample in range(1, FORCED_SAMPLES + 1):
                edit(UNITS, f'forced-{sample}')
                run('forced-rebuild', sample, True, True)
        finally:
            restore(UNITS, 'restore-all')
        run('restore-all')
        for sample in range(1, 3):
            run('noop', sample, True)
        for phase, unit in [('cache-edit', 'ds4_state_cache.c'), ('cli-edit', 'ds4_cli.c')]:
            try:
                for sample in range(1, 3):
                    edit([unit], f'{phase}-{sample}')
                    run(phase, sample, True, True)
            finally:
                restore([unit], 'restore-' + phase)
            run('restore-' + phase)
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        try:
            restore(originals, 'final-restore')
            report['all_source_bytes_restored'] = all((SOURCE / name).read_bytes() == data for name, data in originals.items())
        except Exception as error:
            report['all_source_bytes_restored'] = False
            report['restoration_error'] = str(error)
        report['summary_seconds'] = {}
        for phase in ['forced-rebuild', 'noop', 'cache-edit', 'cli-edit']:
            values = [row['seconds'] for row in report['runs'] if row['phase'] == phase and
                      row['returncode'] == 0 and row['measured'] and (phase == 'noop' or row['compile_commands'] > 0)]
            if values:
                report['summary_seconds'][phase] = {'samples': len(values), 'mean': statistics.mean(values), 'values': values}
        save()


if __name__ == '__main__':
    main()
