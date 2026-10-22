#!/usr/bin/env python3
"""Run only when ROOT releases the compiler window; changes isolated copies only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = OUT / 'zig-source'
RESULTS = OUT / 'profile-build'
JOBS = [12, 8, 18, 12]

spec = importlib.util.spec_from_file_location('build_measurement', OUT / 'measure_builds.py')
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)
UNITS = measurement.NATIVE_C_UNITS + measurement.NATIVE_ZIG_UNITS
TARGETS = measurement.TARGETS


def main():
    RESULTS.mkdir(exist_ok=True)
    zig = shutil.which('zig')
    if zig is None:
        raise RuntimeError('zig is not installed/in PATH')
    originals = {}
    for name in UNITS:
        path = SOURCE / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'expected isolated regular source: {path}')
        originals[name] = path.read_bytes()
    report = {
        'scope': 'Native all-five content-forced builds; shared warm local/global cache; scheduling only.',
        'cwd': str(SOURCE),
        'jobs_order': JOBS,
        'selected_units': UNITS,
        'source_build_zig_sha256': hashlib.sha256((SOURCE / 'build.zig').read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in originals.items()},
        'runs': [],
        'status': 'running',
    }
    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(ROOT / '.workspaces/zig-build/global-cache')
    result_file = RESULTS / 'results.json'

    def save():
        result_file.write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        for arm, jobs in enumerate(JOBS, 1):
            label = f'{arm}-j{jobs}'
            unique = f'scheduling probe {label} {time.time_ns()}'
            for name, data in originals.items():
                comment = f'\n// {unique}\n' if name.endswith('.zig') else f'\n/* {unique} */\n'
                (SOURCE / name).write_bytes(data + comment.encode())
            command = [zig, 'build', f'-j{jobs}', '-Dnative=true', '-Doptimize=ReleaseFast',
                       '--cache-dir', str(OUT / 'zig-native-local-cache'),
                       '--prefix', str(OUT / 'zig-native'), '--summary', 'all', *TARGETS]
            stdout_path = RESULTS / f'{label}.stdout.log'
            stderr_path = RESULTS / f'{label}.stderr.log'
            with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                start = time.perf_counter()
                process = subprocess.run(command, cwd=SOURCE, env=env, stdout=stdout, stderr=stderr)
                seconds = time.perf_counter() - start
            row = {'arm': arm, 'jobs': jobs, 'command': command, 'seconds': seconds,
                   'returncode': process.returncode,
                   'stdout': str(stdout_path.relative_to(OUT)),
                   'stderr': str(stderr_path.relative_to(OUT))}
            report['runs'].append(row)
            save()
            print(json.dumps(row), flush=True)
            if process.returncode:
                raise RuntimeError(f'{label} failed; inspect retained logs')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        for name, data in originals.items():
            (SOURCE / name).write_bytes(data)
        report['all_source_bytes_restored'] = all((SOURCE / name).read_bytes() == data for name, data in originals.items())
        save()


if __name__ == '__main__':
    main()
