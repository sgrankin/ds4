#!/usr/bin/env python3
"""Freeze/run a fixed odd-shift prototype comparison beneath the MAIN guard."""
import argparse
import csv
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

ROOT = Path('/Volumes/Models/ds4')
GUARD = ROOT / 'speed-bench/ideas/pressure_run.py'
GUARD_SHA = '461e8c517e82a687b4579fc450c1e7128b1510192d088bab719682b33c946c3c'
PYTHON = Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python')
PYTHON_SHA = 'b80d260a48404da13a32dd44d7283aadc80f6a6e56ed6e03ed5597ec6f46d48d'
MODEL = dict(path=str(ROOT/'gguf/DeepSeek-V4.1-Flash-Q2.gguf'), bytes=365713686528, mtime_ns=1789237692837653163)
POLICY = dict(DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024', DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_ROUTE_MODE='native')
GEOMETRY = dict(prefix=1409, shift=65, span=512, tail=128, decode=16, ctx=4096, chunk=512, cache_gb=64, capacity=1024)
ARMS = [('old-native-before', 'baseline-probe', 'native'),
        ('new-native', 'new-probe', 'native'),
        ('new-tape-native', 'new-probe', 'tape-native'),
        ('odd65-reuse', 'new-probe', 'reuse'),
        ('old-native-after', 'baseline-probe', 'native')]


def sha(path):
    path = Path(path)
    if path.suffix.lower() == '.gguf':
        raise ValueError('Model hashing forbidden')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def command(output, digest):
    return [str(PYTHON), str(GUARD), str(output/'pressure'), '--', str(PYTHON),
            str(output/'run_suffix_pair_tape_pilot.py'), '--run', str(output),
            '--manifest-sha256', digest]


def verify(output, digest):
    if sha(output/'manifest.json') != digest:
        raise ValueError('Explicit prepared manifest differs')
    plan = json.loads((output/'manifest.json').read_text())
    if plan['geometry'] != GEOMETRY or plan['arms'] != [list(x) for x in ARMS] or plan['policy'] != POLICY or plan['model_from_prior'] != MODEL:
        raise ValueError('Frozen geometry, order, model or policy differs')
    required = {'baseline-probe', 'new-probe', 'build-manifest.json',
                'auxiliary-object-proof.json', 'baseline-core-proof.json', 'final-link-cpu-proof.json',
                'run_suffix_pair_tape_pilot.py', 'corpus.txt',
                'source/ds4.c', 'source/ds4.h', 'source/ds4_suffix_pair_tape.h',
                'source/speed-bench/suffix_pair_tape_bench.c'}
    if not required <= set(plan['files']) or len([p for p in plan['files'] if p.startswith('metal/') and p.endswith('.metal')]) != 26:
        raise ValueError('Required frozen asset inventory incomplete')
    for name, identity in plan['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Prepared asset escapes directory')
        path = output/name
        if path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
            raise ValueError('Frozen asset differs: '+name)
        if name in ('baseline-probe', 'new-probe') and not os.access(path, os.X_OK):
            raise ValueError('Prepared probe is not executable: '+name)
    if sha(GUARD) != GUARD_SHA or sha(PYTHON) != PYTHON_SHA:
        raise ValueError('Guard/interpreter differs')
    prepared = json.loads((output/'prepared-command.json').read_text())
    if prepared != dict(manifest_sha256=digest, command=command(output, digest), cwd=str(ROOT)):
        raise ValueError('Frozen direct-parent command differs')
    return plan


def require_guard(output, digest):
    expected = command(output, digest)
    args = shlex.split(subprocess.check_output(['/bin/ps', '-p', str(os.getppid()), '-o', 'command='], text=True, timeout=2).strip())
    if not args or args[1:] != expected[1:] or not os.path.samefile(args[0], PYTHON) or sha(PYTHON) != PYTHON_SHA or sha(GUARD) != GUARD_SHA:
        raise RuntimeError('Exact pinned MAIN guard parent required')
    with (ROOT/'speed-bench/ideas/artifacts/model-job.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
    raise RuntimeError('MAIN model lock must be held')


def model_identity():
    path = Path(MODEL['path'])
    if path.resolve() != path:
        raise ValueError('Explicit Q2 path must not be an alias')
    value = path.stat()
    identity = dict(path=str(path), bytes=value.st_size, mtime_ns=value.st_mtime_ns)
    if identity != MODEL:
        raise ValueError('Prior model identity changed')
    return identity


def raw_reference(path, approximate):
    with Path(path).open('rb') as stream:
        header = struct.unpack('<8I', stream.read(32))
        expected = (0x53505431, 129280, GEOMETRY['decode'], GEOMETRY['prefix'], GEOMETRY['shift'], GEOMETRY['span'], GEOMETRY['tail'], int(approximate))
        if header != expected:
            raise ValueError('Raw header differs')
        targets = stream.read(header[2]*4)
        if len(targets) != header[2]*4 or any(t >= header[1] for t in struct.unpack('<'+str(header[2])+'I', targets)):
            raise ValueError('Raw teacher-token metadata differs')
        rows = header[2]+2
        remaining = rows*header[1]*4
        digest = hashlib.sha256()
        while remaining:
            part = stream.read(min(remaining, 1<<20))
            if not part:
                raise ValueError('Incomplete native vectors')
            digest.update(part)
            remaining -= len(part)
        candidate_bytes = (header[2]+1)*header[1]*4 if approximate else 0
        if Path(path).stat().st_size != 32+len(targets)+rows*header[1]*4+candidate_bytes:
            raise ValueError('Raw record extent differs')
    return dict(reference_rows=rows, reference_sha256=digest.hexdigest(), teacher_token_hex=targets.hex(), full_sha256=sha(path), bytes=Path(path).stat().st_size)


def validate_quality_csv(path, summary, token_hex):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != GEOMETRY['decode']+1:
        raise ValueError('Candidate quality rows incomplete')
    targets = struct.unpack('<'+str(GEOMETRY['decode'])+'I', bytes.fromhex(token_hex)) + (-1,)
    values = []
    for step, row in enumerate(rows):
        if int(row['step']) != step or int(row['target']) != targets[step]:
            raise ValueError('Quality row order/target differs')
        quality = {k:float(row[k]) for k in ['delta_nll', 'kl', 'tv']}
        if not all(math.isfinite(x) for x in quality.values()) or quality['kl'] < -1e-10 or not 0 <= quality['tv'] <= 1 or any(int(row[k]) not in (0, 1) for k in ['argmax_flip', 'bit_equal']):
            raise ValueError('Nonfinite/out-of-range quality row')
        if step == GEOMETRY['decode'] and quality['delta_nll'] != 0:
            raise ValueError('Terminal row must not claim teacher NLL')
        values.append(quality)
    totals = dict(mean_kl=sum(x['kl'] for x in values)/len(values), mean_tv=sum(x['tv'] for x in values)/len(values), sum_delta_nll=sum(x['delta_nll'] for x in values))
    if any(not math.isclose(summary[k], v, rel_tol=1e-8, abs_tol=1e-10) for k,v in totals.items()) or summary['argmax_flips'] != sum(int(r['argmax_flip']) for r in rows) or summary['bit_equal_rows'] != sum(int(r['bit_equal']) for r in rows):
        raise ValueError('CSV/summary quality arithmetic differs')
    return dict(rows=len(rows), teacher_rows=GEOMETRY['decode'], quality_rows=values, checked_summary=totals)


def diagnostic_state(path, canonical):
    path, canonical = Path(path), Path(canonical)
    with path.open('rb') as stream:
        header = stream.read(52)
    with canonical.open('rb') as stream:
        native = stream.read(52)
    if len(header) != 52 or len(native) != 52 or native[:4] != b'DSV4' or header[:4] != b'DSSD' or header[4:] != native[4:] or path.stat().st_size != canonical.stat().st_size:
        raise ValueError('Private diagnostic state tag/layout differs')
    return dict(sha256=sha(path), bytes=path.stat().st_size, diagnostic_magic='DSSD', canonical_export=False)


def validate_summary(summary, mode):
    integer_fields = ['prefix', 'shift', 'span', 'tail', 'decode', 'raw_reference_rows', 'raw_candidate_rows',
                      'tape_buffer_and_metadata_bytes', 'immutable_tape_bytes', 'peak_staging_bytes_including_reserve',
                      'staging_budget_bytes', 'pair_inputs_reused', 'pair_rows_rebuilt', 'pair_boundary_rows',
                      'reused_tokens', 'fresh_tail_tokens', 'argmax_flips', 'bit_equal_rows']
    if any(type(summary[k]) is not int or summary[k] < 0 for k in integer_fields) or any(type(summary[k]) is not bool for k in ['coverage_rejections_preserve_state', 'precancel_preserves_state', 'partial_cancel_invalidates', 'approximate_export_blocked']) or any(type(summary[k]) not in (int, float) or not math.isfinite(summary[k]) for k in ['fresh_ms_diagnostic', 'reuse_ms_diagnostic', 'mean_kl', 'mean_tv', 'sum_delta_nll']):
        raise ValueError('Summary types/nonfinite fields differ')
    if summary['mode'] != mode or any(summary[k] != GEOMETRY[k] for k in ['prefix', 'shift', 'span', 'tail', 'decode']) or summary['raw_reference_rows'] != GEOMETRY['decode']+2 or summary['raw_candidate_rows'] != (GEOMETRY['decode']+1 if mode=='reuse' else 0):
        raise ValueError('Summary mode/geometry/row completeness differs')
    if summary['peak_staging_bytes_including_reserve'] >= summary['staging_budget_bytes'] or summary['staging_budget_bytes'] != 128<<20:
        raise ValueError('Staging budget differs')
    if mode=='reuse' and (not all(summary[k] for k in ['coverage_rejections_preserve_state', 'precancel_preserves_state', 'partial_cancel_invalidates', 'approximate_export_blocked']) or summary['reused_tokens'] != GEOMETRY['span'] or summary['fresh_tail_tokens'] != GEOMETRY['tail'] or summary['pair_inputs_reused'] != 3*GEOMETRY['span'] or summary['pair_rows_rebuilt'] != 3*((GEOMETRY['prefix']+GEOMETRY['span'])//2-GEOMETRY['prefix']//2) or summary['pair_boundary_rows'] != 3):
        raise ValueError('Odd-boundary/control contract differs')


def prepare(output, build, revision):
    if output.exists():
        raise ValueError('Prepared directory already exists')
    if sha(GUARD) != GUARD_SHA or sha(PYTHON) != PYTHON_SHA:
        raise ValueError('Guard identity differs')
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2]
    for name in ['baseline-probe', 'new-probe', 'manifest.json', 'auxiliary-object-proof.json', 'baseline-core-proof.json', 'final-link-cpu-proof.json']:
        destination = 'build-manifest.json' if name == 'manifest.json' else name
        shutil.copy2(build/name, output/destination)
    shutil.copyfile(Path(__file__), output/'run_suffix_pair_tape_pilot.py')
    shutil.copyfile(source/'speed-bench/promessi_sposi.txt', output/'corpus.txt')
    for path in sorted((source/'metal').glob('*.metal')):
        (output/'metal').mkdir(exist_ok=True)
        shutil.copyfile(path, output/'metal'/path.name)
    for name in ['ds4.c', 'ds4.h', 'ds4_suffix_pair_tape.h', 'speed-bench/suffix_pair_tape_bench.c']:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, destination)
    plan = dict(source_revision=revision, geometry=GEOMETRY, arms=ARMS, policy=POLICY, model_from_prior=MODEL,
                no_model_metadata_in_preparation=True,
                staging_scope='128MiB unique additional owned tape/snapshot/vector/token buffers plus8MiB I/O/64KiB overhead reserve; base graph/model/cache and backend driver metadata monitored separately',
                scope='Single fixed teacher-forced literary trajectory; native parity gate before odd65 approximation, full-vocabulary KL and deltaNLL. Timing diagnostic. Not general task accuracy, forgetting or default promotion.',
                files={str(p.relative_to(output)):dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()})
    put(output/'manifest.json', plan)
    digest = sha(output/'manifest.json')
    put(output/'prepared-command.json', dict(manifest_sha256=digest, command=command(output, digest), cwd=str(ROOT)))
    print(json.dumps(dict(status='PREPARED', files=len(plan['files']), manifest_sha256=digest, command_sha256=sha(output/'prepared-command.json'), model_metadata_touched=False)))


def run(output, digest):
    plan = verify(output, digest)
    require_guard(output, digest)
    identity = model_identity()  # First model resolve/stat, after guard ownership.
    directory = output/'run'
    directory.mkdir(exist_ok=False)
    put(directory/'run-proof.json', dict(model_currently_verified=identity, manifest_sha256=digest, source_revision=plan['source_revision'], guarded=True))
    results = []
    control = None
    environment = {k:v for k,v in os.environ.items() if not k.startswith('DS4')}
    environment.update(POLICY)
    for name, binary, mode in ARMS:
        verify(output, digest)
        require_guard(output, digest)
        if model_identity() != identity:
            raise ValueError('Model changed')
        base = directory/name
        argv = [str(output/binary), '--model', identity['path'], '--corpus', str(output/'corpus.txt'), '--base', str(base), '--mode', mode]
        for key, value in GEOMETRY.items():
            argv += ['--'+key.replace('_','-'), str(value)]
        put(directory/(name+'-command.json'), dict(argv=argv, policy=POLICY))
        print(json.dumps(dict(event='arm_start', arm=name)), flush=True)
        with (directory/(name+'.csv')).open('x') as out, (directory/(name+'.log')).open('x') as err:
            process = subprocess.run(argv, cwd=output, env=environment, stdout=out, stderr=err)
        put(directory/(name+'-exit.json'), dict(returncode=process.returncode))
        if process.returncode:
            raise RuntimeError(name+' failed rc='+str(process.returncode))
        verify(output, digest)
        if model_identity() != identity:
            raise ValueError('Model changed during arm')
        summary = json.loads(Path(str(base)+'.summary.json').read_text())
        validate_summary(summary, mode)
        raw = raw_reference(str(base)+'.raw.bin', mode=='reuse')
        state = {x:sha(str(base)+x) for x in ['.source.state.bin', '.fresh.state.bin']}
        parity = dict(reference_sha256=raw['reference_sha256'], teacher_token_hex=raw['teacher_token_hex'], state=state)
        if control is None:
            control = parity
        elif parity != control:
            raise ValueError(name+' differs from unchanged native core')
        csv_checks = validate_quality_csv(directory/(name+'.csv'), summary, raw['teacher_token_hex']) if mode=='reuse' else None
        private_state = diagnostic_state(str(base)+'.private.state.bin', str(base)+'.fresh.state.bin') if mode=='reuse' else None
        result = dict(arm=name, summary=summary, raw=raw, state=state, csv_checks=csv_checks, private_state=private_state, native_parity=True)
        results.append(result)
        put(directory/(name+'-checked.json'), result)
        print(json.dumps(dict(arm=name, native_parity=True, mean_kl=summary['mean_kl'], sum_delta_nll=summary['sum_delta_nll'], peak_staging_bytes=summary['peak_staging_bytes_including_reserve'])), flush=True)
    put(directory/'summary.json', dict(status='COMPLETE', arms=results, native_controls_exact=True, quality_scope=plan['scope'], timing_promoted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--source-revision')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.run):
        parser.error('Choose prepare or run')
    if args.prepare:
        if not args.build or not args.source_revision or args.manifest_sha256:
            parser.error('Preparation requires build+source revision, no manifest SHA')
        prepare(args.prepare.resolve(), args.build.resolve(), args.source_revision)
    else:
        if not args.manifest_sha256 or args.build or args.source_revision:
            parser.error('Run requires explicit frozen manifest SHA only')
        run(args.run.resolve(), args.manifest_sha256)
