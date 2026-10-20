#!/usr/bin/env python3
"""Freeze/run an endpoint-aligned suffix follow-up comparison beneath the MAIN guard."""
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
CASES = {
    'zero-odd-identity-wrap': dict(prefix=2017, shift=0, span=65, tail=1, decode=16, ctx=4096, chunk=512, cache_gb=64, capacity=1024),
    'inverse-parity-pending': dict(prefix=1408, shift=65, span=65, tail=1, decode=16, ctx=4096, chunk=512, cache_gb=64, capacity=1024),
    'wrap-odd-chunk': dict(prefix=1951, shift=65, span=129, tail=1, decode=16, ctx=4096, chunk=65, cache_gb=64, capacity=1024),
}
ARMS = [('old-native-before', 'baseline-probe', 'native'),
        ('new-native', 'new-probe', 'native'),
        ('new-tape-native', 'new-probe', 'tape-native'),
        ('suffix-reuse', 'new-probe', 'reuse'),
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


def command(output, digest, case):
    return [str(PYTHON), str(GUARD), str(output/'pressure'), '--', str(PYTHON),
            str(output/'run_suffix_pair_tape_followup.py'), '--case', case, '--run', str(output),
            '--manifest-sha256', digest]


def verify(output, digest, case):
    geometry = CASES[case]
    if sha(output/'manifest.json') != digest:
        raise ValueError('Explicit prepared manifest differs')
    plan = json.loads((output/'manifest.json').read_text())
    if plan['case'] != case or plan['geometry'] != geometry or plan['arms'] != [list(x) for x in ARMS] or plan['policy'] != POLICY or plan['model_from_prior'] != MODEL:
        raise ValueError('Frozen geometry, order, model or policy differs')
    required = {'baseline-probe', 'new-probe', 'build-manifest.json',
                'auxiliary-object-proof.json', 'baseline-core-proof.json', 'final-link-cpu-proof.json',
                'run_suffix_pair_tape_followup.py', 'corpus.txt',
                'source/ds4.c', 'source/ds4.h', 'source/ds4_suffix_pair_tape.h',
                'source/speed-bench/suffix_pair_tape_followup_bench.c'}
    if not required <= set(plan['files']) or len([p for p in plan['files'] if p.startswith('metal/') and p.endswith('.metal')]) != 26:
        raise ValueError('Required frozen asset inventory incomplete')
    for name, identity in plan['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Prepared asset escapes directory')
        path = output/name
        if path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
            raise ValueError('Frozen asset differs: '+name)
    if any(not os.access(output/name, os.X_OK) for name in ['baseline-probe', 'new-probe']):
        raise ValueError('Prepared probe is not executable')
    if sha(GUARD) != GUARD_SHA or sha(PYTHON) != PYTHON_SHA:
        raise ValueError('Guard/interpreter differs')
    prepared = json.loads((output/'prepared-command.json').read_text())
    if prepared != dict(manifest_sha256=digest, command=command(output, digest, case), cwd=str(ROOT)):
        raise ValueError('Frozen direct-parent command differs')
    return plan


def require_guard(output, digest, case):
    expected = command(output, digest, case)
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


def raw_reference(path, approximate, geometry):
    with Path(path).open('rb') as stream:
        header = struct.unpack('<8I', stream.read(32))
        expected = (0x53505431, 129280, geometry['decode'], geometry['prefix'], geometry['shift'], geometry['span'], geometry['tail'], int(approximate))
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


def validate_quality_csv(path, summary, token_hex, geometry):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['step', 'target', 'delta_nll', 'kl', 'tv', 'argmax_flip', 'bit_equal']:
            raise ValueError('Quality CSV header differs')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Quality CSV field extent differs')
    if len(rows) != geometry['decode']+1:
        raise ValueError('Candidate quality rows incomplete')
    targets = struct.unpack('<'+str(geometry['decode'])+'I', bytes.fromhex(token_hex)) + (-1,)
    values = []
    for step, row in enumerate(rows):
        if int(row['step']) != step or int(row['target']) != targets[step]:
            raise ValueError('Quality row order/target differs')
        quality = {k:float(row[k]) for k in ['delta_nll', 'kl', 'tv']}
        if not all(math.isfinite(x) for x in quality.values()) or quality['kl'] < -1e-10 or not 0 <= quality['tv'] <= 1 or any(int(row[k]) not in (0, 1) for k in ['argmax_flip', 'bit_equal']):
            raise ValueError('Nonfinite/out-of-range quality row')
        if step == geometry['decode'] and quality['delta_nll'] != 0:
            raise ValueError('Terminal row must not claim teacher NLL')
        values.append(quality)
    totals = dict(mean_kl=sum(x['kl'] for x in values)/len(values), mean_tv=sum(x['tv'] for x in values)/len(values), sum_delta_nll=sum(x['delta_nll'] for x in values))
    if any(not math.isclose(summary[k], v, rel_tol=1e-8, abs_tol=1e-10) for k,v in totals.items()) or summary['argmax_flips'] != sum(int(r['argmax_flip']) for r in rows) or summary['bit_equal_rows'] != sum(int(r['bit_equal']) for r in rows):
        raise ValueError('CSV/summary quality arithmetic differs')
    return dict(rows=len(rows), teacher_rows=geometry['decode'], quality_rows=values, checked_summary=totals)


def diagnostic_state(path, canonical):
    path, canonical = Path(path), Path(canonical)
    with path.open('rb') as stream:
        header = stream.read(52)
    with canonical.open('rb') as stream:
        native = stream.read(52)
    if len(header) != 52 or len(native) != 52 or native[:4] != b'DSV4' or header[:4] != b'DSSD' or header[4:] != native[4:] or path.stat().st_size != canonical.stat().st_size:
        raise ValueError('Private diagnostic state tag/layout differs')
    return dict(sha256=sha(path), bytes=path.stat().st_size, diagnostic_magic='DSSD', canonical_export=False)


def zero_identity_records(base, approximate, raw, state, geometry):
    """Check retained byte evidence; never infer identity from rounded metrics."""
    if geometry['shift'] != 0:
        raise ValueError('Identity records require a frozen zero-shift geometry')
    if state['.source.state.bin'] != state['.fresh.state.bin']:
        raise ValueError('Zero-shift exact source/fresh graph bytes differ')
    row_bytes = 129280*4
    first = 32 + geometry['decode']*4
    path = Path(str(base)+'.raw.bin')
    with path.open('rb') as a, path.open('rb') as b:
        a.seek(first); b.seek(first+row_bytes)
        if a.read(row_bytes) != b.read(row_bytes):
            raise ValueError('Zero-shift exact source/fresh initial vectors differ')
        if approximate:
            a.seek(first+row_bytes)
            b.seek(first+(geometry['decode']+2)*row_bytes)
            remaining = (geometry['decode']+1)*row_bytes
            while remaining:
                n = min(remaining, 1<<18)
                left, right = a.read(n), b.read(n)
                if len(left) != n or left != right:
                    raise ValueError('Zero-shift repaired full vectors differ')
                remaining -= n
    normalized_state = None
    if approximate:
        with Path(str(base)+'.private.state.bin').open('rb') as stream:
            if stream.read(4) != b'DSSD':
                raise ValueError('Actual diagnostic magic missing')
            digest = hashlib.sha256(b'DSV4')
            while part := stream.read(1<<18):
                digest.update(part)
        normalized_state = digest.hexdigest()
        if normalized_state != state['.fresh.state.bin']:
            raise ValueError('Zero-shift repaired graph bytes differ beyond magic')
    return dict(source_fresh_native_state_exact=True, source_fresh_native_vector_exact=True,
                repaired_full_vectors_exact=bool(approximate),
                repaired_normalized_state_sha256=normalized_state,
                canonical_normalization='Validated4-byte DSSD→DSV4 magic only' if approximate else None,
                retained_raw_sha256=raw['full_sha256'])


def validate_summary(summary, mode, geometry):
    integer_fields = ['prefix', 'shift', 'span', 'tail', 'decode', 'raw_reference_rows', 'raw_candidate_rows',
                      'tape_buffer_and_metadata_bytes', 'immutable_tape_bytes', 'peak_staging_bytes_including_reserve',
                      'staging_budget_bytes', 'pair_inputs_reused', 'pair_rows_rebuilt', 'pair_boundary_rows',
                      'reused_tokens', 'fresh_tail_tokens', 'argmax_flips', 'bit_equal_rows']
    if any(type(summary[k]) is not int or summary[k] < 0 for k in integer_fields) or any(type(summary[k]) is not bool for k in ['coverage_rejections_preserve_state', 'precancel_preserves_state', 'partial_cancel_invalidates', 'approximate_export_blocked', 'endpoint_partition_aligned', 'identity_native_checked', 'identity_native_state_equal', 'identity_native_logits_equal', 'identity_reuse_state_equal']) or any(type(summary[k]) not in (int, float) or not math.isfinite(summary[k]) for k in ['fresh_ms_diagnostic', 'reuse_ms_diagnostic', 'mean_kl', 'mean_tv', 'sum_delta_nll']):
        raise ValueError('Summary types/nonfinite fields differ')
    if summary['mode'] != mode or any(summary[k] != geometry[k] for k in ['prefix', 'shift', 'span', 'tail', 'decode']) or summary['raw_reference_rows'] != geometry['decode']+2 or summary['raw_candidate_rows'] != (geometry['decode']+1 if mode=='reuse' else 0):
        raise ValueError('Summary mode/geometry/row completeness differs')
    if summary['peak_staging_bytes_including_reserve'] >= summary['staging_budget_bytes'] or summary['staging_budget_bytes'] != 128<<20:
        raise ValueError('Staging budget differs')
    if mode=='reuse' and (not all(summary[k] for k in ['coverage_rejections_preserve_state', 'precancel_preserves_state', 'partial_cancel_invalidates', 'approximate_export_blocked']) or summary['reused_tokens'] != geometry['span'] or summary['fresh_tail_tokens'] != geometry['tail'] or summary['pair_inputs_reused'] != 3*geometry['span'] or summary['pair_rows_rebuilt'] != 3*((geometry['prefix']+geometry['span'])//2-geometry['prefix']//2) or summary['pair_boundary_rows'] != (3 if geometry['prefix'] % 2 else 0)):
        raise ValueError('Odd-boundary/control contract differs')
    zero = geometry['shift'] == 0
    if not summary['endpoint_partition_aligned'] or summary['identity_native_checked'] != zero:
        raise ValueError('Endpoint partition/identity pre-gate contract differs')
    if zero:
        if not summary['identity_native_state_equal'] or not summary['identity_native_logits_equal']:
            raise ValueError('Zero identity native source/fresh partition parity failed')
        if mode == 'reuse' and (not summary['identity_reuse_state_equal'] or summary['bit_equal_rows'] != geometry['decode']+1 or summary['argmax_flips'] != 0):
            raise ValueError('Zero identity repaired state/vector byte parity failed')
    elif any(summary[k] for k in ['identity_native_state_equal', 'identity_native_logits_equal', 'identity_reuse_state_equal']):
        raise ValueError('Nonzero shift must not claim identity checks')
    if mode != 'reuse' and summary['identity_reuse_state_equal']:
        raise ValueError('Native arm must not claim repaired identity state')


def prepare(output, build, revision, case):
    geometry = CASES[case]
    if output.exists():
        raise ValueError('Prepared directory already exists')
    if sha(GUARD) != GUARD_SHA or sha(PYTHON) != PYTHON_SHA:
        raise ValueError('Guard identity differs')
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2]
    for name in ['baseline-probe', 'new-probe', 'manifest.json', 'auxiliary-object-proof.json', 'baseline-core-proof.json', 'final-link-cpu-proof.json']:
        destination = 'build-manifest.json' if name == 'manifest.json' else name
        shutil.copy2(build/name, output/destination)
    if any(not os.access(output/name, os.X_OK) for name in ['baseline-probe', 'new-probe']):
        raise ValueError('Copied probe is not executable')
    shutil.copy2(Path(__file__), output/'run_suffix_pair_tape_followup.py')
    shutil.copy2(source/'speed-bench/promessi_sposi.txt', output/'corpus.txt')
    for path in sorted((source/'metal').glob('*.metal')):
        (output/'metal').mkdir(exist_ok=True)
        shutil.copy2(path, output/'metal'/path.name)
    for name in ['ds4.c', 'ds4.h', 'ds4_suffix_pair_tape.h', 'speed-bench/suffix_pair_tape_followup_bench.c']:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, destination)
    plan = dict(source_revision=revision, case=case, geometry=geometry, arms=ARMS, policy=POLICY, model_from_prior=MODEL,
                no_model_metadata_in_preparation=True,
                staging_scope='128MiB unique additional owned tape/snapshot/vector/token buffers plus8MiB I/O/64KiB overhead reserve; base graph/model/cache and backend driver metadata monitored separately',
                scope='Endpoint-aligned fixed teacher-forced literary trajectory; native parity gate, zero-edit byte identity, full-vocabulary KL and deltaNLL. Per-case controls; configured chunk changes source/fresh as well as repair partition. Timing diagnostic. Not task accuracy, forgetting or default promotion.',
                files={str(p.relative_to(output)):dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()})
    put(output/'manifest.json', plan)
    digest = sha(output/'manifest.json')
    put(output/'prepared-command.json', dict(manifest_sha256=digest, command=command(output, digest, case), cwd=str(ROOT)))
    print(json.dumps(dict(status='PREPARED', files=len(plan['files']), manifest_sha256=digest, command_sha256=sha(output/'prepared-command.json'), model_metadata_touched=False)))


def run(output, digest, case):
    geometry = CASES[case]
    plan = verify(output, digest, case)
    require_guard(output, digest, case)
    identity = model_identity()  # First model resolve/stat, after guard ownership.
    directory = output/'run'
    directory.mkdir(exist_ok=False)
    put(directory/'run-proof.json', dict(model_currently_verified=identity, manifest_sha256=digest, source_revision=plan['source_revision'], guarded=True))
    results = []
    control = None
    environment = {k:v for k,v in os.environ.items() if not k.startswith('DS4')}
    environment.update(POLICY)
    for name, binary, mode in ARMS:
        verify(output, digest, case)
        require_guard(output, digest, case)
        if model_identity() != identity:
            raise ValueError('Model changed')
        base = directory/name
        argv = [str(output/binary), '--model', identity['path'], '--corpus', str(output/'corpus.txt'), '--base', str(base), '--mode', mode]
        for key, value in geometry.items():
            argv += ['--'+key.replace('_','-'), str(value)]
        put(directory/(name+'-command.json'), dict(argv=argv, policy=POLICY))
        print(json.dumps(dict(event='arm_start', arm=name)), flush=True)
        with (directory/(name+'.csv')).open('x') as out, (directory/(name+'.log')).open('x') as err:
            process = subprocess.run(argv, cwd=output, env=environment, stdout=out, stderr=err)
        put(directory/(name+'-exit.json'), dict(returncode=process.returncode))
        if process.returncode:
            raise RuntimeError(name+' failed rc='+str(process.returncode))
        verify(output, digest, case)
        if model_identity() != identity:
            raise ValueError('Model changed during arm')
        summary = json.loads(Path(str(base)+'.summary.json').read_text())
        validate_summary(summary, mode, geometry)
        raw = raw_reference(str(base)+'.raw.bin', mode=='reuse', geometry)
        state = {x:sha(str(base)+x) for x in ['.source.state.bin', '.fresh.state.bin']}
        parity = dict(reference_sha256=raw['reference_sha256'], teacher_token_hex=raw['teacher_token_hex'], state=state)
        if control is None:
            control = parity
        elif parity != control:
            raise ValueError(name+' differs from unchanged native core')
        csv_checks = validate_quality_csv(directory/(name+'.csv'), summary, raw['teacher_token_hex'], geometry) if mode=='reuse' else None
        private_state = diagnostic_state(str(base)+'.private.state.bin', str(base)+'.fresh.state.bin') if mode=='reuse' else None
        identity_checks = zero_identity_records(base, mode=='reuse', raw, state, geometry) if geometry['shift']==0 else None
        result = dict(arm=name, summary=summary, raw=raw, state=state, csv_checks=csv_checks, private_state=private_state, identity_checks=identity_checks, native_parity=True)
        results.append(result)
        put(directory/(name+'-checked.json'), result)
        print(json.dumps(dict(arm=name, native_parity=True, mean_kl=summary['mean_kl'], sum_delta_nll=summary['sum_delta_nll'], peak_staging_bytes=summary['peak_staging_bytes_including_reserve'])), flush=True)
    put(directory/'summary.json', dict(status='COMPLETE', case=case, geometry=geometry, arms=results, native_controls_exact=True, quality_scope=plan['scope'], timing_promoted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=tuple(CASES))
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--source-revision')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if sum(bool(x) for x in [args.prepare, args.run, args.preflight]) != 1:
        parser.error('Choose prepare, preflight or run')
    if args.preflight:
        if not args.manifest_sha256 or args.build or args.source_revision:
            parser.error('Preflight requires explicit frozen manifest SHA only')
        plan = verify(args.preflight.resolve(), args.manifest_sha256, args.case)
        print(json.dumps(dict(status='PREFLIGHT_PASS', case=args.case, files=len(plan['files']), model_metadata_touched=False)))
    elif args.prepare:
        if not args.build or not args.source_revision or args.manifest_sha256:
            parser.error('Preparation requires build+source revision, no manifest SHA')
        prepare(args.prepare.resolve(), args.build.resolve(), args.source_revision, args.case)
    else:
        if not args.manifest_sha256 or args.build or args.source_revision:
            parser.error('Run requires explicit frozen manifest SHA only')
        run(args.run.resolve(), args.manifest_sha256, args.case)
