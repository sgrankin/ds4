#!/usr/bin/env python3
"""Archive completed fixed confirmation, verifying retained arrays once."""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import statistics

ROOT = Path('/Volumes/Models/ds4')
WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.workspaces/idea-resident-three-confirmation-fixed'
BASE = SOURCE / 'speed-bench/ideas/artifacts/resident-three-confirmation-prepared-20261002-v2'
BUILD = ROOT / '.workspaces/idea-resident-three-confirmation/speed-bench/ideas/artifacts/resident-three-confirmation-build'
OUT = WORKSPACE / 'speed-bench/ideas/evidence/resident-three-confirmation-completed-20261002-v2'
MANIFEST_SHA = 'b80ecd4abdee67976e4b3824892ff34514a7214fa58687dd3746e692e4657df9'
COMMAND_SHA = '009da3bdbc763311e7e5d83b055e5cdb9450cb8bbda6ca022b7ed2dd1e4af7c1'
BUILD_SHA = '4ef05d5c09c9f20d1ea5fcf47b103d4b1f7a8fbcbdbf2f0b14f6ec7a86a8255b'
HELPER_SHA = '1ced202ca2ddf3d3394007b1bfafae4cea7ae4c27c8f4173c11b650fdb1dacd5'
FROZEN_TEST_SHA = 'b1ccef6fcecc518849a25ccc6585567bbed8b3bf4b8e9085eb2d5179b4e83084'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def main():
    verified = {}

    def verify(path, expected):
        path = path.resolve()
        assert path.suffix.lower() != '.gguf'
        if str(path) not in verified:
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            verified[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size}
        assert verified[str(path)]['sha256'] == expected, path
        return verified[str(path)]

    verify(BASE / 'manifest.json', MANIFEST_SHA)
    verify(BASE / 'prepared-command.json', COMMAND_SHA)
    plan = load(BASE / 'manifest.json')
    assert len(plan['files']) == 64 and plan['binary_sha256'] == HELPER_SHA
    assert plan['runtime_commit'] == 'b9408c3b8e3ec5b22ad7efff3e365fd81818c317'
    for name, expected in plan['files'].items():
        verify(BASE / name, expected)
    verify(BASE / 'binary-manifest.json', BUILD_SHA)
    build = load(BASE / 'binary-manifest.json')
    assert build['objects_unchanged'] and not build['core_rebuilt']
    for name, expected in build['files'].items():
        verify(BUILD / name, expected)
    verify(SOURCE / 'speed-bench/ideas/run_resident_confirmation.py', plan['files']['run_resident_confirmation.py'])
    cpu_proof = load(BASE / 'cpu-proof.json')
    assert cpu_proof['source_commit'] == '38c758fe730f7d3780d93bad11ca27bf07aeadd4'
    verify(SOURCE / 'tests/test_resident_confirmation.py', FROZEN_TEST_SHA)
    verify(BASE / 'cpu-test.json', cpu_proof['cpu_test_sha256'])
    test_pin_mismatch = {'original_cpu_proof_field': cpu_proof['cpu_test_sha256'],
                         'clarified_field_identifies': 'cpu_stdout_json_sha256 for original cpu-test.json',
                         'frozen38_jj_blob_and_disk_sha256': FROZEN_TEST_SHA,
                         'bytes': 9125, 'original_cpu_proof_unchanged': True,
                         'first_archiver_refusal_before_large_raw_hash_or_archive_creation': True}
    cpu_checks = WORKSPACE / 'speed-bench/ideas/evidence/confirmation-archive-cpu-checks-20261002'
    fresh_cpu = load(cpu_checks / 'focused-cpu-proof-after-tempdir.json')
    assert fresh_cpu['returncode'] == 0 and fresh_cpu['test_source_sha256'] == FROZEN_TEST_SHA
    assert fresh_cpu['driver_source_sha256'] == plan['files']['run_resident_confirmation.py']
    verify(cpu_checks / 'focused-cpu-test-after-tempdir.log', fresh_cpu['test_log_sha256'])
    assert load(cpu_checks / 'focused-cpu-test-after-tempdir.log')['trials'] == 37
    verify(Path(plan['guard_command_template'][1]), plan['guard_sha256'])
    interpreter = plan['guard_interpreter_identity']
    verify(Path(interpreter['path']), interpreter['sha256'])
    freeze = load(BASE / 'input-freeze.json')
    assert [row['case'] for row in freeze['cases']] == [f'case_{i:03d}' for i in range(2, 96, 6)]
    assert plan['inputs'] == freeze['cases']
    run = BASE / 'run'
    summary = load(run / 'summary.json')
    assert [row['name'] for row in summary] == ['native-before', 'broader3', 'native-after']
    before, candidate, after = summary
    checks = load(run / 'checks.json')
    assert checks['native_before_after_reference_fullraw_exact'] and checks['native_demand_io_equal']
    assert checks['policy_locked_before_confirmation'] and not checks['timing_promoted']
    assert all(row['tokens'] == 512 and row['cases'] == 16 for row in summary)
    assert before['io'] == after['io'] and before['raw_sha256'] == after['raw_sha256'] == checks['reference_sha256']
    for row in (before, after):
        assert row['bit_equal_rows'] == 512 and all(row[name] == 0 for name in ('delta_nll', 'kl_base_to_raw', 'tv', 'argmax_flips'))
    assert load(BASE / 'pressure/result.json') == {'status': 'complete', 'child_returncode': 0}
    release = load(BASE / 'root-release-proof.json')
    assert release['root_session'] == 7261 and release['guard_returncode'] == 0
    assert release['root_lock_reacquired'] and release['matching_live_model_processes'] == []
    assert release['samples'] == 140 and release['adverse_samples'] == 0 and release['swap_growth_gib'] == 0
    assert len((BASE / 'pressure/pressure.jsonl').read_text().splitlines()) == 140
    launch_log = ROOT / 'speed-bench/ideas/artifacts/confirmation16-three-root-launch-20261002.log'
    launch_rows = [json.loads(line) for line in launch_log.read_text().splitlines()]
    assert launch_rows[0] == load(BASE / 'root-launch-proof.json')
    assert launch_rows[-1] == load(BASE / 'pressure/result.json')
    assert launch_rows[1:-1] == [{k: v for k, v in row.items() if k != 'per_prompt'} for row in summary]
    root_preflight = load(ROOT / 'speed-bench/ideas/artifacts/confirmation16-three-root-preflight-20261002.json')
    assert root_preflight['manifest_sha256'] == MANIFEST_SHA and root_preflight['command_sha256'] == COMMAND_SHA
    assert root_preflight['files_checked'] == 64 and root_preflight['runtime_commit'] == plan['runtime_commit']

    retained_arrays = {}
    for name, expected in [('reference.bin', checks['reference_sha256'])] + [(row['name'] + '-raw.bin', row['raw_sha256']) for row in summary]:
        identity = verify(run / name, expected)
        assert identity['bytes'] == 264777744
        retained_arrays[name] = {'path': str(run / name), **identity,
                                'verified_by_archive_full_streaming_sha_once': True,
                                'copied': False, 'layout': before['raw_layout']}
    rows = list(csv.DictReader((run / 'broader3.csv').open()))
    groups = {}
    for row in rows:
        groups.setdefault(row['case'], []).append(row)
    assert len(rows) == 512 and list(groups) == [row['case'] for row in freeze['cases']]
    assert all(len(group) == 32 for group in groups.values())
    prompt_means = [statistics.mean(float(row['delta_raw_nll']) for row in group) for group in groups.values()]
    rng = random.Random(20261002)
    boots = sorted(statistics.mean(rng.choices(prompt_means, k=16)) for _ in range(2000))
    mean = statistics.mean(prompt_means)
    interval = [boots[49], boots[1949]]
    assert mean == candidate['prompt_mean_delta_nll'] and interval == candidate['prompt_bootstrap_95']
    assert mean > 0 and interval[0] < 0 < interval[1]
    kl = sorted(float(row['raw_kl']) for row in rows)
    assert kl[506] == candidate['kl_p99'] and kl[-1] == candidate['kl_max']
    native_bytes = before['io']['requested_bytes']
    candidate_bytes = candidate['io']['requested_bytes']
    assert native_bytes == before['io']['returned_bytes'] == before['io']['cache_loader_bytes'] == 207277056000
    assert candidate_bytes == candidate['io']['returned_bytes'] == candidate['io']['cache_loader_bytes'] == 147995320320
    reduction = (1 - candidate_bytes / native_bytes) * 100

    OUT.mkdir(parents=True, exist_ok=False)
    files = {}
    originals = {}

    def put(name, data, original=None):
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
        files[name] = {'sha256': digest(data), 'bytes': len(data)}
        if original is not None:
            originals[name] = {'path': str(original), **files[name]}

    def copy(name, original, expected=None):
        data = original.read_bytes()
        if expected is not None:
            assert digest(data) == expected, original
        put(name, data, original)

    def write(name, value):
        put(name, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())

    for source in sorted(run.iterdir()):
        if source.suffix != '.bin':
            copy('run/' + source.name, source)
    for name in ('manifest.json', 'prepared-command.json', 'input-freeze.json', 'fixtures.tsv',
                 'binary-manifest.json', 'cpu-proof.json', 'cpu-test.json',
                 'root-launch-proof.json', 'root-release-proof.json'):
        copy('prepared/' + name, BASE / name)
    for name in ('command.json', 'pressure.jsonl', 'result.json'):
        copy('guard/' + name, BASE / 'pressure' / name)
    for name in ('run_resident_confirmation.py', 'raw_helpers.py'):
        copy('source/' + name, BASE / name, plan['files'][name])
    copy('source/test_resident_confirmation.py', SOURCE / 'tests/test_resident_confirmation.py', FROZEN_TEST_SHA)
    copy('source/pressure_run.py', Path(plan['guard_command_template'][1]), plan['guard_sha256'])
    for source in sorted((BASE / 'inputs').iterdir()):
        copy('inputs/' + source.name, source, plan['files'][str(source.relative_to(BASE))])
    for name in ('source/quality_probe_io.c', 'source/ds4_resident_route.h',
                 'source/resident_route_bfcde97a.h', 'source/test_resident_three.c',
                 'parent-runtime-manifest.json', 'compile.log', 'self-test.log'):
        copy('build-proof/' + name, BUILD / name, build['files'][name])
    copy('root-launch.log', launch_log)
    copy('root-preflight.json', ROOT / 'speed-bench/ideas/artifacts/confirmation16-three-root-preflight-20261002.json')
    copy('source/archive_resident_confirmation.py', Path(__file__).resolve())
    for source in sorted(cpu_checks.iterdir()):
        copy('cpu-independent/' + source.name, source)
    write('retained-raw-arrays.json', retained_arrays)
    write('cpu-test-pin-discrepancy.json', test_pin_mismatch)
    write('verified-original-asset-inventory.json', verified)
    per_prompt = io.StringIO()
    writer = csv.writer(per_prompt)
    writer.writerow(['case', 'tokens', 'mean_delta_nll', 'mean_kl', 'max_delta_nll', 'max_kl', 'argmax_flips'])
    for name, group in groups.items():
        writer.writerow([name, len(group), statistics.mean(float(row['delta_raw_nll']) for row in group),
                         statistics.mean(float(row['raw_kl']) for row in group),
                         max(float(row['delta_raw_nll']) for row in group),
                         max(float(row['raw_kl']) for row in group), sum(int(row['argmax_flip']) for row in group)])
    put('per-prompt-quality.csv', per_prompt.getvalue().encode())
    result = {'status': 'COMPLETE_RAW_NATIVE_CONTROLS_PASS_APPROXIMATE_QUALITY_CHANGED',
              'root_session': 7261, 'producer_and_guard_returncode': 0,
              'frozen_source': '38c758fe730f7d3780d93bad11ca27bf07aeadd4',
              'frozen_manifest_sha256': MANIFEST_SHA, 'command_sha256': COMMAND_SHA,
              'helper_sha256': HELPER_SHA, 'inputs_selection': freeze['selection'],
              'input_outcome_selection': False, 'inputs': 16, 'teacher_forced_rows': 512,
              'mean_delta_nll': mean, 'paired_prompt_bootstrap_95': interval,
              'bootstrap': plan['bootstrap'], 'worst_prompt_delta_nll': max(prompt_means),
              'mean_kl': statistics.mean(kl), 'kl_p99': kl[506], 'kl_max': kl[-1],
              'mean_tv': candidate['tv'], 'argmax_flips': candidate['argmax_flips'],
              'bit_equal_rows': candidate['bit_equal_rows'],
              'whole_prefill_plus_decode_logical_native_bytes': native_bytes,
              'whole_prefill_plus_decode_logical_candidate_bytes': candidate_bytes,
              'logical_bytes_reduction_percent': reduction,
              'logical_bytes_difference': native_bytes - candidate_bytes,
              'quality_interpretation': 'Positive mean deltaNLL; paired prompt bootstrap interval crosses zero. This does not establish equal quality or generalization.',
              'io_scope': 'Cumulative explicit expert-loader requested=returned=cache-loader bytes over all prefill and decode; no background worker. Logical read bytes, not physical SSD traffic.',
              'timing_claim': False, 'equal_quality_claim': False, 'defaults_promoted': False,
              'independent_resident_arithmetic_review': 'PASS mean, KL, worst prompt, paired bootstrap, fixed 2mod6 case order and requested/returned sums',
              'archive_full_raw_streaming_sha_verified_once': True,
              'original_cpu_test_pin_mismatch': test_pin_mismatch,
              'independent_actual_frozen38_cpu_tests': {'trials': 37, 'returncode': 0,
                  'source_sha256': FROZEN_TEST_SHA, 'log_sha256': fresh_cpu['test_log_sha256'],
                  'first_missing_temp_parent_invocation_preserved': True},
              'model_metadata_opened_by_archive': False, 'model_opened_by_archive': False,
              'build_or_gpu_called_by_archive': False, 'original_run_mutated': False}
    write('result.json', result)
    text = f'''# Completed held-out broader-three confirmation

ROOT session 7261 and the MAIN guard completed with rc0. Fresh native-before,
native-after and reference arrays have identical full SHA and all 512 raw rows
are bit-exact; both native whole-engine IO snapshots are identical. The candidate
uses the same helper/runtime/shaders, with the broader-three policy frozen before
these confirmation outcomes.

| Measure | Native | Broader three |
| --- | --- | --- |
| Mean teacher-forced NLL | {before['avg_nll']:.12f} | {candidate['avg_nll']:.12f} |
| Whole P+D logical requested/returned bytes | {native_bytes} | {candidate_bytes} |

Candidate mean deltaNLL is **+{mean:.9f}** (positive means worse target likelihood).
The paired 16-prompt bootstrap 95% interval is [{interval[0]:+.9f}, {interval[1]:+.9f}],
crossing zero. It uses 2000 resamples, seed 20261002, sorted indices 49/1949.
Worst prompt mean deltaNLL is +{max(prompt_means):.9f}; mean KL is {statistics.mean(kl):.9f},
KL p99 {kl[506]:.9f}, maximum {kl[-1]:.9f}; TV mean {candidate['tv']:.9f};
argmax flips are {candidate['argmax_flips']}/512 and only {candidate['bit_equal_rows']}/512 rows are bit-identical.
Every row and every per-prompt result is retained; no tail or case was removed.

Whole prefill-plus-decode logical demanded bytes decrease by **{reduction:.6f}%**
({native_bytes - candidate_bytes} bytes). Requested, returned and cache-loader counters
agree within each arm. Route-footer skipped bytes are a different counter and are
not added to this total. This is not measured physical SSD traffic. Diagnostic
timings remain in original outputs but support no clean speed claim here. The
positive mean and interval crossing zero do not establish equal quality.

The untouched inputs are the first 16 case indices 2 mod 6 from the frozen 96-case
set, distinct from calibration/earlier holdouts. Their exact 32 input files, hashes,
original TSV and selection proof are archived without outcome-based selection.
Settings: ctx4096/cache64 GiB/chunk512, TILE64/LIMIT1024, 32 teacher-forced tokens
per prompt; broader3 gap.30/mass.20/total.40/rank48/MAX_SUBS3/apply/replace.
All inherited DS4 names were cleared by the frozen driver. Policy stays opt-in.

The copied compact evidence includes all summaries, 512-row CSVs, per-prompt CSV,
checks, reference metadata, strict commands/exits/logs, ROOT launch/preflight/release,
MAIN guard result and 140 normal pressure samples with zero adverse samples/swap
growth. Driver/tests/frontend and source/build/binary/object/shader identities are
verified and inspectable. The ambiguously named original CPU test-hash field pins
captured cpu-test.json stdout, not test source. Its clarification and the first
archiver refusal are retained; the archived test is pinned to verified immutable38
SHA. A fresh run of the actual38 focused CPU tests passed all 37 trials and emitted
the identical stdout SHA. Its first invocation, which failed before trials because
the new workspace lacked the ignored temporary-artifacts parent, is preserved;
only that empty directory was created before retry. No test source changed.
The original proof is unchanged. None of the 64 run assets changed.
All four 264777744-byte raw arrays were independently
stream-hashed once and remain at their original exact paths; they are not copied.
The inventory records full hashes, sizes and row layout. The original run remains
unchanged. The archiver opened no model metadata or tokenizer/model payload,
executed no build/helper/GPU work, and acquired no model-job lock. Model identity
is historical run data.
No default promotion or generalization claim is made.
'''
    put('README.md', text.encode())
    write('original-copied-file-inventory.json', originals)
    manifest = {'status': 'ARCHIVED_COMPLETED_FIXED_CONFIRMATION', 'files': files,
                'retained_large_arrays': retained_arrays,
                'bytes_excluding_manifest': sum(row['bytes'] for row in files.values()),
                'source_commit': result['frozen_source'], 'runtime_commit': plan['runtime_commit'],
                'prepared_manifest_sha256': MANIFEST_SHA, 'originals_preserved': True}
    with (OUT / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for name, identity in files.items():
        data = (OUT / name).read_bytes()
        assert digest(data) == identity['sha256'] and len(data) == identity['bytes'], name
    print(json.dumps({'path': str(OUT), 'files': len(files), 'bytes': manifest['bytes_excluding_manifest'],
                      'manifest_sha256': digest((OUT / 'manifest.json').read_bytes()),
                      'independent_mean_delta_nll': mean, 'interval': interval,
                      'whole_PD_logical_bytes_reduction_percent': reduction,
                      'large_arrays_sha_verified_once': len(retained_arrays)}, indent=2))


if __name__ == '__main__':
    main()
