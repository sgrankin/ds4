#!/usr/bin/env python3
"""Archive completed native25 evidence without running or rescoring a producer."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Volumes/Models/ds4')
WORKSPACE = Path(__file__).resolve().parents[2]
EFFORT = ROOT / '.workspaces/idea-quant-effort'
ANALYSIS = ROOT / '.workspaces/idea-quant-native-only-analysis'
LABEL = 'native-effort25-counting-four-20261002-v1'
OUT = WORKSPACE / 'speed-bench/ideas/evidence/native-effort25-completed-20261002-v1'
ANALYSIS_SHA = 'ce2ca66defb63131ab9ad170c9eee84291d91bb48da7e03aed710133588da769'
ANALYSIS_SOURCE_SHA = '7830f64402b9972bac164b06dde1e3a6aa4a251cae3187fc29086c0cc9cefcf5'
QUEUE_SHA = '2af91c43a8c701cdcec392ef9041607cbf068180ea32528521b9afc01cc495b6'
HELPER_SHA = 'f0596bf4fb3035871c4bd7b1393161b005377fca2bc0a6e28b868b852cd3f571'
LAUNCHER_SHA = '98204b286e48ab100c6f5cfa1fa4565530f7a74f32ae5931fe351c230262640d'
CASES = ['circular10', 'restricted_derangements8', 'lattice8x6', 'subset_mod7']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def main():
    run = EFFORT / 'speed-bench/ideas/artifacts' / LABEL
    pressure = run.with_name(LABEL + '-pressure')
    analysis = ANALYSIS / 'speed-bench/experiments/native-effort25-counting-analysis-20261002'
    narrow = ANALYSIS / 'speed-bench/experiments/narrow-half-counting-analysis-20261002'
    launcher = ANALYSIS / 'speed-bench/experiments/overthink-queue-launcher-20261002'
    producer_manifest = run.with_name(LABEL + '-manifest.json')
    producer = load(producer_manifest)
    checked_bytes = (analysis / 'checked-summary.json').read_bytes()
    assert digest(checked_bytes) == ANALYSIS_SHA
    checked = json.loads(checked_bytes)
    summary = checked['summaries'][0]
    arm = summary['arms']['0.0']
    assert summary['complete'] and summary['records'] == 4
    assert arm['completed_correct'] == 2 and arm['invalid'] == 2
    assert arm['cap_hits'] == 2 and arm['tokens'] == 5985 and arm['reasoning_tokens'] == 5982
    assert summary['execution_returncode'] == 3
    assert summary['strict_outcome_status'] == 'INVALID_PRESENT'
    assert summary['frozen_manifest_sha256'] == digest(producer_manifest.read_bytes())
    guard_result = load(pressure / 'result.json')
    release = load(run / 'root-release-proof.json')
    assert guard_result == {'status': 'complete', 'child_returncode': 3}
    assert release['root_session'] == 39064 and release['guard_returncode'] == 3
    assert release['root_lock_reacquired'] and release['matching_live_model_processes'] == []
    assert release['samples'] == 294 and release['adverse_samples'] == 0
    assert release['swap_growth_gib'] == 0
    records = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    assert [row['case'] for row in records] == CASES
    assert [row['status'] for row in records] == ['CORRECT', 'CORRECT', 'INVALID', 'INVALID']
    assert [row['cap_hit'] for row in records] == [False, False, True, True]
    assert [row['stopped'] for row in records] == [True, True, False, False]
    assert [row['think_closed'] for row in records] == [True, True, False, False]
    assert sum(row['generated_tokens'] for row in records) == 5985
    assert sum(row['reasoning_tokens'] for row in records) == 5982
    assert all(row['think_level'] == 25 and row['lambda'] == 0 and
               row['max_tokens'] == 2048 and row['seed'] == 20261002 for row in records)
    preflight = load(launcher / 'native-effort25-preflight.json')
    assert preflight['queue_sha256'] == QUEUE_SHA
    assert len(preflight['verified_files']) == 98
    queue = EFFORT / 'speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/generation-queue-v1.json'
    assert digest(queue.read_bytes()) == QUEUE_SHA
    binary = Path(producer['binary']['path'])
    assert digest(binary.read_bytes()) == HELPER_SHA == producer['binary']['sha256']
    assert digest((ANALYSIS / 'speed-bench/summarize_overthink_checked.py').read_bytes()) == ANALYSIS_SOURCE_SHA
    native75_path = narrow / 'checked-summary.json'
    native75_bytes = native75_path.read_bytes()
    native75 = json.loads(native75_bytes)['summaries'][0]
    native75_arm = native75['arms']['0.0']
    assert native75_arm['tokens'] == 6681 and native75_arm['completed_correct'] == 1
    assert native75['configuration']['think_level'] == 75
    pressure_rows = [json.loads(line) for line in (pressure / 'pressure.jsonl').read_text().splitlines()]
    assert len(pressure_rows) == 294

    OUT.mkdir(parents=True, exist_ok=False)
    files = {}
    source_inventory = {}

    def put(relative, data, original=None):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
        files[relative] = {'sha256': digest(data), 'bytes': len(data)}
        if original is not None:
            source_inventory[relative] = {'path': str(original), **files[relative]}

    def copy(relative, source, expected=None):
        data = source.read_bytes()
        if expected is not None:
            assert digest(data) == expected, source
        put(relative, data, source)

    def write(relative, value):
        put(relative, (json.dumps(value, indent=2) + '\n').encode())

    for source in sorted(run.iterdir()):
        assert source.is_file(), source
        copy('producer-output/' + source.name, source)
    for name in ('command.json', 'pressure.jsonl', 'result.json'):
        copy('guard/' + name, pressure / name)
    copy('frozen-producer-manifest.json', producer_manifest)
    copy('frozen-generation-queue.json', queue, QUEUE_SHA)
    copy('native25-98-file-preflight.json', launcher / 'native-effort25-preflight.json')
    copy('launcher-frozen-manifest.json', launcher / 'manifest.json')
    for name in ('checked-summary.json', 'analysis-run.json', 'analysis.log'):
        copy('analysis/' + name, analysis / name)
    copy('source/summarize_overthink_checked.py', ANALYSIS / 'speed-bench/summarize_overthink_checked.py', ANALYSIS_SOURCE_SHA)
    launcher_source = Path(load(launcher / 'manifest.json')['launcher_path'])
    copy('source/run_overthink_queue.py', launcher_source, LAUNCHER_SHA)
    copy('source/pressure_run.py', Path(producer['guard']['path']), producer['guard']['sha256'])
    for name in ('speed-bench/overthink_pilot.c', 'speed-bench/prepare_overthink_run.py',
                 'speed-bench/overthink_diagnostics.h', 'speed-bench/overthink_counting_cases.h',
                 'speed-bench/overthink_counting_gold.py', 'speed-bench/overthink_gold.py',
                 'ds4_overthink_markers.h'):
        copy('source/' + name, EFFORT / name, producer['source_sha256'][name])
    fixture_name = 'speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/cases-v1.json'
    copy('cases-v1.json', EFFORT / fixture_name, producer['source_sha256'][fixture_name])
    core_proof = Path(load(launcher / 'manifest.json')['core_proof_path'])
    copy('existing-core-object-proof.json', core_proof, preflight['core_proof_sha256'])
    copy('source/archive_native_effort25.py', Path(__file__).resolve())
    options = {'producer_configuration': summary['configuration'],
               'producer_argv': producer['pilot_argv'],
               'policy_environment': producer['policy_environment'],
               'records': [{key: row[key] for key in ('case', 'prompt', 'seed', 'temperature',
                          'top_p', 'min_p', 'max_tokens', 'cache_gib_requested', 'lambda',
                          'scope', 'think_level', 'overthink_after', 'marker_profile',
                          'case_offset', 'case_limit', 'generated_tokens', 'reasoning_tokens',
                          'status', 'parse_reason', 'cap_hit', 'stopped', 'think_closed')}
                         for row in records]}
    write('record-options-and-completion.json', options)
    native75_projection = {'source_path': str(native75_path), 'source_sha256': digest(native75_bytes),
                           'analysis_source_sha256': ANALYSIS_SOURCE_SHA,
                           'configuration': native75['configuration'],
                           'penalty': 0.0, 'arm': native75_arm,
                           'cases': [{'case': row['case'], 'arm': row['arms']['0.0']}
                                     for row in native75['cases']]}
    write('native75-native-only-comparison-input.json', native75_projection)
    result = {'execution_status': 'COMPLETE_WITH_INVALID_CAPPED_RESULTS',
              'producer_returncode': 3, 'guard_returncode': 3,
              'record_integrity_status': summary['record_integrity_status'],
              'strict_outcome_status': summary['strict_outcome_status'],
              'completed_correct': 2, 'invalid': 2, 'cap_hits': 2,
              'tokens': 5985, 'reasoning_tokens': 5982, 'native25': arm,
              'native75_penalty_zero': native75_arm,
              'tokens_difference_percent': (5985 / 6681 - 1) * 100,
              'comparison_scope': 'Four identical counting problems and one seed; effort changes the thinking prompt. Descriptive token/completion comparison only, no cross-effort KL, raw-logit equivalence, causal or generalization claim.',
              'native50_partial_outcomes_read': False,
              'timing_scope': summary['timing_qualification'],
              'pressure': {'samples': 294, 'normal_samples': 294, 'adverse_samples': 0,
                           'swap_growth_gib': 0.0, 'host_pressure': summary['host_pressure']},
              'release': release,
              'no_rescoring_or_analysis_rerun': True,
              'accepted_analysis_commit': 'f27b7d5a942d4ea5e23849f9209e47ed4dc5a3a7',
              'checked_summary_commit': '7c6e0d98',
              'checked_summary_sha256': ANALYSIS_SHA,
              'helper_sha256': HELPER_SHA, 'queue_launcher_sha256': LAUNCHER_SHA,
              'queue_sha256': QUEUE_SHA, 'preflight_verified_files': 98,
              'archive_model_metadata_read': False, 'archive_model_payload_read': False,
              'archive_helper_executed': False, 'defaults_promoted': False}
    write('result.json', result)
    readme = '''# Completed native effort-25 counting run

The producer and MAIN guard completed with **rc3**: two completed correct answers
and two INVALID capped answers. The latter reached 2048 tokens with reasoning
unclosed and no stop token. Integrity checks passed; this is not an all-correct
run. All four full raw outputs, token JSONL traces, original typed records and
generation options are preserved here without scoring changes.

The run used the first four frozen counting problems, seed 20261002, temperature
0.6/top-p 0.95/min-p 0, cache 64 GiB, hesitation markers, diagnostic stride 32,
native-only penalty zero, and effort 25. It generated 5985 tokens, including 5982
reasoning tokens. ROOT session 39064 released ownership after 294 normal pressure
samples, zero adverse samples and zero swap growth. Its release proof records
the reacquired lock and no matching live model processes.

The comparison uses only the penalty-zero native arm of the previously completed
effort-75 narrow pair: 6681 tokens and one completed correct answer. Native 25
used 10.4176% fewer tokens and completed two correct answers on these four problems.
The effort prefix changes the thinking prompt, so this single-seed comparison
does not establish causality, generalization, cross-effort KL or equivalent raw
logits. Diagnostic CPU work and warm cache reuse prevent a clean speed claim.
Native 50 was still running and no partial outcomes were read for this archive.

CLM's accepted f27 checked analysis and frozen native25 summary were reused;
analysis was not rerun. Producer source, guard, queue launcher, fixture, generation
queue, manifest, existing-object proof and all 98 preflight identity records are
retained. The helper's actual SHA is f0596bf4fb3035871c4bd7b1393161b005377fca2bc0a6e28b868b852cd3f571.
Model identity is copied from the original run manifest; no current model stat,
tokenizer/model open, model payload hash, build or helper execution was performed
by the archiver. Original outputs remain untouched. No default is promoted.
'''
    put('README.md', readme.encode())
    write('original-file-inventory.json', source_inventory)
    manifest = {'status': 'ARCHIVED_COMPLETE_RC3_INVALID_PRESENT',
                'archive_source': str(Path(__file__).resolve()),
                'files': files, 'total_file_bytes_excluding_manifest': sum(x['bytes'] for x in files.values()),
                'originals_preserved': True, 'no_model_access': True,
                'binary_identity': producer['binary'], 'source_identity': producer['source_sha256'],
                'original_producer_revision': producer['revision'],
                'accepted_analysis_revision': 'f27b7d5a942d4ea5e23849f9209e47ed4dc5a3a7'}
    with (OUT / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    for name, identity in files.items():
        data = (OUT / name).read_bytes()
        assert digest(data) == identity['sha256'] and len(data) == identity['bytes'], name
    print(json.dumps({'path': str(OUT), 'files': len(files),
                      'bytes': manifest['total_file_bytes_excluding_manifest'],
                      'manifest_sha256': digest((OUT / 'manifest.json').read_bytes()),
                      'producer_and_guard_returncode': 3,
                      'completed_correct': 2, 'invalid_capped': 2}, indent=2))


if __name__ == '__main__':
    main()
