#!/usr/bin/env python3
"""Retain completed native50 evidence and descriptive native effort counts."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Volumes/Models/ds4')
WORKSPACE = Path(__file__).resolve().parents[2]
EFFORT = ROOT / '.workspaces/idea-quant-effort'
ANALYSIS = ROOT / '.workspaces/idea-quant-native-only-analysis'
PRIOR = ROOT / '.workspaces/idea-native-effort25-archive/speed-bench/ideas/evidence/native-effort25-completed-20261002-v1'
LABEL = 'native-effort50-counting-four-20261002-v1'
OUT = WORKSPACE / 'speed-bench/ideas/evidence/native-effort50-completed-20261002-v1'
ANALYSIS_SHA = '2eba56b1f26b823a145c83359c801699ce15de682b149b278c19e5dc9bd49ff5'
ANALYSIS_SOURCE_SHA = '7830f64402b9972bac164b06dde1e3a6aa4a251cae3187fc29086c0cc9cefcf5'
QUEUE_SHA = '2af91c43a8c701cdcec392ef9041607cbf068180ea32528521b9afc01cc495b6'
HELPER_SHA = 'f0596bf4fb3035871c4bd7b1393161b005377fca2bc0a6e28b868b852cd3f571'
LAUNCHER_SHA = '98204b286e48ab100c6f5cfa1fa4565530f7a74f32ae5931fe351c230262640d'
PRIOR_MANIFEST_SHA = '5ed09bcc06ea1e7dd8b9f102f9bc5225693394630968c0bc84462dc6b14b0370'
CASES = ['circular10', 'restricted_derangements8', 'lattice8x6', 'subset_mod7']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def main():
    run = EFFORT / 'speed-bench/ideas/artifacts' / LABEL
    pressure = run.with_name(LABEL + '-pressure')
    analysis = WORKSPACE / 'speed-bench/experiments/native-effort50-counting-analysis-20261002'
    launcher = ANALYSIS / 'speed-bench/experiments/overthink-queue-launcher-20261002'
    manifest_path = run.with_name(LABEL + '-manifest.json')
    producer = load(manifest_path)
    checked_bytes = (analysis / 'checked-summary.json').read_bytes()
    assert digest(checked_bytes) == ANALYSIS_SHA
    summary = json.loads(checked_bytes)['summaries'][0]
    arm = summary['arms']['0.0']
    records = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['records'] == 4
    assert summary['execution_returncode'] == 3 and summary['strict_outcome_status'] == 'INVALID_PRESENT'
    assert summary['frozen_manifest_sha256'] == digest(manifest_path.read_bytes())
    assert [row['case'] for row in records] == CASES
    assert [row['status'] for row in records] == ['CORRECT', 'INVALID', 'INVALID', 'INVALID']
    assert [row['cap_hit'] for row in records] == [False, True, True, True]
    assert [row['stopped'] for row in records] == [True, False, False, False]
    assert [row['think_closed'] for row in records] == [True, False, False, False]
    tokens = sum(row['generated_tokens'] for row in records)
    reasoning = sum(row['reasoning_tokens'] for row in records)
    assert tokens == arm['tokens'] == 6336 and reasoning == arm['reasoning_tokens'] == 6335
    assert arm['completed_correct'] == 1 and arm['invalid'] == 3 and arm['cap_hits'] == 3
    assert all(row['think_level'] == 50 and row['lambda'] == 0 and
               row['max_tokens'] == 2048 and row['seed'] == 20261002 for row in records)
    assert load(pressure / 'result.json') == {'status': 'complete', 'child_returncode': 3}
    release = load(run / 'root-release-proof.json')
    assert release['root_session'] == 34982 and release['guard_returncode'] == 3
    assert release['time'] == 1790983622.190748 and release['root_lock_reacquired']
    assert release['matching_live_model_processes'] == []
    assert release['samples'] == 301 and release['adverse_samples'] == 0 and release['swap_growth_gib'] == 0
    assert len((pressure / 'pressure.jsonl').read_text().splitlines()) == 301
    preflight_path = launcher / 'native-effort50-preflight.json'
    preflight = load(preflight_path)
    assert preflight['queue_sha256'] == QUEUE_SHA and len(preflight['verified_files']) == 98
    queue = EFFORT / 'speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/generation-queue-v1.json'
    assert digest(queue.read_bytes()) == QUEUE_SHA
    assert digest(Path(producer['binary']['path']).read_bytes()) == HELPER_SHA == producer['binary']['sha256']
    assert digest((WORKSPACE / 'speed-bench/summarize_overthink_checked.py').read_bytes()) == ANALYSIS_SOURCE_SHA
    assert digest((PRIOR / 'manifest.json').read_bytes()) == PRIOR_MANIFEST_SHA
    prior_manifest = load(PRIOR / 'manifest.json')

    def prior_value(name):
        data = (PRIOR / name).read_bytes()
        assert digest(data) == prior_manifest['files'][name]['sha256']
        return json.loads(data)

    prior25 = prior_value('result.json')['native25']
    prior75 = prior_value('native75-native-only-comparison-input.json')
    assert prior25['tokens'] == 5985 and prior25['completed_correct'] == 2
    assert prior75['arm']['tokens'] == 6681 and prior75['arm']['completed_correct'] == 1
    assert [row['case'] for row in prior75['cases']] == CASES
    OUT.mkdir(parents=True, exist_ok=False)
    files = {}
    original_files = {}

    def put(name, data, original=None):
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
        files[name] = {'sha256': digest(data), 'bytes': len(data)}
        if original is not None:
            original_files[name] = {'path': str(original), **files[name]}

    def copy(name, original, expected=None):
        data = original.read_bytes()
        if expected is not None:
            assert digest(data) == expected, original
        put(name, data, original)

    def write(name, value):
        put(name, (json.dumps(value, indent=2) + '\n').encode())

    for source in sorted(run.iterdir()):
        assert source.is_file(), source
        copy('producer-output/' + source.name, source)
    for name in ('command.json', 'pressure.jsonl', 'result.json'):
        copy('guard/' + name, pressure / name)
    for name in ('checked-summary.json', 'analysis-run.json', 'analysis.log'):
        copy('analysis/' + name, analysis / name)
    copy('frozen-producer-manifest.json', manifest_path)
    copy('frozen-generation-queue.json', queue, QUEUE_SHA)
    copy('native50-98-file-preflight.json', preflight_path)
    copy('launcher-frozen-manifest.json', launcher / 'manifest.json')
    launcher_manifest = load(launcher / 'manifest.json')
    copy('source/run_overthink_queue.py', Path(launcher_manifest['launcher_path']), LAUNCHER_SHA)
    copy('source/summarize_overthink_checked.py', WORKSPACE / 'speed-bench/summarize_overthink_checked.py', ANALYSIS_SOURCE_SHA)
    copy('source/pressure_run.py', Path(producer['guard']['path']), producer['guard']['sha256'])
    for name in ('speed-bench/overthink_pilot.c', 'speed-bench/prepare_overthink_run.py',
                 'speed-bench/overthink_diagnostics.h', 'speed-bench/overthink_counting_cases.h',
                 'speed-bench/overthink_counting_gold.py', 'speed-bench/overthink_gold.py',
                 'ds4_overthink_markers.h'):
        copy('source/' + name, EFFORT / name, producer['source_sha256'][name])
    fixture_name = 'speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/cases-v1.json'
    copy('cases-v1.json', EFFORT / fixture_name, producer['source_sha256'][fixture_name])
    copy('existing-core-object-proof.json', Path(launcher_manifest['core_proof_path']), preflight['core_proof_sha256'])
    copy('source/archive_native_effort50.py', Path(__file__).resolve())
    keys = ('case', 'prompt', 'seed', 'temperature', 'top_p', 'min_p', 'max_tokens',
            'cache_gib_requested', 'lambda', 'scope', 'think_level', 'overthink_after',
            'marker_profile', 'case_offset', 'case_limit', 'generated_tokens', 'reasoning_tokens',
            'status', 'parse_reason', 'cap_hit', 'stopped', 'think_closed')
    write('record-options-and-completion.json', {'producer_configuration': summary['configuration'],
          'producer_argv': producer['pilot_argv'], 'policy_environment': producer['policy_environment'],
          'records': [{key: row[key] for key in keys} for row in records]})
    comparison = []
    for effort, outcome in ((25, prior25), (50, arm), (75, prior75['arm'])):
        comparison.append({'effort': effort, **{key: outcome[key] for key in
            ('cases', 'completed_correct', 'invalid', 'cap_hits', 'tokens', 'reasoning_tokens')}})
    write('descriptive-native-effort-comparison.json', {'rows': comparison, 'case_ids': CASES,
          'seed': 20261002, 'native25_archive_commit': 'e12b453e7098942a54d73cf0c58ea31ead03ffa4',
          'native25_archive_path': str(PRIOR), 'native25_archive_manifest_sha256': PRIOR_MANIFEST_SHA,
          'native75_comparison_input_path': str(PRIOR / 'native75-native-only-comparison-input.json'),
          'native75_comparison_input_sha256': prior_manifest['files']['native75-native-only-comparison-input.json']['sha256'],
          'scope': 'Four fixed counting problems and one seed. Each effort changes the thinking prompt. Descriptive counts only; no cross-effort KL, raw-logit equivalence, causal quantization, generalization or speed claim.'})
    write('result.json', {'execution_status': 'COMPLETE_WITH_INVALID_CAPPED_RESULTS',
          'producer_returncode': 3, 'guard_returncode': 3, 'native50': arm,
          'record_integrity_status': summary['record_integrity_status'],
          'strict_outcome_status': summary['strict_outcome_status'],
          'computed_record_totals': {'generated_tokens': tokens, 'reasoning_tokens': reasoning},
          'pressure': {'samples': 301, 'normal_samples': 301, 'adverse_samples': 0,
                       'swap_growth_gib': 0, 'host_pressure': summary['host_pressure']},
          'root_release': release, 'accepted_analysis_commit': 'f27b7d5a942d4ea5e23849f9209e47ed4dc5a3a7',
          'checked_summary_sha256': ANALYSIS_SHA, 'helper_sha256': HELPER_SHA,
          'source_revision': producer['revision'], 'queue_launcher_sha256': LAUNCHER_SHA,
          'queue_sha256': QUEUE_SHA, 'preserved_preflight_identity_records': 98,
          'rescored': False, 'model_metadata_read': False, 'model_payload_read': False,
          'helper_executed': False, 'model_guard_acquired': False, 'defaults_promoted': False})
    put('README.md', b'''# Completed native effort-50 counting run

Producer and MAIN guard completed with **rc3**: one completed correct answer and
three INVALID answers capped at 2048 tokens with unclosed reasoning and no stop.
The accepted f27 checker validated all four records and token traces without
changing the producer's strict statuses. Total generated tokens are 6336, including
6335 reasoning tokens. The circular10 answer completed in 192 tokens; the other
three each used 2048. All four raw outputs, token JSONL, original records, exact
prompts/options, stop/cap/closure fields and diagnostic sampling records are kept.

| Native effort | Completed correct / 4 | INVALID caps | Tokens | Reasoning tokens |
| --- | --- | --- | --- | --- |
| 25 | 2 | 2 | 5985 | 5982 |
| 50 | 1 | 3 | 6336 | 6335 |
| 75 | 1 | 3 | 6681 | 6680 |

This is a descriptive comparison of the same four frozen problems at one seed.
The thinking prompt changes with effort. It supports no cross-effort KL, equivalent
logit, quantization-causality, generalization or speed claim. The 75 row uses only
the penalty-zero native arm of the completed narrow pair. The 25 archive is linked
in the comparison JSON rather than copied again.

Settings were native-only penalty zero, cache 64 GiB, max output 2048, seed
20261002, temperature 0.6/top-p 0.95/min-p 0, hesitation markers and stride 32.
ROOT session 34982 release time 1790983622.190748 follows guard rc3, with the lock reacquired
and no matching live model processes. All 301 pressure samples were normal;
there were zero adverse samples and zero swap growth.

The original source revision cee2d1426766064ec18740621c34faf006f9919a, actual
helper SHA f0596bf4fb3035871c4bd7b1393161b005377fca2bc0a6e28b868b852cd3f571,
queue launcher, full 98-file preflight identities, frozen queue, fixture, source
and existing-object proof are retained. Preflight hashes are preserved evidence;
the archiver reverified the helper, copied sources and archived files, not every
historical object. No current model stat, tokenizer/model open, model payload hash,
guard acquisition, build or helper execution was performed. No default is promoted.
''')
    write('original-file-inventory.json', original_files)
    manifest = {'status': 'ARCHIVED_COMPLETE_RC3_INVALID_PRESENT', 'files': files,
                'total_file_bytes_excluding_manifest': sum(row['bytes'] for row in files.values()),
                'originals_preserved': True, 'binary_identity': producer['binary'],
                'source_identity': producer['source_sha256'], 'model_accessed': False,
                'accepted_analysis_revision': 'f27b7d5a942d4ea5e23849f9209e47ed4dc5a3a7'}
    with (OUT / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    for name, identity in files.items():
        data = (OUT / name).read_bytes()
        assert digest(data) == identity['sha256'] and len(data) == identity['bytes'], name
    print(json.dumps({'archive_path': str(OUT), 'files': len(files),
          'bytes': manifest['total_file_bytes_excluding_manifest'],
          'manifest_sha256': digest((OUT / 'manifest.json').read_bytes()),
          'producer_and_guard_returncode': 3, 'table': comparison}, indent=2))


if __name__ == '__main__':
    main()
