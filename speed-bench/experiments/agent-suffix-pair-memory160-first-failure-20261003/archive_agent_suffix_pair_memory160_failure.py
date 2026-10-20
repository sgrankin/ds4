#!/usr/bin/env python3
"""Copy completed first-attempt evidence; never invoke a native executable."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat

BASE = Path('/Volumes/Models/ds4')
ACTUAL = BASE / '.workspaces/idea-agent-suffix-pair-memory160/speed-bench/ideas/artifacts/agent-suffix-pair-memory160-prepared-20261003-v1'
CHOICE = BASE / '.workspaces/idea-agent-suffix-pair-memory160-choice/speed-bench/ideas/evidence/agent-suffix-pair-memory160-first-choice-refusal-v1'
WORK = BASE / '.workspaces/idea-agent-suffix-pair-memory160-archive'
OUT = WORK / 'speed-bench/experiments/ds41-ssd/agent-suffix-pair-actual-evidence/first-memory160-tool-contract-failure-v1'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()

def load(p):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d: raise ValueError('duplicate key: ' + k)
            d[k] = v
        return d
    return json.loads(p.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def pin(p):
    assert p.is_file() and not p.is_symlink(), str(p)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': digest(p), 'mode': stat.S_IMODE(p.stat().st_mode)}

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')

assert not OUT.exists(), 'exclusive output already exists'
assert digest(ACTUAL / 'manifest.json') == 'd13eaf4e51b10d9dc0849b26e7648ea790173a96f90e6d234256df63742e4d63'
assert digest(ACTUAL / 'prepared-command.json') == 'ac4501a2bfb61b9e3e91163dbbfe75670cd704cfb50044f6d8c545c40684562b'
assert digest(ACTUAL / 'root-release-proof.json') == '0147aede0d8e4a0076760efc68fdccd9bf5832bbd5cbec251cc9cd442d431397'
assert digest(CHOICE / 'manifest.json') == 'b882dc9f8d0314148b5fa0d0895b4d4485f03567c93fe6a440cedf3b18f2a43b'
prepared = load(ACTUAL / 'manifest.json')
for name, want in prepared['files'].items():
    got = pin(ACTUAL / name)
    assert all(got[k] == want[k] for k in ('bytes', 'sha256', 'mode')), name
assert len(prepared['files']) == 117
release = load(ACTUAL / 'root-release-proof.json')
assert release['guard_result'] == {'status': 'complete', 'child_returncode': 3}
assert release['lock_reacquired'] is True and release['live_model_processes'] == []
outcome = load(ACTUAL / 'run/controller-outcome.json')
assert outcome['status'] == 'FIRST_FAILURE_RETAINED'
assert outcome['overall_success'] is False and outcome['completed'] == [1, 2]
assert outcome['error'] == 'ControlFailure: First tool was not the single frozen updater invocation'
assert outcome['persistence'] == 'NOT_EXERCISED' and outcome['first_attempt'] is True and outcome['no_retries'] is True
controls = [load(ACTUAL / f'run/turn-{i}-check.json') for i in (1, 2)]
assert controls == [{'turn': 1, 'answer': 'CODE=2718', 'passed': True},
                    {'turn': 2, 'answer': 'READY', 'passed': True}]
rounds = [json.loads(x) for x in (ACTUAL / 'run/private/rounds.jsonl').read_text().splitlines()]
assert [(r['turn'], r['tool_round']) for r in rounds] == [(1, 0), (2, 0)] + [(3, i) for i in range(7)]
assert all(r['at_limit'] is False and r['context_limited'] is False and r['interrupted'] is False for r in rounds)
grader = load(ACTUAL / 'run/private/grader-command-result.json')
assert grader == {'returncode': 0, 'passed': True, 'stdout': 'PASS\n', 'stderr': ''}
attempts = [json.loads(x) for x in (ACTUAL / 'run/private/grader-attempts.jsonl').read_text().splitlines()]
assert len(attempts) == 1 and attempts[0]['passed'] is True
trace_bytes = (ACTUAL / 'run/trace.log').read_bytes()
trace = trace_bytes.decode('utf-8', errors='surrogateescape')
assert trace.encode('utf-8', errors='surrogateescape') == trace_bytes
assert 'context pair suffix APPROXIMATE' not in trace
frontiers = [int(x) for x in re.findall(r'context frontier captured tokens=(\d+)', trace)]
assert frontiers[:5] == [2010, 2363, 2987, 3545, 4192]
assert 'live context edit committed tokens=4193->4182' in trace
assert 'prefill tool_round=2 transcript=4245 prompt=4245 cached=0 suffix=4245 think=none' in trace
pressure = [json.loads(x) for x in (ACTUAL / 'pressure/pressure.jsonl').read_text().splitlines()]
assert len(pressure) == release['normal_pressure_samples'] == 243
assert all(p['level'] == 1 and p['reason'] is None and p['consecutive_bad'] == 0 for p in pressure)
assert max(p['swap_gb'] for p in pressure) - pressure[0]['swap_gb'] == release['swap_growth_gb'] == 0
assert max(p['wired_gb'] + p['compressor_gb'] for p in pressure) == release['peak_wired_compressor_gib']

OUT.mkdir(parents=True)
external = {}
for origin, target in [(ACTUAL, OUT / 'prepared'), (CHOICE, OUT / 'choice-refusal')]:
    for f in sorted(origin.rglob('*')):
        if not f.is_file(): continue
        got = pin(f)
        rel = f.relative_to(origin)
        if got['bytes'] > 20 * 1024 * 1024:
            external[str(f)] = dict(got, scope='retained native sysprompt KV; not GGUF; not copied due20MiB file limit')
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        assert all(pin(dest)[k] == got[k] for k in ('bytes', 'sha256', 'mode'))
launchlog = BASE / 'speed-bench/ideas/artifacts/agent-suffix-pair-memory160-root-launch-20261003-v1.log'
shutil.copy2(launchlog, OUT / 'root-launch.log')
script = Path(__file__)
shutil.copy2(script, OUT / 'archive_agent_suffix_pair_memory160_failure.py')
shutil.copytree(WORK / 'speed-bench/ideas/artifacts/memory160-failure-archive-attempt-v1',
                OUT / 'first-CPU-archive-decode-refusal')

write('independent-analysis.json', {
    'status': 'RETAINED_INVALID_AGENT_TOOL_CONTRACT_NO_SUFFIX_REUSE',
    'source_freeze': '86f834866f81ae8cb33c9c9f00e4df86a3b7a95c',
    'prepared_freeze': '3d410056fd401b540845a9a311de9c6a2d3506f3',
    'controls': controls, 'controller_status': outcome['status'], 'controller_error': outcome['error'],
    'guard_returncode': 3, 'agent_returncode': outcome['agent_returncode'],
    'first_third_tool': rounds[2]['calls'], 'third_tool_rounds': 7,
    'source_frontier_tokens': 2987, 'following_frontier_tokens_before_edit': [3545, 4192],
    'configured_frontier_slots': 2, 'source_eviction_from_FIFO_policy': True,
    'actual_import_tokens': [4193, 4182], 'next_prefill_cached': 0, 'next_prefill_fresh': 4245,
    'approximate_pair_trace_count': 0, 'semantic_grader_attempts': 1, 'semantic_grader_passed': True,
    'overall_success': False, 'strict_updater_and_read_audit': 'FAILED_AT_FIRST_TOOL; subsequent live-context reads also retained in parsed rounds',
    'no_forgetting_or_suffix_semantic_success_or_speed_claim': True,
    'model_stop_cap_context': {'observed_rounds': len(rounds), 'at_limit_rounds': 0,
                               'context_limited_rounds': 0, 'interrupted_rounds': 0,
                               'last_model_stopped': rounds[-1]['model_stopped']},
    'persistence': 'NOT_EXERCISED', 'choice_sha256': '480057204ff830e9a5428f1af75f398377f1055354fc80a85adc720ae1b2e32a',
    'choice_before_outcome_archive': True,
    'baseline': 'SOURCE_REFUSAL_UNRUN: prepared-specific live-file path is protected system text and differs across proposed arms',
    'release_scope': 'ROOT proof at release time before subsequent ragged job; no present-time process/flock claim',
    'release': release, 'pressure_independently_recomputed': True,
    'no_native_execution_or_GGUF_access_during_archive': True,
    'trace_decode': 'lossless UTF8 surrogateescape; raw trace bytes copied unchanged',
    'first_CPU_archive_decode_refusal_commit': '5067a46d',
})
write('source-diagnosis.json', {
    'source': pin(BASE / '.workspaces/idea-agent-suffix-pair-memory160/ds4_agent.c'),
    'system_livepath': {'function': 'agent_worker_build_system_tokens', 'lines': [5210, 5220, 5223, 5224],
                        'fact': 'Literal configured live_context_path is appended to protected system tokens.'},
    'cwd': {'CLI_parse_lines': [937, 938], 'main_chdir_line': 14594,
            'bash_spawn_lines': [9883, 9905], 'fact': 'CLI chdir uses run/project; tool sh -c inherits process cwd. Frozen updater existed there. Only live file under run is system-advertised.',
            'motivation_inference_not_claimed': True},
    'eviction': {'capture_function_line': 5593, 'oldest_drop_lines': [5630, 5631, 5632],
                 'capture_before_tool_lines': [11659, 11661, 11662],
                 'fact': 'Two new generation-end frontiers remove original2987 frontier from two-slot FIFO before updater imports edit.'},
    'original_refusal_preserved': 'choice-refusal/source-refusal.json',
    'repair_or_retry': False,
})
files = {}
for f in sorted(OUT.rglob('*')):
    if f.is_file():
        d = pin(f); d.pop('path'); files[str(f.relative_to(OUT))] = d
assert 'prepared/manifest.json' in files and 'prepared/provenance/cpu/manifest.json' in files
write('archive-index.json', {
    'status': 'IMMUTABLE_FIRST_MEMORY160_TOOL_CONTRACT_FAILURE_ARCHIVE',
    'root_model_session': 70948, 'first_attempt': True, 'no_retries': True,
    'source_outcome_files_unchanged': True, 'file_index_excludes_only_actual_top_index': True,
    'prepared_pinned_files_verified': 117, 'files': files, 'external_retained_files': external,
    'historical_fixed128_failure_index': prepared['prior128_failure_index_path'],
    'historical_fixed128_failure_index_sha256': prepared['prior128_failure_index_sha256'],
    'historical_fixed128_archive_commit': prepared['prior128_archive_commit'],
    'CPU_archive_path': prepared['cpu_archive_path'], 'CPU_archive_sha256': prepared['cpu_archive_sha256'],
})
physical = {str(f.relative_to(OUT)) for f in OUT.rglob('*') if f.is_file()}
assert physical - {'archive-index.json'} == set(files)
print(json.dumps({'status': 'PASS', 'archive': str(OUT), 'sha256': digest(OUT / 'archive-index.json'),
                  'local_payload_files': len(files), 'external_files': len(external),
                  'local_payload_bytes': sum(p['bytes'] for p in files.values())}))
