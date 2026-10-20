#!/usr/bin/env python3
"""Analysis-only validated native/paired counting summaries; never launches models."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

QUEUE = Path('/Volumes/Models/ds4/.workspaces/idea-quant-effort/speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/generation-queue-v1.json')
QUEUE_SHA = '2af91c43a8c701cdcec392ef9041607cbf068180ea32528521b9afc01cc495b6'
CASES = QUEUE.parent / 'cases-v1.json'
CASES_SHA = '0e4b4eff5154cf199984829019e03678686deef50e6abb5fff02eeeff854b5e1'
OLD_SHA = 'dbd8d472b86cb34c696a1ee4ae217a8dfcf83e6a17e51c2891883427faae746a'
GOLDS = {'circular10': 123, 'restricted_derangements8': 10860, 'lattice8x6': 2002, 'subset_mod7': 132}
ALLOWED = ('narrow-half-effort75-pair', 'native-effort25', 'native-effort50')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ': expected bounded integer')
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + ': expected finite number')
    return value


def boolean(value, label):
    require(type(value) is bool, label + ': expected boolean')
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    raw = path.read_bytes()
    require(not raw or raw.endswith(b'\n'), str(path) + ': truncated final JSONL line')
    rows = []
    for line in raw.splitlines():
        require(bool(line), str(path) + ': empty JSONL row')
        row = json.loads(line)
        require(type(row) is dict, str(path) + ': object required')
        rows.append(row)
    return rows


def old_module():
    path = Path(__file__).with_name('summarize_overthink.py')
    require(digest(path) == OLD_SHA, 'unchanged original aggregation source hash mismatch')
    spec = importlib.util.spec_from_file_location('unchanged_overthink_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(manifest, fixture):
    require(manifest['suite'] == 'counting', 'only frozen counting suite supported')
    cases = fixture['cases']
    require(type(cases) is list and len(cases) == 4, 'four typed counting cases required')
    require({c['id']: c['answer'] for c in cases} == GOLDS, 'case IDs/golds mismatch')
    for case in cases:
        require(type(case['id']) is str and type(case['answer']) is int
                and type(case['prompt']) is str, 'typed case ID/gold/prompt required')
    offset = integer(manifest.get('case_offset', 0), 'case_offset')
    limit = integer(manifest['case_limit'], 'case_limit', 1)
    require(offset + limit <= len(cases), 'case range outside fixture')
    native = boolean(manifest.get('native_only', False), 'native_only')
    candidate = manifest['candidate']
    if candidate is not None:
        require(0 < number(candidate, 'candidate') <= 4, 'candidate outside frozen range')
    require(not native or candidate is None, 'native_only excludes candidate')
    penalties = [0.0] if native else [0.0, float(candidate)] if candidate is not None else [0.0, .5, 1.0]
    cap = integer(manifest['max_tokens'], 'max_tokens', 1)
    effort = integer(manifest['think_level'], 'think_level', 1)
    require(cap <= 4096 and effort <= 100, 'cap/effort outside pilot range')
    integer(manifest['overthink_after'], 'overthink_after')
    stride = integer(manifest.get('diagnostic_stride', 0), 'diagnostic_stride', 1)
    require(stride <= 4096, 'diagnostic stride outside pilot range')
    number(manifest['cache_gib_requested'], 'cache_gib_requested')
    require(manifest['cache_gib_requested'] > 0 and manifest['marker_profile'] in ('paper', 'hesitation'), 'cache/marker configuration')
    return cases[offset:offset+limit], penalties


def local_artifact(directory, name):
    require(type(name) is str and name and Path(name).name == name,
            'trace must be a plain local filename')
    return directory / name


def validate_trace(directory, row, stride):
    diagnostics = row['diagnostics']
    require(type(diagnostics) is dict, 'typed diagnostics required')
    require(type(diagnostics['stride']) is int and diagnostics['stride'] == stride, 'diagnostic stride mismatch')
    require(number(diagnostics['temperature_float_value'], 'temperature float') == 0.6000000238418579, 'native float temperature mismatch')
    trace_path = local_artifact(directory, diagnostics['token_trace'])
    raw_path = local_artifact(directory, row['raw_trace'])
    trace = jsonl(trace_path)
    generated = row['generated_tokens']
    require(len(trace) == generated + int(row['stopped']), 'token trace truncated/extra sampling steps')
    raw = bytearray(); final = bytearray(); reasoning = 0; closed = False
    counts = dict(active_steps=0, thinking_marker_selections=0, active_marker_selections=0, pre_filter_samples=0, sampled_argmax_changes=0)
    previous_rng = f'{row['seed']:016x}'
    for index, token in enumerate(trace):
        require(type(token['step']) is int and token['step'] == index, 'noncontiguous/duplicate token step')
        require(integer(token['token_id'], 'token_id') <= 2147483647, 'token_id outside signed C int range')
        for key in ('stop', 'inside_thinking', 'policy_active', 'selected_marker'):
            boolean(token[key], key)
        thinking = integer(token['generated_thinking_tokens'], 'generated_thinking_tokens')
        require(thinking <= index, 'generated_thinking_tokens exceeds prior generated step')
        require(token['policy_active'] == (row['lambda'] > 0 and token['inside_thinking'] and thinking >= row['overthink_after']), 'policy activity does not match frozen think-only gate')
        for key in ('rng_before', 'rng_after'):
            require(type(token[key]) is str and re.fullmatch('[0-9a-f]{16}', token[key]), 'invalid RNG identity')
        require(previous_rng is None or token['rng_before'] == previous_rng, 'RNG chain mismatch')
        previous_rng = token['rng_after']
        require(type(token['piece_hex']) is str and re.fullmatch('(?:[0-9a-f]{2})*', token['piece_hex']), 'invalid piece hex')
        piece = bytes.fromhex(token['piece_hex'])
        require(token['stop'] == (row['stopped'] and index == generated), 'stop token placement mismatch')
        counts['active_steps'] += token['policy_active']
        counts['thinking_marker_selections'] += token['selected_marker'] and token['inside_thinking']
        counts['active_marker_selections'] += token['selected_marker'] and token['policy_active']
        require(('raw_marker_mass_pre_filter' in token) == (index % stride == 0 and token['inside_thinking']), 'missing/extra sampled diagnostic fields')
        if 'raw_marker_mass_pre_filter' in token:
            require(index % stride == 0 and token['inside_thinking'], 'diagnostic sample placement mismatch')
            mass = number(token['raw_marker_mass_pre_filter'], 'marker mass')
            require(0 <= mass <= 1, 'marker mass out of range')
            for key in ('kl_raw_to_policy_pre_filter','selected_delta_nll_pre_filter','marker_best_minus_other_logit'):
                number(token[key], key)
            for key in ('raw_argmax', 'policy_argmax'):
                require(integer(token[key], key) <= 2147483647, key + ' outside signed C int range')
            counts['pre_filter_samples'] += 1
            counts['sampled_argmax_changes'] += token['raw_argmax'] != token['policy_argmax']
        if token['stop']:
            continue
        raw.extend(piece)
        if not closed:
            reasoning += 1
        # Frozen producer resets final_offset on EVERY exact closing piece,
        # while its reasoning counter stops after the first closure.
        if piece == b'</think>':
            closed = True
            final.clear()
        elif closed:
            final.extend(piece)
    require(bytes(raw) == raw_path.read_bytes(), 'raw trace bytes do not match token pieces')
    require(row['think_closed'] == closed and row['reasoning_tokens'] == reasoning, 'record reasoning/closure mismatch')
    require(type(row['final']) is str and row['final'].encode() == bytes(final), 'record final bytes mismatch')
    require(integer(diagnostics['sampling_steps'], 'sampling_steps') == len(trace), 'sampling_steps mismatch')
    for key, value in counts.items():
        require(integer(diagnostics[key], key) == value, key + ' mismatch')
    return dict(token_trace_sha256=digest(trace_path), raw_trace_sha256=digest(raw_path), sampling_steps=len(trace), reconstructed_raw_bytes=len(raw))


def summarize(directory, manifest, fixture, original=None):
    directory = Path(directory)
    cases, penalties = contract(manifest, fixture)
    rows = jsonl(directory / 'results.jsonl')
    expected = {(c['id'], p) for c in cases for p in penalties}
    by_id = {c['id']: c for c in cases}; seen = set(); traces = []
    options = {key:manifest[key] for key in ('suite','think_level','max_tokens','cache_gib_requested','overthink_after','case_limit','marker_profile')}
    options.update(case_offset=manifest.get('case_offset',0), seed=20261002, temperature=.6, top_p=.95, min_p=0, scope='think', scorer='strict-final-v2')
    for row in rows:
        case = row['case']; penalty = number(row['lambda'], 'lambda')
        require(type(case) is str and (case,penalty) in expected, 'unexpected case/arm')
        require((case,penalty) not in seen, 'duplicate case/arm record'); seen.add((case,penalty))
        gold = by_id[case]
        require(type(row['expected']) is str and row['expected'] == str(gold['answer']), 'wrong gold')
        require(type(row['prompt']) is str and row['prompt'] == gold['prompt'], 'wrong fixture prompt')
        for key,want in options.items():
            if type(want) is int: integer(row[key], key)
            elif type(want) is float: number(row[key], key)
            else: require(type(row[key]) is str, key + ': string required')
            require(row[key] == want, 'record configuration mismatch: ' + key)
        for key in ('parse_ok','think_closed','stopped','cap_hit','correct_exact'):
            boolean(row[key], key)
        n = integer(row['generated_tokens'], 'generated_tokens')
        reasoning = integer(row['reasoning_tokens'], 'reasoning_tokens')
        require(reasoning <= n <= manifest['max_tokens'], 'token counts out of bounds')
        require(row['cap_hit'] == (n == manifest['max_tokens']), 'cap flag mismatch')
        require(row['stopped'] != row['cap_hit'], 'must stop or reach cap, exclusively')
        for key in ('prefill_seconds','decode_seconds'):
            require(number(row[key],key) >= 0, 'negative timing')
        require(row['status'] in ('CORRECT','INCORRECT','INVALID'), 'unknown strict status')
        require(type(row['parsed_answer']) is str and type(row['parse_reason']) is str, 'typed scoring result required')
        # Check recorded score consistency; never reparse/rewrite final answers.
        if row['status'] == 'INVALID':
            require(not row['parse_ok'] and not row['correct_exact'] and row['parse_reason'] in ('unclosed_reasoning','empty_final','unsupported_final_format'), 'inconsistent INVALID record')
        else:
            require(row['parse_ok'] and row['think_closed'] and row['parse_reason']=='none', 'inconsistent parsed record')
            require(row['correct_exact'] == (row['status']=='CORRECT'), 'correctness flag mismatch')
            require((row['parsed_answer']==row['expected']) == row['correct_exact'], 'recorded canonical answer/status mismatch')
        traces.append(dict(case=case, penalty=penalty, **validate_trace(directory,row,manifest['diagnostic_stride'])))
    require(seen == expected, 'missing case/arm records')
    # All maths, strict labels, cap counts and paired token diagnostics come
    # from the unchanged summarizer. Only native-only completeness is corrected.
    result = (original or old_module()).summarize(directory)
    for penalty in penalties:
        group = [r for r in rows if r['lambda'] == penalty]
        arm = result['arms'][str(float(penalty))]
        arm['completed_correct'] = sum(r['status'] == 'CORRECT' and not r['cap_hit'] and r['stopped'] and r['think_closed'] for r in group)
        arm['capped_correct'] = sum(r['status'] == 'CORRECT' and r['cap_hit'] for r in group)
    result['completed_correct_rule'] = 'Recorded CORRECT plus !cap_hit, stopped and think_closed; producer strict status/counts remain unchanged.'
    result['expected_records'] = len(expected)
    result['complete'] = True
    result['expected_arms'] = len(penalties)
    result['native_only'] = manifest.get('native_only',False)
    result['validated_case_ids'] = [c['id'] for c in cases]
    result['trace_integrity'] = traces
    result['original_aggregation_sha256'] = OLD_SHA
    result['analysis_scope'] = 'Recorded strict-final-v2 outcomes unchanged; no scoring repair, model access or cross-effort KL comparison.'
    guard_path = Path(manifest['pressure_dir']) / 'result.json'
    require(guard_path.is_file(), 'completed guard result required')
    guard = json.loads(guard_path.read_text())
    require(guard.get('status')=='complete' and type(guard.get('child_returncode')) is int, 'guard incomplete/malformed')
    result['guard'] = guard
    invalid = sum(row['status']=='INVALID' for row in rows)
    require(guard['child_returncode'] == (3 if invalid else 0), 'guard rc does not match frozen pilot INVALID-only return rule')
    result['execution_returncode'] = guard['child_returncode']
    result['record_integrity_status'] = 'PASS'
    result['strict_outcome_status'] = 'INVALID_PRESENT' if invalid else 'VALID_INCORRECT_PRESENT' if any(r['status']=='INCORRECT' for r in rows) else 'ALL_CORRECT'
    result['configuration']['native_only'] = result['native_only']
    return result


def frozen_job(label, preflight=False):
    require(label in ALLOWED, 'job is not one of the three scheduled frozen jobs')
    require(digest(QUEUE)==QUEUE_SHA and digest(CASES)==CASES_SHA, 'frozen queue/cases changed')
    queue = json.loads(QUEUE.read_text()); fixture = json.loads(CASES.read_text())
    job = next(j for j in queue['jobs'] if j['label']==label)
    path = Path(job['manifest']); require(digest(path)==job['manifest_sha256'], 'frozen manifest changed')
    manifest = json.loads(path.read_text())
    require(manifest['think_level']==job['effort'] and manifest['case_limit']==job['case_limit'] and manifest['case_offset']==job['case_offset'], 'manifest/queue options mismatch')
    require(manifest['binary']['sha256']==job['binary_sha256'], 'manifest/queue binary identity mismatch')
    require(manifest['source_sha256']['speed-bench/experiments/ds41-ssd/overthink-provenance/counting-study/cases-v1.json']==CASES_SHA, 'manifest fixture source mismatch')
    directory = Path(manifest['results_dir'])
    adjacent = directory.parent / (directory.name+'-manifest.json')
    require(digest(adjacent)==job['manifest_sha256'], 'aggregation adjacent manifest mismatch')
    cases, penalties = contract(manifest, fixture)
    require(penalties == job['penalties'], 'manifest/queue arm mismatch')
    if preflight:
        return dict(status='FROZEN_ANALYSIS_CONTRACT_PASS_NO_OUTCOMES_READ', job=label, manifest_sha256=job['manifest_sha256'], expected_records=len(cases)*len(penalties), expected_case_ids=[c['id'] for c in cases], expected_golds={c['id']:str(c['answer']) for c in cases}, penalties=penalties, native_only=manifest.get('native_only',False), think_level=manifest['think_level'], old_aggregation_sha256=digest(Path(__file__).with_name('summarize_overthink.py')), model_metadata_accessed=False, results_read=False)
    result=summarize(directory,manifest,fixture)
    result.update(job=label, frozen_manifest_sha256=job['manifest_sha256'], frozen_queue_sha256=QUEUE_SHA, fixture_sha256=CASES_SHA)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',action='append',required=True,choices=ALLOWED)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true',help='validate only frozen typed analysis contracts; no outcomes or model access')
    args=parser.parse_args()
    try:
        require(len(set(args.job))==len(args.job),'duplicate requested job')
        results=[frozen_job(label,preflight=args.preflight) for label in args.job]
        comparison=[] if args.preflight else [dict(job=r['job'], effort=r['configuration']['think_level'], **r['arms']['0.0']) for r in results]
        text=json.dumps(dict(status='FROZEN_CONTRACT_PREFLIGHT_PASS' if args.preflight else 'ANALYSIS_INTEGRITY_PASS',summaries=results,native_effort_comparison=comparison,
            comparison_scope='Strict accuracy/caps/reasoning tokens on the same four cases; prompts differ by effort prefix, so no cross-effort KL or raw-logit equivalence claim.'),indent=2)+'\n'
        if args.output:
            with args.output.open('x') as stream: stream.write(text)
        else: print(text,end='')
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
        print(json.dumps(dict(status='ANALYSIS_INTEGRITY_REJECTED',error=str(error))))
        return 2
    return 0

if __name__=='__main__':
    raise SystemExit(main())
