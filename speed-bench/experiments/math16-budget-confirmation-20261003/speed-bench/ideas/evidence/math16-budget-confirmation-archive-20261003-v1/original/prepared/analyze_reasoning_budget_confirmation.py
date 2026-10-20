#!/usr/bin/env python3
"""Strict protocol-boundary first attempts; no model access or distribution scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import reasoning_budget_signed64 as scorer

CAP = 2048
VOCAB = 129280
SEED = 20261002
TEMP = 0.6000000238418579
TOP_P = 0.949999988079071
CLOSE_ID = 128822
GOLDS = {'circular14_five': '196', 'linear12_four_gap2': '15', 'permutations9_two_forbidden': '43387', 'deranged8_first_two_odd': '3099', 'dyck7_three_interior_returns': '48', 'bounded10x8_avoid5x4': '20565', 'subset15_seven_mod5': '1287', 'subset16_six_nonconsecutive_threeodd': '100', 'tiles2x8_domino_square': '171', 'composition18_five_max6': '780', 'balls9_four_bins_max4': '80', 'cycle9_three_colors': '510', 'ladder2x5_three_colors': '486', 'words_A3B4C3_noAB': '700', 'BANANA_no_adjacent_A': '12', 'digits8_sum12_no_equal_neighbors': '1492'}
NAMES = ['native-before', 'budget1024', 'native-after']
TOKEN_COMMON = ('step', 'token_id', 'stop', 'inside_thinking', 'generated_thinking_tokens',
                'policy_active', 'rng_before', 'rng_after', 'piece_hex')
TOKEN_NEW = set(TOKEN_COMMON) | {'penalty', 'intervention'}
ROW_NEW = set('case policy_type reasoning_budget forced_think_close suite think_level native_penalty think_enabled '
              'prompt_supplied_think_close generated_think_close status parse_ok parse_reason scorer seed '
              'temperature_float_value top_p_float_value min_p max_tokens cache_gib_requested context_size '
              'prefill_chunk prompt_tokens generated_tokens reasoning_tokens think_closed stopped cap_hit '
              'correct_exact prefill_seconds decode_seconds parsed_answer final expected prompt raw_trace '
              'token_trace prompt_trace'.split())
ROW_COMMON = tuple('case status parse_ok parse_reason scorer seed think_level max_tokens '
                   'cache_gib_requested generated_tokens reasoning_tokens think_closed stopped cap_hit correct_exact '
                   'parsed_answer final expected prompt'.split())


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.suffix.lower() != '.gguf', 'model files are never hashed by analysis')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def integer(value, label, maximum=2147483647):
    require(type(value) is int and 0 <= value <= maximum, 'typed bounded integer: ' + label)
    return value


def boolean(value, label):
    require(type(value) is bool, 'literal boolean: ' + label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric: ' + label)
    return value


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=object_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def load(path):
    return decode_json(Path(path).read_bytes())


def jsonl(path):
    data = Path(path).read_bytes()
    require(data.endswith(b'\n') and data, 'complete nonempty newline-terminated JSONL')
    values = [decode_json(line) for line in data.splitlines()]
    require(all(type(value) is dict for value in values), 'JSONL object rows')
    return values


def local(root, name):
    require(type(name) is str and name and name not in ('.', '..') and Path(name).name == name,
            'plain local artifact name')
    path = Path(root) / name
    require(not path.is_symlink() and path.resolve().parent == Path(root).resolve(), 'artifact path escape')
    return path


def next_rng(value):
    # Frozen native sample_rng_next state update, excluding its returned multiplier.
    x = value or 0x9e3779b97f4a7c15
    x ^= x >> 12
    x ^= (x << 25) & ((1 << 64) - 1)
    x ^= x >> 27
    return x & ((1 << 64) - 1)


def fixture_check(fixture):
    require(type(fixture) is dict and set(fixture) == {'suite', 'cases'}, 'fixture schema')
    require(fixture['suite'] == 'math-budget-confirmation16-v1', 'fixed unused16 suite')
    cases = fixture['cases']
    require(type(cases) is list and len(cases) == 16 and [c['case'] for c in cases] == list(GOLDS),
            'sixteen original ordered confirmation cases')
    for case in cases:
        require(set(case) == {'case', 'prompt', 'answer'} and type(case['prompt']) is str and case['prompt'],
                'literal frozen case')
        require(type(case['answer']) is str and case['answer'] == GOLDS[case['case']], 'original gold')
    return cases


def trace_check(root, row, budget, historical=False):
    token_name = row['diagnostics']['token_trace'] if historical else row['token_trace']
    trace_path = local(root, token_name)
    raw_path = local(root, row['raw_trace'])
    trace = jsonl(trace_path)
    require(len(trace) == row['generated_tokens'] + int(row['stopped']), 'all sampled rows, including stop')
    previous = SEED
    inside = True
    count = 0
    closed = False
    generated_close = False
    reasoning = 0
    forced_at = None
    raw = bytearray()
    final = bytearray()
    common = []
    for step, token in enumerate(trace):
        if not historical:
            require(set(token) == TOKEN_NEW, 'exact protocol token schema')
        else:
            require(set(TOKEN_COMMON) <= set(token), 'historical common token schema')
            require(number(token['applied_logit_delta'], 'historical zero delta') == 0,
                    'historical native has no logit intervention')
        require(integer(token['step'], 'step') == step, 'contiguous unique ordered step')
        integer(token['token_id'], 'token ID', VOCAB - 1)
        for key in ('stop', 'inside_thinking', 'policy_active'):
            boolean(token[key], key)
        prior = integer(token['generated_thinking_tokens'], 'prior reasoning count')
        require(prior <= step and token['inside_thinking'] == inside and prior == count,
                'exact pre-step phase and control-excluding counter')
        for key in ('rng_before', 'rng_after'):
            require(type(token[key]) is str and re.fullmatch('[0-9a-f]{16}', token[key]), 'fixed RNG hex')
        before, after = int(token['rng_before'], 16), int(token['rng_after'], 16)
        require(before == previous, 'continuous RNG state')
        require(type(token['piece_hex']) is str and re.fullmatch('(?:[0-9a-f]{2})*', token['piece_hex']),
                'literal complete piece bytes')
        piece = bytes.fromhex(token['piece_hex'])
        require((token['token_id'] == CLOSE_ID) == (piece == b'</think>'),
                'bidirectional validated closing token/piece identity')
        require(token['stop'] == (row['stopped'] and step == row['generated_tokens']), 'final stop position')
        forced = token['policy_active'] if not historical else False
        if not historical:
            require(number(token['penalty'], 'zero penalty') == 0, 'protocol is not a logit penalty')
            require(token['intervention'] == ('forced-thinking-close' if forced else 'native-sample'),
                    'explicit intervention type')
        if forced:
            require(budget == 1024 and forced_at is None and inside and count >= 1024,
                    'one eligible forced close at budget')
            require(CAP - step >= 2 and not token['stop'] and token['token_id'] == CLOSE_ID
                    and piece == b'</think>' and after == before, 'legal evaluated forced close and unchanged RNG')
            forced_at = step
        else:
            require(not token['policy_active'], 'native historical/control activity')
            require(not (budget and forced_at is None and inside and count >= budget),
                    'first eligible reasoning step must force')
            require(after == next_rng(before), 'one original native RNG advance')
        if forced_at is not None and step > forced_at:
            require(not inside, 'outside thinking after forced close before sampling')
        previous = after
        common.append({key: token[key] for key in TOKEN_COMMON})
        if token['stop']:
            continue
        raw.extend(piece)
        if not closed:
            reasoning += 1  # Preserve producer's first-close convention, including close itself.
        if piece == b'</think>':
            require(token['token_id'] == CLOSE_ID, 'validated closing token identity')
            inside = False
            closed = True
            generated_close = True
            final.clear()  # Every close resets final offset; reasoning stopped at first close.
        else:
            if closed:
                final.extend(piece)
            if piece == b'<think>':
                require(forced_at is None, 'no reopened reasoning following forced close')
                inside = True
            elif inside:
                count += 1
    require(bytes(raw) == raw_path.read_bytes(), 'raw reconstruction from every non-stop piece')
    require(row['reasoning_tokens'] == reasoning and row['think_closed'] == closed
            and row.get('generated_think_close', generated_close) == generated_close, 'closure/count identity')
    require(type(row['final']) is str and row['final'].encode('utf-8') == bytes(final), 'last-close final bytes')
    if not historical:
        require(row['forced_think_close'] == (forced_at is not None), 'published force matches actual trace')
        prompt_path = local(root, row['prompt_trace'])
        data = prompt_path.read_bytes()
        ids = data.splitlines()
        require(data.endswith(b'\n') and ids and len(ids) == row['prompt_tokens'], 'complete prompt IDs')
        require(all(re.fullmatch(b'[0-9]+', value) and int(value) < VOCAB for value in ids), 'prompt ID extent/range')
        prompt_sha = sha(prompt_path)
    else:
        prompt_sha = None  # Historical producer did not export prompt IDs or their count.
    return {'trace': common, 'raw_sha256': sha(raw_path), 'token_sha256': sha(trace_path),
            'prompt_sha256': prompt_sha, 'force_step': forced_at}


def validate_arm(root, budget, fixture, rc, historical=False):
    require(type(budget) is int and budget in (0, 1024) and historical is False, 'pilot budget')
    cases = fixture_check(fixture)
    rows = jsonl(Path(root) / 'results.jsonl')
    require(len(rows) == 16 and [row['case'] for row in rows] == [case['case'] for case in cases],
            'complete sixteen first attempts in frozen order')
    infos = []
    for row, case in zip(rows, cases):
        if not historical:
            require(set(row) == ROW_NEW, 'exact producer result schema')
            options = dict(policy_type='protocol-boundary-token', reasoning_budget=budget, suite='math-budget-confirmation16-v1',
                           think_level=25, native_penalty=0, think_enabled=True, prompt_supplied_think_close=False,
                           scorer='strict-final-v2', seed=SEED, temperature_float_value=TEMP, top_p_float_value=TOP_P,
                           min_p=0, max_tokens=CAP, cache_gib_requested=64, context_size=8192, prefill_chunk=512)
            for key, value in options.items():
                require(type(row[key]) is type(value) and row[key] == value, 'typed fixed option: ' + key)
            boolean(row['forced_think_close'], 'forced_think_close')
            boolean(row['generated_think_close'], 'generated_think_close')
            require(0 < integer(row['prompt_tokens'], 'prompt_tokens') <= 8192 - CAP,
                    'complete prompt plus full generation cap fits frozen context')
        else:
            options = dict(suite='counting', marker_profile='closing', mapped_markers=1, closing_boost=0,
                           policy_direction='ADD', closing_token_id=CLOSE_ID, scope='think', engine_penalty=0,
                           engine_marker_profile='hesitation', engine_overthink_after=0, think_level=25, seed=SEED,
                           temperature=0.6, top_p=0.95, min_p=0, max_tokens=CAP, cache_gib_requested=64,
                           closing_after=1024, case_offset=0, case_limit=4, scorer='strict-final-v2')
            for key, value in options.items():
                require(type(row[key]) is type(value) and row[key] == value, 'historical typed option: ' + key)
            require(number(row['diagnostics']['temperature_float_value'], 'historical T') == TEMP
                    and integer(row['diagnostics']['active_steps'], 'historical active') == 0, 'old native semantics')
        require(type(row['prompt']) is str and row['prompt'] == case['prompt']
                and type(row['expected']) is str and row['expected'] == case['answer'], 'frozen prompt/gold')
        for key in ('parse_ok', 'think_closed', 'stopped', 'cap_hit', 'correct_exact'):
            boolean(row[key], key)
        n = integer(row['generated_tokens'], 'generated', CAP)
        require(integer(row['reasoning_tokens'], 'reasoning') <= n
                and row['cap_hit'] == (n == CAP) and row['stopped'] != row['cap_hit'], 'stop/cap/count contract')
        for key in ('prefill_seconds', 'decode_seconds'):
            require(number(row[key], key) >= 0, 'nonnegative diagnostic time')
        info = trace_check(root, row, budget, historical)
        parsed = scorer.parsed_integer(row['final']) if row['think_closed'] else None
        status = 'INVALID' if parsed is None else 'CORRECT' if parsed == case['answer'] else 'INCORRECT'
        reason = 'none' if parsed is not None else 'unclosed_reasoning' if not row['think_closed'] else \
                 'empty_final' if not row['final'].strip(' \t\n\r\v\f') else 'unsupported_final_format'
        require(type(row['parsed_answer']) is str and row['parsed_answer'] == (parsed or '')
                and row['parse_ok'] == (parsed is not None) and row['status'] == status
                and row['parse_reason'] == reason and row['correct_exact'] == (status == 'CORRECT'), 'exact signed64 scorer')
        infos.append(info)
    invalid = sum(row['status'] == 'INVALID' for row in rows)
    require(type(rc) is int and rc == (3 if invalid else 0), 'actual INVALID-only producer return code')
    report = dict(budget=budget, producer_returncode=rc, cases=16, completed_correct=sum(
        row['status'] == 'CORRECT' and row['stopped'] and not row['cap_hit'] for row in rows),
        incorrect=sum(row['status'] == 'INCORRECT' for row in rows), invalid=invalid,
        capped_correct=sum(row['status'] == 'CORRECT' and row['cap_hit'] for row in rows),
        cap_hits=sum(row['cap_hit'] for row in rows), forced_cases=sum(info['force_step'] is not None for info in infos),
        generated_tokens=sum(row['generated_tokens'] for row in rows),
        reasoning_tokens=sum(row['reasoning_tokens'] for row in rows),
        per_case=[dict(case=row['case'], status=row['status'], stopped=row['stopped'], cap_hit=row['cap_hit'],
                  generated_tokens=row['generated_tokens'], reasoning_tokens=row['reasoning_tokens'],
                  parsed_answer=row['parsed_answer'], force_step=info['force_step'], raw_sha256=info['raw_sha256'],
                  token_sha256=info['token_sha256'], prompt_sha256=info['prompt_sha256']) for row, info in zip(rows, infos)])
    return report, rows, infos


def exact_native(first, second, historical=False):
    _, arows, ainfos = first
    _, brows, binfos = second
    require(len(arows) == len(brows) == 16, 'complete native controls')
    for a, b, ai, bi in zip(arows, brows, ainfos, binfos):
        require(all(type(a[key]) is type(b[key]) and a[key] == b[key] for key in ROW_COMMON), 'native common result identity')
        require(ai['trace'] == bi['trace'] and ai['raw_sha256'] == bi['raw_sha256'], 'native all token/RNG/piece/raw identity')
        if not historical:
            require(ai['prompt_sha256'] == bi['prompt_sha256'], 'new native prompt ID identity')
    return True


def pre_force(native, candidate):
    _, nrows, ninfos = native
    _, crows, cinfos = candidate
    gates = []
    for n, c, ni, ci in zip(nrows, crows, ninfos, cinfos):
        require(ni['prompt_sha256'] == ci['prompt_sha256'], 'candidate unchanged exact prompt IDs')
        stop = ci['force_step']
        if stop is None:
            require(all(n[key] == c[key] for key in ROW_COMMON) and ni['trace'] == ci['trace']
                    and ni['raw_sha256'] == ci['raw_sha256'], 'no-force natural closure retains whole native attempt')
        else:
            require(stop < len(ni['trace']) and ni['trace'][:stop] == ci['trace'][:stop], 'all pre-force native tokens exact')
            for key in ('step', 'inside_thinking', 'generated_thinking_tokens', 'rng_before'):
                require(ni['trace'][stop][key] == ci['trace'][stop][key], 'same intervention source state')
        gates.append({'case': c['case'], 'force_step': stop, 'pre_force_exact': True,
                      'whole_attempt_native_exact': stop is None})
    return gates


def validate_release(release, guard_result):
    require(type(release) is dict and release['guard_result'] == guard_result, 'actual ROOT/pressure result agreement')
    require(guard_result['status'] == 'complete' and type(guard_result['child_returncode']) is int
            and guard_result['child_returncode'] in (0, 3), 'complete guard, actual typed producer status')
    require(release['lock_reacquired'] is True and type(release['live_model_processes']) is list
            and release['live_model_processes'] == [], 'ROOT released lock and no live model')
    require(number(release['time'], 'release time') > 0 and integer(release['normal_pressure_samples'], 'normal') > 0
            and integer(release['bad_pressure_samples'], 'bad') == 0
            and number(release['swap_growth_gb'], 'swap') == 0, 'released normal pressure')


def validate_pressure(plan, plan_sha, release):
    import run_reasoning_budget_confirmation as driver
    pressure = Path(plan['pressure'])
    command = load(pressure / 'command.json')
    require(command['argv'] == driver.rendered_command(plan, plan_sha)[4:],
            'actual pressure guard launched the exact frozen child command')
    initial = command['initial']
    require(integer(initial['level'], 'initial pressure') == 1
            and number(initial['swap_gb'], 'initial swap') >= 0, 'normal initial pressure')
    samples = jsonl(pressure / 'pressure.jsonl')
    require(len(samples) == release['normal_pressure_samples'], 'actual pressure sample balance')
    previous = number(initial['time'], 'initial time')
    for sample in samples:
        require(integer(sample['level'], 'sample pressure') == 1 and sample['reason'] is None
                and integer(sample['consecutive_bad'], 'sample bad') == 0,
                'every actual pressure sample normal')
        stamp = number(sample['time'], 'sample time')
        require(previous <= stamp <= release['time'], 'ordered completed pressure samples before ROOT release')
        previous = stamp
        require(number(sample['swap_gb'], 'sample swap') == initial['swap_gb'], 'no actual swap growth')


def paired(native, candidate):
    nreport,nrows,_=native;creport,crows,_=candidate
    require(len(nrows)==len(crows)==16,'all16 paired first attempts')
    rows=[]
    for n,c in zip(nrows,crows):
        require(n['case']==c['case'],'fixed pair identity')
        good=lambda r:r['status']=='CORRECT' and r['stopped'] and not r['cap_hit']
        ng,cg=good(n),good(c)
        rows.append(dict(case=n['case'],native_status=n['status'],budget_status=c['status'],
            native_cap=n['cap_hit'],budget_cap=c['cap_hit'],native_completed_correct=ng,budget_completed_correct=cg,
            native_generated=n['generated_tokens'],budget_generated=c['generated_tokens'],
            native_correct_budget_failed=ng and not cg,native_failed_budget_correct=not ng and cg))
    return dict(per_case=rows,native_correct_budget_failed=sum(x['native_correct_budget_failed'] for x in rows),
        native_failed_budget_correct=sum(x['native_failed_budget_correct'] for x in rows),
        generated_token_delta=creport['generated_tokens']-nreport['generated_tokens'],
        completed_correct_delta=creport['completed_correct']-nreport['completed_correct'],
        interpretation='All16 predetermined pairs; capped correct remains separately reported, and all INVALID/capped/wrong attempts remain first attempts.')


def analyze(plan_path, expected_sha, release_path, release_sha, output):
    # Release proof and guard result are read before any new producer outcome.
    import run_reasoning_budget_confirmation as driver
    require(sha(plan_path) == expected_sha and sha(release_path) == release_sha, 'explicit analysis authority')
    plan = load(plan_path)
    driver.validate_plan(plan, plan_path)
    release = load(release_path)
    guard_result = load(Path(plan['pressure']) / 'result.json')
    validate_release(release, guard_result)
    validate_pressure(plan, expected_sha, release)
    driver.check_assets(plan)
    status = load(Path(plan['output']) / 'complete.json')
    require(status['status'] == 'COMPLETE' and status['arms'] == NAMES and status['returncode'] == guard_result['child_returncode'],
            'complete first run, no partial analysis')
    require(type(status['returncode']) is int, 'typed complete producer status')
    fixture = load(plan['fixture'])
    validated = []
    for arm in plan['arms']:
        process = load(Path(plan['output']) / (arm['name'] + '-process.json'))
        require(set(process) == {'returncode'} and type(process['returncode']) is int, 'actual producer status')
        validated.append(validate_arm(arm['results_dir'], arm['budget'], fixture, process['returncode']))
    require(status['returncode'] == (3 if any(arm[0]['invalid'] for arm in validated) else 0),
            'aggregate first-attempt INVALID status agrees with actual guard return code')
    exact_native(validated[0], validated[2])
    gates = pre_force(validated[0], validated[1])
    result = dict(status='COMPLETE_FIRST_ATTEMPTS', plan_sha256=expected_sha, release_sha256=release_sha,
                  guard_child_returncode=guard_result['child_returncode'], arms=[x[0] for x in validated],
                  native_before_and_after_exact_full_typed_fields=True, pre_force_controls=gates,
                  fixed_unused16_cases=True, all_first_attempts=48, paired=paired(validated[0], validated[1]), policy_chosen_before_budget_outcomes=True,
                  counters='API threshold excludes control markers; producer reasoning_tokens includes first closing token.',
                  limitations='One fixed independently authored confirmation set, one seed. Wrong/capped/INVALID attempts retained. No broad generalization, cross-prefix KL/NLL, equal-quality, quantization-causality or speed claim.')
    driver.put(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--release-proof', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args.plan, args.plan_sha256, args.release_proof, args.release_sha256, args.output),
                         allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, 'reasoning-budget analysis: ' + str(error) + '\n')
