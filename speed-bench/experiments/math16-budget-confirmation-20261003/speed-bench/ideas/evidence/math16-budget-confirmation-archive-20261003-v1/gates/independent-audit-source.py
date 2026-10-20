#!/usr/bin/env python3
"""Independent first-attempt final scoring and token-work audit; no model access."""
import argparse
import hashlib
import json
import re
from pathlib import Path

GOLDS = (196, 15, 43387, 3099, 48, 20565, 1287, 100, 171, 780, 80, 510, 486, 700, 12, 1492)
CAP, SEED, CLOSE = 2048, 20261002, 128822


def insist(value, why):
    if not value:
        raise ValueError(why)


def digest(path):
    path = Path(path)
    insist(path.suffix.lower() != '.gguf', 'model access forbidden')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pairs(items):
    out = {}
    for key, value in items:
        insist(key not in out, 'duplicate JSON field')
        out[key] = value
    return out


def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def load(path):
    return decode(Path(path).read_bytes())


def lines(path):
    raw = Path(path).read_bytes()
    insist(raw and raw.endswith(b'\n'), 'incomplete JSONL')
    return [decode(line) for line in raw.splitlines()]


def local(directory, name):
    insist(isinstance(name, str) and name not in ('', '.', '..') and Path(name).name == name,
           'plain artifact name')
    out = Path(directory) / name
    insist(not out.is_symlink() and out.resolve().parent == Path(directory).resolve(), 'artifact escape')
    return out


def final_number(text):
    # Independent bounded conversion; retain the production wrapper order.
    text = text.strip(' \t\n\r\v\f')
    if text.startswith('```') and text.endswith('```') and len(text) >= 6:
        text = text[3:-3]
        for language in ('python\n', 'text\n'):
            if text.startswith(language):
                text = text[len(language):]
                break
        text = text.strip(' \t\n\r\v\f')
    for left, right, minimum in (('**', '**', 4), ('`', '`', 2), ('\\boxed{', '}', 9)):
        if len(text) >= minimum and text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip(' \t\n\r\v\f')
    if re.fullmatch(r'[+-]?[0-9]+', text) is None:
        return None
    negative = text[0] == '-'
    digits = text.lstrip('+-').lstrip('0') or '0'
    if len(digits) > 19:
        return None
    value = int(digits) * (-1 if negative else 1)
    return value if -(1 << 63) <= value < (1 << 63) else None


def rng_after(state):
    state = state or 0x9e3779b97f4a7c15
    state ^= state >> 12
    state ^= (state << 25) & ((1 << 64) - 1)
    state ^= state >> 27
    return state & ((1 << 64) - 1)


def attempt(directory, row, gold, budget):
    trace = lines(local(directory, row['token_trace']))
    insist(len(trace) == row['generated_tokens'] + int(row['stopped']), 'sample extent')
    raw, final = bytearray(), bytearray()
    previous, reasoning, closed, force = SEED, 0, False, None
    for step, token in enumerate(trace):
        insist(token['step'] == step and 0 <= token['token_id'] < 129280, 'token order/range')
        before, after = int(token['rng_before'], 16), int(token['rng_after'], 16)
        insist(before == previous, 'RNG continuity')
        piece = bytes.fromhex(token['piece_hex'])
        active = token['policy_active']
        insist(type(active) is bool and token['penalty'] == 0, 'typed protocol-only intervention')
        if active:
            insist(budget == 1024 and force is None and not closed and
                   token['generated_thinking_tokens'] == 1024 and step <= CAP - 2,
                   'first eligible single forced close')
            insist(token['token_id'] == CLOSE and piece == b'</think>' and not token['stop'] and
                   after == before and token['intervention'] == 'forced-thinking-close', 'forced close semantics')
            force = step
        else:
            insist(after == rng_after(before) and token['intervention'] == 'native-sample', 'native RNG step')
        previous = after
        insist(token['stop'] == (row['stopped'] and step == row['generated_tokens']), 'actual stop position')
        if token['stop']:
            continue
        raw.extend(piece)
        if not closed:
            reasoning += 1
        if piece == b'</think>':
            insist(token['token_id'] == CLOSE, 'closing identity')
            closed = True
            final.clear()
        else:
            if force is not None:
                insist(piece != b'<think>' and not token['inside_thinking'], 'no resumed thinking')
            if closed:
                final.extend(piece)
    insist(bytes(raw) == local(directory, row['raw_trace']).read_bytes(), 'all raw token-piece bytes')
    insist(row['final'].encode() == bytes(final) and closed == row['think_closed'] and
           reasoning == row['reasoning_tokens'], 'final-region and work reconstruction')
    parsed = final_number(row['final']) if closed else None
    status = 'INVALID' if parsed is None else 'CORRECT' if parsed == gold else 'INCORRECT'
    insist(row['status'] == status and row['parse_ok'] == (parsed is not None) and
           row['correct_exact'] == (status == 'CORRECT') and
           row['parsed_answer'] == ('' if parsed is None else str(parsed)), 'independent final score')
    insist(type(row['generated_tokens']) is int and 0 <= row['generated_tokens'] <= CAP and
           row['cap_hit'] == (row['generated_tokens'] == CAP) and row['stopped'] != row['cap_hit'], 'actual cap contract')
    return {'case': row['case'], 'status': status, 'completed_correct': status == 'CORRECT' and row['stopped'],
            'capped_correct': status == 'CORRECT' and row['cap_hit'], 'cap': row['cap_hit'],
            'generated': row['generated_tokens'], 'reasoning': reasoning, 'force_step': force}, trace


def self_test():
    for text, expected in [(' +000196 ', 196), ('```text\n**`\\boxed{15}`**\n```', 15),
                           ('-9223372036854775808', -(1 << 63)), ('9223372036854775807', (1 << 63) - 1),
                           ('+' + '0' * 5000 + '12', 12), ('9223372036854775808', None),
                           ('-9223372036854775809', None), ('1,287', None), ('answer: 15', None), ('', None)]:
        insist(final_number(text) == expected, 'signed64 self-check')
    insist(rng_after(SEED) != SEED, 'RNG state self-check')


def audit(args):
    insist(digest(args.plan) == args.plan_sha256 and digest(args.release) == args.release_sha256 and
           digest(args.summary) == args.summary_sha256, 'explicit frozen authorities')
    plan, release, summary = load(args.plan), load(args.release), load(args.summary)
    guard = load(Path(plan['pressure']) / 'result.json')
    insist(guard == release['guard_result'] and guard['status'] == 'complete' and
           type(guard['child_returncode']) is int and guard['child_returncode'] in (0, 3), 'complete guard')
    insist(release['lock_reacquired'] is True and release['live_model_processes'] == [] and
           release['bad_pressure_samples'] == 0 and release['swap_growth_gb'] == 0 and
           release['normal_pressure_samples'] > 0, 'released normal serial model job')
    complete = load(Path(plan['output']) / 'complete.json')
    insist(complete['status'] == 'COMPLETE' and complete['returncode'] == guard['child_returncode'], 'all first attempts complete')
    fixture = load(plan['fixture'])['cases']
    insist(len(fixture) == 16 and tuple(int(x['answer']) for x in fixture) == GOLDS, 'fixed sixteen golds')
    arms, token_traces, native_rows = [], [], []
    for arm in plan['arms']:
        rows = lines(Path(arm['results_dir']) / 'results.jsonl')
        insist(len(rows) == 16 and [r['case'] for r in rows] == [c['case'] for c in fixture], 'complete ordered case inventory')
        results, traces = [], []
        for row, case, gold in zip(rows, fixture, GOLDS):
            insist(row['prompt'] == case['prompt'] and row['expected'] == case['answer'], 'unchanged prompt/gold')
            result, trace = attempt(arm['results_dir'], row, gold, arm['budget'])
            results.append(result)
            traces.append(trace)
        arms.append(results)
        token_traces.append(traces)
        native_rows.append(rows)
    insist([x['name'] for x in plan['arms']] == ['native-before', 'budget1024', 'native-after'], 'fixed arm order')
    insist(arms[0] == arms[2] and token_traces[0] == token_traces[2], 'native full token/RNG/piece/work repeat')
    for n, c, nt, ct, nr, cr in zip(arms[0], arms[1], token_traces[0], token_traces[1], native_rows[0], native_rows[1]):
        np = local(plan['arms'][0]['results_dir'], nr['prompt_trace']).read_bytes()
        cp = local(plan['arms'][1]['results_dir'], cr['prompt_trace']).read_bytes()
        insist(np == cp, 'common opaque prompt IDs')
        if c['force_step'] is None:
            insist(n == c and nt == ct, 'unforced attempt exact')
        else:
            at = c['force_step']
            insist(nt[:at] == ct[:at] and nt[at]['rng_before'] == ct[at]['rng_before'], 'entire pre-force native token/RNG prefix')
    counts = []
    for results in arms:
        counts.append({'completed_correct': sum(x['completed_correct'] for x in results),
                       'incorrect': sum(x['status'] == 'INCORRECT' for x in results),
                       'invalid': sum(x['status'] == 'INVALID' for x in results),
                       'capped_correct': sum(x['capped_correct'] for x in results),
                       'cap_hits': sum(x['cap'] for x in results),
                       'generated_tokens': sum(x['generated'] for x in results),
                       'reasoning_tokens': sum(x['reasoning'] for x in results),
                       'forced_cases': sum(x['force_step'] is not None for x in results)})
    for own, producer in zip(counts, summary['arms']):
        insist(all(own[k] == producer[k] for k in own), 'independent complete arm recount')
    lost = sum(n['completed_correct'] and not c['completed_correct'] for n, c in zip(arms[0], arms[1]))
    gained = sum(c['completed_correct'] and not n['completed_correct'] for n, c in zip(arms[0], arms[1]))
    insist(summary['paired']['native_correct_budget_failed'] == lost and
           summary['paired']['native_failed_budget_correct'] == gained, 'both paired discordances')
    result = {'status': 'ROOT_INDEPENDENT48_FINAL_SCORE_TOKEN_WORK_AND_PAIRED_AUDIT_PASS',
              'plan_sha256': args.plan_sha256, 'release_sha256': args.release_sha256,
              'producer_summary_sha256': args.summary_sha256, 'source_sha256': digest(__file__),
              'all48_attempts': True, 'counts': counts, 'native_correct_budget_failed': lost,
              'native_failed_budget_correct': gained,
              'scope': 'Final answers, full token/RNG/piece traces and work. Raw artifacts here are text bytes, not full-vocabulary logits. No cross-prefix KL/NLL or speed/quantization-causality claim.',
              'model_or_helper_or_guard_executed': False}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    for name in ('plan', 'plan-sha256', 'release', 'release-sha256', 'summary', 'summary-sha256', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.self_test:
        print('ROOT math16 independent scorer self-checks: PASS')
    else:
        insist(all(getattr(args, n) for n in ('plan', 'plan_sha256', 'release', 'release_sha256', 'summary', 'summary_sha256', 'output')), 'explicit audit authority arguments')
        audit(args)
