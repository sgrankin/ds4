#!/usr/bin/env python3
"""Model-free independent scripted attempts and real subprocess driver tests."""
import contextlib
import copy
import errno
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'speed-bench/ideas'))
import analyze_reasoning_budget as a
import run_reasoning_budget as d

ART = WORK / 'speed-bench/ideas/artifacts'
ROOT = ART / ('reasoning-budget-cpu-fixtures-' + str(time.time_ns()))
ROOT.mkdir()
CASE_IDS = ['circular10', 'restricted_derangements8', 'lattice8x6', 'subset_mod7']
GOLDS = ['123', '10860', '2002', '132']
TEXT = (WORK / 'speed-bench/reasoning_budget_cases.h').read_text()
FIXTURE = {'suite': 'counting-v1', 'cases': [dict(case=c, prompt=p, answer=g) for c, p, g in
    re.findall(r'\{"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*ANSWER_INTEGER\}', TEXT)]}
SERIAL = 0
NEGATIVE_TRIALS = 0


def dump(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


def lines(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def fork(source):
    global SERIAL
    SERIAL += 1
    target = ROOT / ('mutation-' + str(SERIAL))
    # Small immutable fixture copies; never use the real helper or model.
    shutil.copytree(source, target)
    return target


def state_advance(state):
    # Independently transcribed frozen native xorshift state, checked against
    # literal seed/first-state vectors, not the analyzer's next_rng function.
    word = state if state else 11400714819323198485
    word = word ^ (word // 4096)
    word = word ^ ((word * 33554432) % 18446744073709551616)
    word = word ^ (word // 134217728)
    return word % 18446744073709551616


def scripted_arm(root, budget, historical=False):
    """Predetermined event timelines, independent of production validation.

    Case0 closes naturally early; case1 is wrong only after intervention;
    case2 retains invalid output; case3 has a strictly correct capped final.
    Published counts come from event endpoints, rather than validator helpers.
    """
    root.mkdir()
    rows = []
    for index, case in enumerate(FIXTURE['cases']):
        close_at = [2, 1100, None, 1500][index] if budget == 0 else [2, 1024, 1024, 1024][index]
        force_at = close_at if budget and index else None
        count = [4, 1102, 2048, 2048][index] if budget == 0 else [4, 1026, 1026, 2048][index]
        final = ['123', '10860' if budget == 0 else '999', '' if budget == 0 else 'not-an-integer', '132'][index]
        stopped = index in (0, 1) or (index == 2 and budget != 0)
        raw = bytearray()
        tokens = []
        rng = 20261002
        for step in range(count + int(stopped)):
            stop = step == count
            inside = close_at is None or step <= close_at
            prior_count = step if close_at is None else min(step, close_at)
            forced = step == force_at
            piece = (b'<eos>' if stop else b'</think>' if step == close_at else
                     final.encode() if close_at is not None and step == close_at + 1 else
                     b' ' if close_at is not None and step > close_at else b'r')
            token_id = 1 if stop else 128822 if step == close_at else 10 if piece == b'r' else 11
            before = rng
            if not forced:
                rng = state_advance(rng)
            token = dict(step=step, token_id=token_id, stop=stop, inside_thinking=inside,
                         generated_thinking_tokens=prior_count, policy_active=forced,
                         rng_before=f'{before:016x}', rng_after=f'{rng:016x}', piece_hex=piece.hex())
            if historical:
                token['applied_logit_delta'] = 0
                token['selected_marker'] = False
            else:
                token['penalty'] = 0
                token['intervention'] = 'forced-thinking-close' if forced else 'native-sample'
            tokens.append(token)
            if not stop:
                raw.extend(piece)
        # Producer retains whitespace after the final number on capped case3.
        if index == 3:
            final += ' ' * (count - close_at - 2)
        parse_ok = index != 2
        status = 'INVALID' if not parse_ok else 'INCORRECT' if index == 1 and budget else 'CORRECT'
        row = dict(case=case['case'], suite='counting-v1', think_level=25, native_penalty=0,
                   think_enabled=True, prompt_supplied_think_close=False,
                   generated_think_close=close_at is not None, status=status, parse_ok=parse_ok,
                   parse_reason='none' if parse_ok else 'unclosed_reasoning' if close_at is None else 'unsupported_final_format',
                   scorer='strict-final-v2', seed=20261002, temperature_float_value=0.6000000238418579,
                   top_p_float_value=0.949999988079071, min_p=0, max_tokens=2048,
                   cache_gib_requested=64, context_size=8192, prefill_chunk=512, prompt_tokens=3,
                   generated_tokens=count, reasoning_tokens=count if close_at is None else close_at + 1,
                   think_closed=close_at is not None, stopped=stopped, cap_hit=count == 2048,
                   correct_exact=status == 'CORRECT', prefill_seconds=0.1, decode_seconds=0.2,
                   parsed_answer='' if not parse_ok else '999' if status == 'INCORRECT' else case['answer'],
                   final=final, expected=case['answer'], prompt=case['prompt'],
                   raw_trace=case['case'] + '.txt', token_trace=case['case'] + '.tokens.jsonl',
                   prompt_trace=case['case'] + '.prompt.ids', policy_type='protocol-boundary-token',
                   reasoning_budget=budget, forced_think_close=force_at is not None)
        (root / row['raw_trace']).write_bytes(raw)
        lines(root / row['token_trace'], tokens)
        (root / row['prompt_trace']).write_text('128000\n42\n128821\n')
        if historical:
            for key in ('native_penalty', 'think_enabled', 'prompt_supplied_think_close', 'generated_think_close',
                        'temperature_float_value', 'top_p_float_value', 'context_size', 'prefill_chunk',
                        'prompt_tokens', 'token_trace', 'prompt_trace', 'policy_type', 'reasoning_budget', 'forced_think_close'):
                row.pop(key)
            row.update(suite='counting', marker_profile='closing', mapped_markers=1, closing_boost=0,
                       policy_direction='ADD', closing_token_id=128822, scope='think', engine_penalty=0,
                       engine_marker_profile='hesitation', engine_overthink_after=0, temperature=0.6, top_p=0.95,
                       closing_after=1024, case_offset=0, case_limit=4,
                       diagnostics=dict(temperature_float_value=0.6000000238418579, active_steps=0,
                                        token_trace=case['case'] + '.tokens.jsonl'))
        rows.append(row)
    lines(root / 'results.jsonl', rows)
    return root


NATIVE = scripted_arm(ROOT / 'native', 0)
CANDIDATE = scripted_arm(ROOT / 'candidate', 1024)
HISTORICAL = scripted_arm(ROOT / 'historical', 0, historical=True)
FAKE = ROOT / 'fake_cpu_producer.py'
FAKE.write_text('''import json, os, pathlib, shutil, sys
source, target, rc, audit = sys.argv[1:]
shutil.copytree(source, target)
with pathlib.Path(audit).open('a') as f:
 f.write(json.dumps({'target':target,'DS4X':os.environ.get('DS4X'),
  'DS4_UNEXPECTED':os.environ.get('DS4_UNEXPECTED'),
  'DS4_OVERTHINK_PENALTY':os.environ.get('DS4_OVERTHINK_PENALTY')})+'\\n')
print('CPU synthetic producer retained first attempt')
sys.exit(int(rc))
''')


def synthetic_plan(name):
    prepared = ROOT / name
    prepared.mkdir()
    dump(prepared / 'fixture.json', FIXTURE)
    audit = prepared / 'audit.jsonl'
    arms = []
    for arm, budget, source in zip(a.NAMES, [0, 1024, 0], [NATIVE, CANDIDATE, NATIVE]):
        output = prepared / (arm + '-result')
        arms.append(dict(name=arm, budget=budget, results_dir=str(output),
                         argv=[sys.executable, str(FAKE), str(source), str(output), '3', str(audit)]))
    plan = dict(output=str(prepared / 'run'), fixture=str(prepared / 'fixture.json'),
                reference_dir=str(HISTORICAL), reference_returncode=3, arms=arms, run_cwd=str(WORK),
                environment=dict(d.ENV), pressure=str(prepared / 'pressure'),
                guard_command_template=['cpu-python', 'cpu-guard', 'cpu-pressure', '--', 'cpu-child', d.PLACEHOLDER])
    path = prepared / 'plan.json'
    dump(path, plan)
    return path, plan


def shape_plan(name):
    # Pure plan/schema fixture: identities are explicit constants, assets are
    # never opened or trusted as production in these schema-only trials.
    prepared = ART / ('reasoning-budget-shape-' + str(time.time_ns()) + '-' + name)
    prepared.mkdir()
    dump(prepared / 'fixture.json', FIXTURE)
    plan_path = prepared / 'plan.json'
    environment = dict(d.ENV)
    for shader in d.SHADER_NAMES:
        environment['DS4_METAL_' + shader + '_SOURCE'] = str(prepared / 'metal' / (shader.lower() + '.metal'))
    source_refs = {str(prepared / 'source' / name): pin for name, pin in d.SOURCES.items()}
    objects = {str(d.ROOT / '.workspaces/idea-quant-effort' / name): pin for name, pin in d.OBJECTS.items()}
    paths = [str(prepared / name) for name in ['run_reasoning_budget.py', 'analyze_reasoning_budget.py',
             'reasoning_budget_signed64.py', 'fixture.json']]
    paths += [str(d.BUILD / name) for name in ['reasoning-budget-probe', 'build-manifest.json', 'compile.log', 'self-test.log']]
    paths += [str(d.GUARD), str(d.PYTHON), str(d.PYTHON_TARGET), str(d.FRAMEWORK)] + list(source_refs) + list(objects)
    paths += [str(d.CPU_PROOF), str(d.CPU_TEST)]
    paths += [value for key, value in environment.items() if key.endswith('_SOURCE')]
    paths += [str(d.HOLDOUT / name) for name in ['plan.json', 'analyze_native_effort_holdout.py']]
    paths += [str(d.CLOSING / name) for name in ['plan.json', 'source/core-proof.json']]
    paths += [str(d.CLOSING / 'source' / name) for name in ['closing_pilot.c', 'closing_policy.h',
                                                        'overthink_counting_cases.h', 'overthink_diagnostics.h']]
    paths += [str(d.ROOT / '.workspaces/idea-quant-closing/speed-bench/closing_pilot.c')]
    paths += [str(d.REFERENCE / name) for name in ['results.jsonl', 'marker-map.json']]
    paths += [str(d.REFERENCE / (case + '-boost0.0' + ext)) for case in CASE_IDS for ext in ['.txt', '.tokens.jsonl']]
    assets = {path: dict(sha256='0' * 64, bytes=1) for path in paths}
    for path, pin in list(objects.items()) + list(source_refs.items()) + [
        (str(d.BUILD / 'reasoning-budget-probe'), d.BINARY_SHA), (str(d.GUARD), d.GUARD_SHA),
        (str(d.PYTHON), d.PYTHON_SHA), (str(d.PYTHON_TARGET), d.PYTHON_SHA), (str(d.FRAMEWORK), d.FRAMEWORK_SHA),
        (str(prepared / 'reasoning_budget_signed64.py'), d.SCORER_SHA)]:
        assets[path]['sha256'] = pin
    arms = []
    for name, budget in zip(a.NAMES, [0, 1024, 0]):
        run_name = prepared.name + '-' + name
        arms.append(dict(name=name, budget=budget, run_name=run_name,
            argv=[str(d.BUILD / 'reasoning-budget-probe'), d.MODEL['path'], run_name, '25', str(budget)],
            results_dir=str(d.WORK / 'speed-bench/ideas/artifacts' / run_name)))
    plan = dict(schema='reasoning-budget-pilot-v1', prepared_only=True, model_grant=False,
        analysis_requires_root_release=True, no_automatic_retry_or_scheduler=True, settings=copy.deepcopy(d.SETTINGS),
        workspace=str(d.WORK), run_cwd=str(d.WORK), root_cwd=str(d.ROOT), model=copy.deepcopy(d.MODEL),
        binary=str(d.BUILD / 'reasoning-budget-probe'), binary_sha256=d.BINARY_SHA,
        build_manifest=str(d.BUILD / 'build-manifest.json'), guard=str(d.GUARD), guard_sha256=d.GUARD_SHA,
        cpu_proof=str(d.CPU_PROOF),
        launch_python=str(d.PYTHON), guard_interpreter_identity=dict(path=str(d.FRAMEWORK), sha256=d.FRAMEWORK_SHA),
        launch_python_resolved_identity=dict(path=str(d.PYTHON_TARGET), sha256=d.PYTHON_SHA),
        guard_command_template=[str(d.PYTHON), str(d.GUARD), str(prepared / 'pressure'), '--', str(d.PYTHON),
            str(prepared / 'run_reasoning_budget.py'), '--run', str(plan_path), '--plan-sha256', d.PLACEHOLDER, '--root-granted'],
        output=str(prepared / 'run'), pressure=str(prepared / 'pressure'), fixture=str(prepared / 'fixture.json'),
        reference_dir=str(d.REFERENCE), reference_returncode=3, arms=arms, environment=environment,
        source_refs=source_refs, core_objects=objects, assets=assets)
    return plan_path, plan


class ProtocolTests(unittest.TestCase):
    def reject(self, callback):
        global NEGATIVE_TRIALS
        NEGATIVE_TRIALS += 1
        with self.assertRaises((ValueError, KeyError, TypeError, OSError)):
            callback()

    def arm(self, root, budget=1024, rc=3):
        return a.validate_arm(root, budget, FIXTURE, rc)

    def mutate_row(self, updates, root=CANDIDATE, index=1, budget=1024, rc=3):
        target = fork(root)
        rows = a.jsonl(target / 'results.jsonl')
        rows[index].update(updates)
        lines(target / 'results.jsonl', rows)
        self.reject(lambda: self.arm(target, budget, rc))

    def mutate_token(self, updates, step=1024, root=CANDIDATE, index=1, budget=1024):
        target = fork(root)
        row = a.jsonl(target / 'results.jsonl')[index]
        tokens = a.jsonl(target / row['token_trace'])
        tokens[step].update(updates)
        lines(target / row['token_trace'], tokens)
        self.reject(lambda: self.arm(target, budget))

    def test_independent_rng_known_vectors(self):
        self.assertEqual(state_advance(20261002), 0x00026a77b178752e)
        self.assertEqual(a.next_rng(20261002), 0x00026a77b178752e)
        for state in [0, 1, 2**64 - 1, 0x123456789abc, 20261002]:
            self.assertEqual(state_advance(state), a.next_rng(state))

    def test_complete_mixed_first_attempts(self):
        native = self.arm(NATIVE, 0)
        candidate = self.arm(CANDIDATE)
        historical = a.validate_arm(HISTORICAL, 0, FIXTURE, 3, historical=True)
        self.assertTrue(a.exact_native(historical, native, historical=True))
        self.assertTrue(a.exact_native(native, self.arm(NATIVE, 0)))
        gates = a.pre_force(native, candidate)
        self.assertEqual([x['force_step'] for x in gates], [None, 1024, 1024, 1024])
        self.assertEqual(candidate[0]['completed_correct'], 1)
        self.assertEqual(candidate[0]['incorrect'], 1)
        self.assertEqual(candidate[0]['invalid'], 1)
        self.assertEqual(candidate[0]['capped_correct'], 1)
        self.assertEqual(candidate[1][1]['reasoning_tokens'], 1025)

    def test_actual_frozen_historical_contract_without_model(self):
        report, rows, infos = a.validate_arm(d.REFERENCE, 0, FIXTURE, 3, historical=True)
        self.assertEqual((report['completed_correct'], report['invalid'], report['generated_tokens']), (2, 2, 5985))
        self.assertTrue(all(info['prompt_sha256'] is None for info in infos))

    def test_typed_fixed_options(self):
        for key, value in [('reasoning_budget', True), ('seed', True), ('temperature_float_value', .6),
            ('top_p_float_value', .95), ('native_penalty', False), ('cache_gib_requested', 64.0),
            ('think_enabled', 1), ('prefill_chunk', 513), ('policy_type', 'closing-boost'), ('suite', 'heldout')]:
            self.mutate_row({key: value})

    def test_literal_summary_types_and_geometry(self):
        for key, value in [('generated_tokens', True), ('reasoning_tokens', -1), ('prompt_tokens', 0),
            ('cap_hit', 1), ('stopped', 1), ('correct_exact', 0), ('think_closed', 1),
            ('forced_think_close', 1), ('generated_think_close', 1), ('generated_tokens', 2049),
            ('cap_hit', True), ('stopped', False), ('prefill_seconds', -1), ('decode_seconds', True)]:
            self.mutate_row({key: value})

    def test_scorer_golds_and_wrong_not_invalid(self):
        for updates in [dict(expected='999'), dict(prompt='changed'), dict(status='CORRECT'),
            dict(parsed_answer='10860'), dict(parse_ok=False), dict(parse_reason='none!'), dict(scorer='tolerant')]:
            self.mutate_row(updates)
        self.reject(lambda: self.arm(CANDIDATE, rc=0))
        self.reject(lambda: self.arm(CANDIDATE, rc=True))
        self.assertEqual(a.scorer.parsed_integer('+000123'), '123')
        self.assertEqual(a.scorer.parsed_integer('```text\n**`\\boxed{123}`**\n```'), '123')
        for value in ['9223372036854775808', '-9223372036854775809', '9' * 5000, '123 trailing']:
            self.assertIsNone(a.scorer.parsed_integer(value))
        self.assertEqual(a.scorer.parsed_integer('-' + '0' * 5000 + '123'), '-123')

    def test_forced_close_eligibility_and_rng(self):
        for updates in [dict(policy_active=False), dict(token_id=1), dict(piece_hex=b'</think>!'.hex()),
            dict(rng_after='0000000000000000'), dict(inside_thinking=False), dict(generated_thinking_tokens=1023),
            dict(stop=True), dict(intervention='native-sample'), dict(penalty=.1)]:
            self.mutate_token(updates)
        self.mutate_token(dict(policy_active=True, intervention='forced-thinking-close'), step=1023)
        self.mutate_token(dict(policy_active=True, intervention='forced-thinking-close'), step=1025)

    def test_postforce_phase_must_be_outside(self):
        self.mutate_token(dict(inside_thinking=True), step=1025)
        self.mutate_token(dict(piece_hex=b'<think>'.hex()), step=1025)
        self.mutate_token(dict(generated_thinking_tokens=1025), step=1025)

    def test_force_final_slot_refuses_even_self_consistent_metadata(self):
        target = fork(CANDIDATE)
        row = a.jsonl(target / 'results.jsonl')[3]
        tokens = a.jsonl(target / row['token_trace'])
        # A caller cannot delay forcing to2047 and publish that unevaluated
        # close. The first-eligible gate or remaining-slot gate must reject.
        token = copy.deepcopy(tokens[1024])
        token.update(step=2047, generated_thinking_tokens=2047,
                     rng_before=tokens[2047]['rng_before'], rng_after=tokens[2047]['rng_before'])
        tokens[2047] = token
        lines(target / row['token_trace'], tokens)
        self.reject(lambda: self.arm(target))

    def test_native_rng_exact_advance_and_chain(self):
        self.mutate_token(dict(rng_after='000000000135288a'), step=0, root=NATIVE, budget=0)
        self.mutate_token(dict(rng_before='0000000000000000'), step=3, root=NATIVE, budget=0)
        self.mutate_token(dict(rng_after='FFFFFFFFFFFFFFFF'), step=3, root=NATIVE, budget=0)

    def test_duplicate_missing_or_truncated_tokens(self):
        for kind in ['duplicate', 'missing', 'truncated', 'duplicate-key', 'nonfinite']:
            target = fork(CANDIDATE)
            row = a.jsonl(target / 'results.jsonl')[1]
            path = target / row['token_trace']
            data = path.read_bytes()
            if kind == 'duplicate':
                values = a.jsonl(path); values[10] = values[9]; lines(path, values)
            elif kind == 'missing':
                values = a.jsonl(path); values.pop(10); lines(path, values)
            elif kind == 'truncated': path.write_bytes(data[:-1])
            elif kind == 'duplicate-key': path.write_bytes(data.replace(b'"step": 0', b'"step": 0, "step": 0', 1))
            else: path.write_bytes(data.replace(b'"penalty": 0', b'"penalty": NaN', 1))
            self.reject(lambda: self.arm(target))

    def test_token_types_and_vocabulary(self):
        for updates in [dict(step=True), dict(token_id=-1), dict(token_id=129280), dict(stop=0),
            dict(policy_active=1), dict(generated_thinking_tokens=True), dict(piece_hex='0'), dict(penalty=True)]:
            self.mutate_token(updates)
        self.mutate_token(dict(token_id=128822), step=0)

    def test_raw_final_prompt_extent_and_path_escape(self):
        for field, content in [('raw_trace', b'wrong'), ('prompt_trace', b'1\n2'), ('prompt_trace', b'1\n129280\n3\n')]:
            target = fork(CANDIDATE)
            row = a.jsonl(target / 'results.jsonl')[1]
            (target / row[field]).write_bytes(content)
            self.reject(lambda: self.arm(target))
        self.mutate_row(dict(raw_trace='../escape.txt'))
        self.mutate_row(dict(token_trace='/absolute.tokens'))
        self.mutate_row(dict(final='10860'))

    def test_complete_results_order_and_gold_fixture(self):
        target = fork(CANDIDATE)
        rows = a.jsonl(target / 'results.jsonl'); rows.reverse(); lines(target / 'results.jsonl', rows)
        self.reject(lambda: self.arm(target))
        for kind in ['missing', 'duplicate', 'changed-gold']:
            fixture = copy.deepcopy(FIXTURE)
            if kind == 'missing': fixture['cases'].pop()
            elif kind == 'duplicate': fixture['cases'][1] = fixture['cases'][0]
            else: fixture['cases'][1]['answer'] = '999'
            self.reject(lambda: a.fixture_check(fixture))

    def test_native_reference_mismatch_and_preforce_mismatch(self):
        native = self.arm(NATIVE, 0)
        candidate = self.arm(CANDIDATE)
        changed = copy.deepcopy(native); changed[2][0]['trace'][0]['token_id'] += 1
        self.reject(lambda: a.exact_native(native, changed))
        changed = copy.deepcopy(candidate); changed[2][1]['trace'][0]['token_id'] += 1
        self.reject(lambda: a.pre_force(native, changed))
        changed = copy.deepcopy(candidate); changed[2][0]['raw_sha256'] = '0' * 64
        self.reject(lambda: a.pre_force(native, changed))

    def test_strict_release_actual_schema(self):
        result = dict(status='complete', child_returncode=3)
        proof = dict(guard_result=result, lock_reacquired=True, live_model_processes=[], time=1,
                     normal_pressure_samples=1, bad_pressure_samples=0, swap_growth_gb=0)
        a.validate_release(proof, result)
        for field, value in [('lock_reacquired', 1), ('live_model_processes', ['helper']), ('time', True),
            ('time', float('nan')), ('normal_pressure_samples', 0), ('bad_pressure_samples', 1),
            ('swap_growth_gb', .01), ('guard_result', dict(status='complete', child_returncode=0))]:
            wrong = copy.deepcopy(proof); wrong[field] = value
            self.reject(lambda: a.validate_release(wrong, result))
        for value in [True, 1, 2, -1]:
            wrong = dict(result, child_returncode=value)
            self.reject(lambda: a.validate_release(dict(proof, guard_result=wrong), wrong))

    def test_actual_pressure_command_sample_balance_and_finite_fields(self):
        path, plan = synthetic_plan('pressure-only')
        pressure = Path(plan['pressure']); pressure.mkdir()
        pin = a.sha(path)
        command = dict(argv=d.rendered_command(plan, pin)[4:], initial=dict(time=.1, level=1, swap_gb=0))
        samples = [dict(time=.2, level=1, swap_gb=0, reason=None, consecutive_bad=0)]
        proof = dict(normal_pressure_samples=1, time=1)
        dump(pressure / 'command.json', command); lines(pressure / 'pressure.jsonl', samples)
        a.validate_pressure(plan, pin, proof)
        for key, value in [('level', True), ('level', 2), ('time', True), ('time', .01), ('time', 2),
                           ('swap_gb', .1), ('reason', 'pressure-stop'), ('consecutive_bad', True)]:
            wrong = copy.deepcopy(samples); wrong[0][key] = value
            lines(pressure / 'pressure.jsonl', wrong)
            self.reject(lambda: a.validate_pressure(plan, pin, proof))
        lines(pressure / 'pressure.jsonl', samples)
        self.reject(lambda: a.validate_pressure(plan, pin, dict(proof, normal_pressure_samples=2)))
        command['argv'].append('--unfrozen'); dump(pressure / 'command.json', command)
        self.reject(lambda: a.validate_pressure(plan, pin, proof))

    def test_plan_shape_and_required_inventory(self):
        path, plan = shape_plan('shape')
        d.validate_plan(plan, path)
        for mutation in [lambda p: p['assets'].pop(p['binary']),
            lambda p: p['assets'].pop(next(iter(p['source_refs']))),
            lambda p: p['assets'].pop(next(iter(p['core_objects']))),
            lambda p: p['assets'].pop(str(path.parent / 'reasoning_budget_signed64.py')),
            lambda p: p.update(fixture=str(path.parent / 'other.json')),
            lambda p: p.update(model_grant=True), lambda p: p.update(prepared_only=1),
            lambda p: p['settings'].update(reasoning_budget=True),
            lambda p: p['settings'].update(selected_tile_override=64),
            lambda p: p['arms'].reverse(), lambda p: p['arms'][1].update(budget=True),
            lambda p: p['arms'][1]['argv'].append('--retry'),
            lambda p: p['guard_command_template'].__setitem__(-2, '0' * 64),
            lambda p: p['environment'].update(DS4_UNEXPECTED='1'),
            lambda p: p['guard_interpreter_identity'].update(path=str(d.PYTHON)),
            lambda p: p['core_objects'].__setitem__(next(iter(p['core_objects'])), '1' * 64),
            lambda p: p['assets'][p['binary']].update(bytes=True)]:
            wrong = copy.deepcopy(plan); mutation(wrong)
            self.reject(lambda: d.validate_plan(wrong, path))

    def test_asset_drift_and_nonexecutability(self):
        executable = ROOT / 'asset.sh'; executable.write_text('fixture'); executable.chmod(0o755)
        plan = dict(binary=str(executable), assets={str(executable): d.asset(executable)})
        d.check_assets(plan)
        executable.chmod(0o644)
        self.reject(lambda: d.check_assets(plan))
        executable.chmod(0o755); executable.write_text('changed')
        self.reject(lambda: d.check_assets(plan))
        self.reject(lambda: a.sha(ROOT / 'not-opened.gguf'))

    def test_pinned_launcher_symlink_only_and_distinct_framework(self):
        self.assertEqual(d.asset(d.PYTHON)['sha256'], d.PYTHON_SHA)
        self.assertFalse(os.path.samefile(d.PYTHON, d.FRAMEWORK))
        with patch.object(d, 'PYTHON_TARGET', ROOT / 'foreign-target'):
            self.reject(lambda: d.asset(d.PYTHON))
        symlink = ROOT / 'foreign-symlink'; symlink.symlink_to(d.PYTHON_TARGET)
        self.reject(lambda: d.asset(symlink))

    def test_guard_exact_framework_suffix_and_held_lock(self):
        path, plan = shape_plan('guard')
        lock_root = ROOT / 'guard-root'; (lock_root / 'speed-bench/ideas/artifacts').mkdir(parents=True)
        (lock_root / 'speed-bench/ideas/artifacts/model-job.lock').touch()
        command = d.rendered_command(plan, 'a' * 64)
        actual = [str(d.FRAMEWORK)] + command[1:]
        def digest(path):
            return d.GUARD_SHA if str(path) == str(d.GUARD) else d.FRAMEWORK_SHA
        def held(fd, flags):
            raise BlockingIOError(errno.EAGAIN, 'held')
        with patch.object(d, 'ROOT', lock_root), patch.object(a, 'sha', side_effect=digest), \
                patch.object(d.os.path, 'samefile', return_value=True), patch.object(d.fcntl, 'flock', side_effect=held), \
                patch.object(d.subprocess, 'check_output', return_value=' '.join(actual)) as ps:
            self.assertTrue(d.require_guard(plan, path, 'a' * 64)['lock_held'])
            self.assertEqual(ps.call_args.kwargs['timeout'], 2)
            for changed in [actual + ['--retry'], actual[:-1], [actual[0]] + actual[1:-2] + ['bad', actual[-1]]]:
                ps.return_value = ' '.join(changed)
                self.reject(lambda: d.require_guard(plan, path, 'a' * 64))
            ps.return_value = ' '.join(actual)
            with patch.object(d.os.path, 'samefile', return_value=False):
                self.reject(lambda: d.require_guard(plan, path, 'a' * 64))
            with patch.object(d.fcntl, 'flock', return_value=None):
                self.reject(lambda: d.require_guard(plan, path, 'a' * 64))

    def test_explicit_grant_and_exclusive_output(self):
        path, plan = shape_plan('grant')
        pin = 'a' * 64
        proof = dict(status='GRANTED', sole_model_grant=True, manifest_sha256=pin,
                     command=d.rendered_command(plan, pin), time=1)
        dump(path.parent / 'root-launch-proof.json', proof)
        d.require_grant(plan, path, pin)
        for field, value in [('status', 'PREPARED'), ('sole_model_grant', 1), ('manifest_sha256', 'b' * 64),
                             ('command', []), ('time', False)]:
            wrong = dict(proof); wrong[field] = value; dump(path.parent / 'root-launch-proof.json', wrong)
            self.reject(lambda: d.require_grant(plan, path, pin))
        d.put(ROOT / 'exclusive.json', dict(first=True))
        self.reject(lambda: d.put(ROOT / 'exclusive.json', dict(second=True)))
        self.assertEqual(a.load(ROOT / 'exclusive.json'), dict(first=True))

    def invoke_synthetic(self, path, plan, guard_failure=False):
        events = []
        def guard(*args):
            events.append('guard')
            if guard_failure: raise ValueError('CPU refusal before metadata')
            return dict(cpu_only=True)
        def grant(*args): events.append('grant')
        def assets(*args): events.append('assets')
        def model(*args):
            self.assertIn('guard', events); self.assertIn('grant', events)
            events.append('fake-model-identity'); return dict(cpu_only=True)
        with patch.object(d, 'validate_plan'), patch.object(d, 'require_guard', side_effect=guard), \
                patch.object(d, 'require_grant', side_effect=grant), patch.object(d, 'check_assets', side_effect=assets), \
                patch.object(d, 'model_identity', side_effect=model), \
                patch.dict(os.environ, {'DS4X': 'remove', 'DS4_UNEXPECTED': 'remove'}), contextlib.redirect_stdout(io.StringIO()):
            result = d.run(path, a.sha(path), root_granted=True)
        return result, events

    def test_real_three_arm_cpu_producers_retaining_rc3_wrong_and_caps(self):
        path, plan = synthetic_plan('three-arms')
        rc, events = self.invoke_synthetic(path, plan)
        self.assertEqual(rc, 3)
        self.assertEqual(events[:4], ['guard', 'grant', 'assets', 'fake-model-identity'])
        complete = a.load(Path(plan['output']) / 'complete.json')
        self.assertEqual(complete['arms'], a.NAMES)
        self.assertEqual(complete['returncode'], 3)
        audits = a.jsonl(path.parent / 'audit.jsonl')
        self.assertEqual(len(audits), 3)
        self.assertTrue(all(row['DS4X'] is None and row['DS4_UNEXPECTED'] is None
                            and row['DS4_OVERTHINK_PENALTY'] == '0' for row in audits))
        candidate = a.load(Path(plan['output']) / 'budget1024-summary.json')
        self.assertEqual((candidate['incorrect'], candidate['invalid'], candidate['capped_correct']), (1, 1, 1))
        self.reject(lambda: self.invoke_synthetic(path, plan))
        self.assertEqual(len(a.jsonl(path.parent / 'audit.jsonl')), 3)

    def test_first_producer_infrastructure_failure_no_next_arm(self):
        path, plan = synthetic_plan('producer-failure')
        plan['arms'][0]['argv'][-2] = '1'; dump(path, plan)
        self.reject(lambda: self.invoke_synthetic(path, plan))
        output = Path(plan['output'])
        self.assertEqual(a.load(output / 'native-before-process.json')['returncode'], 1)
        self.assertEqual(a.load(output / 'first-failure.json')['completed_arms'], [])
        self.assertFalse(Path(plan['arms'][1]['results_dir']).exists())
        self.assertFalse((output / 'complete.json').exists())

    def test_parent_refusal_before_any_model_identity_or_producer(self):
        path, plan = synthetic_plan('guard-refusal')
        with patch.object(d, 'model_identity', side_effect=AssertionError('must never touch metadata')):
            self.reject(lambda: self.invoke_synthetic(path, plan, guard_failure=True))
        self.assertFalse(Path(plan['output']).exists())
        self.assertFalse(Path(plan['arms'][0]['results_dir']).exists())

    def test_analysis_release_gate_precedes_outcomes_and_aggregate_rc(self):
        path, plan = synthetic_plan('offline')
        self.invoke_synthetic(path, plan)
        pressure = Path(plan['pressure']); pressure.mkdir()
        guard = dict(status='complete', child_returncode=3); dump(pressure / 'result.json', guard)
        dump(pressure / 'command.json', dict(argv=d.rendered_command(plan, a.sha(path))[4:],
                                             initial=dict(time=.1, level=1, swap_gb=0)))
        lines(pressure / 'pressure.jsonl', [dict(time=.2, level=1, swap_gb=0, reason=None, consecutive_bad=0),
                                          dict(time=.3, level=1, swap_gb=0, reason=None, consecutive_bad=0)])
        release_path = path.parent / 'root-release.json'
        release = dict(guard_result=guard, lock_reacquired=True, live_model_processes=[], time=1,
                       normal_pressure_samples=2, bad_pressure_samples=0, swap_growth_gb=0)
        dump(release_path, release)
        with patch.object(d, 'validate_plan'), patch.object(d, 'check_assets'):
            summary = a.analyze(path, a.sha(path), release_path, a.sha(release_path), path.parent / 'checked.json')
            self.assertEqual(summary['guard_child_returncode'], 3)
            self.assertEqual(summary['arms'][1]['invalid'], 1)
            # Forge both top-level statuses to0 while preserving actual armrc3.
            guard['child_returncode'] = 0; dump(pressure / 'result.json', guard)
            release['guard_result'] = guard; dump(release_path, release)
            complete = a.load(Path(plan['output']) / 'complete.json'); complete['returncode'] = 0
            dump(Path(plan['output']) / 'complete.json', complete)
            self.reject(lambda: a.analyze(path, a.sha(path), release_path, a.sha(release_path), path.parent / 'forged.json'))
            self.assertFalse((path.parent / 'forged.json').exists())
            release['lock_reacquired'] = False; dump(release_path, release)
            with patch.object(a, 'validate_arm', side_effect=AssertionError('outcome read before release')):
                self.reject(lambda: a.analyze(path, a.sha(path), release_path, a.sha(release_path), path.parent / 'unreleased.json'))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    proof = dict(cpu_only=True, model_opened=False, model_metadata_touched=False, real_helper_executed=False,
                 tests_run=result.testsRun, negative_trials=NEGATIVE_TRIALS, failures=len(result.failures),
                 errors=len(result.errors), pass_=result.wasSuccessful(), fixtures=str(ROOT),
                 fake_producer='Actual Python subprocesses copy independently scripted token/piece/RNG timelines; production analyzer and driver arm loop execute unchanged.')
    dump(ROOT / 'result.json', proof)
    print(json.dumps(proof))
    sys.exit(0 if result.wasSuccessful() else 1)
