#!/usr/bin/env python3
"""Offline task and typed trace evidence; never constructs a model replay."""
import argparse
import csv
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PREPARED = Path('/Volumes/Models/ds4/.workspaces/idea-agent-resident3-live/speed-bench/ideas/artifacts/resident-agent-three-live-prepared-20261002-v3')
PLAN_SHA = 'b6a4b9649054454f985c083c72a692203dc442accca487e8c66f26a331f48e49'
REQUIRED = ('README.md', 'data.csv', 'ledger/__init__.py', 'ledger/io.py',
            'ledger/money.py', 'ledger/report.py', 'ledger/cli.py',
            'tests/test_report.py', 'result.json', 'recent.json')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    require(Path(path).suffix.lower() != '.gguf', 'Model access forbidden')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def integer(value):
    return type(value) is int and value >= 0

def validate_outcome(outcome, summary):
    require(type(summary.get('process_returncode')) is int and summary['process_returncode'] == 1,
            'Original private frontend rc1 not recorded')
    require(outcome['status'] == 'FAILED' and outcome['phase'] == 'trace_validation' and outcome['error_type'] == 'ValueError'
            and outcome['error'] == 'Compaction requires a separate replay format', 'Original failure differs')
    require(type(outcome['agent_returncode']) is int and outcome['agent_returncode'] == 0 and
            all(type(outcome[k]) is int and outcome[k] == 3 for k in ('turns_submitted', 'turns_observed', 'checks_recorded')),
            'Agent exit/owned wait or task turn counts failed')

def validate_recorded_checks(checks):
    require(type(checks) is list and len(checks) == 3, 'Missing ordered checks')
    for index, check in enumerate(checks):
        require(type(check) is dict and type(check['stage']) is int and check['stage'] == index, 'Stage order/type')
        require(type(check['held_out_rc']) is int and check['held_out_rc'] == 0 and
                type(check['tests_rc']) is int and check['tests_rc'] == 0, 'Recorded check failure')
        require(check['held_out'] == f'PASS held-out stage {index}\n', 'Recorded checker output differs')
        matches = re.findall(r'Ran (\d+) tests? in [\d.]+s\n\nOK\n', check['tests'])
        require(len(matches) == 1 and int(matches[0]) >= (5 if index == 0 else 7), 'Recorded unit suite incomplete')

def parse_trace(data):
    """Preserve all traced P/D/C tokens, with C an opaque live-state boundary."""
    require(data.endswith(b'\n'), 'Truncated trace')
    events, history = [], []
    block = current = compact = startup = boundary = None
    epoch = 0
    users = tool_calls = 0
    user_messages = []
    prior_stamp = None
    last_generation = None
    expected_round = 0
    expected_carried = 0
    worker_seen = False

    def finish_block():
        nonlocal block, history, compact
        if block['label'] == 'initial_system_prompt':
            require(startup is not None and not history and not events and epoch == 0,
                    'Initial prompt missing startup sync')
            require(block['start'] == 0 and block['end'] == startup['prompt'], 'Initial prompt geometry')
            history = [t['id'] for t in block['tokens']]
            events.append(dict(kind='P', epoch=epoch, label=block['label'], cached=0,
                               count=len(history), tokens=block['tokens'], sync=startup))
        elif block['label'] == 'prefill_suffix':
            require(current is not None and not current.get('block'), 'Unowned/duplicate suffix')
            require((block['start'], block['end']) == (current['cached'], current['prompt']), 'Suffix geometry')
            current['block'] = block
        elif block['label'] == 'compacted_transcript':
            require(compact is not None and compact.get('summary') and not compact.get('block'),
                    'Compacted block missing summary or duplicated')
            require(block['start'] == 0, 'Compacted block must be complete')
            compact['block'] = block
        block = None

    for lineno, line in enumerate(data.decode('utf-8').splitlines(), 1):
        require(len(line) > 24 and line[23] == ' ', 'Malformed timestamp')
        stamp = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S.%f')
        require(prior_stamp is None or stamp >= prior_stamp, 'Trace time moved backwards')
        prior_stamp = stamp
        text = line[24:]
        m = re.fullmatch(r'token index=(\d+) id=(\d+) bytes=(\d+) text=(".*") hex=([0-9a-f]*)', text)
        if m:
            index, token, n = map(int, m.group(1, 2, 3))
            require(token < 129280 and len(m[5]) == n * 2, 'Token ID/byte geometry')
            require(type(json.loads(m[4])) is str, 'Token text must be JSON string')
            record = dict(index=index, id=token, bytes=n, piece_hex=m[5])
            if block is not None:
                require(index == block['start'] + len(block['tokens']), 'Token block index/order')
                block['tokens'].append(record)
                if len(block['tokens']) == block['end'] - block['start']:
                    finish_block()
            else:
                require(current is not None and current.get('sync') and not compact, 'Token outside active decode')
                require(index == len(current['decoded']) + 1, 'Decode token index/order')
                current['decoded'].append(record)
            continue
        require(block is None, 'Interrupted/truncated token block')
        if text.startswith('agent worker start '):
            require(not worker_seen and lineno == 1, 'Worker start order')
            worker_seen = True
        elif (m := re.fullmatch(r'tokens label=(\w+) start=(\d+) len=(\d+)', text)):
            label, start, end = m[1], int(m[2]), int(m[3])
            require(label in ('initial_system_prompt', 'prefill_suffix', 'compacted_transcript'), 'Unknown token block')
            require(end >= start, 'Token block range')
            block = dict(label=label, start=start, end=end, tokens=[], line=lineno)
            if end == start:
                finish_block()
        elif (m := re.fullmatch(r'prefill sync done prompt=(\d+) cached=(\d+) suffix=(\d+) rc=(\d+) ([\d.]+) ms', text)):
            prompt, cached, suffix, rc = map(int, m.group(1, 2, 3, 4))
            require(rc == 0 and cached == 0 and suffix == prompt and current is None, 'Standalone sync geometry/status')
            sync = dict(prompt=prompt, cached=cached, suffix=suffix, rc=rc, recorded_ms=float(m[5]), stamp=line[:23])
            if compact is not None:
                require(compact.get('block') and not compact.get('sync') and compact['block']['end'] == prompt,
                        'Compaction missing full block or duplicate/wrong rebuild')
                compact['sync'] = sync
            else:
                require(not events and startup is None and worker_seen, 'Unexpected standalone sync')
                startup = sync
        elif (m := re.fullmatch(r'prefill tool_round=(\d+) transcript=(\d+) prompt=(\d+) cached=(\d+) suffix=(\d+) think=(\w+)', text)):
            require(current is None and compact is None and boundary is None and bool(events), 'Prefill order')
            tool_round, transcript, prompt, cached, suffix = map(int, m.group(1, 2, 3, 4, 5))
            require(tool_round == expected_round and prompt == transcript and prompt - cached == suffix and cached <= len(history),
                    'Prefill/history geometry or tool-round order')
            current = dict(tool_round=tool_round, prompt=prompt, cached=cached, suffix=suffix,
                           think=m[6], started=line[:23], decoded=[], epoch=epoch, calls=0)
        elif (m := re.fullmatch(r'prefill sync done tool_round=(\d+) prompt=(\d+) cached=(\d+) suffix=(\d+) rc=(\d+) ([\d.]+) ms', text)):
            require(current is not None and current.get('block') and not current.get('sync'), 'Round sync without complete block')
            values = tuple(map(int, m.group(1, 2, 3, 4)))
            require(values == tuple(current[k] for k in ('tool_round', 'prompt', 'cached', 'suffix')) and int(m[5]) == 0,
                    'Round sync metadata/status')
            current['sync'] = dict(rc=0, recorded_ms=float(m[6]), stamp=line[:23])
            history = history[:current['cached']] + [t['id'] for t in current['block']['tokens']]
            events.append(dict(kind='P', epoch=epoch, tool_round=current['tool_round'], cached=current['cached'],
                               count=current['suffix'], tokens=current['block']['tokens'], sync=current['sync']))
        elif text == 'dsml tool start detected':
            require(current is not None and current.get('sync'), 'Tool start outside decode')
        elif (m := re.fullmatch(r'dsml done calls=(\d+)', text)):
            require(current is not None and current.get('sync') and int(m[1]) > 0, 'Tool calls outside decode')
            current['calls'] += int(m[1]); tool_calls += int(m[1])
        elif (m := re.fullmatch(r'generation finished tool_round=(\d+) generated=(\d+) carried=(\d+) context_limited=([01])', text)):
            require(current is not None and current.get('sync') and int(m[1]) == current['tool_round'], 'Generation finish order')
            require(len(current['decoded']) == int(m[2]), 'Generation count differs')
            require(int(m[3]) == expected_carried, 'Carried generation count differs')
            last_generation = dict(kind='D', epoch=epoch, tool_round=current['tool_round'], tokens=current['decoded'],
                                   count=int(m[2]), carried=int(m[3]), context_limited=bool(int(m[4])),
                                   calls=current['calls'], started=current['started'], prefilled=current['sync']['stamp'], finished=line[:23])
            events.append(last_generation); history += [t['id'] for t in current['decoded']]
            expected_round = current['tool_round'] + 1
            expected_carried = 0
            current = None
        elif (m := re.fullmatch(r'generation context boundary: generated=(\d+) partial_tool=([01])', text)):
            require(current is None and compact is None and boundary is None and last_generation is not None and
                    last_generation['context_limited'] and int(m[1]) == last_generation['count'], 'Context boundary order')
            boundary = dict(generated=int(m[1]), partial_tool=bool(int(m[2])), stamp=line[:23])
            expected_round = last_generation['tool_round']
            expected_carried = last_generation['carried'] + last_generation['count']
        elif (m := re.fullmatch(r'compaction summary prefix=(\d+) total=(\d+) retained_unsummarized=(\d+)', text)):
            require(current is None and compact is None and last_generation is not None, 'Compaction start order')
            prefix, total, retained = map(int, m.group(1, 2, 3))
            require(prefix + retained == total, 'Compaction prefix/retained geometry')
            compact = dict(prefix=prefix, old=total, retained_unsummarized=retained, boundary=boundary, started=line[:23])
        elif text.startswith('compaction-summary='):
            require(compact is not None and not compact.get('summary'), 'Unowned/duplicate summary')
            summary = json.loads(text[len('compaction-summary='):])
            require(type(summary) is str and bool(summary), 'Empty/non-string summary')
            compact['summary'] = summary
            compact['summary_sha256'] = hashlib.sha256(summary.encode()).hexdigest()
        elif (m := re.fullmatch(r'compaction summary stopped before control token id=(\d+)', text)):
            require(compact is not None and not compact.get('summary') and int(m[1]) < 129280, 'Summary stop order')
            compact['untraced_summary_stop_id'] = int(m[1])
        elif (m := re.fullmatch(r'compacted reason=(".*") old=(\d+) new=(\d+) tail_start=(\d+) tail=(\d+)', text)):
            require(compact is not None and compact.get('sync') and compact.get('summary'), 'Compaction missing rebuild/summary')
            reason = json.loads(m[1]); old, new, tail_start, tail = map(int, m.group(2, 3, 4, 5))
            require(type(reason) is str and old == compact['old'] and new == compact['block']['end'] and
                    tail_start + tail == old and tail_start <= compact['prefix'], 'Compaction commit geometry')
            epoch += 1
            events.append(dict(kind='C', epoch_before=epoch-1, epoch_after=epoch, old=old, new=new,
                               tail_start=tail_start, tail=tail, reason=reason, summary=compact['summary'],
                               summary_sha256=compact['summary_sha256'], tokens=compact['block']['tokens'],
                               rebuild_sync=compact['sync'], context_boundary=compact['boundary'],
                               internal_summary_token_ids_traced=False, cold_reset_replay_proven=False))
            history = [t['id'] for t in compact['block']['tokens']]
            compact = boundary = None
        elif text.startswith('user='):
            require(current is None and compact is None and boundary is None, 'User arrived during active operation')
            message = json.loads(text[5:])
            require(type(message) is str, 'Malformed user text')
            user_messages.append(message)
            users += 1; expected_round = expected_carried = 0
        elif text.startswith('datetime-context='):
            require(current is None and compact is None and type(json.loads(text[17:])) is str, 'Datetime context order/type')
        elif text.startswith('sysprompt kv stored '):
            require(startup is not None and not events, 'System cache store order')
        else:
            raise ValueError(f'Unknown trace event at line {lineno}: {text[:100]}')
    require(worker_seen and block is None and current is None and compact is None and boundary is None,
            'Incomplete trace operation')
    require(users == 3 and tool_calls >= 6 and last_generation is not None, 'Missing task turns/tool use')
    return dict(format='DS4_TYPED_LIVE_TRACE_V1', events=events, epochs=epoch+1, users=users, user_messages=user_messages,
                tool_calls=tool_calls, final_context_tokens=len(history),
                generation_rounds=sum(e['kind'] == 'D' for e in events),
                compactions=sum(e['kind'] == 'C' for e in events),
                all_traced_token_blocks_complete=True, replay_created=False,
                cold_reset_replay_proven=False, latency_parity_proven=False, io_parity_proven=False,
                limitation='Internal summary generation is not token-traced; its live KV/cache effects are not reproduced by a cold rebuild.')

def gold_json(data, since=None):
    result, seen = {}, set()
    for row in csv.DictReader(data.decode().splitlines()):
        if row['event_id'] in seen:
            continue
        seen.add(row['event_id'])
        if (since and row['date'] < since) or row['kind'] == 'void':
            continue
        require(row['kind'] in ('charge', 'refund'), 'Unknown fixture event kind')
        cents = Decimal(row['amount']) * 100
        require(cents == cents.to_integral_exact(), 'Fractional fixture cents')
        amount = int(cents) * (-1 if row['kind'] == 'refund' else 1)
        result[row['account']] = result.get(row['account'], 0) + amount
    return result

def analyze(arm, prepared=PREPARED, run_final_checks=True):
    require(sha(prepared/'plan.json') == PLAN_SHA, 'Frozen V3 plan differs')
    plan = json.loads((prepared/'plan.json').read_text())
    require(arm.parent == Path(plan['output']) and arm.name in ('native-before', 'broader3', 'native-after'), 'Foreign arm')
    files = ['harness-outcome.json', 'checks.json', 'trace.log', 'stdout.log', 'prompts.json', 'command.json', 'provenance.json']
    original = {name: sha(arm/name) for name in files}
    original_arm_summary = arm.parent/(arm.name+'-summary.json')
    original['../'+original_arm_summary.name] = sha(original_arm_summary)
    require(not (arm/'summary.json').exists() and not (arm/'replay.txt').exists(), 'Original arm unexpectedly has a completed flat summary/replay')
    outcome = json.loads((arm/'harness-outcome.json').read_text())
    validate_outcome(outcome, json.loads(original_arm_summary.read_text()))
    # The immutable frontend sets trace_validation only after pipe EOF and its
    # own successful wait(). This is observed provenance, not inferred from PID.
    frozen_inputs = {}
    for name in ('live.py', 'check.py', 'prompts.json', 'trace_session.py'):
        require(sha(prepared/name) == plan['asset_sha256'][str(prepared/name)], 'Frozen frontend/checker identity differs')
        frozen_inputs[prepared/name] = sha(prepared/name)
    require(sha(arm/'prompts.json') == sha(prepared/'prompts.json'), 'Prompts differ')
    checks = json.loads((arm/'checks.json').read_text())
    validate_recorded_checks(checks)
    project = arm/'project-final'; fixture = prepared/'fixture'
    before_project = {str(p.relative_to(project)): sha(p) for p in sorted(project.rglob('*'))
                      if p.is_file() and '__pycache__' not in p.parts}
    for relative in REQUIRED:
        require((project/relative).is_file(), 'Missing final file: '+relative)
    for source in sorted(fixture.rglob('*')):
        if source.is_file():
            require(sha(source) == plan['asset_sha256'][str(source)], 'Frozen initial fixture differs')
            frozen_inputs[source] = sha(source)
    for relative in ('data.csv', 'tests/test_report.py'):
        require((project/relative).read_bytes() == (fixture/relative).read_bytes(), 'Immutable/gold fixture mutated')
    fixture_data = (fixture/'data.csv').read_bytes()
    json_checks = {}
    for name, since in [('result.json', None), ('recent.json', '2026-08-15')]:
        actual = json.loads((project/name).read_text()); expected = gold_json(fixture_data, since)
        require(type(actual) is dict and all(type(v) is int for v in actual.values()) and actual == expected,
                'Independent integer-cent JSON gold failed')
        json_checks[name] = dict(expected=expected, actual=actual, exact=True)
    cpu = []
    if run_final_checks:
        env = {k: v for k, v in os.environ.items() if not k.startswith('DS4')}
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        commands = [[sys.executable, '-B', str(prepared/'check.py'), str(project), str(stage)] for stage in range(3)]
        commands += [[sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
        for command in commands:
            result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=20)
            cpu.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            require(result.returncode == 0, 'Independent final-project CPU check failed')
    ledger = parse_trace((arm/'trace.log').read_bytes())
    require(ledger['compactions'] > 0, 'Expected recorded compaction absent')
    require(ledger['user_messages'] == [p+'\n' for p in json.loads((prepared/'prompts.json').read_text())],
            'Trace task prompts differ from frozen three turns')
    file_hashes = {str(p.relative_to(project)): sha(p) for p in sorted(project.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    require(file_hashes == before_project, 'Generated project changed during CPU analysis')
    require(original == {name: sha(arm/name) for name in original}, 'Original arm changed during analysis')
    require(all(sha(p) == h for p, h in frozen_inputs.items()), 'Frozen checker/fixture changed during analysis')
    return dict(status='SEPARATE_OFFLINE_TASK_COMPLETION_PASS', original_harness_status='FAILED',
                original_private_frontend_returncode=1, original_harness_outcome=outcome,
                actual_agent_returncode=0, pipe_eof_and_owned_successful_wait_observed=True,
                original_harness_success_rewritten=False, source_arm=str(arm), source_sha256=original,
                prepared_plan_sha256=PLAN_SHA, ordered_three_recorded_checks_pass=True,
                required_final_files=list(REQUIRED), generated_project_sha256=file_hashes,
                independent_json_gold=json_checks, independent_final_project_cpu_checks=cpu,
                typed_trace=ledger, model_opened=False, model_metadata_read=False,
                limitation=ledger['limitation'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.arm.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False); target.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'original_harness_status', 'actual_agent_returncode', 'model_opened')}))

if __name__ == '__main__':
    main()
