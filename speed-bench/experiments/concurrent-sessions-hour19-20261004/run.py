#!/usr/bin/env python3
"""Two concurrent clients, four identities; ROOT owns the server and restart."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('hour17_suffix_workflow',
    ROOT / 'speed-bench/experiments/server-confirmation-hour17-20261004/suffix_workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
FACTS = json.loads((HERE / 'fixture.json').read_text())['sessions']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--phase', choices=['before', 'after'], required=True)
    p.add_argument('--mode', choices=['native', 'reuse'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shared', type=Path, required=True)
    p.add_argument('--server-log', type=Path)
    a = p.parse_args()
    a.output, a.shared = a.output.resolve(), a.shared.resolve()
    state_file = a.output / 'histories.json'
    if a.phase == 'before' and state_file.exists():
        p.error('before requires a fresh history namespace')
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / a.phase
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    rows, groups, edits = [], [], []
    # Concurrent log slices belong to a group, not to either individual client.
    offset = a.server_log.stat().st_size if a.server_log and a.server_log.exists() else 0
    client = SimpleNamespace(base_url=a.base_url, server_log=None)
    if a.phase == 'before':
        state = {'mode': a.mode, 'shared': str(a.shared), 'histories': {}, 'results': [], 'groups': []}
        for session, fact in FACTS.items():
            directory = a.shared / session
            directory.mkdir(parents=True, exist_ok=True)
            active, archive = directory / 'active.json', directory / 'archive.json'
            data = {key: fact[key] for key in ['batches', 'items_per_batch', 'reserved_items']}
            data['notes'] = [f'Dispatch note {i:02d}: current batch counts are authoritative; obsolete shipment logs may be archived.' for i in range(24)]
            workflow.dump(active, data)
            workflow.dump(archive, {'archived_code': None})
            block = (workflow.BEGIN + '\nObsolete dispatch label: ' + fact['old_code'] + '.\n'
                     + 'old shipment note checked.\n' * 64 + workflow.END)
            initial = ('These two marked dispatch spans are obsolete notes; their labels will move to the archive file.\n'
                       + block + '\n' + block + '\n'
                       + f'Read {active} once using read with whole=true and raw=true. '
                       'Retain the inventory numbers and complete unchanged tool result. Then reply READY only.')
            state['histories'][session] = [{'role': 'user', 'content': initial}]
    else:
        state = json.loads(state_file.read_text())
        if state['mode'] != a.mode or state['shared'] != str(a.shared):
            p.error('after must continue the same mode and advertised paths')

    order = [('warm-AB', ['A', 'B'], False), ('warm-CD', ['C', 'D'], False),
             ('edited-AB', ['A', 'B'], True)] if a.phase == 'before' else [('edited-CD', ['C', 'D'], True)]
    workflow.dump(out / 'execution.json', {'phase': a.phase, 'mode': a.mode,
        'base_url': a.base_url, 'shared': str(a.shared), 'max_workers': 2,
        'groups': [group for group, _, _ in order], 'tools': workflow.TOOLS,
        'instructions': workflow.INSTRUCTIONS, 'server_started_by_driver': False,
        'request_style': 'full input, store:false, temperature:0, effort:none, stable tools/instructions',
        'keys': {session: 'ds4-hour17-confirmation-' + session for session in FACTS}})

    def task(group, session, edited, submitted):
        destination = out / session
        destination.mkdir(exist_ok=True)
        fact, history = FACTS[session], state['histories'][session]
        name = session + ('-edited' if edited else '-warm')
        edit = None
        if edited:
            edit_started = time.monotonic()
            archive = a.shared / session / 'archive.json'
            code = fact['old_code'] if session in ['A', 'B'] else fact['new_code']
            workflow.dump(archive, {'archived_code': code})
            edit = workflow.remove_first_block(history, destination / name)
            edit.update(session=session, seconds=time.monotonic() - edit_started)
            history.append({'role': 'user', 'content':
                f'Read {archive} again now with whole=true and raw=true. '
                'A fresh read is required even if an earlier answer contained a code. '
                'Use the unchanged active inventory tool result to calculate batches * items_per_batch - reserved_items. '
                'Reply only JSON with active_total (integer) and archived_code (string) from the current file.'})
            gold = {'active_total': fact['batches'] * fact['items_per_batch'] - fact['reserved_items'],
                    'archived_code': code}
            wanted_read = archive
        else:
            gold, wanted_read = 'READY', a.shared / session / 'active.json'
        row, _ = workflow.turn(client, destination, name, session, history, wanted_read, gold, 0)
        row.update(group=group, queue_inclusive_seconds=time.monotonic() - submitted)
        if edited:
            row.update(seconds_including_edit=row['queue_inclusive_seconds'], edit_seconds=edit['seconds'])
        workflow.dump(destination / (name + '-result.json'), row)
        return row, edit

    with ThreadPoolExecutor(max_workers=2) as pool:
        for group, sessions, edited in order:
            group_started = time.monotonic()
            futures = [pool.submit(task, group, session, edited, time.monotonic()) for session in sessions]
            group_rows = []
            for future in futures:
                row, edit = future.result()
                rows.append(row)
                group_rows.append(row)
                state['results'].append(row)
                if edit is not None:
                    edits.append(edit)
            makespan = time.monotonic() - group_started
            log, offset = workflow.read_log(a.server_log, offset)
            (out / (group + '-server.log')).write_text(log)
            record = {'group': group, 'sessions': sessions, 'seconds': makespan,
                'steps': [row['step'] for row in group_rows], 'pass': all(row['pass'] for row in group_rows),
                'server_log_lines': log.splitlines(),
                'suffix_log_lines': [line for line in log.splitlines() if 'suffix' in line.lower()],
                'exact_hit_log_lines': [line for line in log.splitlines() if 'state cache exact hit' in line],
                'slot_eviction_log_lines': [line for line in log.splitlines() if 'evicting checkpoint' in line]}
            groups.append(record)
            state['groups'].append(record)
            workflow.dump(state_file, state)
    result = {'phase': a.phase, 'mode': a.mode, 'steps': rows, 'groups': groups, 'edits': edits,
        'seconds': time.monotonic() - started, 'pass': all(row['pass'] for row in rows),
        'fresh_reads': sum(len([call for call in row['calls'] if call['read']]) for row in rows),
        'restart_owned_by_launcher': True, 'individual_server_logs': 'Concurrent logs attributed to groups only.'}
    workflow.dump(out / 'result.json', result)
    fields = ['step', 'session', 'group', 'seconds', 'queue_inclusive_seconds', 'seconds_including_edit',
              'correct', 'fresh_read', 'complete', 'pass', 'text']
    with (out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'phase': a.phase, 'mode': a.mode, 'pass': result['pass'],
                      'seconds': result['seconds'], 'fresh_reads': result['fresh_reads']}), flush=True)
    return 0 if result['pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
