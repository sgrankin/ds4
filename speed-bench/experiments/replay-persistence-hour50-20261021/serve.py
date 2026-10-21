#!/usr/bin/env python3
"""Two-process replay-cache HTTP qualification. Run under pressure_run.py."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('item47',
    ROOT/'speed-bench/experiments/decoder-replay-hour47-20261020/serve.py')
item47 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(item47)
save, stop, scenario = item47.save, item47.stop, item47.scenario


def arm(args, label, fixture):
    out = args.output/label
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    env.update(DS4_V41_DECODER_REPLAY_ROWS='128', DS4_V41_DECODER_REPLAY_SELECTED='1',
        DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS='1', DS4_SERVER_CONTEXT_SUFFIX_REUSE='0',
        DS4_STATE_CACHE_DIR=str(args.output/'cache'), DS4_STATE_CACHE_HOT_MB='0',
        DS4_STATE_CACHE_SSD_MB='4096', DS4_SERVER_RESPONSE_STORE_DIR=str(out/'responses'),
        DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    # The default unbatched session syncs the whole prompt, enabling wide replay.
    command = [str(args.server), '--model', str(args.model), '--host', '127.0.0.1',
        '--port', str(port), '--ssd-streaming', '--ssd-streaming-cache-experts', '64GB',
        '--ctx', '16384', '--prefill-chunk', '8192', '--mtp-draft', '1']
    execution = {'command': command, 'cwd': str(ROOT),
        'env': {k: v for k, v in env.items() if k.startswith('DS4')},
        'server_sha256': hashlib.sha256(args.server.read_bytes()).hexdigest(),
        'batched_sessions': 0, 'application_ssd_kv_cache': 'shared, disk-only',
        'timing': 'workflow includes three HTTP requests and actual note read; startup separate'}
    save(out/'execution.json', execution)
    result = {'arm': label, 'pass': False, 'error': None}
    proc = None
    started = time.monotonic()
    try:
        with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            result['pid'] = execution['pid'] = proc.pid
            save(out/'execution.json', execution)
            base = f'http://127.0.0.1:{port}/v1'
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server startup exit {proc.returncode}')
                if time.monotonic()-started > args.timeout:
                    raise TimeoutError('startup deadline')
                try:
                    with urllib.request.urlopen(base+'/models', timeout=1) as response:
                        models = json.load(response)['data']
                    if models:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(.2)
            result['startup_seconds'] = time.monotonic()-started
            common = {'model': models[0]['id'], 'temperature': 0,
                'reasoning': {'effort': 'none'}, 'max_output_tokens': 128, 'store': False,
                'tools': [{'type': 'tool_search', 'execution': 'client',
                    'description': 'Discover a tool by capability or name. Return matching tool definitions.',
                    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                                   'required': ['query']}}]}
            result['workflow'] = scenario(args, base, common, fixture, out/'workflow')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['server_returncode'] = stop(proc)
    result['arm_seconds'] = time.monotonic()-started
    lines = (out/'stderr.log').read_text(errors='replace').splitlines()
    result['runtime_replay_lines'] = [s for s in lines if 'decoder replay' in s]
    result['state_cache_lines'] = [s for s in lines if 'state cache' in s]
    result['replay_store_lines'] = [s for s in lines if 'state cache replay store' in s]
    result['replay_hit_lines'] = [s for s in lines if 'state cache replay hit' in s]
    first_prefill_end = next((i for i, s in enumerate(lines)
        if 'ds4-server: chat ctx=' in s and ' prompt done ' in s), len(lines))
    first_hits = [s for s in lines[:first_prefill_end]
                  if 'state cache replay hit' in s]
    result['first_request_replay_hit_tokens'] = max(
        (int(re.search(r'tokens=(\d+)', s).group(1)) for s in first_hits), default=0)
    result['replay_executed'] = (
        any('decoder replay rows=128 ' in s for s in result['runtime_replay_lines']) and
        any('selected_layers=20 ' in s for s in result['runtime_replay_lines']))
    workflow = result.get('workflow', {})
    requests = workflow.get('requests', [])
    result['usage'] = [r.get('usage') for r in requests]
    first_usage = (requests[0].get('usage') or {}) if requests else {}
    result['eligible_prompt'] = first_usage.get('input_tokens', 0) >= 8192
    result['first_request_cached_tokens'] = first_usage.get('input_tokens_details', {}).get('cached_tokens', 0)
    result['usage_totals_match'] = all(
        u['input_tokens_details']['cached_tokens'] + u['input_tokens_details']['cache_write_tokens'] == u['input_tokens']
        for u in result['usage'])
    result['usage_matches_restore'] = (label == '01cold' or
        result['first_request_cached_tokens'] == result['first_request_replay_hit_tokens'])
    result['successful_replay_store'] = any('state cache replay store ok ' in s for s in lines)
    persistence_pass = (result['replay_executed'] and result['successful_replay_store']
                        if label == '01cold' else
                        result['first_request_replay_hit_tokens'] > 0 and result['usage_matches_restore'])
    result['pass'] = (not result['error'] and workflow.get('pass', False) and
        not workflow.get('error') and result['eligible_prompt'] and persistence_pass and result['usage_totals_match'])
    save(out/'result.json', result)
    print(json.dumps({'arm': label, 'pid': result.get('pid'), 'pass': result['pass'],
                      'error': result['error']}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--server', type=Path, default=ROOT/'ds4-server')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=600)
    args = p.parse_args()
    args.model, args.server, args.output = args.model.resolve(), args.server.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.output/'fixtures'
    work.mkdir()
    path = work/'note.json'
    path.write_text(json.dumps({'archived_code': 'MAPLE-4826', 'receipt': 'READ50'})+'\n')
    logs = ''.join(f'historical audit {i:04d}: module={i%8} count={(i*37)%997:03d} '
                   'state=reviewed immutable=true; no current decisions.\n' for i in range(360))
    prompt = ('Historical context only:\n<log>\n'+logs+'</log>\nCurrent key: SABLE-1937.\n'
        f'Task: use tool_search to discover read_note, then use that tool to read {path}. '
        'After actually reading it, reply with exactly one JSON object with archived_code and receipt '
        'from the file, and current_key from this request. Do not answer before the read. No prose or Markdown.')
    fixture = {'case': 'http-replay-persistence', 'history_mode': 'full',
        'system': 'Follow the requested tool workflow. Use only defined tools, and do not invent tool results.',
        'prompt': prompt, 'note_path': str(path), 'namespace': None,
        'discovered': [{'type': 'function', 'name': 'read_note',
            'description': 'Read the exact local note file at path; returns its JSON contents.',
            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                           'required': ['path'], 'additionalProperties': False}}],
        'expected': {'archived_code': 'MAPLE-4826', 'receipt': 'READ50', 'current_key': 'SABLE-1937'}}
    save(work/'fixture.json', fixture)

    def interrupt(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    result = {'pass': False, 'arms': []}
    for label in ('01cold', '02restart'):
        result['arms'].append(arm(args, label, fixture))
        save(args.output/'result.json', result)
        if not result['arms'][-1]['pass']:
            break
    if len(result['arms']) == 2 and all(r['pass'] for r in result['arms']):
        first = json.loads((args.output/'01cold/workflow/01.request.json').read_text())
        second = json.loads((args.output/'02restart/workflow/01.request.json').read_text())
        result['identical_first_request'] = first == second
        result['distinct_processes'] = result['arms'][0]['pid'] != result['arms'][1]['pid']
        result['pass'] = all(r['pass'] for r in result['arms']) and result['identical_first_request'] and result['distinct_processes']
    save(args.output/'result.json', result)
    return 0 if result['pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
