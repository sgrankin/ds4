#!/usr/bin/env python3
"""Unbatched HTTP tool-workflow bracket. Run under pressure_run.py."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('item46',
    ROOT/'speed-bench/experiments/incremental-tools-hour46-20261020/run.py')
item46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(item46)
save, stop, scenario = item46.save, item46.stop, item46.scenario


def arm(args, index, rows, fixture):
    out = args.output/f'{index:02d}-rows{rows}'
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    env.update(DS4_V41_DECODER_REPLAY_ROWS=str(rows), DS4_V41_DECODER_REPLAY_SELECTED='1',
        DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS='1', DS4_SERVER_CONTEXT_SUFFIX_REUSE='0',
        DS4_SERVER_RESPONSE_STORE_DIR=str(out/'responses'),
        DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    # Omit --batched-session: its default zero selects whole-prompt sync.
    command = [str(args.server), '--model', str(args.model), '--host', '127.0.0.1',
        '--port', str(port), '--ssd-streaming', '--ssd-streaming-cache-experts', '64GB',
        '--ctx', '16384', '--prefill-chunk', '8192', '--mtp-draft', '1']
    save(out/'execution.json', {'command': command, 'cwd': str(ROOT),
        'env': {k: v for k, v in env.items() if k.startswith('DS4')},
        'server_sha256': hashlib.sha256(args.server.read_bytes()).hexdigest(),
        'batched_sessions': 0, 'application_ssd_kv_cache': 'disabled',
        'timing': 'workflow includes three HTTP requests and actual note read; startup separate'})
    result = {'arm': index, 'rows': rows, 'pass': False, 'error': None}
    proc = None
    started = time.monotonic()
    try:
        with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
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
            result['pass'] = result['workflow']['pass'] and not result['workflow']['error']
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['server_returncode'] = stop(proc)
    result['arm_seconds'] = time.monotonic()-started
    result['runtime_replay_lines'] = [line for line in (out/'stderr.log').read_text(errors='replace').splitlines()
                                      if 'decoder replay' in line]
    requests = result.get('workflow', {}).get('requests', [])
    first_usage = (requests[0].get('usage') or {}) if requests else {}
    result['eligible_prompt'] = first_usage.get('input_tokens', 0) >= 8192
    result['replay_executed'] = (
        any('decoder replay rows=128 ' in line for line in result['runtime_replay_lines']) and
        any('selected_layers=20 ' in line for line in result['runtime_replay_lines']))
    result['pass'] = result['pass'] and result['eligible_prompt'] and (rows == 0 or result['replay_executed'])
    save(out/'result.json', result)
    print(json.dumps({'arm': index, 'rows': rows, 'pass': result['pass'], 'error': result['error']}), flush=True)
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
    path.write_text(json.dumps({'archived_code': 'MAPLE-4826', 'receipt': 'READ47'})+'\n')
    logs = ''.join(f'historical audit {i:04d}: module={i%8} count={(i*37)%997:03d} '
                   'state=reviewed immutable=true; no current decisions.\n' for i in range(360))
    prompt = ('Historical context only:\n<log>\n'+logs+'</log>\nCurrent key: SABLE-1937.\n'
        f'Task: use tool_search to discover read_note, then use that tool to read {path}. '
        'After actually reading it, reply with exactly one JSON object with archived_code and receipt '
        'from the file, and current_key from this request. Do not answer before the read. No prose or Markdown.')
    fixture = {'case': 'http-replay', 'history_mode': 'full',
        'system': 'Follow the requested tool workflow. Use only defined tools, and do not invent tool results.',
        'prompt': prompt, 'note_path': str(path), 'namespace': None,
        'discovered': [{'type': 'function', 'name': 'read_note',
            'description': 'Read the exact local note file at path; returns its JSON contents.',
            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                           'required': ['path'], 'additionalProperties': False}}],
        'expected': {'archived_code': 'MAPLE-4826', 'receipt': 'READ47', 'current_key': 'SABLE-1937'}}
    save(work/'fixture.json', fixture)

    def interrupt(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    results = []
    for index, rows in enumerate((0, 128, 0), 1):
        results.append(arm(args, index, rows, fixture))
        save(args.output/'result.json', results)
        if not results[-1]['pass']:
            break
    return 0 if len(results) == 3 and all(r['pass'] for r in results) else 3


if __name__ == '__main__':
    raise SystemExit(main())
