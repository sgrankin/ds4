#!/usr/bin/env python3
"""Two-process legacy KVC SSD cache HTTP qualification. Run under pressure_run.py."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import time
import urllib.request

ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location('item47',
    ROOT/'speed-bench/experiments/decoder-replay-hour47-20261020/serve.py')
item47 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(item47)
save, stop, scenario = item47.save, item47.stop, item47.scenario


def arm(args, label, fixture):
    server = args.restart_server if label == '02restart' else args.server
    out = args.output/label
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    env.update(DS4_V41_DECODER_REPLAY_ROWS='0', DS4_V41_DECODER_REPLAY_SELECTED='1',
        DS4_SERVER_INCREMENTAL_TOOL_SCHEMAS='1', DS4_SERVER_CONTEXT_SUFFIX_REUSE='0',
        DS4_STATE_CACHE_DIR='', DS4_SERVER_RESPONSE_STORE_DIR=str(out/'responses'),
        DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    # Keep the original unbatched C server, with bounded 512-row prefill chunks.
    command = [str(server), '--model', str(args.model), '--host', '127.0.0.1',
        '--port', str(port), '--ssd-streaming', '--ssd-streaming-cache-experts', args.expert_cache,
        '--ctx', '16384', '--prefill-chunk', '512', '--mtp-draft', '1',
        '--kv-disk-dir', str(args.output/'cache'), '--kv-disk-space-mb', '4096']
    execution = {'command': command, 'cwd': str(ROOT),
        'env': {k: v for k, v in env.items() if k.startswith('DS4')},
        'server_sha256': hashlib.sha256(server.read_bytes()).hexdigest(),
        'batched_sessions': 0, 'application_ssd_kv_cache': 'legacy KVC shared directory',
        'state_cache': 'disabled',
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
    result['legacy_store_lines'] = [s for s in lines if 'kv cache stored tokens=' in s]
    result['legacy_hit_lines'] = [s for s in lines if 'kv cache hit text' in s]
    result['cache_files'] = []
    for path in sorted((args.output/'cache').glob('*.kv')):
        with path.open('rb') as fp:
            header = fp.read(52)
        row = {'path': str(path), 'file_bytes': path.stat().st_size, 'valid_payload': False}
        if len(header) == 52 and header[:4] == b'KVC\x01' and header[20] == 2:
            row.update(tokens=struct.unpack_from('<I', header, 8)[0],
                hits=struct.unpack_from('<I', header, 12)[0],
                payload_bytes=struct.unpack_from('<Q', header, 40)[0],
                text_bytes=struct.unpack_from('<I', header, 48)[0])
            row['valid_payload'] = (row['tokens'] > 0 and row['payload_bytes'] > 0 and
                row['file_bytes'] >= 52 + row['text_bytes'] + row['payload_bytes'])
        result['cache_files'].append(row)
    first_prefill_end = next((i for i, s in enumerate(lines)
        if 'ds4-server: chat ctx=' in s and ' prompt done ' in s), len(lines))
    first_hits = [s for s in lines[:first_prefill_end] if 'kv cache hit text' in s]
    result['first_request_legacy_hit_tokens'] = max(
        (int(re.search(r'tokens=(\d+)', s).group(1)) for s in first_hits), default=0)
    result['replay_executed'] = bool(result['runtime_replay_lines'])
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
        result['first_request_cached_tokens'] == result['first_request_legacy_hit_tokens'])
    payloads_present = any(row['valid_payload'] for row in result['cache_files'])
    persistence_pass = (payloads_present and bool(result['legacy_store_lines']) if label == '01cold' else
        payloads_present and result['first_request_legacy_hit_tokens'] > 0 and result['usage_matches_restore'])
    result['pass'] = (not result['error'] and workflow.get('pass', False) and
        not workflow.get('error') and result['eligible_prompt'] and persistence_pass and result['usage_totals_match'] and not result['replay_executed'] and not result['state_cache_lines'])
    save(out/'result.json', result)
    print(json.dumps({'arm': label, 'pid': result.get('pid'), 'pass': result['pass'],
                      'error': result['error']}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--server', type=Path, default=ROOT/'.workspaces/zig-kvstore-20261022/c/bin/ds4-server')
    p.add_argument('--restart-server', type=Path, help='Second process executable (defaults to --server)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--expert-cache', default='64GB', help='SSD expert-cache size (default: 64GB)')
    args = p.parse_args()
    args.model, args.server, args.output = args.model.resolve(), args.server.resolve(), args.output.resolve()
    args.restart_server = (args.restart_server or args.server).resolve()
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
    fixture = {'case': 'http-legacy-kvc-persistence', 'history_mode': 'full',
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
