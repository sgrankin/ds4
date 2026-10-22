#!/usr/bin/env python3
"""One cumulative two-slot HTTP/cancellation/durable-restart check; ROOT runs it."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def request(base, output, name, method, path, payload=None, expected=200):
    save(output/(name+'.request.json'), dict(method=method, path=path, body=payload))
    message = urllib.request.Request(base+path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(message, timeout=180) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    (output/(name+'.response.json')).write_bytes(raw)
    body = json.loads(raw)
    record = dict(name=name, status=status, expected=expected,
                  seconds=time.monotonic()-started, pass_status=status == expected)
    save(output/(name+'.result.json'), record)
    if status != expected:
        raise RuntimeError(f'{name}: HTTP{status}, expected{expected}')
    return body, raw


def text(response):
    return ''.join(part.get('text', '') for item in response.get('output', [])
        if item.get('type') == 'message' for part in item.get('content', [])
        if part.get('type') == 'output_text').strip()


def stop(process):
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return process.returncode


INSTRUCTIONS = 'Follow the current request. Return only the requested JSON object, with no prose or Markdown.'
GOLD = {'A': {'code': 'IVY-46', 'count': 37}, 'B': {'code': 'CORAL-82', 'count': 231}}


def graded(body, gold, cap):
    try:
        actual = json.loads(text(body))
    except ValueError:
        actual = None
    usage = body.get('usage') or {}
    return {'expected': gold, 'actual': actual, 'response_status': body.get('status'),
        'usage': usage, 'incomplete_details': body.get('incomplete_details'),
        'pass': actual == gold and body.get('status') == 'completed' and usage.get('output_tokens', 0) < cap}


def streamed_disconnect(base, output, payload, first_delta, close_now):
    """Close an actual in-generation SSE socket, retaining its received bytes."""
    save(output / 'cancel.request.json', payload)
    host, port = base.removeprefix('http://').split(':')
    raw, started, first_at = bytearray(), time.monotonic(), None
    encoded = json.dumps(payload).encode()
    with socket.create_connection((host, int(port)), timeout=5) as connection:
        connection.sendall((f'POST /v1/responses HTTP/1.1\r\nHost: {host}:{port}\r\n'
            f'Content-Type: application/json\r\nContent-Length: {len(encoded)}\r\nConnection: close\r\n\r\n').encode() + encoded)
        connection.settimeout(1)
        while time.monotonic() - started < 180:
            try:
                chunk = connection.recv(8192)
            except socket.timeout:
                continue
            if not chunk:
                raise RuntimeError('cancel stream ended before a nonempty output-text delta')
            raw.extend(chunk)
            (output / 'cancel.received.http').write_bytes(raw)
            for line in raw.split(b'\n'):
                if not line.startswith(b'data:'):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if event.get('type') == 'response.output_text.delta' and event.get('delta'):
                    first_at = time.monotonic()
                    first_delta.set()
                    break
            if first_at is not None:
                break
        if first_at is None:
            raise TimeoutError('no generated SSE delta within180 seconds')
        if not close_now.wait(timeout=10):
            raise TimeoutError('surviving client was not submitted')
        connection.shutdown(socket.SHUT_RDWR)
    (output / 'cancel.received.http').write_bytes(raw)
    row = {'submitted': started, 'first_delta': first_at, 'disconnected': time.monotonic(),
        'received_bytes': len(raw), 'actual_text_delta': True,
        'closed_before_completed': b'"type":"response.completed"' not in raw and b'event: response.completed' not in raw}
    save(output / 'cancel.result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--restart-server', type=Path)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.server, args.model, args.output = args.server.resolve(), args.model.resolve(), args.output.resolve()
    args.restart_server = (args.restart_server or args.server).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    store, cache = args.output / 'responses', args.output / 'cache'
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
        (ROOT / 'ds4_metal.m').read_text()))
    env = {key: value for key, value in os.environ.items() if not key.startswith(('DS4', 'PT_'))}
    env.update({key: str(ROOT / value) for key, value in shaders.items()})
    env.update(DS4_SERVER_RESPONSE_STORE_DIR=str(store), DS4_STATE_CACHE_DIR=str(cache),
        DS4_STATE_CACHE_HOT_MB='0', DS4_STATE_CACHE_SSD_MB='512', DS4_SERVER_BATCH_LOG='1',
        DS4_SERVER_DECODE_COALESCE_US='2000', DS4_SERVER_CONTEXT_SUFFIX_REUSE='0',
        DS4_V41_DECODER_REPLAY_ROWS='0', DS4_V41_COMPRESSED_KV_F16='0',
        DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0')
    result = {'pass': False, 'phases': [], 'checks': {}, 'quality': [], 'error': None}
    parents, parent_bytes = {}, {}
    conversation, items_before, survivor = None, None, None
    for phase, binary in [('before', args.server), ('after', args.restart_server)]:
        out = args.output / phase
        out.mkdir()
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        command = [str(binary), '--model', str(args.model), '--host', '127.0.0.1', '--port', str(port),
            '--ssd-streaming', '--ssd-streaming-cache-experts', '32GB', '--ctx', '8192',
            '--prefill-chunk', '512', '--batched-session', '2', '--mixed-prefill-quantum', '512',
            '--mtp-draft', '1', '--tokens', '128', '--trace', str(out / 'trace.log')]
        save(out / 'execution.json', {'command': command, 'cwd': str(ROOT),
            'env': {key: value for key, value in env.items() if key.startswith('DS4')},
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()})
        process, started = None, time.monotonic()
        row = {'phase': phase, 'error': None}
        try:
            with (out / 'server.stdout').open('wb') as stdout, (out / 'server.stderr').open('wb') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                row['pid'] = process.pid
                base = f'http://127.0.0.1:{port}'
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'server startup exit{process.returncode}')
                    if time.monotonic() - started > 120:
                        raise TimeoutError('120-second startup deadline')
                    try:
                        with urllib.request.urlopen(base + '/v1/models', timeout=1) as response:
                            model = json.load(response)['data'][0]['id']
                        break
                    except (OSError, ValueError, KeyError, IndexError):
                        time.sleep(.2)
                row['startup_seconds'] = time.monotonic() - started
                common = {'model': model, 'store': True, 'stream': False, 'temperature': 0,
                    'reasoning': {'effort': 'none'}, 'max_output_tokens': 128, 'instructions': INSTRUCTIONS}
                if phase == 'before':
                    conversation, _ = request(base, out, 'conversation-create', 'POST', '/v1/conversations', {'metadata': {'qualification': 'cumulative-server'}})
                    cid = conversation['id']
                    barrier = threading.Barrier(2)
                    def initial(side):
                        fact = GOLD[side]
                        notes = ''.join(f'Old audit {index}: record verified; no new operational facts.\n' for index in range(32))
                        payload = common | {'prompt_cache_key': 'cumulative-' + side,
                            'input': f'{notes}Current record: code={fact["code"]}; count={fact["count"]}.\nRemember these facts. Return JSON with code, count, and audit equal to the integer array 0 through 15.'}
                        if side == 'B':
                            payload['conversation'] = cid
                        barrier.wait(timeout=10)
                        submitted_at = time.monotonic()
                        body, raw = request(base, out, side + '-initial', 'POST', '/v1/responses', payload)
                        return body, raw, {'submitted': submitted_at, 'finished': time.monotonic()}
                    row['concurrent_clients'] = {}
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = {side: pool.submit(initial, side) for side in ('A', 'B')}
                        for side, future in futures.items():
                            parents[side], parent_bytes[side], row['concurrent_clients'][side] = future.result()
                            result['quality'].append(graded(parents[side], GOLD[side] | {'audit': list(range(16))}, 128))
                    concurrent = row['concurrent_clients'].values()
                    result['checks']['client_requests_overlap'] = max(item['submitted'] for item in concurrent) < min(item['finished'] for item in concurrent)
                    for side in ('A', 'B'):
                        fetched, raw = request(base, out, side + '-initial-get', 'GET', '/v1/responses/' + parents[side]['id'])
                        result['checks'][side + '_stored_exact'] = fetched == parents[side] and raw == parent_bytes[side]
                    first_delta, close_now, submitted = threading.Event(), threading.Event(), threading.Event()
                    def surviving():
                        submitted.set()
                        return request(base, out, 'B-surviving', 'POST', '/v1/responses', common | {
                            'conversation': cid, 'prompt_cache_key': 'cumulative-B',
                            'input': 'Return JSON with only the remembered code and count, without audit.'})
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        cancelled = pool.submit(streamed_disconnect, base, out, common | {
                            'store': False, 'stream': True, 'max_output_tokens': 512,
                            'prompt_cache_key': 'cumulative-cancel',
                            'input': 'Count every integer from 1 through 1000 in order, separated by spaces. Do not summarize or stop early.'}, first_delta, close_now)
                        try:
                            while not first_delta.wait(timeout=.2):
                                if cancelled.done():
                                    cancelled.result()
                                    raise RuntimeError('cancel client returned without a text delta')
                            active = pool.submit(surviving)
                            if not submitted.wait(timeout=10):
                                raise TimeoutError('surviving request not submitted')
                            close_now.set()
                            row['disconnect'] = cancelled.result()
                            survivor, survivor_raw = active.result()
                        finally:
                            close_now.set()
                    result['quality'].append(graded(survivor, GOLD['B'], 128))
                    fetched, raw = request(base, out, 'B-surviving-get', 'GET', '/v1/responses/' + survivor['id'])
                    result['checks']['survivor_stored_exact'] = fetched == survivor and raw == survivor_raw
                    items_before, _ = request(base, out, 'conversation-items', 'GET', f'/v1/conversations/{cid}/items?order=asc&limit=100')
                else:
                    for side in ('A', 'B'):
                        fetched, raw = request(base, out, side + '-restart-get', 'GET', '/v1/responses/' + parents[side]['id'])
                        result['checks'][side + '_restart_exact'] = fetched == parents[side] and raw == parent_bytes[side]
                    fetched, _ = request(base, out, 'conversation-get', 'GET', '/v1/conversations/' + cid)
                    result['checks']['conversation_get'] = fetched.get('id') == cid
                    items, _ = request(base, out, 'conversation-items', 'GET', f'/v1/conversations/{cid}/items?order=asc&limit=100')
                    result['checks']['conversation_restart_items_exact'] = items['data'] == items_before['data']
                    for side in ('A', 'B'):
                        payload = common | {'prompt_cache_key': 'cumulative-' + side,
                            'input': 'Return JSON with only the remembered code and count, without audit.'}
                        if side == 'A':
                            payload['previous_response_id'] = parents['A']['id']
                        else:
                            payload['conversation'] = cid
                        child, child_raw = request(base, out, side + '-continued', 'POST', '/v1/responses', payload)
                        result['quality'].append(graded(child, GOLD[side], 128))
                        fetched, raw = request(base, out, side + '-continued-get', 'GET', '/v1/responses/' + child['id'])
                        result['checks'][side + '_child_exact'] = fetched == child and raw == child_raw
                        deleted, _ = request(base, out, side + '-child-delete', 'DELETE', '/v1/responses/' + child['id'])
                        result['checks'][side + '_child_deleted'] = deleted.get('deleted') is True
                        request(base, out, side + '-child-missing', 'GET', '/v1/responses/' + child['id'], expected=404)
                    appended, _ = request(base, out, 'item-append', 'POST', f'/v1/conversations/{cid}/items', {'items': [{'type': 'message', 'role': 'user', 'content': 'Obsolete diagnostic note; no facts.'}]})
                    item_id = appended['data'][0]['id']
                    item, _ = request(base, out, 'item-get', 'GET', f'/v1/conversations/{cid}/items/{item_id}')
                    result['checks']['item_get'] = item.get('id') == item_id
                    deleted, _ = request(base, out, 'item-delete', 'DELETE', f'/v1/conversations/{cid}/items/{item_id}')
                    result['checks']['item_deleted'] = deleted.get('id') == cid and deleted.get('object') == 'conversation'
                    request(base, out, 'item-missing', 'GET', f'/v1/conversations/{cid}/items/{item_id}', expected=404)
                    deleted, _ = request(base, out, 'conversation-delete', 'DELETE', '/v1/conversations/' + cid)
                    result['checks']['conversation_deleted'] = deleted.get('deleted') is True
                    request(base, out, 'conversation-missing', 'GET', '/v1/conversations/' + cid, expected=404)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            result['error'] = row['error']
        finally:
            row['server_returncode'] = stop(process)
            row['seconds'] = time.monotonic() - started
        logs = (out / 'server.stderr').read_text(errors='replace')
        trace = (out / 'trace.log').read_text(errors='replace') if (out / 'trace.log').exists() else ''
        row['batch_lines'] = [line for line in logs.splitlines() if 'decode batch count=' in line]
        row['batch_counts'] = [int(count) for count in re.findall(r'decode batch count=(\d+)', logs)]
        row['cancellation_lines'] = [line for line in trace.splitlines() if '--- trace: cancelled ' in line]
        row['cache_hit_lines'] = [line for line in logs.splitlines() if 'state cache exact hit' in line]
        row['cache_manifests'] = len(list(cache.glob('*.manifest')))
        row['store_files'] = [str(path.relative_to(store)) for path in store.rglob('*.json')] if store.exists() else []
        if phase == 'before':
            result['checks']['actual_disconnect_during_generation'] = bool(row.get('disconnect', {}).get('actual_text_delta') and row.get('disconnect', {}).get('closed_before_completed') and row['cancellation_lines'])
            result['checks']['coordinator_decode_observed'] = bool(row['batch_counts'])
        result['phases'].append(row)
        save(out / 'result.json', row)
        save(args.output / 'result.json', result)
        if row['error']:
            break
    result['checks']['distinct_processes'] = len(result['phases']) == 2 and result['phases'][0].get('pid') != result['phases'][1].get('pid')
    result['pass'] = not result['error'] and len(result['quality']) == 5 and all(row['pass'] for row in result['quality']) and all(result['checks'].values())
    save(args.output / 'result.json', result)
    return 0 if result['pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
