#!/usr/bin/env python3
"""One local model server beneath MAIN pressure_run.py; installed SDK/Pi smoke."""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--server', type=Path, default=ROOT / 'ds4-server')
    args = p.parse_args()
    parent = subprocess.check_output(['/bin/ps', '-p', str(os.getppid()), '-o', 'command='], text=True)
    if 'speed-bench/ideas/pressure_run.py' not in parent:
        p.error('launch as the direct child of MAIN speed-bench/ideas/pressure_run.py')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                              (ROOT / 'ds4_metal.m').read_text()))
    env.update({key: str(ROOT / value) for key, value in shaders.items()})
    env.update(DS4_STATE_CACHE_DIR=str(out / 'cache'), DS4_STATE_CACHE_HOT_MB='0',
               DS4_STATE_CACHE_SSD_MB='1024', DS4_SERVER_RESPONSE_STORE_DIR=str(out / 'store'),
               DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_SELECTED_TILE='64',
               DS4_METAL_V41_SELECTED_LIMIT='1024', DS4_SERVER_CONTEXT_SUFFIX_REUSE='0')
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    command = [str(args.server.resolve()), '--model', str(ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'),
               '--host', '127.0.0.1', '--port', str(port), '--ssd-streaming',
               '--ssd-streaming-cache-experts', '64GB', '--ctx', '8192',
               '--prefill-chunk', '512', '--batched-session', '1', '--tokens', '256']
    save(out / 'execution.json', {'command': command, 'cwd': str(ROOT),
         'environment': {key: value for key, value in env.items() if key.startswith('DS4')},
         'server_count': 1, 'sdk_version': '6.26.0', 'pi_version': '0.83.0',
         'scope': 'Local installed-client compatibility and tiny retrieval correctness; no timing comparison.'})
    started = time.monotonic()
    result = {'returncode': None}
    with (out / 'server.stdout.log').open('wb') as stdout, (out / 'server.stderr.log').open('wb') as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server startup exit {proc.returncode}')
                if time.monotonic() - started >= 120:
                    raise TimeoutError('server startup timeout')
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                try:
                    conn.request('GET', '/v1/models')
                    response = conn.getresponse()
                    response.read()
                    if response.status == 200:
                        break
                except OSError:
                    pass
                finally:
                    conn.close()
                time.sleep(.2)
            save(out / 'startup.json', {'seconds': time.monotonic() - started, 'pid': proc.pid, 'port': port})
            runner = ['/opt/homebrew/bin/node', str(HERE / 'client-smoke.mjs'),
                      '--base-url', f'http://127.0.0.1:{port}/v1', '--output', str(out / 'checks')]
            with (out / 'client.stdout.log').open('wb') as client_stdout, (out / 'client.stderr.log').open('wb') as client_stderr:
                completed = subprocess.run(runner, cwd=ROOT, env=env, timeout=600,
                                           stdout=client_stdout, stderr=client_stderr)
            result.update(argv=runner, returncode=completed.returncode)
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            result.update(seconds=time.monotonic() - started, server_returncode_after_shutdown=proc.returncode)
            save(out / 'result.json', result)
    return result['returncode'] if result['returncode'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
