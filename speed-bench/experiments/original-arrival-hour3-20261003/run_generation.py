"""Four fixed generation cases, invoked only as a MAIN pressure-guard child."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/Volumes/Models/ds4')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.workspaces/idea-resident-original-deadline-native'
BUILD = WORK / 'speed-bench/ideas/artifacts/original-deadline-admitted-direct-v1'
FIXTURES = ROOT / 'speed-bench/experiments/resident-functional-hour2-20261003/fixtures'
MODEL = ROOT / 'gguf/DeepSeek-V4.1-Flash-Q2.gguf'
IDS = ['functional24_01_cursor_pages', 'functional24_07_little_endian_words',
       'functional24_16_weighted_median', 'functional24_24_literal_unicode_note']
ARMS = ['native', 'affinity-strong', 'deadline-strong-age1', 'deadline-strong-age8']


def require_guard():
    parent = subprocess.check_output(['/bin/ps', '-p', str(os.getppid()), '-o', 'args='],
                                     text=True, timeout=2)
    assert str(ROOT / 'speed-bench/ideas/pressure_run.py') in shlex.split(parent), 'MAIN guard parent required'
    with (ROOT / 'speed-bench/ideas/artifacts/model-job.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
        raise RuntimeError('MAIN model-job lock is not held')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs='*')
    args = parser.parse_args()
    if any(arm not in ARMS for arm in args.arms):
        parser.error('arms must be chosen from ' + ', '.join(ARMS))
    require_guard()  # Before any GGUF metadata or native child.
    st = MODEL.stat()
    assert (st.st_size, st.st_mtime_ns) == (365713686528, 1789237692837653163)
    out = HERE / 'generation'
    out.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT'))}
    env.update({
        'DS4_CLI_FORCE_SESSION': '1',
        'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.90',
        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.25',
        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS': '.50',
        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '64',
        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': '3',
        'DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS': '1',
        'DS4_METAL_V41_RESIDENT_ROUTE_POLICY': 'replace',
        'DS4_METAL_V41_SELECTED_TILE': '64',
        'DS4_METAL_V41_SELECTED_LIMIT': '1024',
        'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1',
        'DS4_METAL_V41_RESIDENT_ROUTE_PROFILE': '1',
    })
    for arm in args.arms or ARMS:
        arm_env = dict(env)
        arm_env['DS4_METAL_V41_RESIDENT_ROUTE_MODE'] = 'native' if arm == 'native' else 'apply'
        arm_env['DS4_METAL_V41_ORIGINAL_DEADLINE'] = '1' if arm.startswith('deadline') else '0'
        cli = BUILD / ('ds4-original-deadline-' + arm.rsplit('-', 1)[1] if arm.startswith('deadline')
                       else 'ds4-original-deadline')
        for case in IDS:
            stem = arm + '-' + case
            command = [str(cli), '--model', str(MODEL), '--prompt-file', str(FIXTURES / (case + '.prompt.txt')),
                       '--ctx', '8192', '--ssd-streaming', '--ssd-streaming-cache-experts', '64GB',
                       '--prefill-chunk', '512', '--tokens', '256', '--temp', '0', '--seed', '20261003', '--nothink']
            record = dict(argv=command, cwd=str(WORK), compiled_join_policy='only admitted original requests', environment={k: v for k, v in arm_env.items() if k.startswith('DS4')},
                          inherited_prefixes_cleared=['DS4', 'PT'], model_bytes=st.st_size, model_mtime_ns=st.st_mtime_ns)
            metadata = out / (stem + '.execution.json')
            with metadata.open('x') as fp:
                json.dump(record, fp, indent=2)
            print('Generating', arm, case, flush=True)
            start = time.monotonic()
            error = None
            with (out / (stem + '.stdout')).open('x') as stdout, (out / (stem + '.stderr')).open('x') as stderr:
                try:
                    result = subprocess.run(command, cwd=WORK, env=arm_env, stdout=stdout, stderr=stderr, timeout=300)
                    rc = result.returncode
                except subprocess.TimeoutExpired:
                    rc, error = 124, 'generation timeout after 300 seconds'
                except OSError as exc:
                    rc, error = 127, str(exc)
            text = (out / (stem + '.stderr')).read_text(errors='replace')
            if rc == 0:
                if 'GENERATION_SUMMARY ' not in text:
                    error = 'session generation footer missing'
                elif arm != 'native' and 'resident-route layers=' not in text:
                    error = 'resident selector was not activated'
                elif arm.startswith('deadline') and 'ORIGINAL_DEADLINE generation=' not in text:
                    error = 'original deadline stage was not activated'
            record.update(returncode=rc, seconds=time.monotonic() - start, activation_error=error,
                          generation_footer=[line for line in text.splitlines() if 'GENERATION_SUMMARY ' in line])
            metadata.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(dict(arm=arm, case=case, returncode=rc, seconds=record['seconds'], error=error)), flush=True)
            if rc or error:
                raise SystemExit(rc or 1)


if __name__ == '__main__':
    main()
