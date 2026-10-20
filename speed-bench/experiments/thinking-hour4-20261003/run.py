"""Small serial experiment runner; ROOT invokes this beneath MAIN pressure_run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/Volumes/Models/ds4')
HERE = Path(__file__).resolve().parent
PLAN = ROOT / '.workspaces/idea-practical-write-observation-v4/speed-bench/ideas/artifacts/practical-write-native-controls-prepared-v4/plan.json'


def owned(argv, cwd, env, out, err, timeout):
    with out.open('xb') as stdout, err.open('xb') as stderr:
        try:
            child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            return dict(returncode=127, timeout=False, error=str(error))
        try:
            return dict(returncode=child.wait(timeout=timeout), timeout=False)
        except subprocess.TimeoutExpired:
            # Signal the still-owned group before reaping the leader.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                if sig == signal.SIGTERM:
                    time.sleep(.5)
            return dict(returncode=child.wait(timeout=5), timeout=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs='+')
    parser.add_argument('--helper')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--case', action='append')
    parser.add_argument('--tasks-file', type=Path)
    parser.add_argument('--grader', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--group', default='runs')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if args.tasks_file:
        plan['tasks'] = json.loads(args.tasks_file.read_text())['tasks']
    if args.grader:
        plan['original_grade']['path'] = str(args.grader.resolve())
    names = {task['key'] for task in plan['tasks']}
    if args.case and any(key not in names for key in args.case):
        parser.error('--case must match a loaded task: ' + ', '.join(sorted(names)))
    output = args.output.resolve() if args.output else HERE
    model = Path(plan['model']['path']).stat()  # Run only inside the held model guard.
    assert (model.st_size, model.st_mtime_ns) == (365713686528, 1789237692837653163)
    helper = args.helper or next(a['helper']['path'] for a in plan['arms'] if a['name'] == 'new-write0')
    overrides = dict(value.split('=', 1) for value in args.env)
    for arm in args.arms:
        env = {k: v for k, v in os.environ.items() if not k.startswith(('DS4', 'PT_'))}
        env.update(plan['shader_env'])
        env.update({'TZ': 'UTC', 'DS4_AGENT_TEST_TIME': '1790985600', 'DS4_MTP_SPEC_DISABLE': '1',
                    'DS4_METAL_V41_SELECTED_TILE': '64', 'DS4_METAL_V41_SELECTED_LIMIT': '1024',
                    'DS4_METAL_STREAMING_EXPERT_IO_METRICS': '1',
                    'DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY': '1',
                    'DS4_METAL_V41_RESIDENT_ROUTE_MODE': 'native',
                    'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '0', 'PT_WRITE_REASONING_BUDGET': '0',
                    'PT_LIGHT_AUDIT': '1'})
        if arm == 'budget256-affinity':
            env.update({'PT_WRITE_REASONING_BUDGET': '256',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MODE': 'apply',
                        'DS4_METAL_V41_RESIDENT_ROUTE_GAP': '.90',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS': '.25',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS': '.50',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK': '64',
                        'DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS': '3',
                        'DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS': '1',
                        'DS4_METAL_V41_RESIDENT_ROUTE_POLICY': 'replace'})
        elif arm.startswith('budget'):
            budget = arm.removeprefix('budget')
            if budget not in ('256', '512', '1024'):
                parser.error('named budgets are 256, 512 or 1024')
            env['PT_WRITE_REASONING_BUDGET'] = budget
        elif arm in ('penalty01', 'penalty05', 'penalty1'):
            env.update({'DS4_OVERTHINK_PENALTY': {'penalty01': '.1', 'penalty05': '.5', 'penalty1': '1'}[arm],
                        'DS4_OVERTHINK_SCOPE': 'all', 'DS4_OVERTHINK_AFTER': '0'})
        env.update(overrides)
        for task in plan['tasks']:
            key = task['key']
            if args.case and key not in args.case:
                continue
            case = output / args.group / arm / key
            case.mkdir(parents=True)  # Retain first attempts; refuse overwrites.
            audit = case / 'audit'
            audit.mkdir()
            visible = output / 'visible' / key
            project = visible / 'project'
            project.mkdir(parents=True, exist_ok=True)
            config = (json.dumps(task['config'], ensure_ascii=False, indent=2) + '\n').encode()
            configfile = project / 'config.json'
            if configfile.exists():
                assert configfile.read_bytes() == config, 'public config changed'
            else:
                configfile.write_bytes(config)
            prompt = visible / 'prompt.txt'
            if prompt.exists():
                assert prompt.read_text() == task['prompt'], 'public prompt changed'
            else:
                prompt.write_text(task['prompt'])
            (project / 'solution.py').unlink(missing_ok=True)
            arm_env = dict(env, PT_AUDIT_OUT=str(audit))
            command = [helper, '--model', plan['model']['path'], *plan['helper_argv_tail'],
                       '--prompt-file', str(prompt), '--system', task['system'],
                       '--trace', str(case / 'legacy.trace')]
            if arm == 'nothink':
                command += ['--nothink']
            record = dict(arm=arm, task=key, argv=command, cwd=str(project),
                          environment={k: v for k, v in arm_env.items() if k.startswith(('DS4', 'PT_'))})
            metadata = case / 'execution.json'
            metadata.write_text(json.dumps(record, indent=2) + '\n')
            print('Running', arm, key, flush=True)
            start = time.monotonic()
            result = owned(command, project, arm_env, case / 'stdout', case / 'stderr', args.timeout)
            record.update(result, seconds=time.monotonic() - start)
            record['config_unchanged'] = configfile.read_bytes() == config
            code = audit / 'first-solution.py'
            if not code.exists() and (project / 'solution.py').exists():
                code = case / 'solution.py'
                code.write_bytes((project / 'solution.py').read_bytes())
            record['snapshot_written'] = code.exists() and (project / 'solution.py').exists() and code.read_bytes() == (project / 'solution.py').read_bytes()
            grade = dict(correct=False, reason='no first solution written')
            if code.exists():
                grade_command = [sys.executable, '-B', plan['original_grade']['path'], key, str(code)]
                grader_env = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
                process = owned(grade_command, case, grader_env, case / 'grader.stdout', case / 'grader.stderr', 4)
                if not process['timeout'] and process['returncode'] in (0, 3):
                    try:
                        grade = json.loads((case / 'grader.stdout').read_text())
                    except ValueError:
                        grade = dict(correct=False, reason='invalid grader output')
                else:
                    grade = dict(correct=False, reason='grader timeout or infrastructure failure')
                record['grader'] = dict(argv=grade_command, **process)
            record['grade'] = grade
            metadata.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(dict(arm=arm, task=key, **result, seconds=record['seconds'], grade=grade)), flush=True)
    import summarize
    summarize.main(here=output)


if __name__ == '__main__':
    main()
