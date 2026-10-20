#!/usr/bin/env python3
"""Invoke the pinned grader outside the project; contains no unit gold."""
import hashlib,json,os,subprocess
from pathlib import Path
if __name__=='__main__':
    output=Path(os.environ['DS4_AGENT_PAIR_ACTUAL_OUTPUT'])
    # Claim the one grading attempt before invocation. A hung/missing/bad first
    # file cannot disappear from the record and be replaced by a later PASS.
    try:
        with (output/'grader-started.json').open('x') as record:
            data=Path('access.py').read_bytes()
            (output/'first-graded-access.py').write_bytes(data)
            json.dump(dict(status='FIRST_GRADING_STARTED',code_sha256=hashlib.sha256(data).hexdigest()),record);record.write('\n')
        result=subprocess.run([os.environ['DS4_AGENT_PAIR_ACTUAL_PYTHON'],os.environ['DS4_AGENT_PAIR_ACTUAL_GRADER'],'.','--attempt-dir',str(output)],capture_output=True,text=True,timeout=3)
        passed=result.returncode==0 and result.stdout.strip()=='PASS' and not result.stderr
        with (output/'grader-command-result.json').open('x') as record:
            json.dump(dict(returncode=result.returncode,passed=passed,stdout=result.stdout,stderr=result.stderr),record);record.write('\n')
    except BaseException as error:
        # Never echo the traceback, source, expected value or checker paths to
        # the agent. The private outcome preserves the actual first failure.
        try:
            with (output/'grader-command-failure.json').open('x') as record:json.dump(dict(type=type(error).__name__,error=str(error)),record);record.write('\n')
        except FileExistsError:pass
        print('FAIL');raise SystemExit(3)
    print('PASS' if passed else 'FAIL');raise SystemExit(0 if passed else 3)
