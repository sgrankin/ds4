"""Prepared C/native actual-agent file transaction pair; root launches under guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
INITIAL = b'item=widgets\ncount=17\nlocation=east\n'
EXPECTED = INITIAL.replace(b'count=17', b'count=18')
ATTRIBUTE = 'com.ds4.agent-file-test'
ATTRIBUTE_VALUE = b'preserve-ORCHID42'
PROMPT = (
    'Use only the builtin read, edit and write tools for this task. First read '
    'inventory.txt once with whole=true and raw=true, and consume its result. '
    'Then use edit on inventory.txt with old="count=17" and new="count=18", '
    'replacing that one exact unique occurrence and preserving every other byte. '
    'Then use write to create receipt.txt containing exactly ORCHID42 '
    '(eight bytes, no newline). Do not use bash, exec, Python, shell commands or '
    'any other tool, and do not retry or repair. After all three tools succeed, '
    'finish with exactly DONE. All paths are relative to the supplied working directory.'
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def generation_rounds(trace):
    rounds, current = [], None
    for line in trace.splitlines():
        if 'prefill sync done tool_round=' in line:
            current = bytearray()
        elif current is not None and ' token index=' in line:
            current.extend(bytes.fromhex(line.rsplit(' hex=', 1)[1]))
        elif current is not None and 'generation finished tool_round=' in line:
            fields = {key: int(value) for key, value in re.findall(
                r'(tool_round|generated|carried|context_limited)=(\d+)', line)}
            fields['text'] = current.decode('utf-8', errors='replace')
            rounds.append(fields)
            current = None
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--c-agent', type=Path,
        default=ROOT/'.workspaces/zig-resident-route-20261022/native/bin/ds4-agent')
    parser.add_argument('--native-agent', type=Path, default=ROOT/'.workspaces/zig-agent-file-20261022/native/bin/ds4-agent')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output/'work'
    work.mkdir()
    inventory, receipt = work/'inventory.txt', work/'receipt.txt'
    prompt_file = output/'prompt.txt'
    prompt_file.write_text(PROMPT + '\n')
    shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(metal/[^\"]+)"\]',
                             (ROOT/'ds4_metal.m').read_text()))
    results = []
    for label, agent in [('c-policy', args.c_agent), ('native', args.native_agent)]:
        arm = output/label
        arm.mkdir()
        inventory.unlink(missing_ok=True)
        inventory.write_bytes(INITIAL)
        inventory.chmod(0o751)
        subprocess.run(['xattr', '-wx', ATTRIBUTE, ATTRIBUTE_VALUE.hex(), str(inventory)], check=True)
        receipt.unlink(missing_ok=True)
        before = inventory.stat()
        (arm/'inventory.before').write_bytes(INITIAL)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('DS4', 'PT_'))}
        env.update({key: str(ROOT/path) for key, path in shaders.items()})
        env.update(DS4_AGENT_CACHE_DIR=str(arm/'cache'),
            DS4_STATE_CACHE_DIR=str(arm/'state-cache'), DS4_STATE_CACHE_HOT_MB='0',
            DS4_STATE_CACHE_SSD_MB='512', DS4_AGENT_CONTEXT_SUFFIX_REUSE='0',
            DS4_AGENT_CONTEXT_SUFFIX_PREROPE='0', DS4_AGENT_TEST_TIME='1790954400',
            DS4_METAL_V41_RESIDENT_ROUTE_GAP='0', DS4_METAL_V41_RESIDENT_PREFILL_GAP='0',
            DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS='0',
            DS4_METAL_V41_SELECTED_TILE='64', DS4_METAL_V41_SELECTED_LIMIT='1024',
            TZ='UTC')
        agent = agent.resolve()
        command = [str(agent), '--model', str(ROOT/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'),
            '--ssd-streaming', '--ssd-streaming-cache-experts', '48GB', '--ctx', '8192',
            '--prefill-chunk', '512', '--mtp-draft', '1', '--nothink', '--temp', '0',
            '--tokens', '512', '--seed', '20261022', '--non-interactive',
            '--prompt-file', str(prompt_file), '--chdir', str(work),
            '--trace', str(arm/'agent.trace')]
        save(arm/'execution.json', dict(command=command, cwd=str(ROOT),
            environment={key: value for key, value in env.items() if key.startswith('DS4') or key == 'TZ'},
            binary_sha256=hashlib.sha256(agent.read_bytes()).hexdigest(),
            source_role='previous otherwise-native C file body' if label == 'c-policy' else 'native file body',
            prompt=PROMPT, before_mode=stat.S_IMODE(before.st_mode), before_uid=before.st_uid,
            before_gid=before.st_gid, xattr=ATTRIBUTE, xattr_value=ATTRIBUTE_VALUE.decode()))
        started = time.monotonic()
        error, returncode = None, None
        with (arm/'stdout').open('wb') as stdout, (arm/'stderr').open('wb') as stderr:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, timeout=600)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                error = '600-second agent deadline exceeded'
        trace = (arm/'agent.trace').read_text(errors='replace') if (arm/'agent.trace').exists() else ''
        rounds = generation_rounds(trace)
        names = re.findall(r'<｜DSML｜\s*invoke\s+name="([^"]+)"', ''.join(row['text'] for row in rounds))
        read_round = next((row['text'] for row in rounds
                           if re.search(r'<｜DSML｜\s*invoke\s+name="read"', row['text'])), '')
        read_whole_raw = all(re.search(r'<｜DSML｜\s*parameter\s+name="' + key +
            r'"[^>]*>\s*true\s*</｜DSML｜\s*parameter>', read_round)
            for key in ('whole', 'raw'))
        after_data = inventory.read_bytes() if inventory.exists() else None
        receipt_data = receipt.read_bytes() if receipt.exists() else None
        after = inventory.stat() if inventory.exists() else None
        try:
            attribute = bytes.fromhex(subprocess.check_output(['xattr', '-px', ATTRIBUTE, str(inventory)], text=True))
        except (OSError, subprocess.CalledProcessError):
            attribute = None
        if after_data is not None:
            (arm/'inventory.after').write_bytes(after_data)
        if receipt_data is not None:
            (arm/'receipt.after').write_bytes(receipt_data)
        row = dict(arm=label, returncode=returncode, error=error,
            seconds=time.monotonic()-started, generations=rounds, tool_names=names,
            read_whole_raw=read_whole_raw,
            inventory_exact=after_data == EXPECTED, receipt_exact=receipt_data == b'ORCHID42',
            mode_preserved=bool(after and stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)),
            owner_preserved=bool(after and (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)),
            xattr_preserved=attribute == ATTRIBUTE_VALUE,
            final_done=bool(rounds and rounds[-1]['text'].strip() == 'DONE'),
            cap=any(row['generated'] + row['carried'] >= 512 or row['context_limited'] for row in rounds),
            tool_error='Tool error:' in trace,
            note='One actual workflow per implementation; timing descriptive, no speed conclusion.')
        row['pass'] = bool(returncode == 0 and not error and names == ['read', 'edit', 'write']
            and read_whole_raw and row['inventory_exact'] and row['receipt_exact'] and row['mode_preserved']
            and row['owner_preserved'] and row['xattr_preserved'] and row['final_done']
            and not row['cap'] and not row['tool_error'])
        save(arm/'result.json', row)
        results.append(row)
        save(output/'results.json', results)
        print(json.dumps(row), flush=True)
    save(output/'comparison.json', dict(all_pass=all(row['pass'] for row in results),
        identical_generated_text=[row['text'] for row in results[0]['generations']] ==
            [row['text'] for row in results[1]['generations']],
        identical_tool_names=results[0]['tool_names'] == results[1]['tool_names'],
        timing_scope='Two serial cold-cache whole processes; one sample each, no performance claim.'))
    return 0 if all(row['pass'] for row in results) else 3


if __name__ == '__main__':
    raise SystemExit(main())
