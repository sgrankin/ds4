#!/usr/bin/env python3
"""Strict six-arm ragged artifacts; complete ROOT release precedes outcome reads."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import ragged_native_tail_validation as contract

VOCAB = 129280
CTX = 32768
SLOTS = 6136
NAMES = ['old0-default', 'new0-default', 'new1-default', 'old0-fast', 'new0-fast', 'new1-fast']
FILES = {'main.csv', 'main.logits.f32', 'main.state.bin', 'dispatch.jsonl', 'branch-targets.csv', 'branch-io.csv'}
for phase in (3, 5):
    FILES |= {f'phase-{phase}.{tail}' for tail in ('anchor.state.bin', 'anchor.logits.f32',
        'restored.state.bin', 'restored.logits.f32', 'branch.logits.f32', 'branch-final.state.bin')}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.suffix.lower() != '.gguf', 'no model hashing')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return contract.load_json(Path(path).read_bytes())


def integer(value, label, maximum=(1 << 64) - 1):
    require(type(value) is int and 0 <= value <= maximum, 'typed integer: ' + label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'finite number: ' + label)
    return value


def decimal(value):
    require(type(value) is str and re.fullmatch('0|[1-9][0-9]*', value), 'literal unsigned decimal')
    return integer(int(value), 'CSV integer')


def csv_rows(path, columns):
    raw = Path(path).read_bytes()
    require(raw and raw.endswith(b'\n'), 'complete CSV')
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == columns, 'exact CSV columns')
        rows = list(reader)
    require(all(set(row) == set(columns) and all(type(v) is str for v in row.values()) for row in rows), 'complete CSV cells')
    return rows


def main_rows(path):
    rows = csv_rows(path, ['phase', 'kind', 'context', 'tokens', 'ms', 'first_decode_ms'])
    require(len(rows) == 9, 'all nine phases')
    result = []
    for phase, row in enumerate(rows):
        require([decimal(row[k]) for k in ('phase', 'context', 'tokens')] ==
                [phase, contract.POSITIONS[phase], contract.COUNTS[phase]] and row['kind'] == contract.KINDS[phase], 'frozen ordered geometry')
        elapsed = number(float(row['ms']), 'phase ms')
        first = number(float(row['first_decode_ms']), 'first decode ms')
        require(elapsed >= 0 and 0 <= first <= elapsed and (row['kind'] == 'D' or first == 0), 'diagnostic clock contract')
        result.append(dict(phase=phase, kind=row['kind'], context=contract.POSITIONS[phase], tokens=contract.COUNTS[phase], ms=elapsed, first_decode_ms=first))
    return result


def branch_io(path):
    columns = ['phase', 'pread_calls', 'requested_bytes', 'returned_bytes', 'hits', 'misses', 'evictions']
    rows = csv_rows(path, columns)
    require(len(rows) == 2, 'two branch IO records')
    result = []
    for phase, row in zip((3, 5), rows):
        value = {k: decimal(v) for k, v in row.items()}
        require(value['phase'] == phase and value['requested_bytes'] == value['returned_bytes'], 'exact branch phase and complete reads')
        result.append(value)
    return result


def cache_records(log):
    main = re.findall(r'^REPLAY_CACHE phase=([0-9]+) configured_slots=([0-9]+) resident_entries=([0-9]+)$', log, re.M)
    branches = re.findall(r'^RAGGED_BRANCH_CACHE phase=([0-9]+) configured_slots=([0-9]+) resident_entries=([0-9]+)$', log, re.M)
    require(len(main) == 9 and len(branches) == 2, 'all phase/branch cache records')
    require(len([line for line in log.splitlines() if line.startswith('REPLAY_CACHE')]) == 9 and
            len([line for line in log.splitlines() if line.startswith('RAGGED_BRANCH_CACHE')]) == 2, 'no malformed extra cache record')
    for rows, phases in ((main, range(9)), (branches, (3, 5))):
        require([int(row[0]) for row in rows] == list(phases), 'ordered unique cache records')
        require(all(int(row[1]) == SLOTS and 0 <= int(row[2]) <= SLOTS for row in rows), 'configured/admitted cache bound')
    marker = 'ragged main final, branches excluded'
    require(log.count(marker) == 1, 'unique main-only memory report')
    region = log.split(marker, 1)[1].split('RAGGED_BRANCH_CACHE', 1)[0]
    footers = re.findall(r'^ds4:   streaming expert cache budget=.*$', region, re.M)
    require(len(footers) == 1, 'unique main-only final cache footer')
    pairs = re.findall(r'([a-z_]+)=([^ ]+)', footers[0])
    fields = dict(pairs)
    require(len(fields) == len(pairs), 'no duplicate cache footer field')
    integers = ('budget', 'entries', 'hits', 'misses', 'wraps', 'evictions', 'buffer_allocs', 'buffer_reuses')
    cache = {key: decimal(fields[key]) for key in integers}
    require(cache['budget'] == SLOTS and cache['entries'] <= SLOTS, 'final cache bound')
    floats = {key: number(float(value), 'rounded cache attribution') for key, value in fields.items() if key not in integers}
    require(all(value >= 0 for value in floats.values()), 'nonnegative rounded attribution')
    return dict(main=[[int(v) for v in row] for row in main], branches=[[int(v) for v in row] for row in branches],
                main_footer=cache, rounded_attribution=floats, exact_main_IO_available=False)


def snapshot(path, history, expected_logits=None, vocab=VOCAB):
    """Validate only canonical envelope/history/logits; preserve opaque sentinels."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'owned canonical file')
    pos = len(history)
    require(0 <= pos < CTX and path.stat().st_size == contract.snapshot_size(pos, vocab), 'full canonical extent')
    with path.open('rb') as stream:
        header = stream.read(52)
        require(struct.unpack('<13I', header) == (0x34565344, 2, CTX, 1, 128, 128, CTX + 1, pos, 40, 512, 128, vocab, 0x413431), 'exact canonical header')
        require(stream.read(pos * 4) == struct.pack('<' + str(pos) + 'i', *history), 'exact canonical history')
        logits = stream.read(vocab * 4)
    contract.finite(logits, vocab)
    require(expected_logits is None or logits == expected_logits, 'canonical logits identity')
    return dict(sha256=sha(path), bytes=path.stat().st_size), logits


def finite_vectors(path, count, vocab=VOCAB):
    path = Path(path)
    require(path.stat().st_size == count * vocab * 4, 'complete full-vocabulary raw extent')
    with path.open('rb') as stream:
        for _ in range(count):
            contract.finite(stream.read(vocab * 4), vocab)
        require(not stream.read(1), 'no extra raw vector')
    return dict(sha256=sha(path), bytes=path.stat().st_size)


def vector(path, index, vocab=VOCAB):
    with Path(path).open('rb') as stream:
        stream.seek(index * vocab * 4)
        result = stream.read(vocab * 4)
    contract.finite(result, vocab)
    return result


def validate_arm(root, stderr_path, replay_text, targets, rc):
    require(type(rc) is int and rc == 0, 'native helper success; first failure is infrastructure')
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and {p.name for p in root.iterdir()} == FILES,
            'complete exact helper output inventory')
    require(all(p.is_file() and not p.is_symlink() for p in root.iterdir()), 'no aliases or nested output')
    blocks = contract.ragged(replay_text)
    phases = main_rows(root / 'main.csv')
    dispatch = contract.dispatch((root / 'dispatch.jsonl').read_bytes())
    cache = cache_records(Path(stderr_path).read_text())
    main = finite_vectors(root / 'main.logits.f32', 9)
    final, _ = snapshot(root / 'main.state.bin', blocks[-1]['history'], vector(root / 'main.logits.f32', 8))
    require((root / 'branch-targets.csv').read_bytes().endswith(b'\n'), 'complete branch target CSV')
    contract.branch_rows(root / 'branch-targets.csv', targets)
    io = branch_io(root / 'branch-io.csv')
    artifacts = {'main.logits.f32': main, 'main.state.bin': final}
    for phase in (3, 5):
        anchor = root / f'phase-{phase}.anchor.logits.f32'
        restored = root / f'phase-{phase}.restored.logits.f32'
        branch = root / f'phase-{phase}.branch.logits.f32'
        for p, count in ((anchor, 1), (restored, 1), (branch, 32)):
            artifacts[p.name] = finite_vectors(p, count)
        phase_logits = vector(root / 'main.logits.f32', phase)
        require(anchor.read_bytes() == restored.read_bytes() == vector(branch, 0) == phase_logits, 'anchor/restored/first score equals phase-end full vector')
        astate, _ = snapshot(root / f'phase-{phase}.anchor.state.bin', blocks[phase]['history'], phase_logits)
        rstate, _ = snapshot(root / f'phase-{phase}.restored.state.bin', blocks[phase]['history'], phase_logits)
        require(astate == rstate, 'full canonical restore byte identity')
        artifacts[f'phase-{phase}.anchor.state.bin'] = astate
        artifacts[f'phase-{phase}.restored.state.bin'] = rstate
        state, _ = snapshot(root / f'phase-{phase}.branch-final.state.bin', blocks[phase]['history'] + targets[phase])
        artifacts[f'phase-{phase}.branch-final.state.bin'] = state
    return dict(phases=phases, artifacts=artifacts, cache=cache, branch_IO=io, dispatch=dispatch,
                nine_phase_vectors=9, score_before_target_vectors=64, teacher_forced=True,
                interpretation='Diagnostic timings; branch restoration/evaluation excluded from main. Main rounded IO attribution is not exact demanded bytes.')


def exact_control(reference, candidate):
    require(reference['artifacts'] == candidate['artifacts'], 'all main/branch raw and full state exact')
    require(reference['cache']['main'] == candidate['cache']['main'] and
            reference['cache']['branches'] == candidate['cache']['branches'] and
            reference['cache']['main_footer'] == candidate['cache']['main_footer'], 'exact native cache counters')
    require(reference['branch_IO'] == candidate['branch_IO'], 'exact separate branch IO controls')
    require(reference['dispatch'] == candidate['dispatch'], 'exact native production dispatch')
    return True


def candidate_gate(native, candidate, native_root, candidate_root):
    for phase in range(3):
        require(vector(Path(native_root) / 'main.logits.f32', phase) ==
                vector(Path(candidate_root) / 'main.logits.f32', phase), 'short pre-long control vectors exact')
    contract.dispatch_gate(native['dispatch'], candidate['dispatch'])
    return dict(first_three_short_vectors_exact=True, both_long_phase_dispatch_differences=True,
                long_phase_vectors_and_states='diagnostic; partition change may change FP32', quiet_run_grant=False)


def diagnostics(native_root, candidate_root, targets, replay_text):
    rows = []
    for phase in (3, 5):
        for step, target in enumerate(targets[phase]):
            before = vector(Path(native_root) / f'phase-{phase}.branch.logits.f32', step)
            after = vector(Path(candidate_root) / f'phase-{phase}.branch.logits.f32', step)
            rows.append(dict(phase=phase, step=step, pos=contract.POSITIONS[phase] + step,
                             **contract.logit_metrics(before, after, target)))
    values = [row['delta_nll'] for row in rows]
    blocks = contract.ragged(replay_text)
    phase_rows = []
    for phase in range(9):
        before = vector(Path(native_root) / 'main.logits.f32', phase)
        after = vector(Path(candidate_root) / 'main.logits.f32', phase)
        # The following phase's first actual teacher token is available for
        # phases0..7. The final phase has no recorded next target: distribution
        # diagnostics remain meaningful, but no target NLL is reported.
        target = blocks[phase + 1]['ids'][0] if phase < 8 else None
        metric = contract.logit_metrics(before, after, target if target is not None else 0)
        metric['target'] = target
        if target is None:
            del metric['delta_nll']
        phase_rows.append(dict(phase=phase, pos=contract.POSITIONS[phase], **metric))
    return dict(rows=rows, phase_end_rows=phase_rows, predictions=64, mean_delta_nll=math.fsum(values) / 64,
        max_delta_nll=max(values), min_delta_nll=min(values), mean_KL=math.fsum(r['kl'] for r in rows) / 64,
        max_KL=max(r['kl'] for r in rows), mean_TV=math.fsum(r['tv'] for r in rows) / 64,
        bit_equal_vectors=sum(r['bit_equal'] for r in rows), argmax_flips=sum(r['argmax_flip'] for r in rows),
        scope='Two fixed artificial ID-extended contexts, 32 original next-D/next-P teacher targets each. No generated semantic or broad equal-quality claim.')


def validate_release(plan, plan_sha, release_path, release_sha):
    import run_ragged_native_tail_qualification as driver
    require(sha(release_path) == release_sha, 'explicit ROOT release proof hash')
    release = load(release_path)
    result = load(Path(plan['pressure']) / 'result.json')
    require(release['guard_result'] == result and result['status'] == 'complete' and
            type(result['child_returncode']) is int and result['child_returncode'] == 0, 'complete actual ROOT guard/helper result')
    require(release['lock_reacquired'] is True and type(release['live_model_processes']) is list and
            release['live_model_processes'] == [], 'released MAIN lock and no live model')
    require(number(release['time'], 'release time') > 0 and integer(release['normal_pressure_samples'], 'normal') > 0 and
            integer(release['bad_pressure_samples'], 'bad') == 0 and number(release['swap_growth_gb'], 'swap') == 0, 'normal completed release')
    command = load(Path(plan['pressure']) / 'command.json')
    require(command['argv'] == driver.command(plan, plan_sha)[4:], 'exact actual frozen guard child')
    initial = command['initial']
    require(integer(initial['level'], 'initial level') == 1 and number(initial['swap_gb'], 'initial swap') >= 0, 'normal initial host')
    samples = Path(plan['pressure']) / 'pressure.jsonl'
    raw = samples.read_bytes()
    require(raw and raw.endswith(b'\n'), 'complete pressure samples')
    values = [contract.load_json(line) for line in raw.splitlines()]
    require(len(values) == release['normal_pressure_samples'], 'exact pressure sample balance')
    previous = number(initial['time'], 'initial time')
    for row in values:
        stamp = number(row['time'], 'sample time')
        require(previous <= stamp <= release['time'] and integer(row['level'], 'sample level') == 1 and
                row['reason'] is None and integer(row['consecutive_bad'], 'bad') == 0 and
                number(row['swap_gb'], 'sample swap') == initial['swap_gb'], 'all ordered actual samples normal/no swap growth')
        previous = stamp
    return release


def analyze(plan_path, plan_sha, release_path, release_sha, output):
    import run_ragged_native_tail_qualification as driver
    require(sha(plan_path) == plan_sha, 'explicit frozen plan')
    plan = load(plan_path)
    driver.validate_plan(plan, plan_path)
    validate_release(plan, plan_sha, release_path, release_sha)  # BEFORE every producer outcome read.
    driver.check_assets(plan)
    complete = load(Path(plan['output']) / 'complete.json')
    require(complete['status'] == 'COMPLETE' and complete['arms'] == NAMES and
            type(complete['returncode']) is int and complete['returncode'] == 0, 'all six first arms completed')
    targets = driver.fixture_check(plan)
    values = {}
    for arm in plan['arms']:
        record = load(Path(plan['output']) / (arm['name'] + '-process.json'))
        require(set(record) == {'returncode'} and type(record['returncode']) is int, 'actual process record')
        values[arm['name']] = validate_arm(arm['output'], Path(plan['output']) / (arm['name'] + '-stderr.log'),
            Path(plan['replay']).read_text(), targets, record['returncode'])
        if arm['exact_reference']:
            exact_control(values[arm['exact_reference']], values[arm['name']])
    candidate_gate(values['new0-fast'], values['new1-fast'], Path(plan['output']) / 'new0-fast', Path(plan['output']) / 'new1-fast')
    quality = diagnostics(Path(plan['output']) / 'new0-fast', Path(plan['output']) / 'new1-fast', targets, Path(plan['replay']).read_text())
    result = dict(status='COMPLETE_SIX_NATIVE_CONTROLS_AND_RAGGED_DIAGNOSTICS', plan_sha256=plan_sha,
        release_sha256=release_sha, all_six_first_attempts_retained=True, controls_exact=True,
        actual_both_long_phase_dispatch_differences=True, arms=values, teacher_diagnostics=quality,
        quiet_timing_grant=False, default_promotion=False,
        limitations='Nine phase-end and64 fixed teacher vectors on one synthetic extension. Diagnostic/profiled timings and rounded main IO attribution; no SSD, semantic correctness, general equal-quality, speed or default claim. Original no-tail controls remain separate.')
    driver.put(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'release-proof', 'release-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args.plan, args.plan_sha256, args.release_proof, args.release_sha256, args.output), allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, contract.Refusal) as error:
        parser.exit(1, 'ragged analysis: ' + str(error) + '\n')
