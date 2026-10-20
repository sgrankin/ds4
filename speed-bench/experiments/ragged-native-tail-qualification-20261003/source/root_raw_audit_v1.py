#!/usr/bin/env python3
"""Independent FP64 audit of released, first-attempt ragged quality vectors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

VOCAB = 129280


def sha(path):
    path = Path(path)
    assert path.suffix.lower() != '.gguf'
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric(a, b, target, vocab=VOCAB):
    x = struct.unpack('<' + str(vocab) + 'f', a)
    y = struct.unpack('<' + str(vocab) + 'f', b)
    assert all(math.isfinite(v) for v in x + y)
    mx, my = max(x), max(y)
    lx = math.log(math.fsum(math.exp(v - mx) for v in x))
    ly = math.log(math.fsum(math.exp(v - my) for v in y))
    px = [v - mx - lx for v in x]
    py = [v - my - ly for v in y]
    kl = math.fsum(math.exp(p) * (p - q) for p, q in zip(px, py))
    tv = math.fsum(abs(math.exp(p) - math.exp(q)) for p, q in zip(px, py)) / 2
    assert kl >= -1e-12 and 0 <= tv <= 1 + 1e-12
    return {'delta_nll': py[target] * -1 + px[target], 'kl': max(kl, 0),
            'tv': tv, 'bit_equal': a == b, 'argmax_flip': x.index(mx) != y.index(my)}


def self_check():
    pack = lambda values: struct.pack('<3f', *values)
    a = pack([0, 1, -2])
    equal = metric(a, a, 1, 3)
    shift = metric(a, pack([8, 9, 6]), 1, 3)
    changed = metric(a, pack([0, 0, -2]), 1, 3)
    assert equal == {'delta_nll': 0., 'kl': 0., 'tv': 0., 'bit_equal': True, 'argmax_flip': False}
    assert shift['delta_nll'] == shift['kl'] == shift['tv'] == 0 and not shift['bit_equal']
    assert changed['delta_nll'] > 0 and changed['kl'] > 0 and changed['tv'] > 0 and changed['argmax_flip']


def vector(path, index):
    with Path(path).open('rb') as stream:
        stream.seek(index * VOCAB * 4)
        raw = stream.read(VOCAB * 4)
    assert len(raw) == VOCAB * 4
    return raw


def main(args):
    self_check()
    assert sha(args.plan) == args.plan_sha256
    assert sha(args.release) == args.release_sha256
    plan = json.loads(Path(args.plan).read_text())
    release = json.loads(Path(args.release).read_text())
    assert release['lock_reacquired'] is True and release['live_model_processes'] == []
    assert release['bad_pressure_samples'] == 0 and release['swap_growth_gb'] == 0
    pressure_result = json.loads((Path(plan['pressure']) / 'result.json').read_text())
    assert pressure_result == release['guard_result'] == {'status': 'complete', 'child_returncode': 0}
    for path, pin in plan['assets'].items():
        p = Path(path)
        assert p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256']
    assert sha(args.summary) == args.summary_sha256
    summary = json.loads(Path(args.summary).read_text())
    assert summary['status'] == 'COMPLETE_SIX_NATIVE_CONTROLS_AND_RAGGED_DIAGNOSTICS'
    assert summary['plan_sha256'] == args.plan_sha256 and summary['release_sha256'] == args.release_sha256
    targets = {}
    for line in Path(plan['targets']).read_text().splitlines():
        words = line.split()
        assert words[0] == 'B' and int(words[2]) == 32 and len(words) == 35
        phase = int(words[1])
        assert phase in (3, 5) and phase not in targets
        targets[phase] = list(map(int, words[3:]))
    assert set(targets) == {3, 5}
    rows = []
    root = Path(plan['output'])
    for phase in (3, 5):
        old = root / 'new0-fast' / f'phase-{phase}.branch.logits.f32'
        new = root / 'new1-fast' / f'phase-{phase}.branch.logits.f32'
        assert old.stat().st_size == new.stat().st_size == 32 * VOCAB * 4
        for step, target in enumerate(targets[phase]):
            expected = summary['teacher_diagnostics']['rows'][len(rows)]
            assert expected['phase'] == phase and expected['step'] == step and expected['target'] == target
            measured = metric(vector(old, step), vector(new, step), target)
            for key, value in measured.items():
                if type(value) is bool:
                    assert expected[key] is value, (phase, step, key)
                else:
                    assert abs(expected[key] - value) <= 1e-9, (phase, step, key)
            rows.append({'phase': phase, 'step': step, 'target': target, **measured})
    result = {'status': 'ROOT_INDEPENDENT_FULL_RAW_FP64_PASS', 'rows': rows,
              'mean_delta_nll': math.fsum(row['delta_nll'] for row in rows) / 64,
              'mean_KL': math.fsum(row['kl'] for row in rows) / 64,
              'max_KL': max(row['kl'] for row in rows),
              'argmax_flips': sum(row['argmax_flip'] for row in rows),
              'bit_equal_vectors': sum(row['bit_equal'] for row in rows),
              'plan_sha256': args.plan_sha256, 'release_sha256': args.release_sha256,
              'producer_summary_sha256': args.summary_sha256, 'audit_source_sha256': sha(__file__),
              'vocab': VOCAB, 'teacher_vectors': 64, 'absolute_tolerance': 1e-9,
              'model_or_native_helper_run': False, 'speed_or_semantic_claim': False}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    for name in ('plan', 'plan-sha256', 'release', 'release-sha256', 'summary', 'summary-sha256', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print('SELF_CHECK_PASS')
    else:
        assert all(getattr(args, name.replace('-', '_')) for name in
                   ('plan', 'plan-sha256', 'release', 'release-sha256', 'summary', 'summary-sha256', 'output'))
        main(args)
