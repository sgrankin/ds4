"""ROOT read-only revalidation; never calls model identity or native helpers."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

PREPARED = Path('/Volumes/Models/ds4/.workspaces/idea-resident-env-calibration-next/speed-bench/ideas/artifacts/resident-env-calibration-prepared-20261003-v2')
MANIFEST_SHA = '1b537cb53a217d25fd13ee5957b405955a9a0ae63da9e372dde23092ccfe7fcd'
DRIVER_SHA = '7184efeb16794d88f59f7f6fef5507d09c750189be708b8187d55895a5a0e8ff'

def sha(path):
    assert Path(path).suffix.lower() != '.gguf'
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main(release_sha):
    assert sha(PREPARED/'driver.py') == DRIVER_SHA
    spec = importlib.util.spec_from_file_location('frozen_calibration', PREPARED/'driver.py')
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    s = cal.module()
    release_path = PREPARED/'root-release-proof.json'
    assert sha(release_path) == release_sha
    release = s.strict_json(release_path.read_text())
    assert release['guard_result'] == dict(status='complete', child_returncode=0)
    assert release['lock_reacquired'] is True and release['live_model_processes'] == []
    assert release['bad_pressure_samples'] == 0 and release['swap_growth_gb'] == 0
    qp, qualified = cal.qualification(s)
    plan, assets = cal.verified(PREPARED, MANIFEST_SHA, s, qp)
    raw = s.raw_module()
    raw.check_frozen_assets(assets)
    ref = cal.QUAL/'run/calibration16-reference.raw'
    metadata = raw.raw_record_metadata(ref, 16, 32, 512)
    s.validate_raw_finite(ref, metadata, raw)
    run = PREPARED/'run'
    summary = s.strict_json((run/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['full_native_brackets_and_anchor_controls_exact'] is True
    assert list(summary['arms']) == [a['name'] for a in cal.arms()]
    import numpy as np
    assert np.__version__ == '2.5.3'
    for file, digest in s.NUMPY_FILES.items():
        assert sha(file) == digest
    independent = {}
    results = {}
    for arm in cal.arms():
        name = arm['name']
        record, scores = run/(name+'.raw'), run/(name+'.csv')
        assert s.strict_json((run/(name+'-exit.json')).read_text()) == {'returncode': 0}
        layout = raw.validate_candidate_record(record, metadata, 16, 32, 512)
        finite = s.validate_raw_finite(record, metadata, raw)
        rows = s.validate_csv(scores, metadata, plan['fixed_fixture']['inputs'], raw)
        text = (run/(name+'.log')).read_text()
        row = s.validate_quality_json((run/(name+'.stdout')).read_text(), rows)
        row.update(name=name, raw_sha256=sha(record), layout=layout, finite_validation=finite,
                   cache=s.parse_cache(text), io=s.parse_io(text), route=s.parse_route(text, arm, 512*40), **raw.summarize(scores))
        assert row == s.strict_json((run/(name+'-checked.json')).read_text()) == summary['arms'][name]
        if arm['mode'] == 'native':
            native = qualified['arms']['calibration16-old-native']
            assert row['raw_sha256'] == cal.REF_SHA and row['bit_equal_rows'] == 512
            assert all(row[k] == 0 for k in ('delta_nll', 'kl_base_to_raw', 'tv', 'argmax_flips'))
            assert row['io'] == native['io'] and row['cache'] == native['cache']
        if name in cal.ANCHOR_RAWS:
            key = 'calibration16-old-approx0' if name == 'anchor-affinity3' else 'calibration16-native-weights1'
            assert row['raw_sha256'] == cal.ANCHOR_RAWS[name]
            assert all(row[k] == qualified['arms'][key][k] for k in ('io', 'cache', 'route'))
        argv = [str(s.BUILD/'quality-new8a47'), plan['model_from_prior']['path'], plan['fixed_fixture']['fixtures'], str(scores), str(ref), 'compare', '--cases', '16', '--tokens', '32', '--ctx', '4096', '--cache-gb', '64', '--chunk', '512', '--candidate-record', str(record)]
        invocation = s.strict_json((run/(name+'-command.json')).read_text())
        assert invocation == dict(argv=argv, DS4_environment={**cal.shader_environment(qp), **cal.policy(arm)}, first_attempt=True)
        with scores.open() as stream:
            csv_rows = list(csv.DictReader(stream))
        worst_error = 0.0
        flips = 0
        with ref.open('rb') as base, record.open('rb') as candidate:
            assert base.read(raw.RAW_HEADER.size) == candidate.read(raw.RAW_HEADER.size) == metadata[0]
            vocab = raw.RAW_HEADER.unpack(metadata[0])[1]
            for key, saved in zip(metadata[1], csv_rows):
                assert base.read(raw.RAW_ROW_META.size) == candidate.read(raw.RAW_ROW_META.size) == key
                _, _, target, _, _ = raw.RAW_ROW_META.unpack(key)
                a = np.frombuffer(base.read(vocab*4), dtype='<f4').astype(np.float64)
                b = np.frombuffer(candidate.read(vocab*4), dtype='<f4').astype(np.float64)
                la = a - a.max(); lb = b - b.max()
                la -= math.log(float(np.exp(la).sum())); lb -= math.log(float(np.exp(lb).sum()))
                pa, pb = np.exp(la), np.exp(lb)
                metrics = {'base_nll': -la[target], 'raw_nll': -lb[target],
                           'delta_raw_nll': la[target]-lb[target],
                           'raw_kl': max(0.0, float(np.sum(pa*(la-lb)))),
                           'raw_tv': float(np.abs(pa-pb).sum())/2}
                for field, value in metrics.items():
                    error = abs(float(saved[field])-float(value))
                    assert math.isfinite(value) and error <= 1e-8, (name, field, error)
                    worst_error = max(worst_error, error)
                flip = int(np.argmax(a) != np.argmax(b))
                assert flip == int(saved['argmax_flip'])
                flips += flip
            assert not base.read(1) and not candidate.read(1)
        independent[name] = dict(full_vocabulary_vectors=512, fp64_metric_max_abs_error=worst_error, argmax_flips=flips)
        results[name] = row
        print(json.dumps({'arm': name, 'revalidated': True, **independent[name]}), flush=True)
    decision = cal.decision(results)
    assert decision == s.strict_json((run/'decision.json').read_text()) == summary['decision']
    raw.check_frozen_assets(assets)
    proof = dict(status='ROOT_RELEASED_ALL_FIRST_ATTEMPTS_REVALIDATED', time=time.time(),
                 manifest_sha256=MANIFEST_SHA, driver_sha256=DRIVER_SHA, release_sha256=release_sha,
                 analysis_source_sha256=sha(__file__), summary_sha256=sha(run/'summary.json'),
                 decision_sha256=sha(run/'decision.json'), decision=decision, independent_fp64=independent,
                 source_qualified_raw_csv_teacher_io_cache_route_checks=True,
                 model_stat_or_open=False, native_execution=False, timing_claim=False)
    with (PREPARED/'root-offline-revalidation.json').open('x') as stream:
        json.dump(proof, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'proof_sha256': sha(PREPARED/'root-offline-revalidation.json'), 'decision': decision}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--release-sha256', required=True)
    main(parser.parse_args().release_sha256)
