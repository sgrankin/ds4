"""Model-free independent FP64 audit of the completed fixed mixed24 study.

This analyzer never opens a model, tokenizes, launches a helper, or chooses a
policy.  The exact ROOT release and original prepared authority precede all
current outcome reads.  Raw/policy equality is a source-qualified property of
the pinned no-policy-hook helper, not a separately recorded policy vector.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys

sys.dont_write_bytecode = True
ORIGINAL = Path('/Volumes/Models/ds4/.workspaces/idea-resident-unused-mixed24')
PREPARED = ORIGINAL/'speed-bench/ideas/artifacts/resident-mixed24-confirmation-prepared-20261003-v1'
SOURCE = ORIGINAL/'speed-bench/ideas/prepare_resident_mixed24_confirmation.py'
SOURCE_SHA = '06059d1332a0a513e1187727c66e6cfbd3c15bb1480d81c1b2ffbc6152498fce'
MANIFEST_SHA = 'b30ff4ebd43adbd030506f605a241bbcd897730a082e472767793f5982d604e3'
COMMAND_SHA = '383106a435b060e1e301f2654b4b3fb996fa2bf3be83da3963aaa744d2ec80e9'
SELECTION_SHA = '78f27d9fe84e961dcde9a0b003994b62606bba684f886c02f68683381c785835'
RELEASE_SHA = 'ac67daecdf956de3692008e5dbd185fdb718932b650a3fe15b639a0e178b6ae7'
HEADER = struct.Struct('=4I')
KEY = struct.Struct('=4IQ')
MAGIC = 0x44535131
CASES, STEPS, ROWS, VOCAB = 24, 32, 768, 129280
TOLERANCE = 1e-8
NAMES = ('native-before', 'anchor-affinity3', 'native-after')
FIELDS = 'case step target prompt_tokens base_nll raw_nll policy_nll delta_raw_nll delta_policy_nll raw_kl policy_kl raw_tv policy_tv argmax_flip decode_ms'.split()


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def sha(path):
    path = Path(path)
    require(path.suffix.lower() != '.gguf', 'model hashing is forbidden')
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load(path):
    return strict_json(Path(path).read_text())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_release(proof):
    keys = {'time', 'guard_result', 'lock_reacquired', 'live_model_processes',
        'normal_pressure_samples', 'bad_pressure_samples', 'swap_growth_gb',
        'peak_wired_compressor_gib'}
    require(type(proof) is dict and set(proof) == keys, 'release schema differs')
    guard = proof['guard_result']
    require(type(guard) is dict and set(guard) == {'status', 'child_returncode'}
        and guard['status'] == 'complete' and type(guard['child_returncode']) is int
        and guard['child_returncode'] == 0, 'study not successfully complete')
    require(proof['lock_reacquired'] is True and type(proof['live_model_processes']) is list
        and proof['live_model_processes'] == [], 'ROOT did not prove model release')
    require(type(proof['normal_pressure_samples']) is int and proof['normal_pressure_samples'] > 0
        and type(proof['bad_pressure_samples']) is int and proof['bad_pressure_samples'] == 0,
        'pressure counts invalid')
    require(finite_number(proof['time']) and proof['time'] > 0
        and finite_number(proof['swap_growth_gb']) and proof['swap_growth_gb'] == 0
        and finite_number(proof['peak_wired_compressor_gib'])
        and proof['peak_wired_compressor_gib'] > 0, 'pressure/time invalid')
    return proof


def release_gate():
    path = PREPARED/'root-release-proof.json'
    require(sha(path) == RELEASE_SHA, 'explicit ROOT release authority changed')
    return validate_release(load(path))


def validate_pressure(command, samples, release, expected_argv, grant_time):
    require(type(command) is dict and set(command) == {'argv', 'initial', 'polling_seconds', 'stop_rule'}
        and command['argv'] == expected_argv and command['polling_seconds'] == 1
        and command['stop_rule'] == 'two consecutive adverse samples; large RAM use alone is allowed',
        'retained guard command differs')
    initial = command['initial']
    fields = {'time', 'level', 'swap_gb', 'physical_gb', 'free_gb', 'file_backed_gb', 'wired_gb', 'compressor_gb'}
    require(type(initial) is dict and set(initial) == fields, 'initial pressure schema differs')
    require(len(samples) == release['normal_pressure_samples'], 'release/sample count differs')
    previous, peak = grant_time, 0.0
    for index, row in enumerate([initial]+samples):
        require(type(row) is dict and set(row) == (fields if index == 0 else fields|{'reason','consecutive_bad'}),
            'pressure sample schema differs')
        require(type(row['level']) is int and row['level'] == 1
            and all(finite_number(row[field]) and row[field] >= 0 for field in fields-{'level'}),
            'pressure sample invalid')
        require(previous <= row['time'] <= release['time'] and row['swap_gb'] == initial['swap_gb'],
            'pressure order/swap differs')
        require(index == 0 or (row['reason'] is None and type(row['consecutive_bad']) is int
            and row['consecutive_bad'] == 0), 'adverse pressure sample')
        previous = row['time']
        peak = max(peak, row['wired_gb']+row['compressor_gb'])
    close(peak, release['peak_wired_compressor_gib'], 'retained pressure peak')


def source_module():
    require(sha(SOURCE) == SOURCE_SHA, 'frozen original driver changed')
    spec = importlib.util.spec_from_file_location('mixed24_original_offline', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected, label):
    require(finite_number(actual) and finite_number(expected)
        and abs(actual - expected) <= TOLERANCE, label + ' differs beyond fixed absolute 1e-8')


def fp64_metrics(base, candidate, target, np):
    """Cast saved FP32 to FP64 before arithmetic, exactly as the C scorer does."""
    require(base.ndim == candidate.ndim == 1 and base.shape == candidate.shape
        and type(target) is int and 0 <= target < base.size, 'metric geometry invalid')
    require(np.isfinite(base).all() and np.isfinite(candidate).all(), 'nonfinite raw vector')
    b = base.astype(np.float64)
    c = candidate.astype(np.float64)
    bz = float(b.max()) + math.log(float(np.exp(b - b.max()).sum(dtype=np.float64)))
    cz = float(c.max()) + math.log(float(np.exp(c - c.max()).sum(dtype=np.float64)))
    bp, cp = b - bz, c - cz
    p, q = np.exp(bp), np.exp(cp)
    bn, cn = bz - float(b[target]), cz - float(c[target])
    return dict(base_nll=bn, raw_nll=cn, delta_raw_nll=cn-bn,
        raw_kl=float(np.sum(p * (bp-cp), dtype=np.float64)),
        raw_tv=float(0.5*np.sum(np.abs(p-q), dtype=np.float64)),
        argmax_flip=int(int(np.argmax(b)) != int(np.argmax(c))))


def compare_csv_metric(row, metrics):
    for key in ('base_nll', 'raw_nll', 'delta_raw_nll', 'raw_kl', 'raw_tv'):
        close(float(row[key]), metrics[key], 'CSV ' + key)
    for policy, raw in [('policy_nll', 'raw_nll'), ('delta_policy_nll', 'delta_raw_nll'),
            ('policy_kl', 'raw_kl'), ('policy_tv', 'raw_tv')]:
        close(float(row[policy]), metrics[raw], 'source-qualified CSV ' + policy)
    require(row['argmax_flip'] == str(metrics['argmax_flip']), 'CSV argmax differs')


def stream_scores(reference, candidate, rows, np, cases=CASES, steps=STEPS, vocab=VOCAB):
    require(len(rows) == cases*steps, 'independent scorer row completeness differs')
    totals = {key: [] for key in ('base_nll', 'raw_nll', 'delta_raw_nll', 'raw_kl', 'raw_tv')}
    scored, bit_equal, flips = [], 0, 0
    with Path(reference).open('rb') as bf, Path(candidate).open('rb') as cf:
        bh, ch = bf.read(HEADER.size), cf.read(HEADER.size)
        require(bh == ch == HEADER.pack(MAGIC, vocab, cases, steps), 'canonical raw header differs')
        prefix = None
        for index, row in enumerate(rows):
            bk, ck = bf.read(KEY.size), cf.read(KEY.size)
            require(len(bk) == KEY.size and bk == ck, 'raw metadata differs/truncated')
            case, step, target, prompt, fingerprint = KEY.unpack(bk)
            require((case, step) == divmod(index, steps) and target < vocab and prompt > 0,
                'independent raw order/teacher keys invalid')
            if step == 0:
                prefix = prompt, fingerprint
            require(prefix == (prompt, fingerprint), 'raw per-case prompt/fingerprint changed')
            require(row['step'] == str(step) and row['target'] == str(target)
                and row['prompt_tokens'] == str(prompt), 'CSV/raw teacher alignment differs')
            bb, cb = bf.read(4*vocab), cf.read(4*vocab)
            require(len(bb) == len(cb) == 4*vocab, 'raw vector truncated')
            metric = fp64_metrics(np.frombuffer(bb, dtype=np.float32),
                np.frombuffer(cb, dtype=np.float32), target, np)
            compare_csv_metric(row, metric)
            bit_equal += bb == cb
            flips += metric['argmax_flip']
            for key in totals:
                totals[key].append(metric[key])
            scored.append(dict(row, **{key: repr(value) for key, value in metric.items()}))
        require(bf.read(1) == cf.read(1) == b'', 'raw trailing bytes')
    result = {key: statistics.mean(values) for key, values in totals.items()}
    result.update(rows=len(scored), bit_equal_rows=bit_equal, argmax_flips=flips)
    return result, scored


def summarize_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['case'], []).append(row)
    delta = [statistics.mean(float(r['delta_raw_nll']) for r in group) for group in groups.values()]
    rng = random.Random(20261002)
    boots = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(2000))
    kl = sorted(float(row['raw_kl']) for row in rows)
    return dict(prompt_mean_delta_nll=statistics.mean(delta),
        prompt_bootstrap_95=[boots[49], boots[1949]], worst_prompt_delta_nll=max(delta),
        kl_p99=kl[min(len(kl)-1, int(len(kl)*.99))], kl_max=max(kl),
        per_prompt={name: dict(tokens=len(group),
            delta_nll=statistics.mean(float(r['delta_raw_nll']) for r in group),
            kl=statistics.mean(float(r['raw_kl']) for r in group)) for name, group in groups.items()})


def nested_close(actual, expected, label='arithmetic'):
    if type(actual) is dict and type(expected) is dict:
        require(list(actual) == list(expected), label + ' schema/order differs')
        for key in actual:
            nested_close(actual[key], expected[key], label + '.' + key)
    elif type(actual) is list and type(expected) is list:
        require(len(actual) == len(expected), label + ' extent differs')
        for i, (a, e) in enumerate(zip(actual, expected)):
            nested_close(a, e, label + '[' + str(i) + ']')
    elif type(actual) is int or type(expected) is int:
        require(type(actual) is type(expected) and actual == expected, label + ' typed count differs')
    elif type(actual) is float or type(expected) is float:
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ' differs')


def audit(output):
    # No current outcome read may precede these release/static authority gates.
    release = release_gate()
    require(sha(PREPARED/'manifest.json') == MANIFEST_SHA
        and sha(PREPARED/'prepared-command.json') == COMMAND_SHA
        and sha(PREPARED/'selection.json') == SELECTION_SHA, 'prepared authority changed')
    original = source_module()
    calibration = original.calibration_module()
    study = calibration.module()
    qualified, _ = calibration.qualification(study)
    plan, assets = original.verified(PREPARED, MANIFEST_SHA, calibration, study, qualified)
    raw = study.raw_module()
    raw.check_frozen_assets(assets)
    grant = calibration.require_root_grant(PREPARED, MANIFEST_SHA, study)
    require(tuple(a['name'] for a in plan['arms']) == NAMES
        and plan['selection']['selected_policy'] == 'anchor-affinity3', 'fixed first-attempt order differs')
    # Import only the source-qualified NumPy after its frozen package pins passed.
    import numpy as np
    require(np.__version__ == '2.5.3', 'frozen NumPy version differs')
    run = PREPARED/'run'
    expected = ['summary.json', 'confirmation.json', 'model-identity.json', 'native-reference-proof.json']
    for name in NAMES:
        expected += [name+suffix for suffix in ('.raw', '.csv', '.stdout', '.log', '-exit.json', '-checked.json', '-command.json')]
    current = {run/name: sha(run/name) for name in expected}
    for name in ('root-release-proof.json', 'root-launch-proof.json', 'pressure/result.json',
            'pressure/command.json', 'pressure/pressure.jsonl'):
        current[PREPARED/name] = sha(PREPARED/name)
    completed = load(run/'summary.json')
    require(completed['status'] == 'complete' and completed['first_attempt_only'] is True
        and completed['full_native_brackets_exact'] is True and completed['no_confirmation_policy_selection'] is True
        and completed['timing_claim'] is False and tuple(completed['arms']) == NAMES,
        'all original complete first attempts required')
    guard = load(PREPARED/'pressure/result.json')
    require(guard['status'] == 'complete' and type(guard['child_returncode']) is int
        and guard['child_returncode'] == 0, 'original guard not complete')
    samples = [strict_json(line) for line in (PREPARED/'pressure/pressure.jsonl').read_text().splitlines()]
    validate_pressure(load(PREPARED/'pressure/command.json'), samples, release,
        study.guard_command(PREPARED, MANIFEST_SHA)[4:], grant['time'])
    require(load(run/'model-identity.json') == dict(identity=plan['model_from_prior'],
        guard_verified_before_stat=True), 'retained model identity/guard evidence differs')
    require(completed['model_identity'] == plan['model_from_prior'], 'summary model identity differs')
    reference = run/'native-before.raw'
    metadata = raw.raw_record_metadata(reference, CASES, STEPS, ROWS)
    require(metadata[2]['vocab'] == VOCAB, 'fixed full-vocabulary extent differs')
    require(load(run/'native-reference-proof.json') == dict(sha256=current[reference],
        metadata=metadata[2], created_before_candidate=True), 'first native reference authority differs')
    results, independent = {}, {}
    for arm in plan['arms']:
        name = arm['name']
        argv = [str(study.BUILD/'quality-new8a47'), plan['model_from_prior']['path'],
            plan['fixed_fixture']['fixtures'], str(run/(name+'.csv')), str(reference),
            'record' if name == 'native-before' else 'compare', '--cases', '24', '--tokens', '32',
            '--ctx', '4096', '--cache-gb', '64', '--chunk', '512']
        if name != 'native-before':
            argv += ['--candidate-record', str(run/(name+'.raw'))]
        require(load(run/(name+'-command.json')) == dict(argv=argv,
            DS4_environment=dict(plan['shader_env'], **calibration.policy(arm)), first_attempt=True),
            'retained first-attempt command/environment differs')
        require(load(run/(name+'-exit.json')) == dict(returncode=0)
            and type(load(run/(name+'-exit.json'))['returncode']) is int, 'first attempt helper failed')
        saved = load(run/(name+'-checked.json'))
        require(saved == completed['arms'][name], 'checked arm/aggregate differs')
        layout = raw.validate_candidate_record(run/(name+'.raw'), metadata, CASES, STEPS, ROWS)
        rows = original.validate_csv(run/(name+'.csv'), metadata, plan['fixed_fixture']['inputs'], raw)
        quality = original.validate_quality_json((run/(name+'.stdout')).read_text(), rows)
        metrics, scored = stream_scores(reference, run/(name+'.raw'), rows, np)
        checks = {'avg_nll': 'raw_nll', 'delta_nll': 'delta_raw_nll',
            'policy_delta_nll': 'delta_raw_nll', 'kl_base_to_raw': 'raw_kl',
            'kl_base_to_policy': 'raw_kl', 'tv': 'raw_tv'}
        for field, key in checks.items():
            close(quality[field], metrics[key], 'independent aggregate ' + field)
        require(quality['argmax_flips'] == metrics['argmax_flips']
            and quality['bit_equal_rows'] == metrics['bit_equal_rows'], 'independent counts differ')
        summary = summarize_rows(scored)
        recorded = raw.summarize(run/(name+'.csv'))
        nested_close(recorded, summary, 'independent prompt/tail/bootstrap')
        text = (run/(name+'.log')).read_text()
        quality.update(name=name, raw_sha256=current[run/(name+'.raw')],
            csv_sha256=current[run/(name+'.csv')], layout=layout,
            finite_validation=dict(vectors=ROWS, fp32_values=ROWS*VOCAB, finite=True), cache=study.parse_cache(text),
            io=study.parse_io(text), route=study.parse_route(text, arm, ROWS*40), **recorded)
        require(quality == saved, 'source-qualified checked arm reconstruction differs')
        native = results.get('native-before', quality)
        if arm['mode'] == 'native':
            require(quality['raw_sha256'] == current[reference] and metrics['bit_equal_rows'] == ROWS
                and all(metrics[key] == 0 for key in ('delta_raw_nll', 'raw_kl', 'raw_tv', 'argmax_flips'))
                and quality['io'] == native['io'] and quality['cache'] == native['cache'],
                'native fullraw/teacher/IO/cache brackets differ')
        results[name] = quality
        independent[name] = dict(metrics=metrics, prompt_metrics=summary,
            raw_sha256=quality['raw_sha256'], io=quality['io'], cache=quality['cache'], route=quality['route'])
    report = original.confirmation(results, 'anchor-affinity3', plan['fixed_fixture']['inputs'],
        original.frozen_spec(), calibration)
    require(report == load(run/'confirmation.json') == completed['confirmation'],
        'fixed confirmation arithmetic differs')
    raw.check_frozen_assets(assets)
    require(all(sha(path) == digest for path, digest in current.items()), 'original results changed during analysis')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='MODEL_FREE_FULL_RAW_FP64_AND_ORIGINAL_FIRST_ATTEMPT_AUDIT_PASS',
        original_guard_returncode=0, original_first_attempts_unchanged=True,
        fixed_selected_policy='anchor-affinity3', confirmation=report, independent=independent,
        raw_rows_per_arm=ROWS, arms=3, total_full_vectors_scored=ROWS*3, vocab=VOCAB,
        metric_tolerance=dict(absolute=TOLERANCE, relative=0), release=release,
        source_authority=dict(path=str(SOURCE), sha256=SOURCE_SHA),
        manifest_authority=dict(path=str(PREPARED/'manifest.json'), sha256=MANIFEST_SHA),
        release_authority=dict(path=str(PREPARED/'root-release-proof.json'), sha256=RELEASE_SHA),
        static_frozen_assets=len(assets), original_result_files={str(path): digest for path, digest in current.items()},
        analyzer_sha256=sha(__file__), no_model_tokenizer_metadata_guard_or_helper_run=True,
        policy_vectors_not_separately_recorded=True, no_new_policy_selection=True,
        io_scope='preclose whole-engine P+D requested/returned logical expert reads; no physical SSD or direct postclose measurement',
        timing_claim=False)
    with (output/'analysis.json').open('x') as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(output),
        confirmation_pass=report['selected_confirmation_pass'],
        failures=report['reports']['anchor-affinity3']['failures'],
        delta_nll=results['anchor-affinity3']['delta_nll'], kl=results['anchor-affinity3']['kl_base_to_raw'],
        native_requested_bytes=results['native-before']['io']['requested_bytes'],
        anchor_requested_bytes=results['anchor-affinity3']['io']['requested_bytes']), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    audit(arguments.output.absolute())
