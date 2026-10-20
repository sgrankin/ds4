#!/usr/bin/env python3
"""CPU-only immutable holdout archive: accepted scorer, no model/guard execution."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path('/Volumes/Models/ds4')
WORK = ROOT / '.workspaces/idea-native-effort-holdout-archive'
SOURCE = ROOT / '.workspaces/idea-native-effort-holdout-signed64'
PREPARED = SOURCE / 'speed-bench/ideas/artifacts/native-effort-holdout-prepared-20261003-v2'
PLAN_SHA = '4ddcb09eeec73b54ba3c823157e463687fc756ca1441b36961c74a5a19828b80'
COMMAND_SHA = 'd25fb20188f24d5a2fb8473fed526040d5707212551b9208baa0387f83787b37'
ANALYSIS_SHA = '73a2263294f2d2257fb3e3c20848c8aa5da99ebcccfc4ec4a6a80aff8bb92d5e'
DEST = WORK / 'speed-bench/experiments/native-effort-holdout-20261003'
BASE = 'b97268bd3583dbaa6a4cf9bba75c40f492cd99b6'


def require(value, label):
    if not value:
        raise ValueError(label)


def digest(path):
    path = Path(path)
    require(path.suffix.lower() != '.gguf', 'model hash forbidden')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def info(path):
    path = Path(path)
    return dict(sha256=digest(path), bytes=path.stat().st_size, mode=stat.S_IMODE(path.stat().st_mode))


def module(name, path, pin=None):
    if pin is not None:
        require(digest(path) == pin, 'accepted analyzer pin')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def native_state(word):
    # Independent frozen xorshift state; returned multiplied random word is
    # deliberately not substituted for the stored state.
    word = word or 0x9e3779b97f4a7c15
    word ^= word >> 12
    word ^= (word << 25) & 0xffffffffffffffff
    word ^= word >> 27
    return word & 0xffffffffffffffff


def run():
    require(digest(PREPARED / 'plan.json') == PLAN_SHA and
            digest(PREPARED / 'prepared-command.json') == COMMAND_SHA, 'frozen launch authority')
    plan = load(PREPARED / 'plan.json')
    pressure = Path(plan['pressure'])
    # Actual ROOT release is verified before any producer outcome is opened.
    release = load(PREPARED / 'root-release-proof.json')
    guard = load(pressure / 'result.json')
    require(release['guard_result'] == guard == dict(status='complete', child_returncode=3), 'actual completed rc3 guard')
    require(release['lock_reacquired'] is True and release['live_model_processes'] == [], 'ROOT release complete')
    require(type(release['normal_pressure_samples']) is int and release['normal_pressure_samples'] > 0
            and release['bad_pressure_samples'] == 0 and release['swap_growth_gb'] == 0, 'normal no-swap release')
    launch = load(PREPARED / 'root-launch-proof.json')
    prepared_command = load(PREPARED / 'prepared-command.json')
    monitor = load(pressure / 'command.json')
    require(prepared_command['plan_sha256'] == PLAN_SHA
            and monitor['argv'] == prepared_command['argv'][4:], 'actual pressure child command exact')
    samples = [load_line for load_line in (json.loads(line) for line in (pressure / 'pressure.jsonl').read_text().splitlines())]
    require(len(samples) == release['normal_pressure_samples'] == 2017, 'all pressure observations retained')
    require(all(type(row['level']) is int and row['level'] == 1 and row['reason'] is None
                and row['consecutive_bad'] == 0 and row['swap_gb'] == monitor['initial']['swap_gb']
                and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in
                        ['time', 'swap_gb', 'physical_gb', 'free_gb', 'file_backed_gb', 'wired_gb', 'compressor_gb'])
                for row in samples), 'actual finite normal pressure/no swap')
    require(samples[-1]['time'] <= release['time'], 'ROOT release follows final observation')
    require(math.isclose(max(row['wired_gb'] + row['compressor_gb'] for row in samples),
                         release['peak_wired_compressor_gib'], rel_tol=0, abs_tol=1e-12), 'peak pressure arithmetic')
    analysis = module('accepted_holdout_analysis', PREPARED / 'analyze_native_effort_holdout.py', ANALYSIS_SHA)
    sys.path.insert(0, str(PREPARED))
    driver = module('accepted_holdout_driver', PREPARED / 'run_native_effort_holdout.py',
                    plan['asset_sha256'][str(PREPARED / 'run_native_effort_holdout.py')])
    driver.validate_plan(plan)
    # All58 approved non-model assets, including the original helper and old
    # native objects, are read/hash verified once. Never execute the helper.
    verified = {}
    for path, pin in plan['asset_sha256'].items():
        actual = info(path)
        require(actual['sha256'] == pin, 'frozen asset drift: ' + path)
        verified[path] = actual
    fixture = load(plan['fixture'])
    original = load(Path(plan['output']) / 'summary.json')
    require(original['status'] == 'COMPLETE' and original['all_first_attempts_retained'] is True
            and original['unique_problem_count'] == 6 and original['first_attempts'] == 36
            and original['exact_repeated_native_levels'] == [75, 25, 0], 'complete original summary')
    first, validated, independent = {}, [], []
    all_trace_rows = 0
    for arm in plan['arms']:
        process = load(Path(plan['output']) / (arm['name'] + '-process.json'))
        expected = load(Path(plan['output']) / (arm['name'] + '-summary.json'))
        result = analysis.validate_arm(arm['results_dir'], arm['level'], fixture, process['returncode'])
        result[0]['arm'] = arm['name']
        require(result[0] == expected == original['arms'][len(validated)], 'retained and rechecked strict summary exact')
        if arm['level'] in first:
            analysis.require_repeat(first[arm['level']], result)
        else:
            first[arm['level']] = result
        rows = result[1]
        # Independent count/status arithmetic, without adding any scorer.
        manual = dict(arm=arm['name'], effort=arm['level'], unique_cases=len(rows),
            correct=sum(row['status'] == 'CORRECT' for row in rows),
            completed_correct=sum(row['status'] == 'CORRECT' and row['stopped'] and not row['cap_hit'] for row in rows),
            incorrect=sum(row['status'] == 'INCORRECT' for row in rows), invalid=sum(row['status'] == 'INVALID' for row in rows),
            caps=sum(row['cap_hit'] for row in rows), generated=sum(row['generated_tokens'] for row in rows),
            reasoning=sum(row['reasoning_tokens'] for row in rows), producer_returncode=process['returncode'])
        require(manual['correct'] == result[0]['correct'] and manual['invalid'] == result[0]['invalid']
                and manual['generated'] == result[0]['generated_tokens'] and manual['caps'] == result[0]['cap_hits'],
                'independent aggregate counters')
        for row, trace_info in zip(rows, result[2]):
            previous = 20261002
            for step, token in enumerate(trace_info['trace']):
                require(token['rng_before'] == f'{previous:016x}', 'native RNG starts/chains')
                previous = native_state(previous)
                require(token['rng_after'] == f'{previous:016x}', 'one native RNG advance per original sample')
                require(token['policy_active'] is False and token['penalty'] == 0, 'no native policy intervention')
                all_trace_rows += 1
        independent.append(manual)
        validated.append(result)
    expected_metrics = {0: (0, 6, 3, 8437), 25: (5, 1, 1, 5002), 75: (4, 2, 2, 5974)}
    for level, (correct, invalid, caps, generated) in expected_metrics.items():
        report = first[level][0]
        require((report['completed_correct'], report['invalid'], report['cap_hits'], report['generated_tokens'])
                == (correct, invalid, caps, generated), 'reported level arithmetic')
    terminal_path = ROOT / 'speed-bench/ideas/artifacts/native-effort-holdout-terminal-review-20261003-v1.json'
    terminal = load(terminal_path)
    require(terminal['primary_scores_changed'] is False and
            terminal['results_sha256'] == digest(Path(plan['arms'][0]['results_dir']) / 'results.jsonl'), 'posthoc observation authority')
    for observation in terminal['observations']:
        row = next(row for row in validated[0][1] if row['case'] == observation['case'])
        require(row['status'] == observation['original_status'] == 'INVALID'
                and row['parse_reason'] == observation['original_parse_reason']
                and row['expected'] == observation['expected'] == observation['observed_terminal_answer']
                and digest(observation['raw_path']) == observation['raw_sha256'], 'manual observation matches original immutable row/raw')
        require(observation['terminal_excerpt'].encode() in Path(observation['raw_path']).read_bytes(), 'manual terminal excerpt exact')
    DEST.mkdir(parents=True, exist_ok=False)
    copies, pointers = {}, {}
    def copy_file(source, relative):
        source = Path(source); target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        old = info(source); require(info(target) == old, 'copy bytes/mode differ')
        copies[str(relative)] = dict(original_path=str(source), **old)
    def copy_tree(source, relative, exclude=()):
        for path in sorted(Path(source).rglob('*')):
            if path.is_file() and path not in exclude:
                copy_file(path, Path(relative) / path.relative_to(source))
    copy_tree(plan['output'], 'run')
    copy_tree(pressure, 'pressure')
    for arm in plan['arms']:
        copy_tree(arm['results_dir'], Path('attempts') / arm['name'])
    # Keep all36 full raw/token/RNG/prompt traces as literal copies. The large
    # helper/object/interpreter binaries retain exact external SHA/size/mode.
    binary_names = {'native-effort-holdout'}
    copy_tree(PREPARED, 'prepared-v2', exclude=(PREPARED / 'native-effort-holdout',))
    for path, actual in verified.items():
        source = Path(path)
        if source.suffix in ('.o', '.so') or source.name in binary_names or not str(source).startswith(str(ROOT)):
            pointers[path] = actual
            continue
        relative = Path('provenance') / source.relative_to(ROOT)
        copy_file(source, relative)
    source_dir = SOURCE / 'speed-bench/experiments/ds41-ssd/native-effort-holdout'
    copy_tree(source_dir, 'original-source-evidence')
    original_v1 = ROOT / '.workspaces/idea-native-effort-holdout/speed-bench/ideas/artifacts/native-effort-holdout-prepared-20261003-v1'
    copy_tree(original_v1, 'preserved-unrun-v1', exclude=(original_v1 / 'native-effort-holdout',))
    if (original_v1 / 'native-effort-holdout').is_file():
        pointers[str(original_v1 / 'native-effort-holdout')] = info(original_v1 / 'native-effort-holdout')
    copy_file(terminal_path, Path('root') / terminal_path.name)
    log = ROOT / 'speed-bench/ideas/artifacts/native-effort-holdout-root-launch-20261003-v2.log'
    require('"status": "complete", "child_returncode": 3' in log.read_text(), 'ROOT launch log retains guard rc3')
    copy_file(log, Path('root') / log.name)
    copy_file(WORK / 'speed-bench/ideas/artifacts/native-effort-holdout-archive-first.log',
              Path('archive-attempts/first-schema-refusal.log'))
    copy_file(Path(__file__), Path('archive_completed_native_effort_holdout.py'))
    # Preserve the original helper build logs/refusals present beside its
    # manifest, with no large executable copy and no broad directory filtering.
    build_dir = Path(plan['build_manifest']).parent
    for path in sorted(build_dir.iterdir()):
        if path.is_file():
            if path.suffix in ('.o', '.so') or path.name == 'native-effort-holdout':
                pointers[str(path)] = info(path)
            else:
                copy_file(path, Path('helper-build') / path.name)
    check = dict(status='VERIFIED_COMPLETE_FIRST_ATTEMPTS', base_commit=BASE,
        parser_source_commit=plan['analyzer_source_commit'], immutable_launch_commit='469d75322cd8cb2ad17a7f021906ca0cf1bd894f',
        plan_sha256=PLAN_SHA, prepared_command_sha256=COMMAND_SHA, analyzer_sha256=ANALYSIS_SHA,
        assets_verified=len(verified), first_attempts=36, unique_problems=6,
        independent_arm_counts=independent, exact_repeats=[0, 25, 75], full_native_rng_steps_verified=all_trace_rows,
        original_strict_scores_changed=False, outer_guard_returncode=3,
        manual_review='Three stopped effort0-before outputs have correct terminal numbers within prose; primary INVALID scores unchanged. This is ROOT posthoc observation, not a secondary scorer or assessment of capped outputs.',
        pressure=dict(samples=len(samples), bad=0, swap_growth_gb=0, peak_wired_compressor_gib=release['peak_wired_compressor_gib']),
        original_run_unmodified=True, model_opened_or_statted_or_guarded=False, helper_or_build_executed=False,
        preserved_archive_refusal='First CPU archive attempt assumed prepared-command.command, but actual frozen schema is argv. Refusal happened before outcomes/DEST creation; only archive reader key corrected, original assets unchanged.',
        limitations=['Six unique problems, one fixed seed;36 attempts include exact deterministic repeats, not36 independent problems.',
                    'Strict format-sensitive scorer remains literal and preregistered; posthoc prose observations remain separate.',
                    'Effort0 uses the genuine no-thinking protocol; displayed prose can still explain its reasoning.',
                    'Different effort prefixes prevent cross-effort KL/NLL or causal quantization inference; no speed/default/quality-generalization claim.'])
    write(DEST / 'checked-summary.json', check)
    write(DEST / 'external-binary-pointers.json', pointers)
    write(DEST / 'original-copy-proof.json', copies)
    text = '''The native-only holdout completed all36 first attempts on six unique problems, in frozen0/25/75/75/25/0 order. Original producer/pressure guard rc3 is retained because INVALID attempts occurred. Exact full prompt/token/RNG/piece/raw/status repeats pass at all three effort settings. The old native core, eight object hashes and MAIN defaults are unchanged.

| Native effort | Strict completed correct /6 | INVALID /6 | Cap hits | Generated tokens per six-case block |
|---|---:|---:|---:|---:|
|0|0|6|3|8437|
|25|5|1|1|5002|
|75|4|2|2|5974|

Each after block repeats its before block exactly. These are six problems and one seed;36 attempts include repeats. The scorer remains the original strict signed64 final-format scorer. ROOT separately inspected three stopped effort0-before outputs and observed correct terminal numbers embedded in explanatory prose; those original INVALID scores remain unchanged. Capped outputs were not rescored or promoted. Effort0 is the actual no-thinking protocol, although its visible answer may contain explanations.

The complete raw text, every sampled token/RNG/piece trace, prompt IDs, original result JSONL, per-arm commands/status/stdout/stderr, prepared source/plan controls, source/header/shader/build/CPU proof, original unrun V1 and lexical signed64 correction are retained. Large unchanged helper/object/interpreter binaries have exact SHA/size/mode pointers. The physical-file manifest excludes only its own top-level manifest, preserving other manifest.json files. Original paths and bytes remain unchanged.

The actual ROOT launch/release and all2017 normal pressure samples are retained: zero adverse samples or swap growth, peak wired+compressor80.44197082519531GiB. This run was diagnostic. Different effort prefixes preclude cross-effort KL/NLL or quantization-causality conclusions. There is no speed, default-promotion or broad accuracy claim.
'''
    (DEST / 'README.md').write_text(text)
    # Final immutable physical inventory: exclude only this exact top-level
    # self-manifest, not every file whose basename is manifest.json.
    physical = {str(path.relative_to(DEST)): info(path) for path in sorted(DEST.rglob('*'))
                if path.is_file() and path != DEST / 'manifest.json'}
    write(DEST / 'manifest.json', dict(files=physical, excluded_only='manifest.json', status='immutable-completed-evidence'))
    print(json.dumps(dict(archive=str(DEST), files_indexed=len(physical), manifest_sha256=digest(DEST / 'manifest.json'),
                          checked_summary_sha256=digest(DEST / 'checked-summary.json'), **{k:check[k] for k in ['first_attempts','unique_problems','full_native_rng_steps_verified']})))


if __name__ == '__main__':
    run()
