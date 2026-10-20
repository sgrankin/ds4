#!/usr/bin/env python3
"""Released math16 evidence packaging. Never launches, rescoring, or model access."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat

LIMIT = 2 * 1024 * 1024
NAMES = ['native-before', 'budget1024', 'native-after']
COUNTS = ('completed_correct', 'incorrect', 'invalid', 'capped_correct', 'cap_hits',
          'generated_tokens', 'reasoning_tokens', 'forced_cases')
AUDIT_STATUS = 'ROOT_INDEPENDENT48_FINAL_SCORE_TOKEN_WORK_AND_PAIRED_AUDIT_PASS'


def require(value, message):
    if not value:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: fail('nonfinite JSON'))


def fail(message):
    raise ValueError(message)


def load(path):
    return decode(Path(path).read_bytes())


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def integer(value, label):
    require(type(value) is int and value >= 0, 'literal nonnegative integer: ' + label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'finite number: ' + label)
    return value


def file_info(path, alias=None):
    path = Path(path)
    require(path.suffix.lower() != '.gguf', 'model access forbidden')
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        require(alias is not None and os.readlink(path) == alias['link_text'] and
                str(path.resolve(strict=True)) == alias['resolved'], 'unapproved leaf alias')
    else:
        require(alias is None and stat.S_ISREG(before.st_mode), 'ordinary file required')
    with path.open('rb') as stream:
        start = os.fstat(stream.fileno())
        require(stat.S_ISREG(start.st_mode), 'regular target required')
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        end = os.fstat(stream.fileno())
    require((start.st_dev, start.st_ino, start.st_size, start.st_mtime_ns) ==
            (end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns), 'source changed while hashing')
    return dict(sha256=digest, bytes=end.st_size, mode=stat.S_IMODE(end.st_mode))


def pinned(path, sha):
    require(type(sha) is str and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
            'literal SHA256 required')
    require(file_info(path)['sha256'] == sha, 'pinned file drift: ' + str(path))
    return load(path)


def jsonl(path):
    raw = Path(path).read_bytes()
    require(raw and raw.endswith(b'\n'), 'complete newline-terminated JSONL')
    return [decode(line) for line in raw.splitlines()]


def local(root, name):
    require(type(name) is str and name not in ('', '.', '..') and Path(name).name == name,
            'plain artifact name')
    path = Path(root) / name
    require(not path.is_symlink() and path.resolve().parent == Path(root).resolve(), 'artifact escape')
    return path


def check_runtime(registry):
    authority = pinned(registry['runtime_authority'], registry['runtime_authority_sha256'])
    values = authority['runtime_and_shaders_unchanged']
    require(type(values) is dict and len(values) == 100, 'exact frozen MAIN runtime100 authority')
    for relative, sha in values.items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'runtime authority path escape')
        require(file_info(Path(registry['root']) / path)['sha256'] == sha, 'MAIN runtime drift: ' + relative)
    return values


def gate(registry, release_path, release_sha, summary_path, summary_sha, audit_path, audit_sha):
    """Only fixed authority/release metadata is opened before the release passes."""
    plan = pinned(registry['plan'], registry['plan_sha256'])
    require(plan['prepared_only'] is True and plan['model_grant'] is False and
            plan['all_first_attempts'] == 48 and type(plan['all_first_attempts']) is int and
            [a['name'] for a in plan['arms']] == NAMES and
            [a['budget'] for a in plan['arms']] == [0, 1024, 0] and
            all(type(a['budget']) is int for a in plan['arms']), 'fixed three-arm plan')
    require(len(plan['assets']) == registry['asset_count'], 'complete frozen static asset inventory')
    release = pinned(release_path, release_sha)
    guard = release['guard_result']
    require(type(guard) is dict and guard['status'] == 'complete' and
            type(guard['child_returncode']) is int and guard['child_returncode'] in (0, 3) and
            release['lock_reacquired'] is True and type(release['live_model_processes']) is list and
            release['live_model_processes'] == [], 'actual complete released ROOT model job')
    require(number(release['time'], 'release time') > 0 and
            integer(release['normal_pressure_samples'], 'normal') > 0 and
            integer(release['bad_pressure_samples'], 'bad') == 0 and
            number(release['swap_growth_gb'], 'swap growth') == 0, 'normal released pressure')
    pressure = Path(plan['pressure'])
    require(load(pressure / 'result.json') == guard, 'actual guard/release agreement')
    command = load(pressure / 'command.json')
    expected = [registry['plan_sha256'] if x == '<PLAN_SHA256>' else x
                for x in plan['guard_command_template']]
    require(command['argv'] == expected[4:], 'exact frozen child command')
    initial = command['initial']
    require(integer(initial['level'], 'initial level') == 1 and
            number(initial['swap_gb'], 'initial swap') >= 0, 'normal initial pressure')
    previous = number(initial['time'], 'initial time')
    samples = jsonl(pressure / 'pressure.jsonl')
    require(len(samples) == release['normal_pressure_samples'], 'pressure sample balance')
    for sample in samples:
        stamp = number(sample['time'], 'sample time')
        require(integer(sample['level'], 'level') == 1 and sample['reason'] is None and
                integer(sample['consecutive_bad'], 'bad') == 0 and previous <= stamp <= release['time'] and
                number(sample['swap_gb'], 'swap') == initial['swap_gb'], 'every sample normal/ordered/no swap')
        previous = stamp
    # Outcome summary and audit are deliberately read only after actual release validation.
    summary = pinned(summary_path, summary_sha)
    audit = pinned(audit_path, audit_sha)
    require(file_info(registry['audit_source'])['sha256'] == registry['audit_source_sha256'],
            'independent audit source drift')
    require(summary['status'] == 'COMPLETE_FIRST_ATTEMPTS' and
            summary['plan_sha256'] == registry['plan_sha256'] and summary['release_sha256'] == release_sha and
            type(summary['guard_child_returncode']) is int and summary['guard_child_returncode'] == guard['child_returncode'] and
            summary['all_first_attempts'] == 48 and type(summary['all_first_attempts']) is int and
            summary['native_before_and_after_exact_full_typed_fields'] is True and
            summary['fixed_unused16_cases'] is True and summary['policy_chosen_before_budget_outcomes'] is True,
            'complete original producer summary')
    require(audit['status'] == AUDIT_STATUS and audit['plan_sha256'] == registry['plan_sha256'] and
            audit['release_sha256'] == release_sha and audit['producer_summary_sha256'] == summary_sha and
            audit['source_sha256'] == registry['audit_source_sha256'] and audit['all48_attempts'] is True and
            audit['model_or_helper_or_guard_executed'] is False, 'independent pinned audit PASS chain')
    require(len(summary['arms']) == len(audit['counts']) == 3, 'three complete audited counts')
    for arm, count, budget in zip(summary['arms'], audit['counts'], [0, 1024, 0]):
        require(type(arm['budget']) is int and arm['budget'] == budget and
                type(arm['cases']) is int and arm['cases'] == 16 and len(arm['per_case']) == 16,
                'sixteen cases per arm')
        for key in COUNTS:
            require(integer(arm[key], key) == integer(count[key], key), 'audit/producer count agreement: ' + key)
        require(type(arm['producer_returncode']) is int and
                arm['producer_returncode'] == (3 if arm['invalid'] else 0), 'INVALID-only producer status retained')
    require(guard['child_returncode'] == (3 if any(a['invalid'] for a in summary['arms']) else 0),
            'aggregate INVALID-only rc3 retained')
    for key in ('native_correct_budget_failed', 'native_failed_budget_correct'):
        require(integer(summary['paired'][key], key) == integer(audit[key], key), 'both discordances retained')
    return plan, release, summary, audit


def validate_attempt_inventory(plan, summary):
    """Bind all files to already-scored cases; no answers are rescored here."""
    fixture = load(plan['fixture'])['cases']
    require(len(fixture) == 16 and len({x['case'] for x in fixture}) == 16, 'sixteen unique frozen cases')
    status = load(Path(plan['output']) / 'complete.json')
    require(status['status'] == 'COMPLETE' and status['arms'] == NAMES and
            type(status['returncode']) is int and status['returncode'] == summary['guard_child_returncode'],
            'all three original first-attempt processes complete')
    for arm, report in zip(plan['arms'], summary['arms']):
        process = load(Path(plan['output']) / (arm['name'] + '-process.json'))
        require(set(process) == {'returncode'} and type(process['returncode']) is int and
                process['returncode'] == report['producer_returncode'], 'original process status')
        rows = jsonl(Path(arm['results_dir']) / 'results.jsonl')
        expected = [x['case'] for x in fixture]
        require(len(rows) == 16 and [x['case'] for x in rows] == expected and
                [x['case'] for x in report['per_case']] == expected, 'all sixteen ordered first attempts')
        used = set()
        for row, saved, case in zip(rows, report['per_case'], fixture):
            require(row['prompt'] == case['prompt'] and row['expected'] == case['answer'], 'frozen input/gold identity')
            for key in ('status', 'stopped', 'cap_hit', 'generated_tokens', 'reasoning_tokens', 'parsed_answer'):
                require(type(row[key]) is type(saved[key]) and row[key] == saved[key], 'summary/row identity: ' + key)
            require(row['status'] in ('CORRECT', 'INCORRECT', 'INVALID') and type(row['stopped']) is bool and
                    type(row['cap_hit']) is bool, 'retained typed first-attempt status')
            require(integer(row['generated_tokens'], 'generated') <= 2048 and
                    integer(row['reasoning_tokens'], 'reasoning') <= row['generated_tokens'], 'bounded work counters')
            for key, digest_key in (('token_trace', 'token_sha256'), ('prompt_trace', 'prompt_sha256'), ('raw_trace', 'raw_sha256')):
                require(row[key] not in used, 'trace reused by multiple cases or roles')
                used.add(row[key])
                require(file_info(local(arm['results_dir'], row[key]))['sha256'] == saved[digest_key],
                        'original full trace identity: ' + key)


def tree_files(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'ordinary evidence directory')
    files = []
    for current, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            require(not (Path(current) / name).is_symlink(), 'evidence directory alias')
        for name in names:
            path = Path(current) / name
            require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), 'evidence special file/alias')
            files.append(path)
    return sorted(files)


def physical_index(root):
    root = Path(root)
    files = {str(p.relative_to(root)): file_info(p) for p in tree_files(root)
             if p != root / 'manifest.json'}
    directories = {str(p.relative_to(root)): stat.S_IMODE(p.stat().st_mode)
                   for p in sorted(root.rglob('*')) if p.is_dir()}
    return dict(files=files, directories=directories)


def archive(registry, destination, release_path, release_sha, summary_path, summary_sha,
            audit_path, audit_sha, root_archive_granted=False):
    require(root_archive_granted is True, 'separate explicit ROOT post-release archive grant required')
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), 'exclusive fresh archive destination')
    plan, release, summary, audit = gate(registry, release_path, release_sha, summary_path, summary_sha, audit_path, audit_sha)
    check_runtime(registry)
    validate_attempt_inventory(plan, summary)
    # Every original prepared/run/pressure file and every sibling arm file is retained.
    sources = {}
    roots = [('prepared', Path(registry['plan']).parent)] + [(a['name'], Path(a['results_dir'])) for a in plan['arms']]
    for _, root in roots + [('procedure', Path(__file__).parent)]:
        require(not destination.resolve().is_relative_to(root.resolve()) and
                not root.resolve().is_relative_to(destination.resolve()), 'destination/source tree overlap')
    for label, root in roots:
        for path in tree_files(root):
            sources[str(path)] = (str(Path('original') / label / path.relative_to(root)), None, None)
    for i, (path, expected) in enumerate(registry['static_files'].items()):
        sources.setdefault(path, (str(Path('static') / ('%03d-' % i + Path(path).name)), expected, expected.get('alias')))
    for label, path in [('release.json', release_path), ('producer-summary.json', summary_path), ('independent-audit.json', audit_path),
                        ('independent-audit-source.py', registry['audit_source']), ('runtime100-authority.json', registry['runtime_authority'])]:
        sources.setdefault(str(path), (str(Path('gates') / label), None, None))
    for path in tree_files(Path(__file__).parent):
        sources.setdefault(str(path), (str(Path('procedure') / path.relative_to(Path(__file__).parent)), None, None))
    # Frozen plan assets are verified even when already included in a prepared tree.
    for path, expected in plan['assets'].items():
        registered = registry['static_files'][path]
        info = file_info(path, registered.get('alias'))
        require(info['sha256'] == expected['sha256'] and info['bytes'] == expected['bytes'] and
                info['mode'] == registered['mode'], 'all original static pins and modes')
    require(not any(destination.resolve() == Path(p).resolve() for p in sources), 'destination/source file overlap')
    destination.mkdir(parents=False)
    references, copied = [], []
    try:
        for source, (relative, expected, alias) in sorted(sources.items()):
            info = file_info(source, alias)
            if expected:
                require(all(info[k] == expected[k] for k in ('sha256', 'bytes', 'mode')), 'frozen static evidence drift')
            record = dict(source=source, **info)
            if alias:
                record['source_alias'] = alias
            if info['bytes'] > LIMIT:
                references.append(record)
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), 'exclusive copied file')
            shutil.copy2(source, target, follow_symlinks=True)
            require(file_info(target) == info, 'copy literal bytes/mode identity')
            copied.append(dict(archive_path=relative, **record))
        put(destination / 'external-payload-references.json', dict(threshold_bytes=LIMIT, strictly_greater=True, files=references))
        put(destination / 'archive-observations.json', dict(
            status='RELEASED_EVIDENCE_ARCHIVE', plan_sha256=registry['plan_sha256'], release_sha256=release_sha,
            producer_summary_sha256=summary_sha, independent_audit_sha256=audit_sha,
            original_guard_child_returncode=summary['guard_child_returncode'], unique_cases=16, first_attempts=48,
            preserved_original_scores=True, rescored=False, main_runtime100_unchanged=True,
            limitations='One fixed set and seed. Original wrong/INVALID/caps and both paired discordances retained. Raw artifacts are text bytes, not vocabulary logits. No cross-prefix KL/NLL, equal-quality, quantization-causality, speed, SSD or default-promotion claim.'))
        check_runtime(registry)
        index = physical_index(destination)
        manifest = dict(schema='math16-released-archive-v1', copied=copied, external_references=references, **index)
        put(destination / 'manifest.json', manifest)
        require(physical_index(destination) == index, 'complete physical index excluding only top selfmanifest')
        return manifest
    except Exception as error:
        # Keep any partial copy and the first refusal; never retry into this destination.
        put(destination / 'ARCHIVE-REFUSAL.json', dict(status='REFUSED', exception=type(error).__name__, message=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-sha256', required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--producer-summary', type=Path, required=True)
    parser.add_argument('--producer-summary-sha256', required=True)
    parser.add_argument('--independent-audit', type=Path, required=True)
    parser.add_argument('--independent-audit-sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--root-archive-granted', action='store_true')
    args = parser.parse_args()
    registry = pinned(Path(__file__).with_name('source-authorities.json'), args.registry_sha256)
    require(Path(args.destination).resolve().is_relative_to(Path(registry['archive_workspace']).resolve()),
            'archive stays in owned persistent workspace')
    result = archive(registry, args.destination, args.release, args.release_sha256, args.producer_summary,
                     args.producer_summary_sha256, args.independent_audit, args.independent_audit_sha256,
                     args.root_archive_granted)
    print(json.dumps(dict(status='ARCHIVED', physical_files=len(result['files']), external_refs=len(result['external_references']))))


if __name__ == '__main__':
    main()
