#!/usr/bin/env python3
"""Stage a small BLINK sample with the unmodified Decision Index converters."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('--count', type=int, default=28)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()

    from decision_index.vision import build, rows

    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    items_path = root / 'full/full_blink/items.jsonl'
    if args.rebuild or not items_path.exists():
        # The upstream CLI's partial rebuild calls write_rows for every set.
        # Call its existing per-set builder/converter directly for this subset.
        report = build.build(root, sets=['full_blink'])
        (root / 'build-summary.json').write_text(json.dumps(report, indent=2) + '\n')
        if 'error' in report['full_blink']:
            raise RuntimeError(report['full_blink']['error'])

    items = rows.select('full_blink', rows.load_items(root, 'full_blink'))
    all_rows = [rows.to_row(root, item) for item in items]
    groups = defaultdict(list)
    for row in all_rows:
        # Keep the first screen within our currently supported still codecs.
        if all(Path(name).suffix.lower() in ('.png', '.jpg', '.jpeg') for name in row['images']):
            groups[row['subtask']].append(row)
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).digest())
    selected = []
    while len(selected) < args.count and any(groups.values()):
        for subtask in sorted(groups):
            if groups[subtask] and len(selected) < args.count:
                selected.append(groups[subtask].pop(0))

    subset = root / f'subset-{args.count}'
    subset.mkdir(exist_ok=True)
    link = subset / 'full'
    if not link.exists():
        link.symlink_to('../full', target_is_directory=True)
    (subset / rows.ROWS_FILE).write_text(''.join(rows.dumps(row) + '\n' for row in selected))
    summary = {
        'sample_only': True,
        'source': 'BLINK-Benchmark/BLINK@a3666eb249237ba3d5eca8db21176cc47967e040',
        'source_rows': len(all_rows),
        'rows': len(selected),
        'seed': args.seed,
        'subtasks': dict(Counter(row['subtask'] for row in selected)),
        'images_per_request': dict(Counter(len(row['images']) for row in selected)),
        'subset': str(subset),
        'note': 'Public BLINK sample, native PNG/JPEG cases; no full vision index.'
    }
    (subset / 'sample-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
