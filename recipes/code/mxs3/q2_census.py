#!/usr/bin/env python3
"""Bounded real Q2 header and four fixed scale samples; no model evaluation."""
from collections import Counter
import json
from pathlib import Path
import struct
import range_sample as r

OUT = Path(__file__).resolve().parent / 'q2-census'
r.FILE = 'DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
r.SIZE = 365713686528
r.URL = f'https://huggingface.co/{r.REPO}/resolve/{r.REVISION}/{r.FILE}'
OUT.mkdir(exist_ok=False)
r.HERE = OUT

header = r.fetch(0, 6 * 1024 * 1024, 'header.bin')
directory = r.directory(header)
(OUT / 'directory.json').write_text(json.dumps(directory, indent=2) + '\n')
results = []
requested = len(header)
# Fixed before scale inspection: first 4096 blocks of expert 0 in two layers.
for layer in (0, 20):
    for part, typ, block_bytes in (('gate', 16, 66), ('down', 10, 84)):
        name = f'blk.{layer}.ffn_{part}_exps.weight'
        tensor = next(t for t in directory['tensors'] if t['name'] == name)
        if tensor['type'] != typ or tensor['extent_until_next_tensor'] < 4096 * block_bytes:
            raise ValueError(f'unexpected tensor layout: {tensor}')
        count = 4096 * block_bytes
        requested += count
        if requested > 8 * 1024 * 1024:
            raise ValueError('8 MiB total bound exceeded')
        data = r.fetch(tensor['absolute_offset'], count, f'layer{layer}-{part}.bin')
        fields = {}
        if typ == 16:
            fields['d_f16_bits'] = [struct.unpack_from('<H', data, b)[0]
                                    for b in range(0, count, block_bytes)]
        else:
            fields['d_f16_bits'] = [struct.unpack_from('<H', data, b + 80)[0]
                                    for b in range(0, count, block_bytes)]
            fields['dmin_f16_bits'] = [struct.unpack_from('<H', data, b + 82)[0]
                                       for b in range(0, count, block_bytes)]
            fields['packed_scale_min_bytes'] = [x for b in range(0, count, block_bytes)
                                                 for x in data[b:b+16]]
        census = {}
        for field, values in fields.items():
            groups = [len(set(values[b:b+4096])) for b in range(0, len(values), 4096)]
            histogram = Counter(values)
            census[field] = {'values': len(values), 'unique': len(histogram),
                'unique_per_4096_values': groups,
                'top8': histogram.most_common(8),
                'all_4096_groups_fit_3bit_dictionary': all(k <= 8 for k in groups)}
        results.append({'tensor': tensor, 'expert': 0, 'blocks': 4096,
                        'bytes': count, 'fields': census})
summary = {'revision': r.REVISION, 'file': r.FILE, 'http_bytes_total': requested,
           'maximum_http_bytes': 8 * 1024 * 1024, 'samples': results}
(OUT / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
