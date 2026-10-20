#!/usr/bin/env python3
"""Lossless dictionary+3-bit MXFP4 scale pilot; leaves all 16 code bytes intact."""
import collections
import hashlib
import json
from pathlib import Path
import struct
import time

HERE = Path(__file__).resolve().parent
MAGIC = b'MXS3P001'
HEADER = struct.Struct('<8sQI')


def encode(raw, group_blocks):
    if len(raw) % 17 or group_blocks <= 0:
        raise ValueError('expected complete 17-byte native MXFP4 blocks')
    blocks = len(raw) // 17
    out = bytearray(HEADER.pack(MAGIC, blocks, group_blocks))
    stats = collections.Counter()
    table_bytes = index_bytes = code_bytes = fallback_bytes = 0
    for first in range(0, blocks, group_blocks):
        count = min(group_blocks, blocks - first)
        group = raw[first*17:(first+count)*17]
        scales = group[0::17]
        alphabet = sorted(set(scales))
        stats[len(alphabet)] += 1
        if len(alphabet) > 8:
            out.append(0)
            out.extend(group)
            fallback_bytes += len(group)
            continue
        out.append(len(alphabet))
        out.extend(alphabet)
        table_bytes += len(alphabet)
        lookup = {x:i for i,x in enumerate(alphabet)}
        packed = bytearray((count*3+7)//8)
        for i, scale in enumerate(scales):
            bit = i*3
            value = lookup[scale]
            packed[bit//8] |= (value << (bit%8)) & 255
            if bit%8 > 5:
                packed[bit//8+1] |= value >> (8-bit%8)
        out.extend(packed)
        index_bytes += len(packed)
        for i in range(count):
            out.extend(group[i*17+1:i*17+17])
        code_bytes += count*16
    return bytes(out), {'group_alphabet_histogram': dict(sorted(stats.items())),
                       'groups': sum(stats.values()), 'group_tag_bytes': sum(stats.values()),
                       'dictionary_bytes': table_bytes, 'index_bytes': index_bytes,
                       'unchanged_code_bytes': code_bytes, 'raw_fallback_bytes': fallback_bytes,
                       'raw_fallback_groups': sum(v for k,v in stats.items() if k > 8),
                       'file_header_bytes': HEADER.size}


def decode(packed):
    magic, blocks, group_blocks = HEADER.unpack_from(packed)
    if magic != MAGIC or not group_blocks:
        raise ValueError('wrong pilot header')
    out = bytearray(blocks*17)
    pos = HEADER.size
    for first in range(0, blocks, group_blocks):
        count = min(group_blocks, blocks-first)
        kinds = packed[pos]
        pos += 1
        if kinds == 0:
            out[first*17:(first+count)*17] = packed[pos:pos+count*17]
            pos += count*17
            continue
        if not 1 <= kinds <= 8:
            raise ValueError('wrong scale dictionary extent')
        alphabet = packed[pos:pos+kinds]
        pos += kinds
        size = (count*3+7)//8
        indices = packed[pos:pos+size]
        pos += size
        for i in range(count):
            bit = i*3
            value = indices[bit//8] >> (bit%8)
            if bit%8 > 5:
                value |= indices[bit//8+1] << (8-bit%8)
            dest = (first+i)*17
            out[dest] = alphabet[value & 7]
            out[dest+1:dest+17] = packed[pos+i*16:pos+i*16+16]
        pos += count*16
    if pos != len(packed) or len(out) != blocks*17:
        raise ValueError('wrong pilot payload extent')
    return bytes(out)


def main():
    plan = json.loads((HERE/'sample-plan.json').read_text())
    rows = []
    for item in plan:
        raw = (HERE/item['sample']).read_bytes()
        proof = json.loads((HERE/(item['sample']+'.json')).read_text())
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != item['bytes'] or digest != proof['sha256']:
            raise ValueError('sample differs from range proof')
        scales = raw[0::17]
        alphabet = sorted(set(scales))
        scale_counts = dict(sorted(collections.Counter(scales).items()))
        for group in (256, 4096, len(scales)):
            started = time.perf_counter()
            packed, stats = encode(raw, group)
            encode_s = time.perf_counter()-started
            started = time.perf_counter()
            restored = decode(packed)
            decode_s = time.perf_counter()-started
            if restored != raw:
                raise AssertionError('lossless byte roundtrip failed')
            if group == 4096:
                (HERE/(item['sample']+'.mxs3')).write_bytes(packed)
            row = {'sample': item['sample'], 'tensor_name': item['tensor']['name'],
                   'tensor_type': item['tensor']['type'], 'tensor_shape': item['tensor']['shape'],
                   'expert': item['expert'], 'source_offset': item['offset'],
                   'input_bytes': len(raw), 'packed_bytes': len(packed),
                   'saved_bytes': len(raw)-len(packed),
                   'saved_percent': 100*(1-len(packed)/len(raw)), 'group_blocks': group,
                   'scale_alphabet': alphabet, 'scale_counts': scale_counts,
                   'encode_seconds': encode_s, 'decode_seconds': decode_s,
                   'encode_input_MiB_per_second': len(raw)/(1<<20)/encode_s,
                   'decode_output_MiB_per_second': len(raw)/(1<<20)/decode_s,
                   'roundtrip_byte_equal': True, 'input_sha256': digest,
                   'restored_sha256': hashlib.sha256(restored).hexdigest(),
                   'packed_sha256': hashlib.sha256(packed).hexdigest(), **stats}
            rows.append(row)
            print(item['sample'], group, f"{row['saved_percent']:.5f}%", 'exact', flush=True)
    aggregate = []
    for group_name in ('256', '4096', 'whole_expert'):
        subset = [r for r in rows if r['group_blocks'] ==
                  (r['input_bytes']//17 if group_name == 'whole_expert' else int(group_name))]
        before, after = sum(r['input_bytes'] for r in subset), sum(r['packed_bytes'] for r in subset)
        es, ds = sum(r['encode_seconds'] for r in subset), sum(r['decode_seconds'] for r in subset)
        aggregate.append({'group': group_name, 'samples':len(subset), 'input_bytes': before,
                          'packed_bytes':after, 'saved_bytes': before-after,
                          'saved_percent':100*(1-after/before),
                          'encode_seconds':es, 'decode_seconds':ds,
                          'encode_input_MiB_per_second': before/(1<<20)/es,
                          'decode_output_MiB_per_second': before/(1<<20)/ds,
                          'raw_fallback_groups':sum(r['raw_fallback_groups'] for r in subset),
                          'groups':sum(r['groups'] for r in subset),
                          'dictionary_bytes':sum(r['dictionary_bytes'] for r in subset),
                          'index_bytes':sum(r['index_bytes'] for r in subset),
                          'group_tag_bytes':sum(r['group_tag_bytes'] for r in subset),
                          'file_header_bytes':sum(r['file_header_bytes'] for r in subset),
                          'all_roundtrips_byte_equal':all(r['roundtrip_byte_equal'] for r in subset)})
    result = {'status':'REAL_NATIVE_MXFP4_LOSSLESS_PILOT_PASS',
              'timing_scope':'one CPU Python in-memory encode/decode pass per sample/grain; excludes network/disk; not Metal or model performance',
              'repo':proof['repo'], 'revision':proof['revision'], 'file':proof['file'],
              'total_real_sample_bytes':sum(x['bytes'] for x in plan),
              'aggregate':aggregate, 'rows':rows}
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__ == '__main__':
    main()
