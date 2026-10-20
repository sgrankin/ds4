#!/usr/bin/env python3
"""Bounded immutable-revision HTTP ranges; no whole-model download."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = 'antirez/deepseek-v4.1-flash-gguf'
REVISION = '9dcee51692e6540f697141fc98b097e3c96492a3'
FILE = 'DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf'
SIZE = 49353784544
URL = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}'
MAX_RANGE = 16 * 1024 * 1024


def fetch(start, count, name):
    if start < 0 or not 0 < count <= MAX_RANGE or start + count > SIZE:
        raise ValueError('invalid bounded range')
    path = HERE / name
    if path.exists() or path.with_suffix(path.suffix + '.json').exists():
        raise FileExistsError(path)
    end = start + count - 1
    req = urllib.request.Request(URL, headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
        'User-Agent': 'DS4-bounded-lossless-codec-pilot/1'})
    with urllib.request.urlopen(req, timeout=60) as response:
        actual = response.headers.get('Content-Range', '')
        expected = f'bytes {start}-{end}/{SIZE}'
        if response.status != 206 or actual != expected:
            raise RuntimeError(f'expected206 {expected}, got {response.status} {actual}')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise RuntimeError('encoded response is not a byte range')
        content_length = response.headers.get('Content-Length')
        if content_length is not None and int(content_length) != count:
            raise RuntimeError('wrong Content-Length')
        data = response.read(count + 1)
        if len(data) != count:
            raise RuntimeError(f'wrong response extent {len(data)} != {count}')
        proof = {'repo': REPO, 'revision': REVISION, 'file': FILE,
                 'url': URL, 'status': response.status, 'content_range': actual,
                 'offset': start, 'bytes': len(data),
                 'sha256': hashlib.sha256(data).hexdigest()}
    path.write_bytes(data)
    path.with_suffix(path.suffix + '.json').write_text(json.dumps(proof, indent=2) + '\n')
    return data


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def get(self, n):
        if self.pos + n > len(self.data):
            raise ValueError(f'header needs more than {len(self.data)} bytes at {self.pos}+{n}')
        b = self.data[self.pos:self.pos+n]
        self.pos += n
        return b

    def scalar(self, fmt):
        return struct.unpack('<' + fmt, self.get(struct.calcsize('<' + fmt)))[0]

    def string(self):
        return self.get(self.scalar('Q')).decode('utf8')

    def value(self, typ):
        if typ == 8:
            return self.string()
        if typ == 9:
            elem, count = self.scalar('I'), self.scalar('Q')
            for _ in range(count):
                self.value(elem)
            return {'array_type': elem, 'count': count}
        return self.scalar({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[typ])


def directory(data):
    r = Reader(data)
    if r.get(4) != b'GGUF' or r.scalar('I') != 3:
        raise ValueError('expected GGUFv3')
    nt, nk = r.scalar('Q'), r.scalar('Q')
    meta = {}
    for _ in range(nk):
        key = r.string()
        meta[key] = r.value(r.scalar('I'))
    tensors = []
    for _ in range(nt):
        name, nd = r.string(), r.scalar('I')
        shape = [r.scalar('Q') for _ in range(nd)]
        typ, offset = r.scalar('I'), r.scalar('Q')
        tensors.append({'name': name, 'shape': shape, 'type': typ, 'relative_offset': offset})
    align = meta.get('general.alignment', 32)
    data_offset = (r.pos + align - 1) // align * align
    for i, t in enumerate(tensors):
        t['absolute_offset'] = data_offset + t['relative_offset']
        next_offset = min((x['relative_offset'] for x in tensors
                           if x['relative_offset'] > t['relative_offset']), default=SIZE-data_offset)
        t['extent_until_next_tensor'] = next_offset - t['relative_offset']
    return {'repo': REPO, 'revision': REVISION, 'file': FILE, 'file_bytes': SIZE,
            'header_bytes': r.pos, 'data_offset': data_offset,
            'metadata': meta, 'tensors': tensors}


def samples():
    d = json.loads((HERE / 'directory.json').read_text())
    # Fixed before inspecting any scale values: two layers, all three projections,
    # and early/middle/late expert IDs. Each range is one complete expert matrix.
    choices = [(0, 'gate', 0), (0, 'up', 191), (0, 'down', 383),
               (5, 'gate', 383), (5, 'up', 0), (5, 'down', 191)]
    plan = []
    for layer, part, expert in choices:
        name = f'blk.{layer}.ffn_{part}_exps.weight'
        t = next(t for t in d['tensors'] if t['name'] == name)
        if t['type'] != 39 or len(t['shape']) != 3 or t['shape'][2] != 384:
            raise ValueError('expected native MXFP4 expert tensor')
        per_expert = t['shape'][0] * t['shape'][1] // 32 * 17
        if per_expert * 384 > t['extent_until_next_tensor']:
            raise ValueError('expert tensor exceeds directory extent')
        plan.append({'tensor': t, 'expert': expert, 'bytes': per_expert,
                     'offset': t['absolute_offset'] + expert * per_expert,
                     'sample': f'layer{layer}-{part}-expert{expert}.mxfp4'})
    with (HERE / 'sample-plan.json').open('x') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    for item in plan:
        fetch(item['offset'], item['bytes'], item['sample'])
        print(item['sample'], item['bytes'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('offset', type=int); f.add_argument('bytes', type=int); f.add_argument('name')
    d = sub.add_parser('directory'); d.add_argument('path', type=Path)
    sub.add_parser('samples')
    a = p.parse_args()
    if a.command == 'fetch':
        fetch(a.offset, a.bytes, a.name)
    elif a.command == 'directory':
        out = directory(a.path.read_bytes())
        (HERE / 'directory.json').write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({k:v for k,v in out.items() if k != 'metadata'}, indent=2))
    else:
        samples()
