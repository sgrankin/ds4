import json
from pathlib import Path
from codec_pilot import encode, decode
cases = []
for name, scales, group in [('partial_group', [0, 1, 42, 255, 118, 127, 119], 3),
                            ('eight_symbols', list(range(8))+[0, 7, 3], 11),
                            ('nine_symbol_fallback', list(range(9))*2+[8], 19)]:
    raw = b''.join(bytes([s])+bytes((i*17+j)%256 for j in range(16)) for i,s in enumerate(scales))
    packed, stats = encode(raw, group)
    assert decode(packed) == raw
    if name == 'nine_symbol_fallback':
        assert stats['raw_fallback_groups'] == 1 and stats['raw_fallback_bytes'] == len(raw)
    cases.append({'name':name,'roundtrip_byte_equal':True,'input_bytes':len(raw),
                  'packed_bytes':len(packed),**stats})
Path(__file__).with_name('sanity-checks.json').write_text(json.dumps(cases,indent=2)+'\n')
print('3 direct codec boundary/fallback checks PASS')
