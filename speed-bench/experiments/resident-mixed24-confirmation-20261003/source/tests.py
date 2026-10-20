"""CPU fixtures only; never reads current pilot outcomes or launches a helper."""
import copy
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
PATH = Path(__file__).resolve().parents[1]/'speed-bench/ideas/analyze_resident_mixed24.py'
spec = importlib.util.spec_from_file_location('mixed24_audit_fixture', PATH)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
import numpy as np


def scalar(base, candidate, target):
    b, c = list(map(float, base)), list(map(float, candidate))
    bz = max(b)+math.log(math.fsum(math.exp(x-max(b)) for x in b))
    cz = max(c)+math.log(math.fsum(math.exp(x-max(c)) for x in c))
    p, q = [math.exp(x-bz) for x in b], [math.exp(x-cz) for x in c]
    return dict(base_nll=bz-b[target], raw_nll=cz-c[target],
        delta_raw_nll=cz-c[target]-(bz-b[target]),
        raw_kl=math.fsum(x*((bl-bz)-(cl-cz)) for x,bl,cl in zip(p,b,c)),
        raw_tv=.5*math.fsum(abs(x-y) for x,y in zip(p,q)),
        argmax_flip=int(b.index(max(b)) != c.index(max(c))))


def fixture(directory, base=None, candidate=None):
    base = [np.array(x, dtype=np.float32) for x in (base or [[0,1,-2], [3,0,1]])]
    candidate = [np.array(x, dtype=np.float32) for x in (candidate or [[.5,.8,-2], [2,0,1]])]
    paths = [Path(directory)/name for name in ('base.raw', 'candidate.raw')]
    rows = []
    for vectors, path in zip((base,candidate),paths):
        with path.open('wb') as f:
            f.write(a.HEADER.pack(a.MAGIC,3,1,2))
            for step, vector in enumerate(vectors):
                f.write(a.KEY.pack(0,step,step,7,99))
                f.write(vector.tobytes())
    for step,(b,c) in enumerate(zip(base,candidate)):
        m=scalar(b,c,step)
        row=dict(case='fixture',step=str(step),target=str(step),prompt_tokens='7',decode_ms='0')
        row.update({k:repr(v) for k,v in m.items()})
        row.update(policy_nll=row['raw_nll'],delta_policy_nll=row['delta_raw_nll'],
            policy_kl=row['raw_kl'],policy_tv=row['raw_tv'])
        rows.append(row)
    return paths, rows


class AuditFixtures(unittest.TestCase):
    def test_release_actual_metadata(self):
        path = a.PREPARED/'root-release-proof.json'
        self.assertEqual(a.sha(path),a.RELEASE_SHA)
        self.assertEqual(a.validate_release(a.load(path))['normal_pressure_samples'],299)

    def test_release_rejects_missing_typed_bad_or_active(self):
        good=a.load(a.PREPARED/'root-release-proof.json')
        bad=[]
        for field in good:
            value=copy.deepcopy(good);del value[field];bad.append(value)
        for key,value in [('lock_reacquired',1),('live_model_processes',['probe']),
                ('normal_pressure_samples',True),('bad_pressure_samples',1),
                ('swap_growth_gb',.1),('time',float('inf')),('peak_wired_compressor_gib',False)]:
            item=copy.deepcopy(good);item[key]=value;bad.append(item)
        for key,value in [('status','failed'),('child_returncode',True),('child_returncode',3)]:
            item=copy.deepcopy(good);item['guard_result'][key]=value;bad.append(item)
        for item in bad:
            with self.subTest(item=item),self.assertRaises(ValueError):a.validate_release(item)

    def test_fp64_against_independent_scalar(self):
        for b,c in [([0,1,-2],[.5,.8,-2]),([10,-1000,2],[9,-1000,3]),([1,1,0],[0,1,1])]:
            b=np.array(b,dtype=np.float32);c=np.array(c,dtype=np.float32)
            expected=scalar(b,c,1);actual=a.fp64_metrics(b,c,1,np)
            for key in expected:self.assertAlmostEqual(actual[key],expected[key],places=13)

    def test_native_exact_and_ties(self):
        x=np.array([1,1,-2],dtype=np.float32)
        self.assertEqual(a.fp64_metrics(x,x,0,np)['argmax_flip'],0)
        for key in ('delta_raw_nll','raw_kl','raw_tv'):
            self.assertEqual(a.fp64_metrics(x,x,0,np)[key],0)

    def test_full_vector_nonfinite_not_just_target(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            x=np.array([1,0,value],dtype=np.float32)
            with self.assertRaises(ValueError):a.fp64_metrics(x,np.zeros(3,dtype=np.float32),0,np)

    def test_absolute_tolerance_boundary(self):
        a.close(a.TOLERANCE,0,'fixture')
        with self.assertRaises(ValueError):a.close(a.TOLERANCE*1.0001,0,'fixture')
        with self.assertRaises(ValueError):a.close(float('nan'),0,'fixture')

    def test_stream_valid_and_bit_counter(self):
        with tempfile.TemporaryDirectory() as d:
            paths,rows=fixture(d)
            summary,scored=a.stream_scores(*paths,rows,np,cases=1,steps=2,vocab=3)
            self.assertEqual(summary['rows'],2);self.assertEqual(summary['bit_equal_rows'],0)
            self.assertEqual(a.summarize_rows(scored)['per_prompt']['fixture']['tokens'],2)
            _,same_rows=fixture(d,candidate=[[0,1,-2],[3,0,1]])
            summary,_=a.stream_scores(paths[0],paths[0],
                same_rows,np,cases=1,steps=2,vocab=3)
            self.assertEqual(summary['bit_equal_rows'],2)

    def test_stream_truncation_trailing_wrong_header_and_duplicate_keys(self):
        for mutate in ('truncated','trailing','header','duplicate','fingerprint'):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as d:
                paths,rows=fixture(d);data=bytearray(paths[1].read_bytes())
                if mutate=='truncated':data=data[:-1]
                if mutate=='trailing':data+=b'x'
                if mutate=='header':data[:a.HEADER.size]=a.HEADER.pack(a.MAGIC,3,1,3)
                offset=a.HEADER.size+a.KEY.size+12
                if mutate=='duplicate':data[offset:offset+a.KEY.size]=a.KEY.pack(0,0,1,7,99)
                if mutate=='fingerprint':data[offset:offset+a.KEY.size]=a.KEY.pack(0,1,1,7,100)
                paths[1].write_bytes(data)
                with self.assertRaises(ValueError):a.stream_scores(*paths,rows,np,cases=1,steps=2,vocab=3)

    def test_stream_wrong_teacher_row_and_metric(self):
        for field,value in [('target','2'),('step','1'),('raw_kl','10'),('policy_nll','10'),('argmax_flip','1')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                paths,rows=fixture(d);rows[0][field]=value
                with self.assertRaises(ValueError):a.stream_scores(*paths,rows,np,cases=1,steps=2,vocab=3)

    def test_strict_json_and_typed_arithmetic(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):a.strict_json(text)
        with self.assertRaises(ValueError):a.nested_close({'count':True},{'count':1})
        with self.assertRaises(ValueError):a.nested_close({'a':0.},{'a':1.})

    def test_pressure_exact_command_samples_order_and_peak(self):
        initial=dict(time=2.,level=1,swap_gb=0.,physical_gb=128.,free_gb=30.,
            file_backed_gb=10.,wired_gb=40.,compressor_gb=4.)
        samples=[dict(initial,time=3.,reason=None,consecutive_bad=0)]
        command=dict(argv=['frozen'],initial=initial,polling_seconds=1,
            stop_rule='two consecutive adverse samples; large RAM use alone is allowed')
        release=dict(normal_pressure_samples=1,time=4.,peak_wired_compressor_gib=44.)
        a.validate_pressure(command,samples,release,['frozen'],1.)
        for field,value in [('level',2),('time',.5),('swap_gb',.1),('reason','bad'),('consecutive_bad',True)]:
            bad=copy.deepcopy(samples);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                a.validate_pressure(command,bad,release,['frozen'],1.)
        with self.assertRaises(ValueError):a.validate_pressure(command,samples,release,['other'],1.)

    def test_no_launch_or_model_identity_api(self):
        source=PATH.read_text()
        self.assertNotIn('subprocess',source)
        self.assertNotIn('raw.identity(',source)
        self.assertLess(source.index('release = release_gate()'),source.index("completed = load(run/'summary.json')"))
        self.assertEqual(a.TOLERANCE,1e-8)
        self.assertEqual((a.CASES,a.STEPS,a.ROWS,a.VOCAB),(24,32,768,129280))


if __name__ == '__main__':
    unittest.main()
