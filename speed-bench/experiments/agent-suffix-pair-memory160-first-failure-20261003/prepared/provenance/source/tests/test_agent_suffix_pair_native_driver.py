"""Model-free parser/authority fixtures only; never invokes a compiler/model.

Run only after ROOT's focused CPU grant. Small synthetic raw vectors exercise
the real driver parser without loading model metadata or runtime APIs.
"""
import copy
import errno
import importlib.util
import json
from pathlib import Path
import shlex
import struct
import tempfile
import unittest
from unittest.mock import patch,mock_open

SOURCE=Path(__file__).parents[1]/'speed-bench/ideas/run_agent_suffix_pair_native.py'
spec=importlib.util.spec_from_file_location('suffix_native_driver',SOURCE)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


class DriverFixtures(unittest.TestCase):
    def setUp(self):
        artifacts=Path(__file__).parents[1]/'speed-bench/ideas/artifacts';artifacts.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='suffix-native-cpu-',dir=artifacts)
        self.p=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def input_bytes(self):
        user=100;old=[1]*2114;old[1409]=2;old[1986]=user;edited=old[:1409]+old[1474:];ids=old+edited+[3]*16
        return struct.pack('<8I',0x41535431,1,129280,user,2114,2049,16,0)+struct.pack('<'+str(len(ids))+'i',*ids)
    def summary(self,mode='reuse'):
        flags=mode=='reuse'
        return dict(global_peak_observed=False,performed_phase_admissions_checked=True,mode=mode,prefix=1409,shift=65,span=512,tail=128,decode=16,user_id=100,latest_user_start=1921,app_budget_bytes=128<<20,fresh_latest_user=True,source_move_no_reload=flags,precancel_preserves_state=flags,postmove_cancel_rolls_back=flags,canonical_and_diagnostic_gates=flags,text_only_save_state_unchanged=flags,transaction_peak_bytes_including_reserves=60<<20 if flags else 0,mean_kl=0.,mean_tv=0.,sum_delta_nll=0.,argmax_flips=0,bit_equal_rows=17 if flags else 0,reuse_ms_diagnostic=0.)
    def raw_bytes(self):
        return struct.pack('<8I',0x41534731,3,16,1409,65,512,128,0)+struct.pack('<16i',*([1]*16))+struct.pack('<57f',*([0.]*57))
    def test_complete_literal_input(self):
        p=self.p/'input.bin';p.write_bytes(self.input_bytes());r=d.read_input(p)
        self.assertEqual(r['latest_user_start'],1921);self.assertEqual(r['edited_ids'],r['old_ids'][:1409]+r['old_ids'][1474:]);self.assertEqual(r['teacher_ids'],[3]*16)
    def test_input_missing_trailing_wrong_marker_wrong_splice(self):
        data=self.input_bytes()
        mutations=[data[:-1],data+b'0']
        for offset,value in [(12,129280),(32+1986*4,101),(32+(2114+4)*4,5),(32+8,100),(32+2114*4, -1)]:
            changed=bytearray(data);struct.pack_into('<i',changed,offset,value);mutations.append(changed)
        for data in mutations:
            p=self.p/'bad.bin';p.write_bytes(data)
            with self.assertRaises(ValueError):d.read_input(p)
    def test_summary_literal_flags_and_finite_bounds(self):
        d.validate_summary(self.summary(),'reuse',100);d.validate_summary(self.summary('native'),'native',100)
        for key,value in [('source_move_no_reload',1),('fresh_latest_user',False),('user_id',True),('transaction_peak_bytes_including_reserves',False),('transaction_peak_bytes_including_reserves',(128<<20)+1),('mean_kl',float('nan')),('mean_tv',-1),('argmax_flips',18),('bit_equal_rows','17')]:
            s=self.summary();s[key]=value
            with self.assertRaises(ValueError):d.validate_summary(s,'reuse',100)
    def test_raw_stream_shape_identity_finite(self):
        p=self.p/'raw.bin';p.write_bytes(self.raw_bytes())
        with patch.object(d,'VOCAB',3):
            r=d.read_raw(p,False,[1]*16);self.assertEqual((r['source_rows'],r['branch_rows']),(2,17))
            for data in [self.raw_bytes()[:-1],self.raw_bytes()+b'0']:
                p.write_bytes(data)
                with self.assertRaises(ValueError):d.read_raw(p,False,[1]*16)
            data=bytearray(self.raw_bytes());struct.pack_into('<f',data,96,float('inf'));p.write_bytes(data)
            with self.assertRaises(ValueError):d.read_raw(p,False,[1]*16)
    def test_raw_mode_and_teacher_mismatch(self):
        p=self.p/'raw.bin';p.write_bytes(self.raw_bytes())
        with patch.object(d,'VOCAB',3):
            with self.assertRaises(ValueError):d.read_raw(p,True,[1]*16)
            with self.assertRaises(ValueError):d.read_raw(p,False,[2]*16)
    def csv_text(self):
        return 'FROZEN_INPUT_READY\nstep,target,delta_nll,kl,tv,argmax_flip,bit_equal\n'+''.join(f'{i},{3 if i<16 else -1},0,0,0,0,1\n' for i in range(17))
    def test_csv_order_completeness_numeric_and_summary(self):
        p=self.p/'stdout';p.write_text(self.csv_text());self.assertEqual(d.quality_csv(p,[3]*16,self.summary())['rows'],17)
        bad=[self.csv_text().rsplit('\n',2)[0]+'\n',self.csv_text().replace('0,3,0,0,0,0,1','1,3,0,0,0,0,1'),self.csv_text().replace('0,3,0,0,0,0,1','0,4,0,0,0,0,1'),self.csv_text().replace('0,3,0,0,0,0,1','0,3,nan,0,0,0,1'),self.csv_text().replace('0,3,0,0,0,0,1','0,3,0,0,0,2,1')]
        for text in bad:
            p.write_text(text)
            with self.assertRaises(ValueError):d.quality_csv(p,[3]*16,self.summary())
        p.write_text(self.csv_text());s=self.summary();s['sum_delta_nll']=1
        with self.assertRaises(ValueError):d.quality_csv(p,[3]*16,s)
    def test_model_hash_rejected_before_open(self):
        with patch.object(Path,'open',side_effect=AssertionError('must not open')):
            with self.assertRaises(ValueError):d.sha(self.p/'model.gguf')
    def preflight_plan(self):
        names=['probe','caller-build-manifest.json','run_agent_suffix_pair_native.py','corpus.txt','source/ds4_agent.c','source/ds4.h','source/ds4.c','source/ds4_suffix_pair_tape.h','source/speed-bench/agent_suffix_pair_native_fixture.c']+['metal/'+v for v in d.SHADERS.values()]
        files={}
        for name in names:
            p=self.p/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode());files[name]=dict(bytes=p.stat().st_size,sha256=d.sha(p))
        (self.p/'probe').chmod(0o755)
        plan=dict(geometry=d.GEOMETRY,arms=[list(x) for x in d.ARMS],policy=d.POLICY,model_from_prior=d.MODEL,input_builder=d.INPUT_BUILDER,shader_env={k:str(self.p/'metal'/v) for k,v in d.SHADERS.items()},files=files)
        d.put(self.p/'manifest.json',plan);digest=d.sha(self.p/'manifest.json');d.put(self.p/'prepared-command.json',dict(manifest_sha256=digest,command=d.command(self.p,digest),cwd=str(d.ROOT)))
        return digest
    def test_preflight_explicit_authority_assets_and_execute_mode(self):
        digest=self.preflight_plan();real_sha=d.sha
        def pinned(p):
            return d.GUARD_SHA if p==d.GUARD else d.PYTHON_SHA if p==d.PYTHON else real_sha(p)
        with patch.object(d,'sha',side_effect=pinned),patch.object(d,'model_identity',side_effect=AssertionError('model metadata forbidden')) as model:
            d.verify(self.p,digest)
            with self.assertRaises(ValueError):d.verify(self.p,'0'*64)
            (self.p/'probe').chmod(0o644)
            with self.assertRaisesRegex(ValueError,'not executable'):d.verify(self.p,digest)
            (self.p/'probe').chmod(0o755);(self.p/'metal'/next(iter(d.SHADERS.values()))).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'asset differs'):d.verify(self.p,digest)
            model.assert_not_called()
    def guard(self,actual=None,held=True,same=True):
        args=actual or d.command(self.p,'a'*64)
        stack=[patch.object(d.subprocess,'check_output',return_value=shlex.join(args)),patch.object(d.os.path,'samefile',return_value=same),patch.object(d,'sha',side_effect=lambda p:d.PYTHON_SHA if p==d.PYTHON else d.GUARD_SHA),patch.object(Path,'open',mock_open()),patch.object(d.fcntl,'flock',side_effect=OSError(errno.EAGAIN,'held') if held else None)]
        return stack
    def test_guard_actual_framework_exact_suffix_and_held_lock(self):
        from contextlib import ExitStack
        with ExitStack() as x:
            for p in self.guard():x.enter_context(p)
            d.require_guard(self.p,'a'*64)
        for args,held,same in [(d.command(self.p,'b'*64),True,True),(None,False,True),(None,True,False)]:
            with ExitStack() as x:
                for p in self.guard(args,held,same):x.enter_context(p)
                with self.assertRaises(RuntimeError):d.require_guard(self.p,'a'*64)
    def test_guard_ps_absolute_and_bounded(self):
        from contextlib import ExitStack
        with ExitStack() as x:
            mocks=[x.enter_context(p) for p in self.guard()];d.require_guard(self.p,'a'*64)
            self.assertEqual(mocks[0].call_args.args[0][0],'/bin/ps');self.assertEqual(mocks[0].call_args.kwargs['timeout'],2)
    def test_run_guard_precedes_model_metadata(self):
        with patch.object(d,'verify',return_value={}),patch.object(d,'require_guard',side_effect=RuntimeError('not guarded')),patch.object(d,'model_identity',side_effect=AssertionError('model touched')) as model:
            with self.assertRaises(RuntimeError):d.run(self.p,'a'*64)
            model.assert_not_called()
    def test_native_full_record_identity(self):
        a=self.p/'a';b=self.p/'b';suffixes=['.prefix.state.bin','.source.endpoint.state.bin','.source.full.state.bin','.fresh.endpoint.state.bin','.fresh.full.state.bin','.fresh.endpoint.logits.bin']
        for s in suffixes:Path(str(a)+s).write_bytes(b'fixed'+s.encode());Path(str(b)+s).write_bytes(b'fixed'+s.encode())
        r=dict(source_sha256='source',branch_sha256='branch',teacher_ids=[3]*16)
        self.assertEqual(d.native_record(a,r),d.native_record(b,r));Path(str(b)+suffixes[0]).write_bytes(b'changed');self.assertNotEqual(d.native_record(a,r),d.native_record(b,r))
    def test_native_failure_kept_and_no_second_arm(self):
        plan={'scope':'fixture','shader_env':{}}
        with patch.object(d,'verify',return_value=plan),patch.object(d,'require_guard'),patch.object(d,'model_identity',return_value=d.MODEL),patch.object(d.subprocess,'run',return_value=type('P',(),{'returncode':3})()) as launch:
            with self.assertRaises(RuntimeError):d.run(self.p,'a'*64)
            self.assertEqual(launch.call_count,1)
            failure=json.loads((self.p/'run/failure.json').read_text());self.assertEqual(failure['completed_arms'],[]);self.assertTrue(failure['no_retries']);self.assertEqual(json.loads((self.p/'run/native-before-exit.json').read_text())['returncode'],3)


if __name__=='__main__':unittest.main()
