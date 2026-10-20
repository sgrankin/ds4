"""CPU fixtures for the actual frozen follow-up driver; never model metadata."""
import copy,errno,hashlib,importlib.util,json,os,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('suffix_followup',ROOT/'speed-bench/ideas/run_suffix_pair_tape_followup.py')
d=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(d)
REQUIRED=['baseline-probe','new-probe','build-manifest.json','auxiliary-object-proof.json','baseline-core-proof.json','final-link-cpu-proof.json','run_suffix_pair_tape_followup.py','corpus.txt','source/ds4.c','source/ds4.h','source/ds4_suffix_pair_tape.h','source/speed-bench/suffix_pair_tape_followup_bench.c']
REAL_SHA=d.sha
class CPU(unittest.TestCase):
 def setUp(self):
  directory=ROOT/'speed-bench/ideas/artifacts/suffix-followup-cpu-v1';directory.mkdir(parents=True,exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(prefix='fixture-',dir=directory);self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
  self.model=patch.object(d,'model_identity',side_effect=AssertionError('No model stat/resolve/open allowed')).start();self.addCleanup(patch.stopall)
 def guarded_sha(self,p):
  p=Path(p)
  if p==d.GUARD:return d.GUARD_SHA
  if p==d.PYTHON:return d.PYTHON_SHA
  self.assertNotEqual(p.suffix.lower(),'.gguf');return REAL_SHA(p)
 def fixture(self,case):
  out=self.base/case;out.mkdir()
  names=REQUIRED+[f'metal/shader{i}.metal' for i in range(26)]
  for name in names:
   p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('immutable mock '+name).encode())
  for name in ['new-probe','baseline-probe']:(out/name).chmod(0o755)
  plan=dict(case=case,geometry=d.CASES[case],arms=[list(x) for x in d.ARMS],policy=d.POLICY,model_from_prior=d.MODEL,files={n:dict(bytes=(out/n).stat().st_size,sha256=REAL_SHA(out/n)) for n in names})
  digest=self.pin(out,plan,case);return out,plan,digest
 def pin(self,out,plan,case):
  (out/'manifest.json').write_text(json.dumps(plan));digest=REAL_SHA(out/'manifest.json')
  (out/'prepared-command.json').write_text(json.dumps(dict(manifest_sha256=digest,command=d.command(out,digest,case),cwd=str(d.ROOT))))
  return digest
 def summary(self,geometry,mode='reuse'):
  reused=mode=='reuse';zero=geometry['shift']==0
  s=dict(mode=mode,endpoint_partition_aligned=True,identity_native_checked=zero,identity_native_state_equal=zero,identity_native_logits_equal=zero,identity_reuse_state_equal=zero and reused,
   raw_reference_rows=18,raw_candidate_rows=17 if reused else 0,tape_buffer_and_metadata_bytes=0,immutable_tape_bytes=0,peak_staging_bytes_including_reserve=50000000,staging_budget_bytes=128<<20,
   coverage_rejections_preserve_state=reused,precancel_preserves_state=reused,partial_cancel_invalidates=reused,approximate_export_blocked=reused,
   pair_inputs_reused=3*geometry['span'] if reused else 0,pair_rows_rebuilt=3*((geometry['prefix']+geometry['span'])//2-geometry['prefix']//2) if reused else 0,
   pair_boundary_rows=(3 if geometry['prefix']%2 else 0) if reused else 0,reused_tokens=geometry['span'] if reused else 0,fresh_tail_tokens=geometry['tail'] if reused else 0,
   argmax_flips=0,bit_equal_rows=17 if zero and reused else 0,fresh_ms_diagnostic=2.0,reuse_ms_diagnostic=1.0 if reused else 0,mean_kl=0.0,mean_tv=0.0,sum_delta_nll=0.0)
  s.update({k:geometry[k] for k in ['prefix','shift','span','tail','decode']});return s
 def reject(self,fn,*args):
  with self.assertRaises((ValueError,RuntimeError,KeyError,struct.error)):fn(*args)
 def test_three_cases_manifest_authority_without_model(self):
  for case in d.CASES:
   out,plan,digest=self.fixture(case)
   with patch.object(d,'sha',self.guarded_sha):self.assertEqual(d.verify(out,digest,case),plan)
  self.model.assert_not_called()
 def test_wrong_case_geometry_order_policy_and_model(self):
  case=next(iter(d.CASES));out,plan,digest=self.fixture(case)
  for key,value in [('case','inverse-parity-pending'),('geometry',dict(plan['geometry'],shift=1)),('arms',list(reversed(plan['arms']))),('policy',dict(d.POLICY,DS4_X='bad')),('model_from_prior',dict(d.MODEL,bytes=1))]:
   changed=copy.deepcopy(plan);changed[key]=value
   with patch.object(d,'sha',self.guarded_sha):self.reject(d.verify,out,self.pin(out,changed,case),case)
 def test_each_probe_mode_drift_rejected_despite_equal_bytes(self):
  case=next(iter(d.CASES));out,plan,digest=self.fixture(case)
  for n in ['new-probe','baseline-probe']:
   before=REAL_SHA(out/n);(out/n).chmod(0o644)
   with patch.object(d,'sha',self.guarded_sha):self.reject(d.verify,out,digest,case)
   self.assertEqual(before,REAL_SHA(out/n));(out/n).chmod(0o755)
 def test_missing_assets_and_same_byte_length_drift(self):
  case=next(iter(d.CASES));out,plan,digest=self.fixture(case)
  for n in REQUIRED:
   changed=copy.deepcopy(plan);del changed['files'][n]
   with patch.object(d,'sha',self.guarded_sha):self.reject(d.verify,out,self.pin(out,changed,case),case)
  digest=self.pin(out,plan,case);p=out/'corpus.txt';p.write_bytes(b'X'*p.stat().st_size)
  with patch.object(d,'sha',self.guarded_sha):self.reject(d.verify,out,digest,case)
 def test_wrong_command_digest_case_suffix_and_cwd(self):
  case=next(iter(d.CASES));out,plan,digest=self.fixture(case);original=json.loads((out/'prepared-command.json').read_text())
  for key,value in [('manifest_sha256','bad'),('cwd',str(out)),('command',original['command']+['extra']),('command',[x if x!=case else 'wrap-odd-chunk' for x in original['command']])]:
   bad=copy.deepcopy(original);bad[key]=value;(out/'prepared-command.json').write_text(json.dumps(bad))
   with patch.object(d,'sha',self.guarded_sha):self.reject(d.verify,out,digest,case)
 def test_guard_fail_before_any_model_or_output(self):
  case=next(iter(d.CASES));out=self.base/'uncreated'
  with patch.object(d,'verify',return_value={}),patch.object(d,'require_guard',side_effect=RuntimeError('No guard')):
   self.reject(d.run,out,'digest',case)
  self.model.assert_not_called();self.assertFalse(out.exists())
 def test_exact_guard_parent_and_absolute_bounded_ps(self):
  case=next(iter(d.CASES));out=self.base/'cpu';args=d.command(out,'digest',case)
  lock=unittest.mock.MagicMock();lock.__enter__.return_value=lock
  with patch.object(d.subprocess,'check_output',return_value=' '.join(args)) as ps,patch.object(d.os.path,'samefile',return_value=True),patch.object(d,'sha',self.guarded_sha),patch.object(Path,'open',return_value=lock),patch.object(d.fcntl,'flock',side_effect=BlockingIOError(errno.EAGAIN,'held')):
   d.require_guard(out,'digest',case);self.assertEqual(ps.call_args.args[0][0],'/bin/ps');self.assertEqual(ps.call_args.kwargs['timeout'],2)
  with patch.object(d.subprocess,'check_output',return_value=' '.join(args+['bad'])),patch.object(d.os.path,'samefile',return_value=True),patch.object(d,'sha',self.guarded_sha):self.reject(d.require_guard,out,'digest',case)
 def test_summaries_all_geometries_modes_and_exact_identity(self):
  for g in d.CASES.values():
   for mode in ['native','tape-native','reuse']:d.validate_summary(self.summary(g,mode),mode,g)
  g=d.CASES['zero-odd-identity-wrap'];s=self.summary(g)
  for k in ['identity_native_checked','identity_native_state_equal','identity_native_logits_equal','identity_reuse_state_equal','endpoint_partition_aligned']:
   bad=dict(s);bad[k]=False;self.reject(d.validate_summary,bad,'reuse',g)
  bad=dict(s);bad['bit_equal_rows']=16;self.reject(d.validate_summary,bad,'reuse',g)
 def test_summaries_reject_false_identity_and_wrong_pair_counts(self):
  for case,g in d.CASES.items():
   s=self.summary(g)
   for k in ['pair_rows_rebuilt','pair_boundary_rows','pair_inputs_reused','raw_candidate_rows']:
    bad=dict(s);bad[k]+=1;self.reject(d.validate_summary,bad,'reuse',g)
   if g['shift']:
    for k in ['identity_native_checked','identity_native_state_equal','identity_native_logits_equal','identity_reuse_state_equal']:
     bad=dict(s);bad[k]=True;self.reject(d.validate_summary,bad,'reuse',g)
 def test_first_nonzero_helper_exit_retained_and_no_next_arm(self):
  case=next(iter(d.CASES));out=self.base/'failed';out.mkdir();identity=dict(path='CPU-NOT-A-MODEL',bytes=0,mtime_ns=0)
  with patch.object(d,'verify',return_value=dict(source_revision='CPU-FIXTURE')),patch.object(d,'require_guard'),patch.object(d,'model_identity',return_value=identity),patch.object(d.subprocess,'run',return_value=type('FakeExit',(),dict(returncode=3))()) as run:
   self.reject(d.run,out,'digest',case);self.assertEqual(run.call_count,1)
  directory=out/'run';self.assertEqual(json.loads((directory/'old-native-before-exit.json').read_text())['returncode'],3)
  self.assertTrue((directory/'old-native-before.log').exists());self.assertTrue((directory/'old-native-before.csv').exists());self.assertFalse((directory/'summary.json').exists())
 def raw(self,base,g,approx):
  p=Path(str(base)+'.raw.bin');n=129280*4
  with p.open('wb') as f:f.write(struct.pack('<8I',0x53505431,129280,g['decode'],g['prefix'],g['shift'],g['span'],g['tail'],int(approx)));f.write(struct.pack('<16I',*range(16)));f.truncate(96+(18+(17 if approx else 0))*n)
  return p
 def test_raw_geometry_and_truncation_all_three_cases(self):
  for case,g in d.CASES.items():
   base=self.base/case;p=self.raw(base,g,True);r=d.raw_reference(p,True,g);self.assertEqual(r['reference_rows'],18)
   with p.open('r+b') as f:f.truncate(p.stat().st_size-1)
   self.reject(d.raw_reference,p,True,g)
 def test_retained_identity_checks_real_raw_and_full_state(self):
  g=d.CASES['zero-odd-identity-wrap'];base=self.base/'identity';p=self.raw(base,g,True)
  canonical=b'DSV4'+bytes(range(52));private=b'DSSD'+canonical[4:]
  for suffix,data in [('.source.state.bin',canonical),('.fresh.state.bin',canonical),('.private.state.bin',private)]:Path(str(base)+suffix).write_bytes(data)
  state={n:REAL_SHA(str(base)+n) for n in ['.source.state.bin','.fresh.state.bin']};raw=d.raw_reference(p,True,g)
  r=d.zero_identity_records(base,True,raw,state,g);self.assertTrue(r['repaired_full_vectors_exact'])
  with p.open('r+b') as f:f.seek(96+18*129280*4+32768);f.write(b'X')
  self.reject(d.zero_identity_records,base,True,raw,state,g)
  self.raw(base,g,True);Path(str(base)+'.private.state.bin').write_bytes(private[:-1]+b'X')
  self.reject(d.zero_identity_records,base,True,raw,state,g)
 def test_csv_wrong_target_order_nonfinite_and_summary(self):
  g=d.CASES['inverse-parity-pending'];s=self.summary(g);p=self.base/'quality.csv'
  rows=['step,target,delta_nll,kl,tv,argmax_flip,bit_equal']+[f'{j},{j if j<16 else -1},0,0,0,0,0' for j in range(17)];p.write_text('\n'.join(rows)+'\n');hexids=struct.pack('<16I',*range(16)).hex()
  d.validate_quality_csv(p,s,hexids,g)
  for replace in [('0,0,0,0,0,0,0','0,99,0,0,0,0,0'),('0,0,0,0,0,0,0','0,0,0,nan,0,0,0'),('16,-1,0,0,0,0,0','16,-1,1,0,0,0,0')]:
   p.write_text(('\n'.join(rows)+'\n').replace(*replace));self.reject(d.validate_quality_csv,p,s,hexids,g)
 def test_csv_header_extra_field_and_missing_field_rejected(self):
  g=d.CASES['inverse-parity-pending'];s=self.summary(g);p=self.base/'quality.csv';hexids=struct.pack('<16I',*range(16)).hex()
  original='step,target,delta_nll,kl,tv,argmax_flip,bit_equal\n'+'\n'.join(f'{j},{j if j<16 else -1},0,0,0,0,0' for j in range(17))+'\n'
  p.write_text(original);d.validate_quality_csv(p,s,hexids,g)
  for text in [original.replace('step,target','target,step'),original.replace('0,0,0,0,0,0,0\n','0,0,0,0,0,0,0,extra\n',1),original.replace('0,0,0,0,0,0,0\n','0,0,0,0,0,0\n',1)]:
   p.write_text(text);self.reject(d.validate_quality_csv,p,s,hexids,g)
if __name__=='__main__':unittest.main(verbosity=2)
