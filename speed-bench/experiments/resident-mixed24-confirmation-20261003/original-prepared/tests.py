"""Model-free strict mixed24 geometry, fixed selection, screens and authority."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'speed-bench/ideas/prepare_resident_mixed24_confirmation.py'
spec=importlib.util.spec_from_file_location('mixed24_confirmation',SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
c=s.calibration_module()
RAW=SimpleNamespace(RAW_ROW_META=struct.Struct('=4IQ'))


def cases():return [dict(case=f'case_{i:03d}',category=('agent','tool','code','math','string','rare')[i//4]) for i in range(24)]


def score_rows():
    rows=[];keys=[]
    for case in range(24):
        for step in range(32):
            rows.append(dict(zip(s.CSV_FIELDS,[f'case_{case:03d}',str(step),str(100+step),'240','2','2','2','0','0','0','0','0','0','0','1'])))
            keys.append(RAW.RAW_ROW_META.pack(case,step,100+step,240,case+1))
    return rows,(b'',keys,{})


def scores(path,rows):
    with path.open('w') as f:
        out=csv.DictWriter(f,fieldnames=s.CSV_FIELDS);out.writeheader();out.writerows(rows)


def quality():return dict(cases=24,tokens=768,argmax_flips=0,bit_equal_rows=768,avg_nll=2.,delta_nll=0.,policy_delta_nll=0.,kl_base_to_raw=0.,kl_base_to_policy=0.,tv=0.,startup_ms=1.,prefill_ms=1.,decode_ms=768.,metrics_ms=1.)


def results(selected='wider-weights3'):
    rows={}
    for arm in s.arms(selected,c):
        rows[arm['name']]=dict(delta_nll=.007,kl_base_to_raw=.01,kl_p99=.1,kl_max=1.,argmax_flips=24,worst_prompt_delta_nll=.1,io=dict(requested_bytes=7000),per_prompt={case['case']:dict(delta_nll=.007,kl=.01,tokens=32) for case in cases()})
    rows['native-before']['io']['requested_bytes']=10000;rows['native-after']['io']['requested_bytes']=10000
    return rows


def calibration_results():
    rows={}
    for arm in c.arms():
        rows[arm['name']]=dict(delta_nll=.007,kl_base_to_raw=.01,kl_p99=.1,kl_max=1.,argmax_flips=15,worst_prompt_delta_nll=.1,io=dict(requested_bytes=7000),per_prompt={f'cal_{i:03d}':dict(delta_nll=.007) for i in range(16)})
    rows['native-before']['io']['requested_bytes']=10000;rows['native-after']['io']['requested_bytes']=10000;rows['wider-weights3']['io']['requested_bytes']=6000
    return rows


class Geometry(unittest.TestCase):
    def root(self):return tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts')

    def test_exact768_ordered_teacher_rows_accepted(self):
        with self.root() as d:
            rows,meta=score_rows();path=Path(d)/'scores.csv';scores(path,rows)
            self.assertEqual(s.validate_csv(path,meta,cases(),RAW),rows)
            self.assertEqual(s.validate_quality_json(json.dumps(quality()),rows),quality())

    def test_old512_geometry_rejected_without_monkeypatch(self):
        with self.root() as d:
            rows,meta=score_rows();path=Path(d)/'scores.csv';scores(path,rows[:512])
            with self.assertRaises(ValueError):s.validate_csv(path,(b'',meta[1][:512],{}),cases(),RAW)
            value=quality();value.update(cases=16,tokens=512)
            with self.assertRaises(ValueError):s.validate_quality_json(json.dumps(value),rows)

    def test_missing_extra_duplicate_or_reordered_csv_refused(self):
        with self.root() as d:
            original,meta=score_rows();path=Path(d)/'scores.csv'
            for kind in ('missing','extra','duplicate','order'):
                rows=copy.deepcopy(original)
                if kind=='missing':rows.pop()
                elif kind=='extra':rows.append(rows[-1])
                elif kind=='duplicate':rows[-1]=rows[-2]
                else:rows[0],rows[1]=rows[1],rows[0]
                scores(path,rows)
                with self.assertRaises(ValueError,msg=kind):s.validate_csv(path,meta,cases(),RAW)

    def test_target_prompt_case_step_and_raw_order_bound(self):
        with self.root() as d:
            original,meta=score_rows();path=Path(d)/'scores.csv'
            for field,value in [('target','999'),('prompt_tokens','241'),('case','case_999'),('step','1')]:
                rows=copy.deepcopy(original);rows[0][field]=value;scores(path,rows)
                with self.assertRaises(ValueError,msg=field):s.validate_csv(path,meta,cases(),RAW)
            changed=(b'',list(meta[1]),{});changed[1][0],changed[1][1]=changed[1][1],changed[1][0];scores(path,original)
            with self.assertRaises(ValueError):s.validate_csv(path,changed,cases(),RAW)

    def test_finite_range_and_nll_arithmetic_before_summary(self):
        with self.root() as d:
            original,meta=score_rows();path=Path(d)/'scores.csv'
            for field,value in [('raw_kl','nan'),('base_nll','inf'),('raw_tv','1.1'),('raw_kl','-.1'),('raw_nll','-1'),('delta_raw_nll','.1'),('decode_ms','-1'),('argmax_flip','True')]:
                rows=copy.deepcopy(original);rows[9][field]=value;scores(path,rows)
                with self.assertRaises(ValueError,msg=field):s.validate_csv(path,meta,cases(),RAW)

    def test_quality_typed_counts_schema_and_arithmetic(self):
        rows,_=score_rows()
        for field,value in [('tokens',512),('cases',True),('argmax_flips',True),('bit_equal_rows',769),('avg_nll',2.1),('tv',float('nan')),('decode_ms',-1)]:
            q=quality();q[field]=value
            with self.assertRaises(ValueError,msg=field):s.validate_quality_json(json.dumps(q),rows)
        with self.assertRaises(ValueError):s.validate_quality_json(json.dumps(dict(quality(),extra=0)),rows)
        with self.assertRaises(ValueError):s.validate_quality_json(json.dumps(quality())[:-1]+',"cases":24}',rows)

    def test_frozen_input77_and_six_categories_bound(self):
        frozen=s.inputs();self.assertEqual(len(frozen['files'])+1,77);self.assertEqual(len(frozen['cases']),24)
        self.assertEqual(frozen['category_counts'],dict(agent=4,tool=4,code=4,math=4,string=4,rare=4))
        self.assertEqual(s.sha(s.SPEC),s.SPEC_SHA);self.assertEqual(s.sha(s.FREEZE),s.FREEZE_SHA)

    def test_all77_frozen_physical_input_files_are_authoritatively_pinned(self):
        frozen=s.inputs();expected=dict(frozen['files'],**{s.FREEZE.name:s.FREEZE_SHA})
        self.assertEqual({p.name for p in s.FREEZE.parent.iterdir() if p.is_file()},set(expected))
        for name,digest in expected.items():self.assertEqual(s.sha(s.FREEZE.parent/name),digest,name)


class FixedReports(unittest.TestCase):
    def test_fixed_order_selected_distinct_or_anchor_only(self):
        self.assertEqual([a['name'] for a in s.arms('wider-weights3',c)],['native-before','anchor-affinity3','wider-weights3','native-after'])
        self.assertEqual([a['name'] for a in s.arms('anchor-affinity3',c)],['native-before','anchor-affinity3','native-after'])
        for name in ('unknown','native-before','native-after'):
            with self.assertRaises(ValueError):s.arms(name,c)
        for arm in s.arms('protected-weights1',c):
            env=c.policy(arm);self.assertEqual(env[c.P+'POLICY'],'replace');self.assertEqual(env['DS4_METAL_V41_SELECTED_TILE'],'64')

    def test_bootstrap24_seed_draws_and_six_category_reports(self):
        report=s.confirmation(results(),'wider-weights3',cases(),s.frozen_spec(),c)
        self.assertTrue(report['selected_confirmation_pass']);self.assertTrue(report['no_policy_selection_from_confirmation'])
        pair=report['reports']['wider-weights3']['paired_candidate_minus_anchor_nll'];self.assertEqual(pair,dict(mean=0.,ci95=[0.,0.],paired_prompts=24,draws=2000,seed=20261002))
        self.assertEqual(len(report['reports']['wider-weights3']['categories']),6)
        for row in report['reports']['wider-weights3']['categories'].values():self.assertEqual((row['prompts'],row['tokens']),(4,128))

    def test_fixed24_flip_integer_and20percent_boundaries(self):
        rows=results();rows['wider-weights3']['io']['requested_bytes']=8000
        self.assertTrue(s.confirmation(rows,'wider-weights3',cases(),s.frozen_spec(),c)['selected_confirmation_pass'])
        rows['wider-weights3']['io']['requested_bytes']=8001;rows['wider-weights3']['argmax_flips']=25
        failures=s.confirmation(rows,'wider-weights3',cases(),s.frozen_spec(),c)['reports']['wider-weights3']['failures'];self.assertEqual(failures,['argmax_flips','native_requested_byte_reduction'])

    def test_all_preregistered_quality_dimensions_preserved(self):
        for field,reason,value in [('delta_nll','mean_delta_nll',.02),('kl_base_to_raw','mean_kl',.02),('kl_p99','p99_kl',.2),('kl_max','max_kl',2.),('worst_prompt_delta_nll','worst_prompt_delta_nll',.11)]:
            rows=results();rows['wider-weights3'][field]=value
            self.assertIn(reason,s.confirmation(rows,'wider-weights3',cases(),s.frozen_spec(),c)['reports']['wider-weights3']['failures'])

    def test_paired_upper_failure_not_repaired_or_reselected(self):
        rows=results()
        for row in rows['wider-weights3']['per_prompt'].values():row['delta_nll']+=.006
        report=s.confirmation(rows,'wider-weights3',cases(),s.frozen_spec(),c)
        self.assertEqual(report['fixed_selected_policy'],'wider-weights3');self.assertFalse(report['selected_confirmation_pass']);self.assertIn('paired_nll_ci_upper',report['reports']['wider-weights3']['failures'])

    def test_missing_reordered_untyped_nonfinite_results_refused(self):
        for kind in ('missing','order','nan','bool','bytes'):
            rows=results()
            if kind=='missing':rows.pop('native-after')
            elif kind=='order':rows=dict(reversed(list(rows.items())))
            elif kind=='nan':rows['wider-weights3']['kl_max']=float('nan')
            elif kind=='bool':rows['wider-weights3']['argmax_flips']=True
            else:rows['native-before']['io']['requested_bytes']=True
            with self.assertRaises(ValueError,msg=kind):s.confirmation(rows,'wider-weights3',cases(),s.frozen_spec(),c)

    def test_paired_prompt_coverage_order_and_finiteness(self):
        rows=results();anchor=rows['anchor-affinity3']
        for kind in ('missing','order','nan'):
            candidate=copy.deepcopy(rows['wider-weights3'])
            if kind=='missing':candidate['per_prompt'].popitem()
            elif kind=='order':candidate['per_prompt']=dict(reversed(list(candidate['per_prompt'].items())))
            else:candidate['per_prompt']['case_023']['delta_nll']=float('nan')
            with self.assertRaises(ValueError,msg=kind):s.paired(candidate,anchor,cases())


class SelectionAndGrant(unittest.TestCase):
    def root(self):return tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts')

    def frozen_calibration(self,root,advance=True):
        (root/'run').mkdir();(root/'manifest.json').write_text('{}')
        rows=calibration_results()
        if not advance:rows['wider-weights3']['io']['requested_bytes']=7000
        decision=c.decision(rows)
        release=dict(guard_result=dict(status='complete',child_returncode=0),lock_reacquired=True,live_model_processes=[])
        summary=dict(status='complete',first_attempt_only=True,full_native_brackets_and_anchor_controls_exact=True,arms=rows,decision=decision)
        (root/'root-release-proof.json').write_text(json.dumps(release));(root/'run/summary.json').write_text(json.dumps(summary));(root/'run/decision.json').write_text(json.dumps(decision))
        proof=dict(status='ROOT_FIXED_CALIBRATION_SELECTION',root_explicit_selection=True,calibration_manifest_sha256=s.sha(root/'manifest.json'),calibration_driver_sha256=s.CAL_SHA,calibration_summary_sha256=s.sha(root/'run/summary.json'),calibration_decision_sha256=s.sha(root/'run/decision.json'),calibration_release_sha256=s.sha(root/'root-release-proof.json'),selected_policy=decision['selected_calibration_policy'] if decision['advance'] else 'anchor-affinity3',time=1790980000.5)
        path=root/'selection.json';path.write_text(json.dumps(proof));return proof,path

    def test_released_exact_advance_policy_accepted(self):
        with self.root() as d:
            root=Path(d);proof,path=self.frozen_calibration(root)
            with patch.object(s,'CAL',root),patch.object(s,'CAL_MANIFEST',proof['calibration_manifest_sha256']):self.assertEqual(s.selection_authority(path,c),proof)

    def test_noadvance_must_confirm_anchor(self):
        with self.root() as d:
            root=Path(d);proof,path=self.frozen_calibration(root,False);self.assertEqual(proof['selected_policy'],'anchor-affinity3')
            with patch.object(s,'CAL',root),patch.object(s,'CAL_MANIFEST',proof['calibration_manifest_sha256']):self.assertEqual(s.selection_authority(path,c),proof)

    def test_missing_forged_untyped_selection_before_outcome_read(self):
        with self.root() as d:
            root=Path(d);proof,path=self.frozen_calibration(root)
            for field,value in [('status','PREPARED'),('root_explicit_selection',1),('calibration_driver_sha256','f'*64),('selected_policy','not-a-policy'),('time',True),('time',0),('time',float('inf'))]:
                changed=dict(proof);changed[field]=value;path.write_text(json.dumps(changed))
                with patch.object(s,'CAL',root),patch.object(s,'CAL_MANIFEST',proof['calibration_manifest_sha256']),patch.object(s,'sha',side_effect=AssertionError('outcome read before valid authority')):
                    with self.assertRaises(ValueError,msg=field):s.selection_authority(path,c)
            path.unlink()
            with self.assertRaises(FileNotFoundError):s.selection_authority(path,c)

    def test_unreleased_control_stops_before_summary_read(self):
        with self.root() as d:
            root=Path(d);proof,path=self.frozen_calibration(root)
            release=json.loads((root/'root-release-proof.json').read_text());release['lock_reacquired']=False;(root/'root-release-proof.json').write_text(json.dumps(release));proof['calibration_release_sha256']=s.sha(root/'root-release-proof.json');path.write_text(json.dumps(proof))
            (root/'run/summary.json').unlink();(root/'run/decision.json').unlink()
            with patch.object(s,'CAL',root),patch.object(s,'CAL_MANIFEST',proof['calibration_manifest_sha256']):
                with self.assertRaisesRegex(ValueError,'released'):s.selection_authority(path,c)

    def test_policy_override_and_changed_decision_refused(self):
        with self.root() as d:
            root=Path(d);proof,path=self.frozen_calibration(root);proof['selected_policy']='protected-weights1';path.write_text(json.dumps(proof))
            with patch.object(s,'CAL',root),patch.object(s,'CAL_MANIFEST',proof['calibration_manifest_sha256']):
                with self.assertRaisesRegex(ValueError,'advance'):s.selection_authority(path,c)
            changed=json.loads((root/'run/decision.json').read_text());changed['advance']=False;(root/'run/decision.json').write_text(json.dumps(changed));proof['calibration_decision_sha256']=s.sha(root/'run/decision.json');path.write_text(json.dumps(proof))
            with patch.object(s,'CAL',root),patch.object(s,'CAL_MANIFEST',proof['calibration_manifest_sha256']):
                with self.assertRaisesRegex(ValueError,'arithmetic'):s.selection_authority(path,c)

    def test_execute_missing_or_forged_grant_before_model_access(self):
        with self.root() as d:
            root=Path(d);events=[];fake=SimpleNamespace(require_main_guard=lambda *a:events.append('guard'),guard_command=lambda p,h:['MAIN',str(p),h],strict_json=s.strict_json,raw_module=lambda:(_ for _ in ()).throw(AssertionError('model helper accessed before grant')))
            cal=SimpleNamespace(module=lambda:fake,qualification=lambda _:({},{}),require_root_grant=c.require_root_grant)
            with patch.object(s,'calibration_module',return_value=cal),patch.object(s,'verified',return_value=({},{})):
                with self.assertRaises(FileNotFoundError):s.execute(root,'e'*64)
                (root/'root-launch-proof.json').write_text(json.dumps(dict(status='GRANTED',sole_model_grant=True,manifest_sha256='f'*64,command=fake.guard_command(root,'e'*64),time=1.)))
                with self.assertRaises(ValueError):s.execute(root,'e'*64)
            self.assertEqual(events,['guard','guard'])

    def test_each_arm_rechecks_grant_before_identity_or_process(self):
        with self.root() as d:
            root=Path(d);events=[];raw=SimpleNamespace(check_frozen_assets=lambda _:events.append('assets'),identity=lambda _:events.append('identity') or dict(path='/forbidden.gguf'))
            fake=SimpleNamespace(require_main_guard=lambda *a:events.append('guard'),raw_module=lambda:raw)
            cal=SimpleNamespace(module=lambda:fake,qualification=lambda _:({},{}),require_root_grant=lambda *a:None)
            plan=dict(model_from_prior=dict(path='/forbidden.gguf'),shader_env={},arms=s.arms('anchor-affinity3',c))
            with patch.object(s,'calibration_module',return_value=cal),patch.object(s,'verified',return_value=(plan,{})),patch.object(cal,'require_root_grant',side_effect=[{},ValueError('revoked')]) as grant,patch.object(s.subprocess,'run',side_effect=AssertionError('helper launched after revoked grant')):
                with self.assertRaisesRegex(ValueError,'revoked'):s.execute(root,'e'*64)
            self.assertEqual(grant.call_count,2);self.assertEqual(events.count('identity'),1);self.assertEqual(events[-1],'guard')

    def test_shader_inventory_no_glob_and_all26_exact(self):
        qual=dict(files={'metal/'+name:'f'*64 for name in c.SHADER_FILES})
        with patch.object(Path,'glob',side_effect=AssertionError('glob forbidden')):env=c.shader_environment(qual)
        self.assertEqual(len(env),26);self.assertEqual(len(set(env)),26)
        qual['files']['metal/Dense.metal']='e'*64
        with self.assertRaises(ValueError):c.shader_environment(qual)

    def test_ungranted_prepare_reads_no_calibration_outcomes(self):
        with self.root() as d:
            root=Path(d)
            with patch.object(s,'inputs',return_value={}),patch.object(s,'cpu_proof',side_effect=AssertionError('CPU/preparation started before selection')):
                with self.assertRaises(FileNotFoundError):s.prepare(root/'future',root/'cpu.json',root/'missing-selection.json')
            self.assertFalse((root/'future').exists())


if __name__=='__main__':unittest.main()
