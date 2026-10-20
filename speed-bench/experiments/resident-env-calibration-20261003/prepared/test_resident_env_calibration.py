"""Model-free fixed-policy, decision and authority mutations for the fast loop."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'speed-bench/ideas/prepare_resident_env_calibration.py'
spec=importlib.util.spec_from_file_location('env_calibration',SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def results():
    rows={}
    for arm in s.arms():
        rows[arm['name']]=dict(delta_nll=.0076,kl_base_to_raw=.017,kl_p99=.15,kl_max=1.,argmax_flips=15,worst_prompt_delta_nll=.10,
            io=dict(requested_bytes=7000),per_prompt={f'case_{i:03d}':dict(delta_nll=.0076) for i in range(16)})
    rows['native-before']['io']['requested_bytes']=10000
    rows['native-after']['io']['requested_bytes']=10000
    return rows


class Decision(unittest.TestCase):
    def test_fixed_six_order_guards_and_native_brackets(self):
        arms=s.arms();self.assertEqual(len(arms),8)
        self.assertEqual([a['name'] for a in arms],['native-before','anchor-affinity3','anchor-weights3','wider-weights3','closer-weights3','protected-weights2','protected-weights1','native-after'])
        self.assertEqual([(a['flag'],a['gap'],a['mass'],a['total'],a['rank'],a['subs']) for a in arms[1:-1]],
            [('0','.30','.20','.40','48','3'),('1','.30','.20','.40','48','3'),('1','.30','.20','.50','48','3'),('1','.15','.20','.40','24','3'),('1','.30','.18','.35','48','2'),('1','.30','.12','.40','24','1')])
        self.assertTrue(arms[-2]['total_mass_ignored'])
        for arm in arms:
            env=s.policy(arm);self.assertEqual(env['DS4_METAL_V41_SELECTED_TILE'],'64');self.assertEqual(env['DS4_METAL_V41_SELECTED_LIMIT'],'1024');self.assertEqual(env[s.P+'POLICY'],'replace')
            self.assertFalse(any('BACKFILL' in k or 'PROFILE' in k for k in env))
        self.assertEqual([a['mode'] for a in (arms[0],arms[-1])],['native','native'])

    def test_minimum_bytes_and_calibration_only(self):
        rows=results();rows['wider-weights3']['io']['requested_bytes']=6000
        out=s.decision(rows);self.assertEqual(out['selected_calibration_policy'],'wider-weights3');self.assertTrue(out['advance'])
        self.assertFalse(out['automatic_holdout_run']);self.assertTrue(out['root_final_policy_selection_required']);self.assertTrue(out['holdout_model_grant_required'])

    def test_inclusive_twenty_percent_integer_boundary(self):
        rows=results();rows['closer-weights3']['io']['requested_bytes']=8000
        self.assertTrue(s.decision(rows)['candidates']['closer-weights3']['eligible'])
        rows['closer-weights3']['io']['requested_bytes']=8001
        self.assertIn('native_requested_byte_reduction',s.decision(rows)['candidates']['closer-weights3']['failures'])

    def test_one_percent_tie_inclusive_then_lower_kl(self):
        rows=results();rows['wider-weights3']['io']['requested_bytes']=6000
        rows['anchor-weights3']['io']['requested_bytes']=6060;rows['anchor-weights3']['kl_base_to_raw']=.016
        self.assertEqual(s.decision(rows)['selected_calibration_policy'],'anchor-weights3')
        rows['anchor-weights3']['io']['requested_bytes']=6061
        self.assertEqual(s.decision(rows)['selected_calibration_policy'],'wider-weights3')

    def test_tie_p99_then_bytes_then_fixed_order(self):
        rows=results();rows['wider-weights3']['kl_p99']=.14
        self.assertEqual(s.decision(rows)['selected_calibration_policy'],'wider-weights3')
        rows['wider-weights3']['kl_p99']=.15;rows['closer-weights3']['io']['requested_bytes']=6999
        self.assertEqual(s.decision(rows)['selected_calibration_policy'],'closer-weights3')
        rows['closer-weights3']['io']['requested_bytes']=7000
        self.assertEqual(s.decision(rows)['selected_calibration_policy'],'anchor-affinity3')

    def test_no_advance_preserves_all_failures(self):
        rows=results();rows['anchor-weights3']['kl_p99']=.14;rows['anchor-weights3']['kl_max']=1.1
        out=s.decision(rows);self.assertEqual(out['selected_calibration_policy'],'anchor-weights3');self.assertFalse(out['advance'])
        self.assertEqual(len(out['candidates']),6)
        rows['anchor-weights3']['io']['requested_bytes']=6999
        self.assertTrue(s.decision(rows)['advance'])

    def test_quality_dominance_can_advance_without_extra_io_gain(self):
        rows=results();rows['anchor-weights3']['kl_base_to_raw']=.016
        self.assertTrue(s.decision(rows)['advance'])

    def test_all_screen_dimensions_and_paired_upper_fail(self):
        mapping=[('delta_nll','mean_delta_nll',.02),('kl_base_to_raw','mean_kl',.03),('kl_p99','p99_kl',.2),('kl_max','max_kl',2.),('argmax_flips','argmax_flips',17),('worst_prompt_delta_nll','worst_prompt_delta_nll',.11)]
        for field,reason,value in mapping:
            rows=results();rows['wider-weights3'][field]=value
            self.assertIn(reason,s.decision(rows)['candidates']['wider-weights3']['failures'])
        rows=results()
        for prompt in rows['wider-weights3']['per_prompt'].values():prompt['delta_nll']+=.006
        self.assertIn('paired_nll_ci_upper',s.decision(rows)['candidates']['wider-weights3']['failures'])

    def test_pareto_keeps_quality_preserving_subtwenty_gain(self):
        rows=results();row=rows['closer-weights3'];row['io']['requested_bytes']=9500
        for k in ('delta_nll','kl_base_to_raw','kl_p99','kl_max','argmax_flips','worst_prompt_delta_nll'):row[k]=0
        out=s.decision(rows)['candidates']['closer-weights3'];self.assertFalse(out['eligible']);self.assertTrue(out['pareto_nondominated'])
        self.assertEqual(out['failures'],['native_requested_byte_reduction'])

    def test_no_eligible_policy_retains_six_rows(self):
        rows=results()
        for a in s.arms()[1:-1]:rows[a['name']]['kl_max']=2
        out=s.decision(rows);self.assertIsNone(out['selected_calibration_policy']);self.assertFalse(out['advance']);self.assertEqual(len(out['candidates']),6)

    def test_paired_prompt_bootstrap_coverage_order_and_finite(self):
        rows=results();a=rows['anchor-affinity3'];b=rows['anchor-weights3']
        self.assertEqual(s.paired(a,b),dict(mean=0.,ci95=[0.,0.],paired_prompts=16))
        for change in ('missing','order','nan'):
            other=copy.deepcopy(b)
            if change=='missing':other['per_prompt'].popitem()
            elif change=='order':other['per_prompt']=dict(reversed(list(other['per_prompt'].items())))
            else:other['per_prompt']['case_015']['delta_nll']=float('nan')
            with self.assertRaises(ValueError):s.paired(other,a)

    def test_attempt_order_bytes_and_nonfinite_refused(self):
        rows=results()
        for changed in [dict(reversed(list(rows.items()))),{k:v for k,v in rows.items() if k!='native-after'}]:
            with self.assertRaises(ValueError):s.decision(changed)
        for where,field,value in [('native-before','requested_bytes',0),('native-before','requested_bytes',True),('wider-weights3','requested_bytes',-1),('wider-weights3','kl_max',float('inf'))]:
            changed=copy.deepcopy(rows)
            if field=='requested_bytes':changed[where]['io'][field]=value
            else:changed[where][field]=value
            with self.assertRaises(ValueError):s.decision(changed)

    def test_guard_environment_and_first_attempt_source_order(self):
        body=SOURCE.read_text().split('def execute(',1)[1]
        self.assertLess(body.index('s.require_main_guard(output,digest)'),body.index('identity=raw.identity(model)'))
        self.assertIn("not k.startswith('DS4')",body)
        self.assertIn("require(completed.returncode==0",body)
        self.assertIn("raw_sha256']==REF_SHA",body);self.assertIn("ANCHOR_RAWS[name]",body)
        self.assertIn("all(row[k]==prior[k] for k in ('io','cache','route'))",body)
        self.assertNotIn('retry',body);self.assertNotIn('heldout16',body)

    def test_manifest_authority_and_fixed_screens(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts') as d:
            root=Path(d);qual=dict(model_from_prior={'path':'prior-only'},groups={'calibration16':{'inputs':[]}},files={'metal/'+name:'f'*64 for name in s.SHADER_FILES})
            fake=SimpleNamespace(strict_json=json.loads,SETTINGS={},command_template=lambda p:['guard','<manifest_sha256>'],guard_command=lambda p,d:['guard',d])
            proof=root/'proof.json';proof.write_text('{}')
            plan=dict(arms=s.arms(),settings={},screen=s.SCREEN,qualification_manifest_sha256=s.QUAL_MANIFEST_SHA,source_sha256=s.sha(SOURCE),command_template=fake.command_template(root),shader_env=s.shader_environment(qual),root_model_grant_required=True,model_from_prior=qual['model_from_prior'],fixed_fixture=qual['groups']['calibration16'],holdout_not_included=True,test_sha256=s.sha(Path(__file__)),cpu_proof=dict(path=str(proof),sha256=s.sha(proof)))
            names={'driver.py':s.sha(SOURCE),'test_resident_env_calibration.py':s.sha(Path(__file__)),'cpu-proof.json':s.sha(proof),'qualification-manifest.json':s.QUAL_MANIFEST_SHA,'qualification-summary.json':s.QUAL_PINS['run/summary.json'],'qualification-release.json':s.QUAL_PINS['root-release-proof.json']}
            plan.update(files=names,external_files={str(s.WORK/'speed-bench/ideas/prepare_resident_env_calibration.py'):s.sha(SOURCE),str(s.TEST_SOURCE):s.sha(Path(__file__)),str(proof):s.sha(proof)})
            def save(data):
                (root/'manifest.json').write_text(json.dumps(data));digest=s.sha(root/'manifest.json')
                (root/'prepared-command.json').write_text(json.dumps(dict(command=fake.guard_command(root,digest),manifest_sha256=digest,launch_requires_root_grant=True,automatic_holdout_run=False)))
                return digest
            digest=save(plan)
            with patch.object(s,'reference_assets',return_value={}),patch.object(s,'cpu_proof',return_value=proof):
                self.assertEqual(s.verified(root,digest,fake,qual)[0],plan)
                for key,value in [('screen',dict(s.SCREEN,mean_kl_max=1.)),('arms',s.arms()[::-1]),('files',{}),('external_files',{})]:
                    changed=copy.deepcopy(plan);changed[key]=value
                    with self.assertRaises(ValueError):s.verified(root,save(changed),fake,qual)
                save(plan)
                with self.assertRaises(ValueError):s.verified(root,'f'*64,fake,qual)


class FixedShadersAndGrant(unittest.TestCase):
    def shader_plan(self):return dict(files={'metal/'+name:'f'*64 for name in s.SHADER_FILES})

    def proof(self,root,digest='e'*64):
        fake=SimpleNamespace(strict_json=json.loads,guard_command=lambda p,d:['framework','MAINguard',str(p/'pressure'),'--','framework','driver','--manifest',d])
        proof=dict(status='GRANTED',sole_model_grant=True,manifest_sha256=digest,command=fake.guard_command(root,digest),time=1790980000.25)
        return fake,proof

    def test_extra_unpinned_shader_alias_cannot_override_environment(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts') as d:
            root=Path(d);(root/'metal').mkdir();(root/'metal/Dense.metal').write_text('UNPINNED CASE ALIAS')
            with patch.object(s,'QUAL',root),patch.object(Path,'glob',side_effect=AssertionError('directory glob forbidden')):
                env=s.shader_environment(self.shader_plan())
            self.assertEqual(len(env),26);self.assertEqual(env['DS4_METAL_DENSE_SOURCE'],str(root/'metal/dense.metal'))
            self.assertNotIn(str(root/'metal/Dense.metal'),env.values())

    def test_manifest_requires_exact_unique_recognized_shader_keys(self):
        for change in ('missing','alias','unknown'):
            plan=self.shader_plan()
            if change=='missing':plan['files'].pop('metal/dense.metal')
            elif change=='alias':plan['files']['metal/Dense.metal']='0'*64
            else:plan['files']['metal/not_a_shader.metal']='0'*64
            with self.assertRaises(ValueError):s.shader_environment(plan)
        self.assertEqual(set(s.shader_environment(self.shader_plan())),{'DS4_METAL_'+Path(n).stem.upper()+'_SOURCE' for n in s.SHADER_FILES})

    def test_explicit_frozen_grant_accepted_without_model_metadata(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts') as d:
            root=Path(d);fake,proof=self.proof(root);(root/'root-launch-proof.json').write_text(json.dumps(proof))
            self.assertEqual(s.require_root_grant(root,'e'*64,fake),proof)

    def test_missing_forged_untyped_or_nonfinite_grant_refused(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts') as d:
            root=Path(d);fake,baseline=self.proof(root)
            with self.assertRaises(FileNotFoundError):s.require_root_grant(root,'e'*64,fake)
            mutations=[('status','PREPARED'),('sole_model_grant',1),('manifest_sha256','f'*64),('command',baseline['command']+['extra']),('time',True),('time',0),('time',-1),('time',float('nan')),('time',float('inf'))]
            for key,value in mutations:
                proof=copy.deepcopy(baseline);proof[key]=value;(root/'root-launch-proof.json').write_text(json.dumps(proof))
                with self.assertRaises(ValueError,msg=key+':'+str(value)):s.require_root_grant(root,'e'*64,fake)
            proof=copy.deepcopy(baseline);proof.pop('time');(root/'root-launch-proof.json').write_text(json.dumps(proof))
            with self.assertRaises(ValueError):s.require_root_grant(root,'e'*64,fake)

    def test_actual_execute_rejects_missing_or_forged_grant_before_first_identity(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts') as d:
            root=Path(d);fake,baseline=self.proof(root);events=[]
            fake.require_main_guard=lambda p,h:events.append('guard')
            fake.raw_module=lambda:(_ for _ in ()).throw(AssertionError('raw/model path accessed before grant'))
            plan=dict(model_from_prior={'path':'/forbidden.gguf'})
            with patch.object(s,'module',return_value=fake),patch.object(s,'qualification',return_value=({},{})),patch.object(s,'verified',return_value=(plan,{})):
                with self.assertRaises(FileNotFoundError):s.execute(root,'e'*64)
                baseline['manifest_sha256']='wrong';(root/'root-launch-proof.json').write_text(json.dumps(baseline))
                with self.assertRaises(ValueError):s.execute(root,'e'*64)
            self.assertEqual(events,['guard','guard'])

    def test_each_arm_rechecks_grant_before_next_identity_or_helper(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent/'artifacts') as d:
            root=Path(d);fake,_=self.proof(root);events=[]
            fake.require_main_guard=lambda p,h:events.append('guard')
            raw=SimpleNamespace(check_frozen_assets=lambda assets:events.append('assets'),identity=lambda model:events.append('identity') or {'path':'/forbidden.gguf'},raw_record_metadata=lambda *args:('header','keys','layout'))
            fake.raw_module=lambda:raw;fake.validate_raw_finite=lambda *args:None;fake.write=lambda p,v:None
            plan=dict(model_from_prior={'path':'/forbidden.gguf'});qualplan=self.shader_plan()
            with patch.object(s,'module',return_value=fake),patch.object(s,'qualification',return_value=(qualplan,{})),patch.object(s,'verified',return_value=(plan,{})),patch.object(s,'require_root_grant',side_effect=[{},ValueError('revoked before arm')]) as grant,patch.object(s.subprocess,'run',side_effect=AssertionError('helper must not launch')):
                with self.assertRaisesRegex(ValueError,'revoked before arm'):s.execute(root,'e'*64)
            self.assertEqual(grant.call_count,2);self.assertEqual(events.count('identity'),1);self.assertEqual(events[-1],'guard')

    def test_policy_screen_selection_and_qualification_anchors_byte_unchanged(self):
        old=(s.QUAL_WORK/'speed-bench/ideas/prepare_resident_env_calibration.py').read_text();new=SOURCE.read_text()
        old_tree=ast.parse(old);new_tree=ast.parse(new)
        for name in ['arms','policy','paired','decision','qualification','reference_assets']:
            left=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            right=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.get_source_segment(old,left),ast.get_source_segment(new,right),name)
        for name in ['QUAL_SOURCE_SHA','QUAL_MANIFEST_SHA','QUAL_COMMAND_SHA','QUAL_PINS','REF_SHA','ANCHOR_RAWS','SCREEN']:
            def value(tree):return next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
            self.assertEqual(ast.dump(value(old_tree),include_attributes=False),ast.dump(value(new_tree),include_attributes=False),name)


if __name__=='__main__':unittest.main()
