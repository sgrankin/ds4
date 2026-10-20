"""Model-free record/authority mutations for the nineteen-arm qualification."""
import csv
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'speed-bench/ideas/prepare_resident_native_weights_study.py'
spec=importlib.util.spec_from_file_location('weights_study',SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class Records(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=SOURCE.parents[0]/'artifacts')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.raw=s.raw_module()
        self.cases=[dict(case=f'case_{i:03d}') for i in range(16)]
        self.keys=[self.raw.RAW_ROW_META.pack(i,j,j+1,17,123456+i) for i in range(16) for j in range(32)]
        self.metadata=(self.raw.RAW_HEADER.pack(0x44535131,8,16,32),self.keys,{})
        self.rows=[dict(zip(s.CSV_FIELDS,[self.cases[i]['case'],str(j),str(j+1),'17','2','2.25','2.25','.25','.25','.03','.03','.04','.04','0','1.125'])) for i in range(16) for j in range(32)]
        self.quality=dict(cases=16,tokens=512,argmax_flips=0,bit_equal_rows=0,avg_nll=2.25,delta_nll=.25,policy_delta_nll=.25,kl_base_to_raw=.03,kl_base_to_policy=.03,tv=.04,startup_ms=1,prefill_ms=2,decode_ms=3,metrics_ms=4)

    def csv(self,rows=None,fields=None,name='quality.csv'):
        path=self.root/name
        with path.open('w') as f:
            out=csv.DictWriter(f,fieldnames=fields or s.CSV_FIELDS);out.writeheader();out.writerows(self.rows if rows is None else rows)
        return path

    def rawfile(self,name='rows.raw'):
        path=self.root/name
        with path.open('wb') as f:
            f.write(self.metadata[0])
            for key in self.keys:f.write(key);f.write(struct.pack('<8I',0,0x80000000,1,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000,0))
        return path

    def test_full_raw_finite_and_signed_subnormals(self):
        path=self.rawfile()
        with patch.object(s,'VOCAB',8):self.assertEqual(s.validate_raw_finite(path,self.metadata,self.raw),dict(vectors=512,fp32_values=4096,finite=True))
        # Mutation is the last FP32 value in the last row, not a scalar summary.
        for bits in (0x7fc00001,0x7f800000,0xff800000):
            with path.open('r+b') as f:f.seek(-4,2);f.write(struct.pack('<I',bits))
            with patch.object(s,'VOCAB',8),self.assertRaises(ValueError):s.validate_raw_finite(path,self.metadata,self.raw)

    def test_raw_truncation_header_metadata_extent(self):
        original=self.rawfile().read_bytes()
        for name,data in [('cut',original[:-1]),('extra',original+b'x'),('header',b'xxxx'+original[4:]),('meta',original[:16]+b'xxxx'+original[20:])]:
            path=self.root/name;path.write_bytes(data)
            with patch.object(s,'VOCAB',8),self.assertRaises(ValueError):s.validate_raw_finite(path,self.metadata,self.raw)

    def test_quality_csv_and_arithmetic(self):
        self.assertEqual(len(s.validate_csv(self.csv(),self.metadata,self.cases,self.raw)),512)
        self.assertEqual(s.validate_quality_json(json.dumps(self.quality),self.rows),self.quality)

    def test_csv_coverage_order_types_finite(self):
        for change in ('duplicate','missing','extra','target','prompt','case','flip','nan','inf','delta','tv'):
            rows=[dict(r) for r in self.rows]
            if change=='duplicate':rows[-1]=rows[-2]
            elif change=='missing':rows.pop()
            elif change=='extra':rows.append(rows[-1])
            else:
                key,value={'target':('target','0'),'prompt':('prompt_tokens','18'),'case':('case','foreign'),'flip':('argmax_flip','2'),'nan':('raw_nll','nan'),'inf':('decode_ms','inf'),'delta':('delta_raw_nll','.5'),'tv':('raw_tv','2')}[change]
                rows[-1][key]=value
            with self.assertRaises(ValueError,msg=change):s.validate_csv(self.csv(rows,name=change+'.csv'),self.metadata,self.cases,self.raw)

    def test_csv_duplicate_header_and_extra_column(self):
        path=self.csv();data=path.read_text()
        for name,mutated in [('duplicate',data.replace('case,step,','case,case,',1)),('extra',data+'case_015,31,32,17,2,2.25,2.25,.25,.25,.03,.03,.04,.04,0,1.125,excess\n')]:
            p=self.root/(name+'.csv');p.write_text(mutated)
            with self.assertRaises(ValueError):s.validate_csv(p,self.metadata,self.cases,self.raw)

    def test_quality_json_typed_schema_duplicate_aggregate(self):
        for key,value in [('tokens',True),('tokens',511),('bit_equal_rows',513),('argmax_flips',1),('avg_nll',float('inf')),('kl_base_to_raw',.1),('startup_ms',-1)]:
            data=dict(self.quality);data[key]=value
            with self.assertRaises(ValueError):s.validate_quality_json(json.dumps(data),self.rows)
        for text in [json.dumps(self.quality)[:-1]+',"tokens":512}',json.dumps(dict(self.quality,extra=0)),json.dumps({k:v for k,v in self.quality.items() if k!='tv'})]:
            with self.assertRaises(ValueError):s.validate_quality_json(text,self.rows)

    def state(self):
        phases=[dict(kind='P',cached=0,tokens=3,live=3),dict(kind='D',cached=0,tokens=2,live=5)]
        fixture=self.root/'fixture.txt';fixture.write_text('P 0 3 1 2 3\nD 0 2 4 5\n')
        logits=self.root/'state.logits';logits.write_bytes(struct.pack('<16f',*([1.]*16)))
        snapshot=self.root/'state.snapshot';pos=5
        h=(0x34565344,2,8192,1,128,128,8193,pos,40,512,128,8,0x413431)
        with patch.object(s,'VOCAB',8):
            data=s.SNAPSHOT_HEADER.pack(*h)+struct.pack('<5I',1,2,3,4,5)+struct.pack('<8f',*([1.]*8))
            # Legal internal sentinel bytes are intentionally not interpreted as logits.
            data+=b'\xff'*(s.snapshot_bytes(pos)-len(data));snapshot.write_bytes(data)
        return fixture,phases,logits,snapshot

    def test_canonical_state_snapshot_and_sentinels(self):
        fixture,phases,logits,snapshot=self.state()
        with patch.object(s,'VOCAB',8):
            proof=s.validate_state_artifacts(logits,snapshot,fixture,phases)
            self.assertEqual(proof['history_tokens'],5);self.assertTrue(proof['internal_state_sentinels_preserved'])
        # Source-derived formula at actual geometry and final replay position.
        self.assertEqual(s.snapshot_bytes(2665),s.SNAPSHOT_HEADER.size+(2665+s.VOCAB)*4+40*128*512*4+3*1332*640*4+2665*640*4+3*2*512*4)

    def test_state_header_history_truncation_nonfinite(self):
        fixture,phases,logits,snapshot=self.state();good=snapshot.read_bytes();vectors=logits.read_bytes()
        for name,data in [('header',b'xxxx'+good[4:]),('token',good[:52]+struct.pack('<I',7)+good[56:]),('truncate',good[:-1]),('extra',good+b'x'),('nan',good[:72]+struct.pack('<I',0x7fc00000)+good[76:]),('different',good[:72]+struct.pack('<f',2)+good[76:])]:
            snapshot.write_bytes(data)
            with patch.object(s,'VOCAB',8),self.assertRaises(ValueError,msg=name):s.validate_state_artifacts(logits,snapshot,fixture,phases)
        snapshot.write_bytes(good)
        for bits in (0x7fc00000,0x7f800000,0xff800000):
            logits.write_bytes(vectors[:-4]+struct.pack('<I',bits))
            with patch.object(s,'VOCAB',8),self.assertRaises(ValueError):s.validate_state_artifacts(logits,snapshot,fixture,phases)

    def test_state_csv_typed_phases_schema(self):
        fixture,phases,_,_=self.state()
        rows=[dict(zip(s.STATE_FIELDS,['0','P','3','3','10','0'])),dict(zip(s.STATE_FIELDS,['1','D','5','2','10','1']))]
        path=self.csv(rows,s.STATE_FIELDS,'state.csv');self.assertEqual(s.validate_state_csv(path,phases),rows)
        for key,value in [('phase','0'),('tokens','3'),('kind','P'),('context','4'),('ms','nan'),('first_decode_ms','11')]:
            altered=[dict(r) for r in rows];altered[1][key]=value
            with self.assertRaises(ValueError):s.validate_state_csv(self.csv(altered,s.STATE_FIELDS,key+'.csv'),phases)
        for altered in [rows[:-1],rows+[rows[-1]],rows[::-1]]:
            with self.assertRaises(ValueError):s.validate_state_csv(self.csv(altered,s.STATE_FIELDS,'count.csv'),phases)
        bad=path.read_text().replace('phase,kind,','phase,phase,',1);path.write_text(bad)
        with self.assertRaises(ValueError):s.validate_state_csv(path,phases)
        fixture.write_text(fixture.read_text()+' 6')
        with self.assertRaises(ValueError):s.state_history(fixture,phases)

    def test_exact_io_schema_and_balance(self):
        good='PROBE_IO enabled=1 pread_calls=3 requested_bytes=90 returned_bytes=90 cache_loader_bytes=90 cache_hits=1 cache_misses=2 cache_evictions=0\n'
        self.assertEqual(s.parse_io(good)['requested_bytes'],90)
        for text in [good+good,good.replace('returned_bytes=90','returned_bytes=89'),good.replace('enabled=1','enabled=0'),good.replace('pread_calls=3','pread_calls=-1'),good.replace('cache_hits=1','cache_hits=1 cache_hits=1'),good+'ds4: resident-backfill x\n',good.replace('cache_evictions=0','extra=0')]:
            with self.assertRaises(ValueError):s.parse_io(text)

    def test_cache_selected_regime(self):
        footer='ds4:   streaming expert cache budget=6136 experts entries=100 expert=9.49 MiB target=60.00 GiB live=1.00 GiB, hits=1 misses=2 hit_rate=0.333 wraps=0 evictions=3 buffer_allocs=4 buffer_reuses=5\n'
        phases=''.join(f'REPLAY_CACHE phase={i} configured_slots=6136 resident_entries=100\n' for i in range(7))
        self.assertEqual(s.parse_cache(footer+phases,True)['misses'],2)
        for text in [footer+footer+phases,(footer+phases).replace('configured_slots=6136','configured_slots=6135',1),footer+phases+phases.splitlines()[0]+'\n',footer.replace('misses=2','misses=2 misses=2')+phases]:
            with self.assertRaises(ValueError):s.parse_cache(text,True)

    def test_fixed_order_and_policy_before_apply(self):
        specs=s.arm_specs();self.assertEqual(len(specs),19)
        first=next(i for i,a in enumerate(specs) if a['flag']=='1' and a['mode']=='apply')
        self.assertEqual(first,11);self.assertEqual([a['name'] for a in specs[:4]],['state-old-native','state-new-native0','state-old-approx0','state-new-approx0'])
        for a in specs:
            env=s.policy(a);self.assertEqual(env['DS4_METAL_V41_SELECTED_TILE'],'64')
            if a['mode']=='apply':self.assertEqual(env[s.P+'MAX_SUBS'],'3');self.assertEqual(env[s.P+'MAX_TOTAL_MASS'],'.40')
        for flag,gap in [('yes','.3'),('1','0'),('1','nan'),('1','inf'),('1','2')]:
            env=s.policy(specs[first]);env[s.P+'NATIVE_WEIGHTS']=flag;env[s.P+'GAP']=gap
            with self.assertRaises(ValueError):s.validate_policy(env)

    def test_original_case_hash_mapping(self):
        p=self.root/'p';c=self.root/'c';p.write_text('prompt');c.write_text('continuation')
        manifest=self.root/'inputs.tsv';manifest.write_text('# comment\n'+'\n'.join(f'case_{i:03d}\t{p}\t{c}' for i in range(16))+'\n')
        info=dict(manifest=str(manifest),sha256=s.sha(manifest),files={str(p):s.sha(p),str(c):s.sha(c)})
        cases=s.original_inputs(info);self.assertEqual(cases[0]['prompt_sha256'],s.sha(p));self.assertEqual(len(cases),16)
        relative=manifest.read_text().replace(str(p),'p').replace(str(c),'c');manifest.write_text(relative);info['sha256']=s.sha(manifest)
        with patch.object(s,'ROOT',self.root):self.assertEqual(s.original_inputs(info),cases)
        manifest.write_text(manifest.read_text().replace('case_015','case_014'))
        with self.assertRaises(ValueError):s.original_inputs(info)

    def test_guard_refuses_before_any_model_stat(self):
        # This is a model-free structural ordering assertion, not a forged live guard.
        source=SOURCE.read_text();body=source[source.index('def execute('):source.index('\ndef main(')]
        self.assertLess(body.index('require_main_guard(output,digest)'),body.index('identity=raw.identity(model)'))
        self.assertLess(body.index('validate_state_artifacts('),body.index('validate_quality_json('))
        self.assertIn('if not k.startswith(\'DS4\')',body)
        with patch.object(s.Path,'cwd',return_value=self.root),self.assertRaises(RuntimeError):s.require_main_guard(self.root,'0'*64)

    def test_exact_parent_suffix_framework_and_lock(self):
        import errno
        lock=self.root/'speed-bench/ideas/artifacts/model-job.lock';lock.parent.mkdir(parents=True);lock.write_text('')
        output=self.root/'prepared';digest='0'*64;argv=s.guard_command(output,digest)
        with patch.object(s,'ROOT',self.root),patch.object(s.Path,'cwd',return_value=self.root),patch.object(s,'sha',side_effect=lambda p:s.GUARD_SHA if p==s.GUARD else s.PYTHON_SHA),patch.object(s.subprocess,'check_output',return_value=__import__('shlex').join(argv)),patch.object(s.os.path,'samefile',return_value=True),patch.object(s.fcntl,'flock',side_effect=OSError(errno.EWOULDBLOCK,'held')):
            s.require_main_guard(output,digest)
        for mutation in ([str(s.PYTHON),'other.py',str(s.GUARD),str(output/'pressure')],argv+['extra'],[str(s.PYTHON)]+argv[1:-1]+['1'*64]):
            with patch.object(s,'ROOT',self.root),patch.object(s.Path,'cwd',return_value=self.root),patch.object(s,'sha',side_effect=lambda p:s.GUARD_SHA if p==s.GUARD else s.PYTHON_SHA),patch.object(s.subprocess,'check_output',return_value=__import__('shlex').join(mutation)),self.assertRaises(RuntimeError):s.require_main_guard(output,digest)
        with patch.object(s,'ROOT',self.root),patch.object(s.Path,'cwd',return_value=self.root),patch.object(s,'sha',side_effect=lambda p:s.GUARD_SHA if p==s.GUARD else s.PYTHON_SHA),patch.object(s.subprocess,'check_output',return_value=__import__('shlex').join(argv)),patch.object(s.os.path,'samefile',return_value=False),self.assertRaises(RuntimeError):s.require_main_guard(output,digest)
        with patch.object(s,'ROOT',self.root),patch.object(s.Path,'cwd',return_value=self.root),patch.object(s,'sha',side_effect=lambda p:s.GUARD_SHA if p==s.GUARD else s.PYTHON_SHA),patch.object(s.subprocess,'check_output',return_value=__import__('shlex').join(argv)),patch.object(s.os.path,'samefile',return_value=True),patch.object(s.fcntl,'flock',return_value=None),self.assertRaises(RuntimeError):s.require_main_guard(output,digest)

    def test_manifest_authority_and_exact_order(self):
        plan=dict(build_sha256=s.BUILD_SHA,source_plan_sha256=s.PLAN_SHA,settings=s.SETTINGS,arms=s.arm_specs(),guard_command_template=s.command_template(self.root),guard_sha256=s.GUARD_SHA,guard_interpreter=dict(path=str(s.PYTHON),sha256=s.PYTHON_SHA))
        s.write(self.root/'manifest.json',plan);digest=s.sha(self.root/'manifest.json')
        command=dict(manifest_sha256=digest,command=s.guard_command(self.root,digest));s.write(self.root/'prepared-command.json',command)
        self.assertEqual(s.verified_plan(self.root,digest),plan)
        altered=dict(plan);altered['arms']=plan['arms'][::-1]
        (self.root/'manifest.json').write_text(json.dumps(altered))
        with self.assertRaises(ValueError):s.verified_plan(self.root,digest)
        revised=s.sha(self.root/'manifest.json');command.update(manifest_sha256=revised,command=s.guard_command(self.root,revised));(self.root/'prepared-command.json').write_text(json.dumps(command))
        with self.assertRaises(ValueError):s.verified_plan(self.root,revised)


if __name__=='__main__':unittest.main()
