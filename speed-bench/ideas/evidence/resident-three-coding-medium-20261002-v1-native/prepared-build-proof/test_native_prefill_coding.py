#!/usr/bin/env python3
"""Model-free control-flow checks of the actual C pilot and frozen driver."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('prefill_coding_driver',ROOT/'speed-bench/ideas/run_native_prefill_coding.py')
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)
STUB=r'''
#include "ds4.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct ds4_engine { int evals; };
struct ds4_session { struct ds4_engine *engine; ds4_tokens tokens; int sampled; int valid; };
void ds4_tokens_push(ds4_tokens *t,int v){if(t->len==t->cap){t->cap=t->cap?t->cap*2:16;t->v=realloc(t->v,(size_t)t->cap*sizeof(int));}t->v[t->len++]=v;}
void ds4_tokens_free(ds4_tokens *t){free(t->v);memset(t,0,sizeof(*t));}
void ds4_tokens_copy(ds4_tokens *d,const ds4_tokens *s){ds4_tokens_free(d);for(int i=0;i<s->len;i++)ds4_tokens_push(d,s->v[i]);}
int ds4_engine_open(ds4_engine **e,const ds4_engine_options *o){assert(!strcmp(o->model_path,"MOCK"));assert(o->context_size==4096&&o->prefill_chunk==512&&o->mtp_draft_tokens==1&&o->ssd_streaming);*e=calloc(1,sizeof(**e));return 0;}
void ds4_engine_close(ds4_engine *e){fprintf(stderr,"mock_eval_total=%d\n",e->evals);free(e);}
const char *ds4_engine_model_name(ds4_engine *e){(void)e;return "DeepSeek V4.1 Flash";}
int ds4_engine_vocab_size(ds4_engine *e){(void)e;return 129280;}
int ds4_session_create(ds4_session **s,ds4_engine *e,int n){assert(n==4096);*s=calloc(1,sizeof(**s));(*s)->engine=e;return 0;}
void ds4_session_free(ds4_session *s){ds4_tokens_free(&s->tokens);free(s);}
void ds4_session_gpu_warmup(ds4_session *s){(void)s;}
int ds4_session_prefill_cap(ds4_session *s){(void)s;return 512;}
void ds4_session_invalidate(ds4_session *s){s->valid=0;s->sampled=0;ds4_tokens_free(&s->tokens);}
int ds4_session_sync(ds4_session *s,const ds4_tokens *t,char *error,size_t n){(void)error;(void)n;if(!s->valid){assert(!getenv("DS4_METAL_V41_SELECTED_TILE")&&!getenv("DS4_METAL_V41_SELECTED_LIMIT"));}else if(getenv("MOCK_CANDIDATE")){assert(!strcmp(getenv("DS4_METAL_V41_SELECTED_TILE"),"64")&&!strcmp(getenv("DS4_METAL_V41_SELECTED_LIMIT"),"1024"));}ds4_tokens_copy(&s->tokens,t);s->valid=1;return 0;}
int ds4_session_pos(ds4_session *s){return s->tokens.len;}
int ds4_session_common_prefix(ds4_session *s,const ds4_tokens *t){if(getenv("MOCK_MISMATCH"))return s->tokens.len-1;int i=0;while(i<s->tokens.len&&i<t->len&&s->tokens.v[i]==t->v[i])i++;return i;}
void ds4_chat_begin(ds4_engine *e,ds4_tokens *t){(void)e;ds4_tokens_push(t,10);}
void ds4_chat_append_think_prefix(ds4_engine *e,ds4_tokens *t,ds4_think_mode m){(void)e;assert(m==DS4_THINK_NONE);ds4_tokens_push(t,11);}
void ds4_chat_append_message(ds4_engine *e,ds4_tokens *t,const char *role,const char *text){(void)e;ds4_tokens_push(t,!strcmp(role,"user")?12:13);int n=strncmp(text,"Project context",15)?3:atoi(getenv("MOCK_APPEND")?getenv("MOCK_APPEND"):"400")-2;for(int i=0;i<n;i++)ds4_tokens_push(t,20+i);}
void ds4_chat_append_assistant_prefix(ds4_engine *e,ds4_tokens *t,ds4_think_mode m){(void)e;assert(m==DS4_THINK_NONE);ds4_tokens_push(t,14);}
int ds4_token_eos(ds4_engine *e){(void)e;return 2;}
bool ds4_token_is_stop_for_think_mode(ds4_engine *e,int token,ds4_think_mode m){(void)e;assert(m==DS4_THINK_NONE);return token==2;}
int ds4_session_sample(ds4_session *s,float temp,int k,float p,float min,uint64_t *rng){assert(temp==0&&k==0&&p==1&&min==0&&*rng==20261002);return ++s->sampled<=2?1:2;}
char *ds4_token_text(ds4_engine *e,int token,size_t *n){(void)e;char *p=malloc(2);p[0]=token==2?'!':'x';p[1]=0;*n=1;return p;}
int ds4_session_eval(ds4_session *s,int token,char *error,size_t n){(void)error;(void)n;s->engine->evals++;ds4_tokens_push(&s->tokens,token);return 0;}
int ds4_session_copy_logits(ds4_session *s,float *f,int n){(void)s;assert(n==129280);for(int i=0;i<n;i++)f[i]=(float)-i;return n;}
'''

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'speed-bench/ideas/artifacts',prefix='coding-cpu-')
        cls.work=Path(cls.tmp.name); stub=cls.work/'mock.c';stub.write_text(STUB)
        cls.binary=cls.work/'mock-pilot'
        subprocess.run(['cc','-O0','-std=c11','-Wall','-Wextra','-I',str(ROOT),
            str(ROOT/'speed-bench/native_prefill_coding_pilot.c'),str(stub),'-o',str(cls.binary)],check=True)
        fixture=json.loads((ROOT/'speed-bench/ideas/resident-coding/fixtures-v1.json').read_text())
        cls.order=fixture['task_order'];cls.prompts=cls.work/'prompts';cls.prompts.mkdir()
        (cls.prompts/'tasks.txt').write_text('\n'.join(cls.order)+'\n')
        for task in fixture['tasks']:(cls.prompts/(task['id']+'.txt')).write_text(task['prompt'])
        assembly=json.loads((ROOT/'speed-bench/ideas/native-prefill-coding/assembly-v1.json').read_text())
        for key,name in [('warm_user','warm-user.txt'),('warm_assistant','warm-assistant.txt'),('append_preamble','preamble.txt')]:
            (cls.prompts/name).write_text(assembly[key])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def pilot(self,name,cap='512',**overrides):
        out=self.work/name
        rc=subprocess.run([str(self.binary),'MOCK',str(out),str(self.prompts),'64',cap],
            env=dict(os.environ,**overrides),capture_output=True,text=True)
        return out,rc
    def test_real_c_success_and_exact_assembly(self):
        out,rc=self.pilot('valid');self.assertEqual(rc.returncode,0,rc.stderr)
        records,meta=D.validate_arm(out,{'task_order':self.order})
        self.assertEqual(len(records),4)
        self.assertTrue(all(r['append_tokens']==400 and r['warm_tokens']==10 for r in records))
        self.assertTrue(all(r['stopped'] and r['eos'] and not r['cap_hit'] and r['generated_tokens']==2 for r in records))
        self.assertIn('mock_eval_total=8',rc.stderr)
        for task in self.order:
            prompt=json.loads((out/(task+'.prompt.json')).read_text())
            self.assertEqual(prompt['ids'][:10],[10,11,12,20,21,22,13,20,21,22])
    def test_candidate_knobs_only_restored_after_each_native_warm(self):
        _,rc=self.pilot('candidate',MOCK_CANDIDATE='1',DS4_METAL_V41_SELECTED_TILE='64',DS4_METAL_V41_SELECTED_LIMIT='1024')
        self.assertEqual(rc.returncode,0,rc.stderr)
        self.assertEqual(rc.stderr.count('pos=0 tile=default limit=default'),4)
        self.assertEqual(rc.stderr.count('pos=10 count=400 tile=64 limit=1024'),4)
    def test_minimum256_is_accepted(self):
        _,rc=self.pilot('min',MOCK_APPEND='256');self.assertEqual(rc.returncode,0,rc.stderr)
    def test_maximum1023_is_accepted(self):
        _,rc=self.pilot('max',MOCK_APPEND='1023');self.assertEqual(rc.returncode,0,rc.stderr)
    def test_low_append_rejected_before_generation(self):
        out,rc=self.pilot('low',MOCK_APPEND='255');self.assertEqual(rc.returncode,2);self.assertFalse((out/(self.order[0]+'.txt')).exists())
    def test_high_append_rejected_before_generation(self):
        out,rc=self.pilot('high',MOCK_APPEND='1024');self.assertEqual(rc.returncode,2);self.assertFalse((out/(self.order[0]+'.txt')).exists())
    def test_existing_prefix_mismatch_rejected(self):
        out,rc=self.pilot('mismatch',MOCK_MISMATCH='1');self.assertEqual(rc.returncode,2);self.assertFalse((out/(self.order[0]+'.txt')).exists())
    def test_last_capped_token_is_not_evaluated(self):
        out,rc=self.pilot('cap',cap='1');self.assertEqual(rc.returncode,0,rc.stderr)
        self.assertIn('mock_eval_total=0',rc.stderr)
        records=[json.loads(s) for s in (out/'generation.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['cap_hit'] and not r['stopped'] and r['generated_tokens']==1 for r in records))
    def test_frozen_asset_tampering_rejected(self):
        asset=self.work/'asset';asset.write_text('before');plan={'asset_sha256':{str(asset):D.sha(asset)}}
        D.check_assets(plan);asset.write_text('after')
        with self.assertRaisesRegex(ValueError,'frozen asset changed'):D.check_assets(plan)
    def test_unheld_guard_rejected(self):
        with patch.object(D.subprocess,'check_output',return_value='/bin/unrelated-parent'):
            with self.assertRaisesRegex(RuntimeError,'MAIN pressure guard'):D.require_guard()
    def test_fixed_text_preserves_original_task_gold_fixture(self):
        fixture=json.loads((ROOT/'speed-bench/ideas/resident-coding/fixtures-v1.json').read_text())
        self.assertEqual(sum(len(t['tests']) for t in fixture['tasks']),34)
        self.assertEqual(D.sha(ROOT/'speed-bench/ideas/resident-coding/fixtures-v1.json'),'d1ea470ff2cb70476ee6504a55a06b8756156a3ca2df67739c556d4941e5f2c7')
        assembly=json.loads((ROOT/'speed-bench/ideas/native-prefill-coding/assembly-v1.json').read_text())
        self.assertTrue(assembly['append_preamble'].endswith('Function contract:\n'))
        self.assertTrue(all(t['prompt']==(self.prompts/(t['id']+'.txt')).read_text() for t in fixture['tasks']))

    def test_native_controls_use_exact_bytes_and_native_decisions(self):
        out,rc=self.pilot('controls');self.assertEqual(rc.returncode,0,rc.stderr)
        expected={task:{suffix:D.sha(out/(task+'.'+suffix)) for suffix in
            ('warm-logits.bin','initial-logits.bin','prompt.json','txt')} for task in self.order}
        for task in self.order:
            expected[task]['native_decision_sha256']=D.decision_hash(out/(task+'.tokens.jsonl'))
        plan={'task_order':self.order,'native_controls':expected}
        D.require_native_control(out,plan)
        target=out/(self.order[0]+'.initial-logits.bin')
        data=bytearray(target.read_bytes());data[-1]^=1;target.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'native historical control differs'):
            D.require_native_control(out,plan)

    def test_native_decision_change_rejected(self):
        out,rc=self.pilot('decision-control');self.assertEqual(rc.returncode,0,rc.stderr)
        task=self.order[0];path=out/(task+'.tokens.jsonl')
        before=D.decision_hash(path)
        rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['token_id']+=1
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        self.assertNotEqual(before,D.decision_hash(path))

    def test_candidate_footer_requires_real_substitutions(self):
        path=self.work/'footer'
        footer='ds4: resident-route layers=40 substitutions=3 native_misses=9 skipped_loads=3 skipped_bytes=10 full_snapshots=40 double_layers=0 planned_substitutions=3 drops=0 triple_layers=1\n'
        path.write_text(footer);self.assertEqual(D.route_footer(path)['triple_layers'],1)
        for bad in ('',footer+footer,footer.replace('substitutions=3','substitutions=0'),footer.replace('drops=0','drops=1'),footer.replace('skipped_loads=3','skipped_loads=2')):
            path.write_text(bad)
            with self.assertRaises(ValueError):D.route_footer(path)

    def test_all_inherited_ds4_keys_removed(self):
        with patch.dict(os.environ,{'DS4': 'bad','DS4_FAKE':'bad','DS4_METAL_V41_RESIDENT_ROUTE_BACKFILL':'1','KEEP':'yes'},clear=True):
            env=D.clean_environment({'metal_environment':{'DS4_METAL_DSV41_SOURCE':'fixed'}})
        self.assertEqual(env,{'KEEP':'yes','DS4_METAL_DSV41_SOURCE':'fixed'})

    def test_frozen_arms_hold_prefill_and_cache_constant(self):
        self.assertEqual([name for name,_,_ in D.ARMS],['native-before','broader3','native-after'])
        self.assertTrue(all(cache==64 and p['DS4_METAL_V41_SELECTED_TILE']=='64' and p['DS4_METAL_V41_SELECTED_LIMIT']=='1024' for _,cache,p in D.ARMS))
        self.assertEqual(D.ARMS[1][2]['DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS'],'3')
        self.assertEqual(D.ARMS[0][2],D.ARMS[2][2])

    def test_model_hashing_forbidden(self):
        with self.assertRaisesRegex(ValueError,'model hashing forbidden'):D.sha('anything.gguf')

if __name__=='__main__':unittest.main(verbosity=2)
