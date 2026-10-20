import copy
import ctypes
import errno
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=load('analysis',ROOT/'speed-bench/analyze_native_effort_holdout.py')
fixture=json.loads((ROOT/'speed-bench/experiments/ds41-ssd/native-effort-holdout/input-freeze-v1.json').read_text())

def lines(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

def synthetic(root,level):
    rows=[]
    for index,case in enumerate(fixture['cases']):
        expected=case['answer'];thinking=level>0
        if index==2:pieces=[b'x']*2048
        elif index==4:pieces=([b'</think>'] if thinking else [])+[b' ']*(2047-int(thinking))+[expected.encode()]
        else:
            text=expected if index in (0,5) else '0' if index==1 else 'answer: '+expected
            if index==5:text='**\\boxed{+000'+expected+'}**'
            pieces=([b'</think>'] if thinking else [])+[text.encode()]
        cap=index in (2,4);trace=[];raw=b'';final=b'';closed=not thinking;reasoning=0;gclose=False
        for step,piece in enumerate(pieces+([] if cap else [b'<eos>'])):
            stop=not cap and step==len(pieces)
            trace.append(dict(step=step,token_id=1 if stop else 128822 if piece==b'</think>' else 42,
                stop=stop,inside_thinking=thinking and not closed,generated_thinking_tokens=step if thinking and not closed else 0,
                policy_active=False,penalty=0,rng_before=f'{20261002+step:016x}',rng_after=f'{20261003+step:016x}',piece_hex=piece.hex()))
            if stop:continue
            raw+=piece
            if not closed:reasoning+=1
            if piece==b'</think>':closed=True;gclose=True;final=b''
            elif closed:final+=piece
        parsed=a.parsed_integer(final.decode()) if closed else None
        status='INVALID' if parsed is None else 'CORRECT' if parsed==expected else 'INCORRECT'
        reason='none' if parsed is not None else 'unclosed_reasoning' if not closed else 'empty_final' if not final.strip() else 'unsupported_final_format'
        (root/(case['case']+'.txt')).write_bytes(raw);lines(root/(case['case']+'.tokens.jsonl'),trace)
        (root/(case['case']+'.prompt.ids')).write_text('0\n42\n')
        row=dict(case=case['case'],suite='independent-counting-v1',think_level=level,native_penalty=0,think_enabled=thinking,
            prompt_supplied_think_close=not thinking,generated_think_close=gclose,status=status,parse_ok=parsed is not None,
            parse_reason=reason,scorer='strict-final-v2',seed=20261002,temperature_float_value=a.TEMP,top_p_float_value=a.TOP_P,
            min_p=0,max_tokens=2048,cache_gib_requested=64,context_size=8192,prefill_chunk=512,prompt_tokens=2,
            generated_tokens=len(pieces),reasoning_tokens=reasoning,think_closed=closed,stopped=not cap,cap_hit=cap,
            correct_exact=status=='CORRECT',prefill_seconds=1.,decode_seconds=1.,parsed_answer=parsed or '',final=final.decode(),
            expected=expected,prompt=case['prompt'],raw_trace=case['case']+'.txt',token_trace=case['case']+'.tokens.jsonl',prompt_trace=case['case']+'.prompt.ids')
        rows.append(row)
    lines(root/'results.jsonl',rows);return rows

class Analysis(unittest.TestCase):
    def setUp(self):
        base=ROOT/'speed-bench/ideas/artifacts/holdout-cpu-fixtures';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def check(self,level,rc=3):return a.validate_arm(self.root,level,fixture,rc)
    def test_complete_first_attempt_caps_wrong_and_offset0(self):
        for level in (0,25,75):
            synthetic(self.root,level);report,rows,_=self.check(level)
            self.assertEqual((report['completed_correct'],report['capped_correct'],report['incorrect'],report['invalid'],report['cap_hits']),(2,1,1,2,2))
            if level==0:
                self.assertEqual(report['reasoning_tokens'],0)
                self.assertEqual(rows[0]['final'],fixture['cases'][0]['answer'])
                self.assertTrue(rows[0]['prompt_supplied_think_close']);self.assertFalse(rows[0]['generated_think_close'])
    def test_native_repeat_exact_but_timing_ignored(self):
        synthetic(self.root,25);first=self.check(25)
        rows=a.jsonl(self.root/'results.jsonl');rows[0]['decode_seconds']=99.;lines(self.root/'results.jsonl',rows)
        self.assertTrue(a.require_repeat(first,self.check(25)))
    def test_repeat_rng_or_final_drift_fails(self):
        synthetic(self.root,25);first=self.check(25);second=copy.deepcopy(first)
        second[2][0]['trace'][0]['rng_after']='0000000000000001'
        with self.assertRaises(ValueError):a.require_repeat(first,second)
    def test_duplicate_missing_reordered_records_rejected(self):
        rows=synthetic(self.root,25)
        for wrong in [rows[:-1],rows[:5]+[rows[0]],list(reversed(rows))]:
            lines(self.root/'results.jsonl',wrong)
            with self.assertRaises(ValueError):self.check(25)
    def test_typed_settings_finite_time_and_gold_tamper(self):
        rows=synthetic(self.root,25)
        for key,value in [('native_penalty',False),('decode_seconds',float('nan')),('temperature_float_value',1.),
                          ('cache_gib_requested',80),('expected','0'),('generated_tokens',True),('prompt','changed')]:
            wrong=copy.deepcopy(rows);wrong[0][key]=value;lines(self.root/'results.jsonl',wrong)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(25)
    def test_rng_chain_and_complete_stop_rows(self):
        synthetic(self.root,0);p=self.root/(fixture['cases'][0]['case']+'.tokens.jsonl');rows=a.jsonl(p)
        wrong=copy.deepcopy(rows);wrong[-1]['rng_before']='0000000000000001';lines(p,wrong)
        with self.assertRaises(ValueError):self.check(0)
        lines(p,rows[:-1])
        with self.assertRaises(ValueError):self.check(0)
    def test_no_thinking_state_and_native_policy_fail_closed(self):
        synthetic(self.root,0);p=self.root/(fixture['cases'][0]['case']+'.tokens.jsonl');rows=a.jsonl(p)
        for key,value in [('inside_thinking',True),('policy_active',True),('penalty',.5),('generated_thinking_tokens',1)]:
            wrong=copy.deepcopy(rows);wrong[0][key]=value;lines(p,wrong)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(0)
    def test_raw_prompt_and_final_region_tamper(self):
        rows=synthetic(self.root,0)
        wrong=copy.deepcopy(rows);wrong[0]['final']='0';lines(self.root/'results.jsonl',wrong)
        with self.assertRaises(ValueError):self.check(0)
        lines(self.root/'results.jsonl',rows);(self.root/rows[0]['raw_trace']).write_bytes(b'0')
        with self.assertRaises(ValueError):self.check(0)
        synthetic(self.root,0);(self.root/rows[0]['prompt_trace']).write_text('0\n')
        with self.assertRaises(ValueError):self.check(0)
    def test_original_invalid_rc_preserved(self):
        synthetic(self.root,25)
        for rc in (0,True,None):
            with self.subTest(rc=rc),self.assertRaises(ValueError):self.check(25,rc)
    def test_strict_integer_not_matching_substring(self):
        for value in ['answer: 43','43 or 44','43 trailing','',str(1<<63),'\u200343\u2003']:
            self.assertIsNone(a.parsed_integer(value))
        for value,want in [('43','43'),('**+00043**','43'),('\\boxed{-4}','-4'),('```text\n43\n```','43')]:
            self.assertEqual(a.parsed_integer(value),want)
    def test_signed64_boundaries_and_long_inputs_match_native_strtoll(self):
        libc=ctypes.CDLL(None,use_errno=True)
        strtoll=libc.strtoll;strtoll.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_char_p),ctypes.c_int];strtoll.restype=ctypes.c_longlong
        limit_before=sys.get_int_max_str_digits()
        cases=['0','+0','-0','9223372036854775807','9223372036854775808',
               '-9223372036854775808','-9223372036854775809','9'*5000,
               '+'+'9'*5000,'-'+'9'*5000,'0'*5000+'43',
               '+'+'0'*5000+'43','-'+'0'*5000+'43','-'+'0'*5000,
               '0'*5000+'9223372036854775807','-'+'0'*5000+'9223372036854775808',
               '0'*5000+'9223372036854775808','-'+'0'*5000+'9223372036854775809']
        for value in cases:
            ctypes.set_errno(0);end=ctypes.c_char_p();n=strtoll(value.encode(),ctypes.byref(end),10)
            error=ctypes.get_errno();self.assertIn(error,(0,errno.ERANGE));self.assertEqual(end.value,b'')
            expected=None if error==errno.ERANGE else str(n)
            with self.subTest(prefix=value[:24],length=len(value)):
                self.assertEqual(a.parsed_integer(value),expected)
                self.assertEqual(a.parsed_integer('**\\boxed{'+value+'}**'),expected)
        self.assertEqual(sys.get_int_max_str_digits(),limit_before)
    def test_long_piece_final_does_not_abort_remaining_first_attempts(self):
        for value,parsed,status in [('9'*5000,'','INVALID'),('0'*5000+fixture['cases'][0]['answer'],fixture['cases'][0]['answer'],'CORRECT')]:
            rows=synthetic(self.root,0);row=rows[0]
            trace=a.jsonl(self.root/row['token_trace']);trace[0]['piece_hex']=value.encode().hex()
            lines(self.root/row['token_trace'],trace);(self.root/row['raw_trace']).write_bytes(value.encode())
            row.update(final=value,parsed_answer=parsed,parse_ok=bool(parsed),status=status,
                       parse_reason='none' if parsed else 'unsupported_final_format',correct_exact=status=='CORRECT')
            lines(self.root/'results.jsonl',rows)
            report,checked,_=self.check(0)
            self.assertEqual(len(checked),6);self.assertEqual(checked[-1]['case'],fixture['cases'][-1]['case'])
            self.assertEqual(checked[0]['status'],status)
            self.assertEqual(report['invalid'],3 if status=='INVALID' else 2)

if __name__=='__main__':unittest.main()
