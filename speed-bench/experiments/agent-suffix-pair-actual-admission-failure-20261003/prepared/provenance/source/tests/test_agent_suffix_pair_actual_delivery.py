"""Exact controller-to-production-input-buffer byte contract; no native model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

WORK=Path(__file__).resolve().parents[1];IDEAS=WORK/'speed-bench/ideas'
spec=importlib.util.spec_from_file_location('actual_delivery_controller',IDEAS/'run_agent_suffix_pair_actual.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
DRAFT=IDEAS/'agent_suffix_pair_actual_v1'
CASES=Path(os.environ['PAIR_ACTUAL_DELIVERY_CASES'])
PROBE=Path(os.environ['PAIR_ACTUAL_DELIVERY_PROBE'])


def function(text,name):
    match=re.search(r'^static [^\n]*\b'+re.escape(name)+r'\(',text,re.M);start=match.start();i=text.index('{',match.end());depth=0
    while i<len(text):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i+1]
        i+=1
    raise AssertionError('unclosed source function')


class Delivery(unittest.TestCase):
    def setUp(self):self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=CASES))

    def literal(self):
        raw=(WORK/'speed-bench/agent_suffix_pair_actual_literals.h').read_text()
        return json.loads(re.search(r'pair_actual_third_user = (".*");',raw).group(1)).encode()

    def test_file_task_literal_and_delivery_bind_exactly(self):
        task=json.loads((DRAFT/'task.json').read_bytes());third=(DRAFT/'third.txt').read_bytes()
        self.assertEqual(len(third),1798);self.assertFalse(third.endswith(b'\n'))
        self.assertEqual(third,task['third_user_text'].encode());self.assertEqual(self.literal(),third+b'\n')
        self.assertEqual(c.delivery_payload(task['third_user_text']),self.literal())
        self.assertEqual(hashlib.sha256(third).hexdigest(),'6e279a665d5389ed95bb59580f1366e8d1dafc4df25b627557772191aac987d0')
        self.assertEqual(c.TASK_SHA,'eb5ff21a8d620e22d67c37d438690e1e8c37b69c7592999d4cf399fc7607e283')

    def test_no_trimming_or_tolerant_lf_handling(self):
        for text in ['a\nb','a\nb\n','\na\nb\n','a\r\nb','é\n漢字']:
            self.assertEqual(c.delivery_payload(text),text.encode()+b'\n')
        third=(DRAFT/'third.txt').read_text()
        self.assertNotEqual(c.delivery_payload(third+'\n'),self.literal())
        self.assertNotEqual(c.delivery_payload(third[:-1]),self.literal())

    def test_nul_and_nonstrings_refused_before_write(self):
        for text in ['\0','prefix\0suffix','a\nb\0',None,b'text',1]:
            with self.assertRaises(c.ControlFailure):c.delivery_payload(text)

    def test_atomic_bound_includes_added_protocol_byte(self):
        self.assertEqual(len(c.delivery_payload('a'*4095)),4096)
        with self.assertRaises(c.ControlFailure):c.delivery_payload('a'*4096)

    def test_exact_production_extractions_have_no_native_dependencies(self):
        proof=json.loads((IDEAS/'agent_suffix_pair_actual_delivery_v1/protocol-extraction-proof.json').read_text())
        production=(WORK/'ds4_agent.c').read_text();fixture=(WORK/'speed-bench/agent_suffix_pair_actual_protocol_fixture.c').read_text()
        for name,pin in proof['functions'].items():
            original=function(production,name);self.assertEqual(function(fixture,name),original,name);self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),pin['sha256'])
        for forbidden in ['ds4_engine_open','ds4_session_eval','ds4_tokenize','Metal','ds4_session_create']:
            self.assertNotIn(forbidden,fixture.replace('No native objects or model.',''))

    def test_actual_source_gate_all_conditions_and_bad_lf(self):
        result=subprocess.run([str(PROBE),'--gate-selftest'],capture_output=True,text=True,timeout=5)
        (self.root/'stdout').write_text(result.stdout);(self.root/'stderr').write_text(result.stderr)
        self.assertEqual(result.returncode,0);self.assertEqual(result.stdout.strip(),'production protocol exact gate: PASS10')

    def protocol(self,prompts):
        output=self.root/'output';project=self.root/'project';private=self.root/'private'
        for p in [output,project,private]:p.mkdir()
        # Only semantic/model-completion validators are replaced in this
        # byte-delivery fixture. The actual controller drive/write/framing,
        # production input read/buffer/take/worker_submit and cleanup run.
        with patch.object(c,'rounds',return_value=[]),patch.object(c,'validate_rounds',return_value=None),patch.object(c,'validate_control',side_effect=lambda trace,records,turn:dict(turn=turn,passed=True,scope='CPU protocol bytes only')):
            rc=c.drive([str(PROBE),'--protocol',str(private)],prompts,output,project,private,self.root/'trace',dict(os.environ),{},lambda records,deadline:dict(turn=3,passed=True,scope='CPU protocol bytes only'),timeout=8,cleanup_reserve=1)
        self.assertEqual(rc,0,(output/'controller-outcome.json').read_text())
        for i,prompt in enumerate(prompts,1):
            self.assertEqual((private/f'received-{i}.bin').read_bytes(),c.delivery_payload(prompt))
            self.assertEqual((private/f'submitted-{i}.bin').read_bytes(),c.delivery_payload(prompt))
        outcome=json.loads((output/'controller-outcome.json').read_text());self.assertEqual(outcome['completed'],[1,2,3]);self.assertEqual(outcome['corrective_infrastructure_attempt'],c.CORRECTION)
        return private

    def test_actual_controller_delivers_all_three_immutable_multiline_prompts(self):
        prompts=[(DRAFT/name).read_text() for name in ('first.txt','second.txt','third.txt')]
        private=self.protocol(prompts);self.assertEqual((private/'gate-3.bin').read_bytes(),b'PASS')

    def test_multiline_and_extra_lf_preserved_then_exact_gate_refuses(self):
        private=self.protocol(['line1\nline2','lead\n\nend\n',(DRAFT/'third.txt').read_text()+'\n'])
        self.assertEqual((private/'gate-3.bin').read_bytes(),b'FAIL')

    def test_ownership_header_and_task_sampling_bytes_unchanged(self):
        original=Path('/Volumes/Models/ds4/.workspaces/idea-agent-suffix-pair')
        for path in ['speed-bench/agent_suffix_pair_actual_ownership.h','speed-bench/agent_suffix_pair_actual.c','ds4_agent.c','ds4.c','ds4.h','speed-bench/ideas/agent_suffix_pair_actual_v1/task.json']:
            self.assertEqual((WORK/path).read_bytes(),(original/path).read_bytes(),path)
        spec=importlib.util.spec_from_file_location('original_controller',original/'speed-bench/ideas/run_agent_suffix_pair_actual.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        self.assertEqual(c.POLICY,old.POLICY);self.assertEqual(c.MODEL,old.MODEL);self.assertEqual(c.TASK_SHA,old.TASK_SHA)

    def test_observation_only_hook_delta_keeps_all_admission_branches_exact(self):
        original=Path('/Volumes/Models/ds4/.workspaces/idea-agent-suffix-pair/speed-bench/agent_suffix_pair_actual_hook.inc').read_text()
        current=(WORK/'speed-bench/agent_suffix_pair_actual_hook.inc').read_text()
        lines=[line for line in current.splitlines(True) if not line.startswith('#include "agent_suffix_pair_actual_admission.h"') and 'if(!pair_actual_admission_record(' not in line]
        self.assertEqual(''.join(lines),original)
        self.assertIn('#include "agent_suffix_pair_actual_admission.h"',current)
        self.assertIn('#include "agent_suffix_pair_actual_admission.h"',(WORK/'speed-bench/agent_suffix_pair_actual_protocol_fixture.c').read_text())

    def test_shared_actual_ledger_typed_reservations_and_boundary_failures(self):
        result=subprocess.run([str(PROBE),'--admission-selftest',str(self.root)],capture_output=True,text=True,timeout=5)
        (self.root/'stdout').write_text(result.stdout);(self.root/'stderr').write_text(result.stderr)
        self.assertEqual(result.returncode,0);self.assertEqual(result.stdout.strip(),'selector forecast ledger: PASS5')
        rows=[c.decode_json(line) for line in (self.root/'admission-ledger.jsonl').read_text().splitlines()]
        self.assertEqual([r['stage'] for r in rows],['before_text','before_splice','inclusive_boundary','one_over'])
        self.assertEqual([r['admitted'] for r in rows],[True,False,True,False])
        terms=['worker_owned_bytes','text_requested_bytes','splice_forecast_bytes','canonical_undo_forecast_bytes','undo_terminator_forecast_bytes','checkpoint_token_forecast_bytes','scratch_reserved_bytes','budget_bytes','temporary_forecast_bytes']
        for row in rows:
            self.assertTrue(all(type(row[k]) is int and 0<=row[k]<2**64 for k in terms))
            temporary=sum(row[k] for k in ['text_requested_bytes','splice_forecast_bytes','canonical_undo_forecast_bytes','undo_terminator_forecast_bytes','checkpoint_token_forecast_bytes'])
            self.assertEqual(row['temporary_forecast_bytes'],temporary)
            self.assertEqual(row['admitted'],row['worker_owned_bytes']+temporary+row['scratch_reserved_bytes']<=row['budget_bytes'])
            self.assertIs(row['canonical_undo_allocated_by_selector'],False);self.assertIs(row['observed_global_peak'],False)
        self.assertFalse(rows[0]['text_allocated']);self.assertTrue(rows[1]['text_allocated'])

    def test_first_failure_is_retained_with_no_edit_or_third_evaluation(self):
        proof=json.loads((IDEAS/'evidence/agent-suffix-pair-actual-original-failure-v1/diagnosis.json').read_text())
        self.assertTrue(proof['no_structural_selection_created']);self.assertTrue(proof['no_third_round_record'])
        self.assertTrue(proof['exact_delivered_equals_literal_plus_one_LF']);self.assertEqual(proof['original_manifest_sha256'],c.CORRECTION['original_manifest_sha256'])


if __name__=='__main__':unittest.main(verbosity=2)
