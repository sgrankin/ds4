import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compaction', ROOT/'speed-bench/ideas/analyze_agent_compaction.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
ARM = d.PREPARED.parent/'resident-agent-three-live-prepared-20261002-v3-run/native-before'

class Trace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (ARM/'trace.log').read_bytes()

    def rejected(self, raw):
        with self.assertRaises(ValueError):
            d.parse_trace(raw)

    def test_actual_full_native_trace(self):
        ledger = d.parse_trace(self.raw)
        self.assertEqual((ledger['epochs'], ledger['generation_rounds'], ledger['compactions'], ledger['users']), (2, 17, 1, 3))
        c = [e for e in ledger['events'] if e['kind'] == 'C'][0]
        self.assertEqual((c['old'], c['new'], c['tail_start'], c['tail']), (6662, 2974, 6578, 84))
        self.assertEqual(len(c['tokens']), c['new'])
        self.assertEqual(ledger['final_context_tokens'], 3861)
        self.assertFalse(ledger['replay_created'])
        self.assertFalse(ledger['cold_reset_replay_proven'])
        self.assertTrue(all(len(e['tokens']) == e.get('count', e.get('new')) for e in ledger['events']))

    def test_truncated_newline(self): self.rejected(self.raw[:-1])

    def test_incomplete_final_decode(self):
        self.rejected(self.raw[:self.raw.rfind(b'generation finished')])

    def test_missing_compaction_sync(self):
        rows = self.raw.splitlines(keepends=True)
        self.rejected(b''.join(r for r in rows if b'prefill sync done prompt=2974 ' not in r))

    def test_wrong_compaction_sync(self):
        self.rejected(self.raw.replace(b'prompt=2974 cached=0 suffix=2974 rc=0', b'prompt=2973 cached=0 suffix=2973 rc=0'))

    def test_wrong_compaction_commit(self):
        self.rejected(self.raw.replace(b'old=6662 new=2974 tail_start=6578', b'old=6662 new=2973 tail_start=6578'))

    def test_missing_compaction_commit(self):
        self.rejected(b''.join(r for r in self.raw.splitlines(keepends=True) if b'compacted reason=' not in r))

    def test_unknown_token_block(self):
        self.rejected(self.raw.replace(b'label=compacted_transcript', b'label=secret_replay'))

    def test_unknown_event(self):
        self.rejected(self.raw+b'2026-10-02 19:43:54.745 hidden reset=1\n')

    def test_interrupted_compaction_token_block(self):
        begin = self.raw.index(b'tokens label=compacted_transcript')
        end = self.raw.index(b'prefill sync done prompt=2974', begin)
        body = self.raw[begin:end]
        body = body.replace(b'token index=100 id=', b'token index=101 id=', 1)
        self.rejected(self.raw[:begin]+body+self.raw[end:])

    def test_missing_full_compaction_row(self):
        rows = self.raw.splitlines(keepends=True)
        ix = next(i for i, r in enumerate(rows) if b'tokens label=compacted_transcript' in r)
        del rows[ix+101]
        self.rejected(b''.join(rows))

    def test_token_byte_count(self): self.rejected(self.raw.replace(b'id=0 bytes=29', b'id=0 bytes=28', 1))
    def test_token_range(self): self.rejected(self.raw.replace(b'id=0 bytes=29', b'id=129280 bytes=29', 1))
    def test_generation_count(self): self.rejected(self.raw.replace(b'generated=76 carried=0', b'generated=77 carried=0', 1))
    def test_carried_count(self): self.rejected(self.raw.replace(b'generated=76 carried=0', b'generated=76 carried=1', 1))
    def test_round_sync_failure(self): self.rejected(self.raw.replace(b'rc=0 2539.386', b'rc=1 2539.386', 1))
    def test_startup_sync_failure(self): self.rejected(self.raw.replace(b'rc=0 50229.959', b'rc=1 50229.959', 1))
    def test_round_order(self): self.rejected(self.raw.replace(b'prefill tool_round=1 transcript=2301', b'prefill tool_round=2 transcript=2301', 1))
    def test_summary_geometry(self): self.rejected(self.raw.replace(b'prefix=6662 total=6662', b'prefix=6663 total=6662', 1))
    def test_empty_summary(self):
        rows = self.raw.splitlines(keepends=True)
        rows = [r[:r.index(b'compaction-summary=')]+b'compaction-summary=""\n' if b'compaction-summary=' in r else r for r in rows]
        self.rejected(b''.join(rows))
    def test_unowned_boundary(self):
        self.rejected(self.raw+b'2026-10-02 19:43:54.745 generation context boundary: generated=169 partial_tool=0\n')

    def test_generation_boundary_preserved(self):
        # Synthetic continuation of the already context-limited round3; no new
        # model output is created. The next round remains3 and carries53 tokens.
        raw = self.raw.replace(b'2026-10-02 19:40:41.962 compaction summary',
             b'2026-10-02 19:40:41.950 generation context boundary: generated=53 partial_tool=0\n2026-10-02 19:40:41.962 compaction summary')
        raw = raw.replace(b'prefill tool_round=4 prompt=', b'prefill tool_round=3 prompt=')
        raw = raw.replace(b'prefill tool_round=4 transcript=3208', b'prefill tool_round=3 transcript=3208')
        raw = raw.replace(b'prefill sync done tool_round=4 prompt=3208', b'prefill sync done tool_round=3 prompt=3208')
        raw = raw.replace(b'generation finished tool_round=4 generated=49 carried=0', b'generation finished tool_round=3 generated=49 carried=53')
        ledger = d.parse_trace(raw)
        c = [e for e in ledger['events'] if e['kind'] == 'C'][0]
        self.assertEqual(c['context_boundary']['generated'], 53)
        self.assertFalse(c['context_boundary']['partial_tool'])
        self.assertFalse(c['cold_reset_replay_proven'])

class Task(unittest.TestCase):
    def test_original_failure_retained(self):
        outcome = json.loads((ARM/'harness-outcome.json').read_text())
        summary = json.loads((ARM.parent/'native-before-summary.json').read_text())
        d.validate_outcome(outcome, summary)
        self.assertEqual(outcome['status'], 'FAILED')
        for key, value in [('agent_returncode', 1), ('agent_returncode', False), ('turns_observed', 2), ('status', 'SUCCESS'), ('phase', 'complete')]:
            broken = dict(outcome); broken[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                d.validate_outcome(broken, summary)
        for value in (0, True, 3):
            with self.subTest(rc=value), self.assertRaises(ValueError):
                d.validate_outcome(outcome, dict(summary, process_returncode=value))

    def test_ordered_checks_fail_closed(self):
        checks = json.loads((ARM/'checks.json').read_text())
        d.validate_recorded_checks(checks)
        for kind in range(6):
            broken = copy.deepcopy(checks)
            if kind == 0: broken.reverse()
            elif kind == 1: broken.pop()
            elif kind == 2: broken[1]['held_out_rc'] = 1
            elif kind == 3: broken[2]['tests_rc'] = False
            elif kind == 4: broken[0]['held_out'] = 'PASS\n'
            else: broken[2]['tests'] = 'Ran 6 tests in 0.00s\n\nOK\n'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                d.validate_recorded_checks(broken)

    def test_independent_gold_dedup_before_filter(self):
        data = b'event_id,account,kind,amount,date\na,x,charge,0.29,2026-08-01\na,x,charge,9.99,2026-08-20\nb,x,refund,0.29,2026-08-15\nc,v,void,99.00,2026-08-15\n'
        self.assertEqual(d.gold_json(data), {'x': 0})
        self.assertEqual(d.gold_json(data, '2026-08-15'), {'x': -29})
        with self.assertRaises(ValueError): d.gold_json(data.replace(b'0.29', b'0.291', 1))

    def test_actual_final_project_and_readonly_sources(self):
        result = d.analyze(ARM)
        self.assertEqual(result['status'], 'SEPARATE_OFFLINE_TASK_COMPLETION_PASS')
        self.assertEqual(result['original_harness_status'], 'FAILED')
        self.assertFalse(result['original_harness_success_rewritten'])
        self.assertEqual(len(result['independent_final_project_cpu_checks']), 4)
        self.assertTrue(all(x['returncode'] == 0 for x in result['independent_final_project_cpu_checks']))
        self.assertFalse(result['model_opened'])

if __name__ == '__main__':
    unittest.main()
