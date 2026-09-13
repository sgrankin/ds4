import tempfile
from pathlib import Path
import unittest
from trace_session import extract

TRACE='''2026-09-13 14:00:00.000 tokens label=initial_system_prompt start=0 len=2
2026-09-13 14:00:00.000 token index=0 id=10
2026-09-13 14:00:00.000 token index=1 id=11
2026-09-13 14:00:00.001 prefill tool_round=0 transcript=3 prompt=3 cached=2 suffix=1
2026-09-13 14:00:00.001 tokens label=prefill_suffix start=2 len=3
2026-09-13 14:00:00.001 token index=2 id=12
2026-09-13 14:00:00.003 prefill sync done tool_round=0 prompt=3 cached=2 suffix=1 rc=0 2.0 ms
2026-09-13 14:00:00.004 token index=1 id=13
2026-09-13 14:00:00.005 dsml done calls=1
2026-09-13 14:00:00.005 generation finished tool_round=0 generated=1 carried=0 context_limited=0
'''

class TraceTests(unittest.TestCase):
    def parse(self,text):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace'; p.write_text(text)
            return extract(p)

    def test_exact_boundaries(self):
        stats,replay=self.parse(TRACE)
        self.assertEqual(replay,'P 0 2 10 11\nP 2 1 12\nD 0 1 13\n')
        self.assertEqual(stats['final_context'],4)
        self.assertEqual(stats['prefill_ms'],2)
        self.assertEqual(stats['decode_ms'],2)

    def test_incomplete_decode_rejected(self):
        with self.assertRaises(AssertionError):
            self.parse(TRACE.replace('generated=1','generated=2'))

    def test_missing_prefix_rejected(self):
        with self.assertRaises(AssertionError):
            self.parse(TRACE.replace('start=2 len=3','start=5 len=6').replace('index=2 id=12','index=5 id=12'))

    def test_compaction_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(TRACE+'2026-09-13 14:00:00.006 tokens label=compacted_transcript start=0 len=1\n')

if __name__=='__main__': unittest.main()
