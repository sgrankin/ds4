import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('g',ROOT/'speed-bench/native_effort_holdout_gold.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

class IndependentGold(unittest.TestCase):
    def test_six_distinct_new_cases_and_independent_methods(self):
        cases=g.verified_cases()
        self.assertEqual(len(cases),6);self.assertEqual(len({c['case'] for c in cases}),6)
        old={'circular10','restricted_derangements8','lattice8x6','subset_mod7'}
        self.assertFalse({c['case'] for c in cases}&old)
        for c in cases:
            self.assertEqual(c['gold_results'][0],c['gold_results'][1])
            self.assertNotEqual(*c['gold_methods']);self.assertTrue(c['answer'].isdigit())
    def test_known_counts_and_recurrence_boundaries(self):
        self.assertEqual(g.fixed_points_formula(),22260)
        self.assertEqual(g.necklace_burnside(),43)
        self.assertEqual(g.onto_inclusion_exclusion(),8400)
        self.assertEqual(g.domino_closed_recurrence(),153)
        self.assertEqual(g.graph_component_recurrence(),5700)
        self.assertEqual(g.partition_recurrence(0,0,7),1)
        self.assertEqual(g.partition_recurrence(1,0,7),0)
        self.assertEqual(g.graph_component_recurrence(1,0),1)
        self.assertEqual(g.graph_component_recurrence(2,0),0)

if __name__=='__main__':unittest.main()
