import tempfile
import unittest
from pathlib import Path
import numpy as np
from score_route_prefetch import score, cache_data

class ScoreTests(unittest.TestCase):
    def test_only_nonresident_predictions_issue_reads(self):
        actual = np.tile(np.arange(6), (2,1))
        resident = np.zeros((2,384), dtype=np.uint8)
        resident[0,:2] = 1; resident[1,:6] = 1
        pred = np.array([[0,2,6],[0,6,7]])
        s = score(pred, np.ones_like(pred,dtype=bool), actual, resident)
        self.assertEqual(s['demand_misses'], 4)
        self.assertEqual(s['predicted_reads'], 4)
        self.assertEqual(s['useful_reads'], 1)
        self.assertEqual(s['unnecessary_reads'], 3)
        self.assertEqual(s['reads_on_native_all_hit_layers'], 2)
        self.assertEqual(s['miss_recall'], .25)
        self.assertEqual(s['read_precision'], .25)
    def test_abstention(self):
        actual=np.arange(6)[None,:]
        s=score(actual,np.zeros((1,6),bool),actual,np.zeros((1,384),np.uint8))
        self.assertEqual(s['predicted_reads'],0)
        self.assertEqual(s['miss_recall'],0)
        self.assertIsNone(s['read_precision'])
    def test_cache_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cache.bin'
            head=np.array([0x44534331,1,40,384],dtype='<u4').tobytes()
            p.write_bytes(head+np.array([10,99,0],dtype='<u4').tobytes()+bytes(384))
            labels=np.array([[10,99,0,0,1,2,3,4,5]],dtype='<u4')
            self.assertEqual(cache_data(p,labels)['resident'].sum(),0)
            labels[0,0]=11
            with self.assertRaisesRegex(ValueError,'alignment'):cache_data(p,labels)

if __name__=='__main__':unittest.main()
