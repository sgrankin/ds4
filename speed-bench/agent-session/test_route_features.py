"""Validate capture pairing and token-level integrity before training."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from train_route_predictor import dataset, inputs

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Path(self.tmp.name)/'features.bin'
        self.r = Path(self.tmp.name)/'routes.bin'
        dt = np.dtype([('meta','<u4',(3,)),('x','<u2',(5120,)),('early','<i4',(6,))])
        self.d = np.zeros(80, dtype=dt)
        self.d['meta'][:, 0] = np.repeat([10, 11], 40)
        self.d['meta'][:, 1] = np.repeat([99, 100], 40)
        self.d['meta'][:, 2] = np.tile(np.arange(40), 2)
        self.d['x'] = 0x3f80
        self.d['early'] = np.arange(6)
        self.labels = np.zeros((80, 9), dtype='<u4')
        self.labels[:, :3] = self.d['meta']; self.labels[:, 3:] = np.arange(6)
        self.write()
    def write(self):
        self.f.write_bytes(np.array([0x44534631,1,40,5120,384,6,2,0],dtype='<u4').tobytes()+self.d.tobytes())
        self.r.write_bytes(np.array([0x44535231,40,384,6],dtype='<u4').tobytes()+self.labels.tobytes())
    def test_exact_bf16(self):
        d, _ = dataset(self.f, self.r)
        np.testing.assert_array_equal(inputs(d, [0]), np.ones((1,5120),dtype=np.float32))
    def test_misaligned_label(self):
        self.labels[1, 0] += 1; self.write()
        with self.assertRaisesRegex(ValueError, 'alignment'): dataset(self.f, self.r)
    def test_partial_token(self):
        self.f.write_bytes(self.f.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete'): dataset(self.f, self.r)
    def test_duplicate_expert(self):
        self.labels[0, 3] = self.labels[0, 4]; self.write()
        with self.assertRaisesRegex(ValueError, 'expert IDs'): dataset(self.f, self.r)
    def test_bad_layer_order(self):
        self.d['meta'][1, 2] = 0; self.labels[1, 2] = 0; self.write()
        with self.assertRaisesRegex(ValueError, 'ordering'): dataset(self.f, self.r)
    def test_nonfinite(self):
        self.d['x'][0, 0] = 0x7f80; self.write()
        d, _ = dataset(self.f, self.r)
        with self.assertRaisesRegex(ValueError, 'nonfinite'): inputs(d, [0])

if __name__ == '__main__': unittest.main()
