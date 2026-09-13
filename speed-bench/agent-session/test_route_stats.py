from array import array
from pathlib import Path
import tempfile
import gzip
import unittest
from route_stats import read_routes


class RouteTests(unittest.TestCase):
    def parse(self, words):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'routes.bin'
            p.write_bytes(array('I', words).tobytes())
            return read_routes(p)

    def test_roundtrip(self):
        result = self.parse([0x44535231, 2, 8, 2, 10, 7, 0, 2, 5, 10, 7, 1, 3, 6])
        self.assertEqual(result, (2, 8, 2, [(10, 7, [[2, 5], [3, 6]])]))

    def test_compressed_routes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'routes.bin.gz'
            p.write_bytes(gzip.compress(array('I', [0x44535231, 1, 8, 2, 10, 7, 0, 2, 5]).tobytes()))
            self.assertEqual(read_routes(p)[3], [(10, 7, [[2, 5]])])

    def test_incomplete_token(self):
        with self.assertRaises(ValueError):
            self.parse([0x44535231, 2, 8, 2, 10, 7, 0, 2, 5])

    def test_wrong_layer(self):
        with self.assertRaises(ValueError):
            self.parse([0x44535231, 1, 8, 2, 10, 7, 1, 2, 5])

    def test_duplicate_expert(self):
        with self.assertRaises(ValueError):
            self.parse([0x44535231, 1, 8, 2, 10, 7, 0, 2, 2])


if __name__ == '__main__':
    unittest.main()
