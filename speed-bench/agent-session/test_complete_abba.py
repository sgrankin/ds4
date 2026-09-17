import json
from pathlib import Path
import tempfile
import unittest
from complete_abba import complete


class CompleteTests(unittest.TestCase):
    def fixture(self, root):
        for folder, variants in [('old', 'A'), ('new', 'BBA')]:
            dest = root / folder
            (dest/'metal').mkdir(parents=True)
            manifest = dict(replay_sha256='tokens', binary_sha256='binary',
                            model='model', routes_sha256='routes', cache_gb=None,
                            env={'DS4_REPLAY_PROFILE_MEMORY':'1',
                                 'DS4_METAL_MOE_SOURCE':str(dest/'moe.metal'),
                                 'DS4_V41_ROUTE_ORACLE':str(dest/'routes.bin')})
            (dest/'manifest.json').write_text(json.dumps(manifest))
            (dest/'metal/sha256.json').write_text(json.dumps({'DS4_METAL_MOE_SOURCE':'shader'}))
            rows = [dict(variant=v, logits_sha256='logits', snapshot_sha256='state',
                         startup_prefill_ms=1, append_prefill_ms=2,
                         decode_ms=8 if v=='A' else 3, model_ms=11 if v=='A' else 6)
                    for v in variants]
            (dest/'summary.json').write_text(json.dumps(rows))
        return root/'old', root/'new', root/'result'

    def test_matching_control_and_mean(self):
        with tempfile.TemporaryDirectory() as d:
            result = complete(*self.fixture(Path(d)))
            self.assertEqual(result['candidate_change_percent']['turn_model_ms'], -50)

    def test_environment_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            old, new, out = self.fixture(Path(d))
            p = new/'manifest.json'
            m = json.loads(p.read_text()); m['env']['DS4_REPLAY_PREFILL_CHUNK']='2048'
            p.write_text(json.dumps(m))
            with self.assertRaises(ValueError): complete(old, new, out)
            self.assertFalse(out.exists())

    def test_shader_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            old, new, out = self.fixture(Path(d))
            (new/'metal/sha256.json').write_text(json.dumps({'DS4_METAL_MOE_SOURCE':'changed'}))
            with self.assertRaises(ValueError): complete(old, new, out)
            self.assertFalse(out.exists())


if __name__ == '__main__': unittest.main()
