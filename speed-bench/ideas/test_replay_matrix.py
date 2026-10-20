"""No-model checks preventing contaminated controls and invalid reference reuse."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_replay_matrix as matrix


class ReplayMatrixTest(unittest.TestCase):
    def setUp(self):
        artifacts=matrix.ROOT/'speed-bench/ideas/artifacts'
        artifacts.mkdir(exist_ok=True)
        self.workspace=tempfile.TemporaryDirectory(prefix='matrix-test-',dir=artifacts)
        self.root=Path(self.workspace.name)
        self.model=self.root/'model.stub'
        self.model.write_bytes(b'not a model')
        self.replay=self.root/'session.txt'
        self.replay.write_text('P 0 1 1\nD 0 1 2\n')
        (self.root/'metal').mkdir()
        (self.root/'metal/example.metal').write_text('// frozen shader\n')
        self.binary=self.root/'probe'
        self.binary.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
assert 'DS4_SENTINEL' not in os.environ
out,ref,mode=map(pathlib.Path,sys.argv[3:6])
if str(mode)=='record': ref.write_bytes(b'reference')
else: assert ref.read_bytes()==b'reference'
out.write_text('case,delta_raw_nll,raw_kl\\nphase_2,0,0\\n')
print(json.dumps(dict(tokens=1,bit_equal_rows=0 if os.environ.get('DS4_BAD_PROBE') else 1)))
''')
        self.binary.chmod(0o755)

    def tearDown(self):
        self.workspace.cleanup()

    def args(self,name):
        return argparse.Namespace(output=self.root/name,binary=self.binary,
            replay=self.replay,source_root=self.root,model=self.model,
            ctx=8192,cache_gb=64,chunk=512,variant=[],
            reference=None,reference_manifest=None)

    def reference(self):
        args=self.args('initial')
        with patch.dict(os.environ,{'DS4_SENTINEL':'must be removed'}):
            results=matrix.run(args)
        self.assertEqual([r['name'] for r in results],['reference','native-before','native-after'])
        args=self.args('reused')
        args.reference=self.root/'initial/reference.bin'
        args.reference_manifest=self.root/'initial/manifest.json'
        return args

    def test_clean_controls_and_valid_reuse(self):
        args=self.reference()
        results=matrix.run(args)
        self.assertEqual(len(results),2)
        self.assertNotIn('prompt_bootstrap_95',results[0])

    def test_changed_replay_rejected_before_inference(self):
        args=self.reference()
        self.replay.write_text('changed input')
        with self.assertRaisesRegex(ValueError,'session_sha256'):
            matrix.run(args)

    def test_changed_reference_rejected_before_inference(self):
        args=self.reference()
        args.reference.write_bytes(b'tampered reference')
        with self.assertRaisesRegex(ValueError,'reference hash'):
            matrix.run(args)

    def test_native_byte_drift_rejected(self):
        args=self.args('drift')
        # The fake helper deliberately emits a nonmatching native control.
        text=self.binary.read_text().replace("os.environ.get('DS4_BAD_PROBE')","True")
        self.binary.write_text(text)
        with self.assertRaisesRegex(RuntimeError,'byte-for-byte'):
            matrix.run(args)

    def test_variant_cannot_replace_shader(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            matrix.variant('bad:DS4_METAL_DS4_SOURCE=/other/source')


if __name__=='__main__':
    unittest.main()
