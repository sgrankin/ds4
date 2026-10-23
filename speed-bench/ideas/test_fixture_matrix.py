"""No-model checks for prompt freezing and reusable-reference identity."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_fixture_matrix as matrix


class FixtureMatrixTest(unittest.TestCase):
    def setUp(self):
        self.output=contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        artifacts=matrix.ROOT/'speed-bench/ideas/artifacts'
        artifacts.mkdir(exist_ok=True)
        self.workspace=tempfile.TemporaryDirectory(prefix='fixture-matrix-test-',dir=artifacts)
        self.root=Path(self.workspace.name)
        self.model=self.root/'model.stub'
        self.model.write_bytes(b'not a model')
        self.prompt=self.root/'prompt.txt'
        self.continuation=self.root/'continuation.txt'
        self.prompt.write_text('question')
        self.continuation.write_text('answer')
        self.fixtures=self.root/'fixtures.tsv'
        self.fixtures.write_text(f'case\t{self.prompt}\t{self.continuation}\n')
        (self.root/'src/metal').mkdir(parents=True)
        (self.root/'src/metal/example.metal').write_text('// frozen\n')
        self.binary=self.root/'probe'
        self.binary.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
assert 'DS4_SENTINEL' not in os.environ
if os.environ.get('DS4_FAIL_PROBE'): sys.exit(1)
fixture,out,ref,mode=map(pathlib.Path,sys.argv[2:6])
fields=fixture.read_text().strip().split('\\t')
assert pathlib.Path(fields[1]).read_text()=='question'
assert pathlib.Path(fields[2]).read_text()=='answer'
if str(mode)=='record': ref.write_bytes(b'reference')
else: assert ref.read_bytes()==b'reference'
out.write_text('case,delta_raw_nll,raw_kl\\ncase,0,0\\n')
print(json.dumps(dict(tokens=1,bit_equal_rows=0 if str(mode)=='compare' and os.environ.get('DS4_BAD_PROBE') else 1)))
''')
        self.binary.chmod(0o755)

    def tearDown(self):
        self.workspace.cleanup()
        self.output.__exit__(None,None,None)

    def args(self,name):
        return argparse.Namespace(output=self.root/name,binary=self.binary,
            manifest=self.fixtures,source_root=self.root,model=self.model,cases=1,
            tokens=1,ctx=4096,cache_gb=64,chunk=512,variant=[],
            reference=None,reference_manifest=None)

    def reference(self):
        with patch.dict(os.environ,{'DS4_SENTINEL':'remove me'}):
            matrix.run(self.args('initial'))
        args=self.args('reused')
        args.reference=self.root/'initial/reference.bin'
        args.reference_manifest=self.root/'initial/manifest.json'
        return args

    def test_frozen_inputs_and_valid_reuse(self):
        args=self.reference()
        rows=matrix.run(args)
        self.assertEqual([row['name'] for row in rows],['native-before','native-after'])
        self.assertEqual((args.output/'inputs/000-prompt.txt').read_text(),'question')

    def test_input_change_rejected_before_model_open(self):
        args=self.reference()
        self.prompt.write_text('changed question')
        with self.assertRaisesRegex(ValueError,'reference inputs'):
            matrix.run(args)

    def test_setting_change_rejected_before_model_open(self):
        args=self.reference()
        args.cache_gb=80
        with self.assertRaisesRegex(ValueError,'reference settings'):
            matrix.run(args)

    def test_reference_tampering_rejected_before_model_open(self):
        args=self.reference()
        args.reference.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'reference hash'):
            matrix.run(args)

    def test_native_drift_rejected(self):
        args=self.args('drift')
        self.binary.write_text(self.binary.read_text().replace("os.environ.get('DS4_BAD_PROBE')","True"))
        with self.assertRaisesRegex(RuntimeError,'native-before differs from reference bytes'):
            matrix.run(args)
        self.assertEqual(json.loads((args.output/'reference.json').read_text())['bit_equal_rows'],1)

    def test_reference_reusable_after_candidate_failure(self):
        args=self.args('failed')
        args.variant=[('failed-policy',{'DS4_FAIL_PROBE':'1'})]
        with self.assertRaisesRegex(RuntimeError,'failed-policy failed rc=1'):
            matrix.run(args)
        prior=json.loads((args.output/'manifest.json').read_text())
        self.assertEqual(prior['reference_sha256'],matrix.sha(args.output/'reference.bin'))
        reused=self.args('recovered')
        reused.reference=args.output/'reference.bin'
        reused.reference_manifest=args.output/'manifest.json'
        rows=matrix.run(reused)
        self.assertEqual([row['name'] for row in rows],['native-before','native-after'])

    def test_frozen_assets_tampering_rejected(self):
        real_run=matrix.subprocess.run
        for kind,relative in [('binary','probe'),('Metal source','metal/example.metal'),
                              ('fixture manifest','fixtures.tsv')]:
            with self.subTest(kind=kind):
                args=self.args('tamper-'+relative.replace('/','-'))
                calls=[]
                def tamper(*positional,**kwargs):
                    calls.append(positional[0])
                    completed=real_run(*positional,**kwargs)
                    asset=args.output/relative
                    asset.chmod(0o755)
                    asset.write_text(asset.read_text()+'\nchanged\n')
                    return completed
                with patch.object(matrix.subprocess,'run',side_effect=tamper):
                    with self.assertRaisesRegex(RuntimeError,'frozen '+kind+' changed'):
                        matrix.run(args)
                self.assertEqual(len(calls),1)

    def test_frozen_fixture_change_between_arms_rejected(self):
        args=self.args('between-arms')
        real_write_json=matrix.write_json
        def tamper(path,value):
            real_write_json(path,value)
            if path==args.output/'summary.json':
                (args.output/'fixtures.tsv').write_text('changed\n')
        with patch.object(matrix,'write_json',side_effect=tamper):
            with self.assertRaisesRegex(RuntimeError,'frozen fixture manifest changed'):
                matrix.run(args)
        self.assertTrue((args.output/'reference.json').exists())
        self.assertFalse((args.output/'native-before-command.json').exists())


if __name__=='__main__': unittest.main()
