"""Source/argv gate only; do not execute the caller or compiler."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'speed-bench/ideas/prepare_reasoning_budget_confirmation_build.py'
spec=importlib.util.spec_from_file_location('build2',SOURCE);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class BuildContract(unittest.TestCase):
    def test_exact_accepted_selftest_before_engine(self):
        text=s.C.read_text();s.selftest_gate(text,'--helper-self-test')
        for flag in ('--self-test','--help','--helper-self-test-extra'):
            with self.assertRaises(ValueError):s.selftest_gate(text,flag)
        with self.assertRaises(ValueError):s.selftest_gate('ds4_engine_open();'+text,'--helper-self-test')

    def test_only_SDK_output_changes_and_unchanged_inputs(self):
        old=json.loads((s.V1/'build-plan.json').read_text());out=Path('/nonexistent/unrun-source-only-v2');argv=s.command(old,out)
        restored=argv[:];restored[restored.index('-isysroot')+1]=old['compile_argv'][old['compile_argv'].index('-isysroot')+1];restored[restored.index('-o')+1]=old['compile_argv'][old['compile_argv'].index('-o')+1]
        self.assertEqual(restored,old['compile_argv'])
        plan=dict(old,compile_argv=argv,selftest_argv=[str(out/'reasoning-budget-confirmation-probe'),s.SELFTEST],sdk_path=str(s.SDK),input_freeze_sha256=s.V1_INPUT_SHA)
        s.validate(plan,old,out,s.C.read_text())
        for key,value in [('sdk_path',str(s.SDK.parent/'MacOSX.sdk')),('selftest_argv',[plan['selftest_argv'][0],'--self-test']),('source_sha256','f'*64),('compile_or_link_granted',True),('core_or_Metal_rebuild',True),('max_concurrent_compilers',True)]:
            changed=copy.deepcopy(plan);changed[key]=value
            with self.assertRaises(ValueError):s.validate(changed,old,out,s.C.read_text())
        for bad_out in (s.V1/'new-output',Path('relative-output')):
            with self.assertRaises(ValueError):s.command(old,bad_out)
        changed=copy.deepcopy(old);changed['compile_argv'].extend(['-o','elsewhere'])
        with self.assertRaises(ValueError):s.command(changed,out)
        self.assertEqual(s.sha(s.V1/'input-freeze.json'),s.V1_INPUT_SHA);self.assertEqual(s.sha(s.C),s.C_SHA);self.assertEqual(s.sha(s.HEADER),s.HEADER_SHA)


if __name__=='__main__':unittest.main()
