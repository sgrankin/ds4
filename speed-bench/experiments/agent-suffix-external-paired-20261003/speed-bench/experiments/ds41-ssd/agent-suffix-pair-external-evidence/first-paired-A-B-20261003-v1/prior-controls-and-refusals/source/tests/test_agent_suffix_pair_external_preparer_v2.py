"""Pure static authority fixtures; never prepare, guard, model or compile."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('external_preparer_v2',WORK/'speed-bench/ideas/prepare_agent_suffix_pair_external_v2.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
class AliasAuthority(unittest.TestCase):
    def test_exact_four_known_compiler_authorities(self):
        entries=v2.compiler_aliases();self.assertEqual(len(entries),4)
        for path,known in entries.items():self.assertEqual(v2.check_compiler_authority(path,known)['sha256'],known['sha256'])
        leaves=[value for value in entries.values() if value['leaf_symlink']];self.assertEqual(len(leaves),1);self.assertEqual(leaves[0]['link'],'libSystem.B.tbd')
    def test_foreign_alias_and_model_refuse_before_stat_or_resolve(self):
        for path in ['/unowned/compiler.tbd','/unowned/model.gguf']:
            with patch.object(Path,'stat',side_effect=AssertionError('unowned metadata')),patch.object(Path,'resolve',side_effect=AssertionError('unowned resolve')):
                with self.assertRaisesRegex(ValueError,'Unpinned'):v2.check_compiler_authority(path,dict(sha256='a'*64))
    def test_changed_authority_digest_rejected_before_resolve(self):
        path=next(iter(v2.compiler_aliases()))
        with patch.object(Path,'resolve',side_effect=AssertionError('unbound resolve')):
            with self.assertRaisesRegex(ValueError,'Unpinned'):v2.check_compiler_authority(path,dict(sha256='a'*64))
    def test_payload_alias_remains_forbidden(self):
        parent=WORK/'speed-bench/ideas/artifacts/agent-suffix-pair-external-preparer-v2-cpu';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            p=Path(tmp);(p/'source').write_bytes(b'authority');(p/'alias').symlink_to('source')
            with self.assertRaisesRegex(ValueError,'Payload symlink'):v2.pin(p/'alias')
if __name__=='__main__':unittest.main(verbosity=2)
