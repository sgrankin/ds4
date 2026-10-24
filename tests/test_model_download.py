#!/usr/bin/env python3
"""Offline downloader checks with small, genuinely hashed artifact fixtures."""
import hashlib
import os
from pathlib import Path
import json
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
Q2 = "DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
Q4 = "DeepSeek-V4.1-Flash-Q4.gguf"
MX_BASE = "DeepSeek-V4.1-Flash-1017-MXFP4-experts"
MX = MX_BASE + ".gguf"
MX_SHARDS = [f"{MX_BASE}-{i:05d}-of-00009.gguf" for i in range(1, 10)]
PART1, PART2 = Q4 + ".part1", Q4 + ".part2"
VISION = "DeepSeek-V4.1-Flash-Vision.gguf"
VISION_1017 = "DeepSeek-V4.1-Flash-1017-vision-BF16.gguf"
QWEN_Q2 = "Qwen3.8-Flash-Next-Q2.gguf"
QWEN_Q4 = "Qwen3.8-Flash-Next-Q4.gguf"
FLINT_REVISION = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
# recursive-oss/rec packaging/flint-int4-mlx.json, qualified INT4 checkpoint.
FLINT_ARTIFACTS = {
    "chat_template.jinja": (2328, "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466"),
    "config.json": (3082, "80f577a984db1bd40dfff88457b611d4c13e807704a945616be27f28809eed21"),
    "generation_config.json": (202, "a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3"),
    "model-00000.safetensors": (4745746152, "1a362742a0136c95ed751e2c56d83e63d4ed9ac674984b23b368f105d16cfe7a"),
    "model-00001.safetensors": (4682022280, "12979140fa019042ff5cdec4c0237cc87db13b56428ce7fe093dd9d33893ea32"),
    "model-00002.safetensors": (4682022280, "01805c6b882fdec008b0184ec68e7516102205b521e63ca5126f9d52b9aed87b"),
    "model-00003.safetensors": (4682022280, "9380ed032d90e0ff566d49672faf292f354b3b0e7320820c150eeed8c75e9d42"),
    "model-00004.safetensors": (3777099608, "c1ba12d2c2a80feb5e554554a01cfde2b71c2638e27fc0ee597b4121399eff99"),
    "special_tokens_map.json": (20192, "be37b302338dda189bc70bfbe902d8faebeb5986d60c1342713439eefc4ca182"),
    "tokenizer.json": (10879128, "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68"),
    "tokenizer_config.json": (30439, "159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf"),
    "flint-int4.json": (1206, "319822f10e32a5916b57e23c7f05f9e92b24d02b3a136d02a3846b5edab78109"),
}
ARTIFACTS = {
    MX: (501609136128, "79608a69cd5bb919d8140c37105edc3a33419692d4e6caf4e68c901e4b359141"),
    MX_SHARDS[0]: (49353784544, "636918b09de6253bf41af29f06aac3ceeda220e46379d8a2058ef92c4e433ed8"),
    MX_SHARDS[1]: (49412635712, "b26d3fc1e6ae1be6c52665542f3c1fee6a33791478f26e6cd90b5a75940a9f96"),
    MX_SHARDS[2]: (49582960352, "20c7fe7adb0578336d8a5e5f9eed521aeb88e8c119ba46e0916cdaab28931c80"),
    MX_SHARDS[3]: (49418206432, "67592ba15870c4292a615666adcc2736b53cc345b2cccb1c36710a0a6caf15b4"),
    MX_SHARDS[4]: (49412829632, "ecd04e0aa71bfab8cfe79ea7fef7f35b6be0b6772455139e2081d6ce1dcc18ed"),
    MX_SHARDS[5]: (49257603776, "88214e0cc80fbae578b80c114527e6ce725199684e0b6849475c7d00b3d0f54c"),
    MX_SHARDS[6]: (2406482112, "38939faf3e4258f3ae333f0fa1954ce89515b75ceee90d37bf4ebe5fafe285e6"),
    MX_SHARDS[7]: (101377628544, "a9d1634da2ac57459a64b10b69ddedfadda7dcde0e71ca6e55bea7c293c4a9f0"),
    MX_SHARDS[8]: (101380404256, "b9832b355288c005da6ef444d02424464dd27f6ebdefeadbec2244264eb65285"),
    **FLINT_ARTIFACTS,
    QWEN_Q2: (147207127040, "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9"),
    QWEN_Q4: (177280286720, "680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a"),
    Q2: (365713686528, "5296e85ff64b42ecb62c8d0c0c54e1caa07e3df8e93112f2365ebbaed62ce93a"),
    Q4: (518596067328, "a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e"),
    PART1: (480000000000, "6442b1f9224079662c02003c0ef9ef6be6e2aff509510f681dab9e6cc41df246"),
    PART2: (38596067328, "7c3e10646c918eeaffbc39305a75ec96117450262c61454ff194cef00d7617f0"),
    VISION: (970555552, "cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15"),
    VISION_1017: (970555552, "861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf"),
}


# MXFP4 binary fixtures/cases follow upstream a1012463383a. Their independent
# writer checks exact restoration of the original 16 KiB-aligned GGUF layout.
def gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def mx_fixtures():
    """Independent GGUF writer: small mixed-type tensors, nontrivial padding."""
    def kv(key, kind, value):
        return gguf_string(key) + struct.pack("<I", kind) + value

    def metadata(alignment):
        return [kv("general.architecture", 8, gguf_string("deepseek41")),
                kv("general.alignment", 4, struct.pack("<I", alignment)),
                kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) +
                   gguf_string("hello") + gguf_string("world"))]

    def write(kvs, tensors, alignment):
        header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
        data = b""
        for name, shape, kind, payload in tensors:
            header += gguf_string(name) + struct.pack("<I", len(shape))
            header += struct.pack("<" + "Q" * len(shape), *shape)
            header += struct.pack("<IQ", kind, len(data))
            data += payload + bytes(-len(payload) % alignment)
        return header + bytes(-len(header) % alignment) + data

    specs = [(0, [3], 12), (1, [5], 10), (8, [32, 3], 102), (39, [32, 2], 34),
             (24, [7], 7), (0, [9], 36), (1, [4], 8), (24, [264, 2], 528),
             (24, [264, 3], 792)]
    tensors = [(f"tensor.{i}.weight", shape, kind, bytes([i + 1]) * size)
               for i, (kind, shape, size) in enumerate(specs)]
    result = {MX: write(metadata(16384), tensors, 16384)}
    for i, name in enumerate(MX_SHARDS):
        kvs = metadata(32) if i == 0 else []
        kvs += [kv("split.no", 2, struct.pack("<H", i)),
                kv("split.tensors.count", 5, struct.pack("<i", len(tensors))),
                kv("split.count", 2, struct.pack("<H", len(MX_SHARDS)))]
        result[name] = write(kvs, [tensors[i]], 32)
    return result


MX_PAYLOADS = mx_fixtures()


def payload(name):
    if name in MX_PAYLOADS:
        return MX_PAYLOADS[name]
    if name == Q4:
        return payload(PART1) + payload(PART2)
    return (name + "\n").encode()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".workspaces"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="download-tests-", dir=work)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "source tree"
        self.root.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.out = self.root / "gguf files"
        self.script = self.root / "download-model.nu"
        shutil.copy2(ROOT / "download-model.nu", self.script)
        tools = self.root / "tools"
        tools.mkdir()
        shutil.copy2(ROOT / "tools/join-model-parts.py", tools)
        merger = self.root / "gguf-tools"
        merger.mkdir()
        shutil.copy2(ROOT / "gguf-tools/merge_deepseek41_shards.py", merger)
        fixtures = self.bin / "fixtures"
        fixtures.mkdir()
        catalog = json.loads((ROOT / "tools/model-downloads.json").read_text())
        for name, (size, sha) in ARTIFACTS.items():
            self.assertEqual(catalog["artifacts"][name], {"bytes": size, "sha256": sha})
            data = payload(name)
            (fixtures / name).write_bytes(data)
            catalog["artifacts"][name] = {
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (tools / "model-downloads.json").write_text(json.dumps(catalog))
        hf = self.bin / "hf"
        hf.write_text("""#!/usr/bin/env python3
import json, os
from pathlib import Path
import sys
args = sys.argv[1:]
assert args[0] == 'download'
if args[1] == 'RecursiveSI/flint-int4':
    assert args[args.index('--revision') + 1] == '3b93f9053b530488405ae9fca51d4900d6ce21f0'
else:
    assert args[1] == ('kyuz0/deepseek-v4.1-flash-gguf' if args[2] == 'DeepSeek-V4.1-Flash-1017-vision-BF16.gguf'
                       else 'antirez/qwen3.8-flash-next-gguf' if args[2].startswith('Qwen')
                       else 'antirez/deepseek-v4.1-flash-gguf')
    assert '--revision' not in args
out = Path(args[args.index('--local-dir') + 1])
if os.environ.get('HF_CALLS'):
    with Path(os.environ['HF_CALLS']).open('a') as log:
        log.write(json.dumps({'repo': args[1], 'file': args[2], 'directory': str(out),
            'revision': args[args.index('--revision') + 1] if '--revision' in args else None}) + '\\n')
if os.environ.get('FAIL_DOWNLOAD') or os.environ.get('FAIL_DOWNLOAD_FILE') == args[2]:
    sys.exit(7)
out.mkdir(parents=True, exist_ok=True)
(out / args[2]).write_bytes((Path(__file__).parent / "fixtures" / args[2]).read_bytes())
""")
        hf.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root / "home"), HF_TOKEN="",
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        DS4_GGUF_DIR=str(self.out))

    def run_download(self, target, ok=True):
        result = subprocess.run(["nu", str(self.script), target], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_quants_vision_and_existing(self):
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q2"))
        link = self.root / "ds4flash.gguf"
        self.assertEqual(link.resolve(), self.out / Q2)
        self.assertIn("Already downloaded", self.run_download("ds41f-q2"))
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q4"))
        self.assertEqual(link.resolve(), self.out / Q4)
        self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))
        for name in (PART1, PART2, Q4 + ".assembling"):
            self.assertFalse((self.out / name).exists())
        self.assertIn("Already downloaded", self.run_download("ds41f-q4"))
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-vision"))
        self.assertEqual(link.resolve(), self.out / Q4)

    def test_empty_output_directory_uses_repo_gguf(self):
        self.env["DS4_GGUF_DIR"] = ""
        self.run_download("ds41f-q2")
        target = self.root / "gguf" / Q2
        self.assertEqual(target.read_bytes(), payload(Q2))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), target)
        self.assertFalse(self.out.exists())

    def test_1017_vision_is_verified_without_changing_link(self):
        self.run_download("ds41f-q2")
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-1017-vision"))
        self.assertEqual((self.out / VISION_1017).read_bytes(), payload(VISION_1017))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        self.assertIn("Already downloaded", self.run_download("ds41f-1017-vision"))

    def test_truncated_and_corrupt_artifacts(self):
        self.out.mkdir()
        for target, name in (("ds41f-q2", Q2), ("ds41f-q4", Q4), ("ds41f-mxfp4", MX), ("ds41f-vision", VISION),
                             ("ds41f-1017-vision", VISION_1017),
                             ("qwen38-q2", QWEN_Q2), ("qwen38-q4k", QWEN_Q4)):
            with self.subTest(target=target):
                path = self.out / name
                path.write_bytes(b"short")
                self.assertIn("Incorrect file size", self.run_download(target, ok=False))
                self.assertFalse((self.root / "ds4flash.gguf").exists())
                path.write_bytes(b"x" * len(payload(name)))
                self.assertIn("Checksum mismatch", self.run_download(target, ok=False))
                self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_qwen_is_one_verified_file(self):
        for target, name in (("qwen38-q2", QWEN_Q2), ("qwen38-q4k", QWEN_Q4)):
            self.assertIn("Verifying SHA-256", self.run_download(target))
            self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / name)
            self.assertEqual((self.out / name).read_bytes(), payload(name))
        self.assertEqual({p.name for p in self.out.iterdir()}, {QWEN_Q2, QWEN_Q4})
        self.assertIn("Already downloaded", self.run_download("qwen38-iq2"))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / QWEN_Q2)
        self.assertNotIn("--ple", self.run_download("--help"))

    def test_failure_does_not_replace_link(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD"] = "1"
        self.run_download("ds41f-q4", ok=False)
        self.run_download("ds41f-vision", ok=False)
        self.run_download("ds41f-1017-vision", ok=False)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)

    def test_flint_pinned_directory_bundle_and_existing(self):
        self.run_download("ds41f-q2")
        calls = self.root / "hf-calls.jsonl"
        self.env["HF_CALLS"] = str(calls)
        text = self.run_download("flint")
        self.assertIn("Verifying SHA-256", text)
        directory = self.out / "flint-int4"
        self.assertEqual({p.name for p in directory.iterdir()}, set(FLINT_ARTIFACTS))
        for name in FLINT_ARTIFACTS:
            self.assertEqual((directory / name).read_bytes(), payload(name))
        self.assertTrue((directory / "flint-int4.json").is_file())
        self.assertFalse((self.out / "config.json").exists())
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        records = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual([row["file"] for row in records], list(FLINT_ARTIFACTS))
        for row in records:
            self.assertEqual(row["repo"], "RecursiveSI/flint-int4")
            self.assertEqual(row["revision"], FLINT_REVISION)
            self.assertEqual(Path(row["directory"]), directory)
        self.env["FAIL_DOWNLOAD"] = "1"
        self.assertEqual(self.run_download("flint-int4").count("Already downloaded"), 12)
        self.assertEqual(len(calls.read_text().splitlines()), 12)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)

    def test_flint_corrupt_metadata_and_resume_missing_shard(self):
        self.run_download("ds41f-q2")
        self.run_download("flint-int4")
        directory = self.out / "flint-int4"
        metadata = directory / "config.json"
        metadata.write_bytes(b"x" * len(payload("config.json")))
        self.assertIn("Checksum mismatch", self.run_download("flint", ok=False))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        metadata.write_bytes(payload("config.json"))
        missing = "model-00003.safetensors"
        (directory / missing).unlink()
        calls = self.root / "hf-calls.jsonl"
        self.env.update(HF_CALLS=str(calls), FAIL_DOWNLOAD_FILE=missing)
        self.run_download("flint-int4", ok=False)
        self.assertFalse((directory / missing).exists())
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        del self.env["FAIL_DOWNLOAD_FILE"]
        self.run_download("flint")
        self.assertEqual((directory / missing).read_bytes(), payload(missing))
        records = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual([row["file"] for row in records], [missing, missing])
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)

    def test_partial_requires_explicit_cleanup(self):
        self.out.mkdir()
        (self.out / (Q2 + ".part")).write_bytes(b"partial")
        self.assertIn("cannot resume", self.run_download("ds41f-q2", ok=False))
        self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_q4_resumes_interrupted_assembly(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD"] = "1"
        pending = self.out / (Q4 + ".assembling")
        for tail in (b"", payload(PART2)[:7], payload(PART2)):
            with self.subTest(tail=len(tail)):
                pending.write_bytes(payload(PART1) + tail)
                (self.out / PART2).write_bytes(payload(PART2))
                self.run_download("ds41f-q4")
                self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))
                self.assertFalse(pending.exists())
                self.assertFalse((self.out / PART2).exists())
                (self.out / Q4).unlink()

    def test_q4_rejects_corrupt_parts_and_assembly(self):
        self.run_download("ds41f-q2")
        link = self.root / "ds4flash.gguf"
        for name in (PART1, PART2):
            with self.subTest(part=name):
                for other in (PART1, PART2):
                    (self.out / other).write_bytes(payload(other))
                (self.out / name).write_bytes(b"x" * len(payload(name)))
                self.assertIn("Checksum mismatch", self.run_download("ds41f-q4", ok=False))
                self.assertEqual(link.resolve(), self.out / Q2)
                self.assertFalse((self.out / Q4).exists())
        (self.out / PART1).unlink()
        (self.out / PART2).write_bytes(payload(PART2))
        pending = self.out / (Q4 + ".assembling")
        pending.write_bytes(b"x" * len(payload(PART1)))
        self.env["FAIL_DOWNLOAD"] = "1"
        self.assertIn("Checksum mismatch", self.run_download("ds41f-q4", ok=False))
        self.assertEqual(link.resolve(), self.out / Q2)
        self.assertFalse((self.out / Q4).exists())
        self.assertTrue(pending.exists())

    def test_q4_no_space_preserves_resumable_parts(self):
        self.run_download("ds41f-q2")
        for name in (PART1, PART2):
            (self.out / name).write_bytes(payload(name))
        (self.bin / "sitecustomize.py").write_text(
            "import shutil\nfrom collections import namedtuple\n"
            "shutil.disk_usage = lambda _: namedtuple('usage', 'total used free')(1, 1, 0)\n")
        self.env["PYTHONPATH"] = str(self.bin)
        self.assertIn("Not enough disk space", self.run_download("ds41f-q4", ok=False))
        self.assertEqual((self.out / (Q4 + ".assembling")).read_bytes(), payload(PART1))
        self.assertEqual((self.out / PART2).read_bytes(), payload(PART2))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        del self.env["PYTHONPATH"]
        self.env["FAIL_DOWNLOAD"] = "1"
        self.run_download("ds41f-q4")
        self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))

    def test_hf_failure_preserves_code_and_hides_token(self):
        self.env["FAIL_DOWNLOAD"] = "1"
        result = subprocess.run(
            ["nu", str(self.script), "ds41f-q2", "--token", "fixture secret"],
            env=self.env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("fixture secret", result.stdout + result.stderr)
        self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_help_and_invalid_target(self):
        help_text = self.run_download("--help")
        self.assertIn("ds41f-q2", help_text)
        self.assertIn("ds41f-q4", help_text)
        self.assertIn("ds41f-mxfp4", help_text)
        self.assertIn("ds41f-vision", help_text)
        self.assertIn("ds41f-1017-vision", help_text)
        for retired in ("ds4f-q2", "ds4f-vision-q2", "glm-antirez-q2", "glm-unsloth-q4"):
            self.assertNotIn(retired, help_text)
            self.assertIn("Unknown model", self.run_download(retired, ok=False))
        self.assertIn("pro-q2-imatrix", help_text)
        self.assertIn("glm53-full-q2", help_text)
        self.assertIn("Unknown model", self.run_download("nonexistent", ok=False))
        self.assertFalse(self.out.exists())


    def test_mxfp4_merges_and_verifies_all_shards(self):
        text = self.run_download("ds41f-mxfp4")
        self.assertEqual(text.count("Verifying SHA-256"), 10)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / MX)
        self.assertEqual((self.out / MX).read_bytes(), payload(MX))
        for name in MX_SHARDS:
            self.assertEqual((self.out / name).read_bytes(), payload(name))
        self.assertFalse((self.out / (MX + ".assembling")).exists())
        self.env["FAIL_DOWNLOAD"] = "1"
        self.assertIn("Already downloaded", self.run_download("ds41f-mxfp4"))

    def test_mxfp4_resumes_download_without_changing_default(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD_FILE"] = MX_SHARDS[4]
        self.run_download("ds41f-mxfp4", ok=False)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        self.assertFalse((self.out / MX).exists())
        for name in MX_SHARDS[:4]:
            self.assertEqual((self.out / name).read_bytes(), payload(name))
        del self.env["FAIL_DOWNLOAD_FILE"]
        self.assertEqual(self.run_download("ds41f-mxfp4").count("Already downloaded"), 4)
        self.assertEqual((self.out / MX).read_bytes(), payload(MX))

    def test_mxfp4_rejects_corrupt_shard_and_keeps_default(self):
        self.run_download("ds41f-q2")
        for data, error in [(b"short", "Incorrect file size"),
                            (b"x" * len(payload(MX_SHARDS[4])), "Checksum mismatch")]:
            (self.out / MX_SHARDS[4]).write_bytes(data)
            self.assertIn(error, self.run_download("ds41f-mxfp4", ok=False))
            self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
            self.assertFalse((self.out / MX).exists())
            self.assertEqual((self.out / MX_SHARDS[4]).read_bytes(), data)

    def test_mxfp4_resumes_merge_and_rejects_corrupt_prefix(self):
        self.run_download("ds41f-q2")
        for name in MX_SHARDS:
            (self.out / name).write_bytes(payload(name))
        self.env["FAIL_DOWNLOAD"] = "1"
        pending = self.out / (MX + ".assembling")
        pending.write_bytes(b"wrong prefix")
        self.assertIn("Partial assembly differs", self.run_download("ds41f-mxfp4", ok=False))
        self.assertEqual(pending.read_bytes(), b"wrong prefix")
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        for boundary in [17, 16390, len(payload(MX)) - 5]:
            with self.subTest(boundary=boundary):
                pending.write_bytes(payload(MX)[:boundary])
                self.run_download("ds41f-mxfp4")
                self.assertEqual((self.out / MX).read_bytes(), payload(MX))
                self.assertFalse(pending.exists())
                (self.out / MX).unlink()

    def test_mxfp4_checks_space_before_downloading(self):
        self.run_download("ds41f-q2")
        catalog_path = self.root / "tools/model-downloads.json"
        catalog = json.loads(catalog_path.read_text())
        catalog["artifacts"][MX]["bytes"] = 1 << 62
        catalog_path.write_text(json.dumps(catalog))
        calls = self.root / "hf-calls.jsonl"
        self.env["HF_CALLS"] = str(calls)
        text = self.run_download("ds41f-mxfp4", ok=False)
        self.assertIn("Not enough disk space", text)
        self.assertNotIn("Downloading", text)
        self.assertFalse(calls.exists())
        self.assertFalse(any((self.out / n).exists() for n in MX_SHARDS))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)

    def test_mxfp4_rejects_bad_final_checksum(self):
        self.run_download("ds41f-q2")
        catalog_path = self.root / "tools/model-downloads.json"
        catalog = json.loads(catalog_path.read_text())
        catalog["artifacts"][MX]["sha256"] = "0" * 64
        catalog_path.write_text(json.dumps(catalog))
        self.assertIn("Checksum mismatch", self.run_download("ds41f-mxfp4", ok=False))
        self.assertFalse((self.out / MX).exists())
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        self.assertTrue((self.out / (MX + ".assembling")).exists())
        for name in MX_SHARDS:
            self.assertEqual((self.out / name).read_bytes(), payload(name))

    def test_mxfp4_merge_rejects_missing_or_reordered_shards(self):
        self.out.mkdir()
        for name in MX_SHARDS:
            (self.out / name).write_bytes(payload(name))
        for names in [MX_SHARDS[:-1], list(reversed(MX_SHARDS))]:
            result = subprocess.run([
                "python3", str(self.root / "gguf-tools/merge_deepseek41_shards.py"),
                "--output", str(self.out / MX), "--expected-bytes", str(len(payload(MX))),
                "--sha256", hashlib.sha256(payload(MX)).hexdigest(),
                *[str(self.out / name) for name in names]],
                env=self.env, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Incorrect shard order/count", result.stderr)
            self.assertFalse((self.out / MX).exists())
            self.assertFalse((self.out / (MX + ".assembling")).exists())


if __name__ == "__main__":
    unittest.main()
