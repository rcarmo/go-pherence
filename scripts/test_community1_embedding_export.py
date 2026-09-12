"""Offline exporter contracts; no torch/model imports or neural execution."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name("community1_embedding_export.py")
spec = importlib.util.spec_from_file_location("embedding_export", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportContract(unittest.TestCase):
    def test_pins(self):
        self.assertEqual(module.REVISION, "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")
        self.assertEqual(module.CONFIG, dict(BaseChannels=32, MelBins=80, EmbedDim=256))
        self.assertEqual(len(module.HASHES), 4)
        for digest in [module.CHECKPOINT_SHA, module.PCM_SHA, module.KALDI_SHA, *module.HASHES.values()]:
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_hash_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"fake.bin"
            p.write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                module.checked(p, module.CHECKPOINT_SHA)
            module.checked(p, module.sha(p))

    def test_restricted_load(self):
        calls = [n for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "torch" and n.func.attr == "load"]
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
        self.assertEqual(kwargs, dict(map_location="cpu", weights_only=True))


if __name__ == "__main__":
    unittest.main()
