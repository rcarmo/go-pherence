"""Offline safety/metadata checks; no torch/model imports or neural execution."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name("community1_checkpoint_export.py")
spec = importlib.util.spec_from_file_location("checkpoint_export", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportContract(unittest.TestCase):
    def test_pinned_metadata(self):
        self.assertEqual(len(module.CHECKPOINT_SHA), 64)
        self.assertEqual(module.REVISION, "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")
        self.assertEqual(module.CONFIG["LSTM"], dict(InputSize=60, HiddenSize=128, NumLayers=4, Bidirectional=True))
        self.assertEqual(module.CONFIG["Head"]["Speakers"], 3)
        self.assertEqual(module.CONFIG["Head"]["MaxActive"], 2)
        self.assertEqual(len(module.HASHES), 5)

    def test_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fake.bin"
            p.write_bytes(b"not a checkpoint")
            with self.assertRaises(ValueError):
                module.checked(p, module.CHECKPOINT_SHA)
            self.assertEqual(module.checked(p, module.sha(p)), p)

    def test_no_unsafe_load(self):
        tree = ast.parse(SOURCE.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch" and n.func.attr == "load"]
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
        self.assertEqual(kwargs, dict(map_location="cpu", weights_only=True))


if __name__ == "__main__":
    unittest.main()
