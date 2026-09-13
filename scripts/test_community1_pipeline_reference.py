"""Source-only reference/scorer safety contracts; no neural imports/execution."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).parent


class PipelineReferenceContract(unittest.TestCase):
    def test_restricted_loading(self):
        tree=ast.parse((ROOT/"community1_pipeline_reference.py").read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=="torch" and n.func.attr=="load"]
        self.assertEqual(len(calls),1)
        self.assertEqual({kw.arg:ast.literal_eval(kw.value) for kw in calls[0].keywords},dict(map_location="cpu",weights_only=True))
    def test_source_pins(self):
        tree=ast.parse((ROOT/"community1_pipeline_reference.py").read_text())
        entry=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=="SOURCE_HASHES")
        pins=ast.literal_eval(entry.value)
        self.assertEqual(len(pins),6)
        self.assertIn("core/plda.py",pins)
        self.assertIn("utils/vbx.py",pins)
        for v in pins.values():self.assertEqual(len(v),64);int(v,16)
    def test_pcm_identity_is_explicit_and_bounded(self):
        text=(ROOT/"community1_pipeline_reference.py").read_text();tree=ast.parse(text)
        parser_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument"]
        flags={ast.literal_eval(n.args[0]) for n in parser_calls if n.args and isinstance(n.args[0],ast.Constant)}
        self.assertTrue({"--public-wav-sha256","--samples","--uri"}.issubset(flags))
        self.assertIn("160000 + 127 * 16000", text)
        self.assertIn("args.samples * 2", text)
    def test_scorer_no_inference(self):
        text=(ROOT/"score_community1_diarization.py").read_text();tree=ast.parse(text)
        imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):imports.append(n.module)
        self.assertNotIn("torch",imports)
        self.assertNotIn("pyannote.audio",imports)
        loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=="np" and n.func.attr=="load"]
        self.assertEqual(len(loads),2)
        for call in loads:self.assertIn(("allow_pickle",False),[(k.arg,ast.literal_eval(k.value)) for k in call.keywords])


if __name__=="__main__":unittest.main()
