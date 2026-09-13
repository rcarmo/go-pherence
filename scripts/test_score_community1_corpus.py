"""Offline contracts for the saved-result Community-1 corpus scorer."""
import gzip
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("community1_corpus", ROOT / "score_community1_corpus.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(value):
    return "0" * 64 if value == 0 else "1" * 64


def valid_manifest(role="diagnostic", redistribution="unverified"):
    return {
        "schema": 1,
        "model_revision": MODULE.MODEL_REVISION,
        "cases": [{
            "id": "case-1", "role": role, "corpus": "fixture", "split": "test", "language": "en",
            "licence": {"spdx": "NOASSERTION", "url": "https://example.invalid/licence", "redistribution": redistribution},
            "audio": {"sha256": sha(0), "sample_rate": 16000, "channels": 1, "samples": 32000},
            "rttm": {"sha256": sha(1), "uri": "fixture"}, "uem": [[0, 1], [1, 2]], "turn_extent_end": 2.1,
            "go_result": {"sha256": sha(1), "format": "community1-go-result-v1"},
            "reference": {"sha256": sha(0), "format": "community1-reference-v1"},
            "collars": [0, .25], "tie_policy": 1, "exclusive_may_overlap": False,
            "gates": {"max_abs_der_delta_pp": 1e-9, "max_abs_jer_delta_pp": 1e-9},
        }],
    }


class CorpusScorerContract(unittest.TestCase):
    def test_checked_manifest_and_selection(self):
        manifest = MODULE.validate_manifest(valid_manifest())
        self.assertEqual(MODULE.select_case(manifest, "case-1")["role"], "diagnostic")
        with self.assertRaises(ValueError):
            MODULE.select_case(manifest, "absent")

    def test_qualifying_requires_verified_redistribution(self):
        with self.assertRaisesRegex(ValueError, "unverified redistribution"):
            MODULE.validate_manifest(valid_manifest("qualifying", "unverified"))
        self.assertEqual(MODULE.validate_manifest(valid_manifest("qualifying", "verified"))["cases"][0]["role"], "qualifying")

    def test_rejects_noncanonical_integer_fields(self):
        for key, value in (("schema", True), ("schema", 1.0)):
            manifest = valid_manifest(); manifest[key] = value
            with self.assertRaises(ValueError): MODULE.validate_manifest(manifest)
        for key, value in (("sample_rate", 16000.0), ("channels", True), ("samples", True)):
            manifest = valid_manifest(); manifest["cases"][0]["audio"][key] = value
            with self.assertRaises(ValueError): MODULE.validate_manifest(manifest)
        for value in (True, 1.0):
            manifest = valid_manifest(); manifest["cases"][0]["tie_policy"] = value
            with self.assertRaises(ValueError): MODULE.validate_manifest(manifest)

    def test_rejects_duplicate_keys_and_bad_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for raw, message in (('{"schema":1,"schema":1}', "duplicate JSON key"), ('{"value":NaN}', "nonfinite JSON value")):
                path.write_text(raw)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.read_json(path)
        for change in (lambda c: c["audio"].update(samples=0), lambda c: c.update(uem=[[0, 3]]), lambda c: c.update(collars=[.25, 0]), lambda c: c.update(tie_policy=2)):
            value = valid_manifest()
            change(value["cases"][0])
            with self.assertRaises(ValueError):
                MODULE.validate_manifest(value)

    def test_saved_turns_plain_and_gzip(self):
        go = {"Config": {key: 0 for key in MODULE.GO_CONFIG_KEYS}, "Result": {key: [] for key in MODULE.GO_RESULT_KEYS}}
        go["Config"]["TiePolicy"] = 1
        post = {key: [] for key in MODULE.GO_POSTPROCESS_KEYS}
        post.update(TrainingRows=1, Clusters=1, FullTurns=[{"Start": 0, "End": 1, "Speaker": 0}], ExclusiveTurns=[])
        post["Timeline"] = {key: [] for key in MODULE.GO_TIMELINE_KEYS}
        post["Timeline"]["AmbiguousFrames"] = []
        go["Result"]["Postprocess"] = post
        reference = {"schema": 1, "model_revision": MODULE.MODEL_REVISION, "source_hashes": {}, "threads": 1, "mkldnn": False, "batch_sizes": 1, "constrained": True, "minimum_embedding_samples": 400, "full": [{"start": 0, "end": 1, "speaker": "A"}], "exclusive": [], "artifacts": {}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            go_path, ref_path = directory / "go.json.gz", directory / "reference.json"
            with gzip.open(go_path, "wt", encoding="utf-8") as stream:
                json.dump(go, stream)
            ref_path.write_text(json.dumps(reference))
            turns, _ = MODULE.load_saved_turns(go_path, 2, True, tie_policy=1)
            self.assertEqual(turns["full"], [(0.0, 1.0, "0")])
            turns, _ = MODULE.load_saved_turns(ref_path, 2, False, expected_hash=MODULE.sha256(ref_path))
            self.assertEqual(turns["full"], [(0.0, 1.0, "A")])
            reference["num_speakers"] = 1
            ref_path.write_text(json.dumps(reference))
            MODULE.load_saved_turns(ref_path, 2, False)
            reference["num_speakers"] = 0
            ref_path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "reference speaker count"):
                MODULE.load_saved_turns(ref_path, 2, False)
            with self.assertRaisesRegex(ValueError, "tie policy"):
                MODULE.load_saved_turns(go_path, 2, True, tie_policy=0)
            for invalid in (True, 1.0):
                go["Config"]["TiePolicy"] = invalid
                with gzip.open(go_path, "wt", encoding="utf-8") as stream: json.dump(go, stream)
                with self.assertRaisesRegex(ValueError, "Go tie policy"):
                    MODULE.load_saved_turns(go_path, 2, True, tie_policy=1)

    def test_saved_go_result_rejects_unknown_schema(self):
        go = {"Config": {key: 0 for key in MODULE.GO_CONFIG_KEYS}, "Result": {key: [] for key in MODULE.GO_RESULT_KEYS}, "extra": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "go.json"
            path.write_text(json.dumps(go))
            with self.assertRaisesRegex(ValueError, "invalid keys"):
                MODULE.load_saved_turns(path, 2, True, tie_policy=0)

    def test_exclusive_turns_must_be_sorted_and_disjoint(self):
        overlapping = [{"Start": 0, "End": 1, "Speaker": 0}, {"Start": .5, "End": 1.5, "Speaker": 1}]
        with self.assertRaisesRegex(ValueError, "exclusive turns overlap"):
            MODULE.parse_turns(overlapping, 2, True, exclusive=True)
        self.assertEqual(len(MODULE.parse_turns(overlapping, 2, True, exclusive=True, may_overlap=True)), 2)
        with self.assertRaisesRegex(ValueError, "turns not sorted"):
            MODULE.parse_turns(list(reversed(overlapping)), 2, True)

    def test_padded_tail_and_numeric_go_speaker_order(self):
        turns = [{"Start": 1.9, "End": 2.05, "Speaker": 2}, {"Start": 1.9, "End": 2.05, "Speaker": 10}]
        self.assertEqual(MODULE.parse_turns(turns, 2.1, True), [(1.9, 2.05, "2"), (1.9, 2.05, "10")])
        with self.assertRaisesRegex(ValueError, "declared extent"):
            MODULE.parse_turns(turns, 2, True)

    def test_metric_value(self):
        item = {"der": {"diarization error rate": .125}, "jer": {"jaccard error rate": .25}}
        self.assertEqual(MODULE.metric_value(item, "der"), .125)
        self.assertEqual(MODULE.metric_value(item, "jer"), .25)


if __name__ == "__main__":
    unittest.main()
