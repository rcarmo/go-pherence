"""Offline contracts for absolute trained Community-1 corpus scoring."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("score_community1_quality", ROOT / "score_community1_quality.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(tie=1, ambiguous=None):
    cfg = {key: 0 for key in MODULE.CORPUS.GO_CONFIG_KEYS}
    cfg.update(WindowSamples=160000, StepSamples=16000, MinimumEmbeddingSamples=400, MinSpeakers=1, MaxSpeakers=4, AHCThreshold=.6, Fa=.07, Fb=.8, TiePolicy=tie)
    post = {key: [] for key in MODULE.CORPUS.GO_POSTPROCESS_KEYS}
    post.update(TrainingRows=2, Clusters=2, KMeansLabels=None, FullTurns=[{"Start": 0.1, "End": 1.0, "Speaker": 0}], ExclusiveTurns=[{"Start": 0.1, "End": 1.0, "Speaker": 0}])
    post["Timeline"] = {key: [] for key in MODULE.CORPUS.GO_TIMELINE_KEYS}
    post["Timeline"].update(Frames=10, Classes=2, Start=0, FrameDuration=.1, FrameStep=.1, AmbiguousFrames=ambiguous or [])
    body = {key: [] for key in MODULE.CORPUS.GO_RESULT_KEYS}
    body.update(Windows=[{"Start": 0, "Samples": 32000, "Padding": 128000}], Postprocess=post)
    return {"Schema": 1, "InputSHA256": "0" * 64, "InputSamples": 32000, "ElapsedNanos": 1, "Config": cfg, "Result": body}


class CommunityQualityContract(unittest.TestCase):
    def test_loads_checked_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result()), encoding="utf-8")
            data, turns = MODULE.load_result(path, "0" * 64, 32000)
            self.assertEqual(data["ElapsedNanos"], 1)
            self.assertEqual(turns["full"], [(0.1, 1.0, "0")])

    def test_rejects_strict_ambiguity_and_identity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result(0, [2])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict result"):
                MODULE.load_result(path, "0" * 64, 32000)
            path.write_text(json.dumps(result()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input hash"):
                MODULE.load_result(path, "1" * 64, 32000)
            with self.assertRaisesRegex(ValueError, "input samples"):
                MODULE.load_result(path, "0" * 64, 1)

    def test_kmeans_labels_follow_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            value = result(); value["Result"]["Postprocess"].update(Path="clustered-kmeans", KMeansLabels=[0, 1])
            path.write_text(json.dumps(value), encoding="utf-8")
            MODULE.load_result(path, "0" * 64, 32000)
            value["Result"]["Postprocess"]["KMeansLabels"] = None
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing KMeans"):
                MODULE.load_result(path, "0" * 64, 32000)

    def test_rejects_unknown_wrapper_and_bad_tie_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            value = result(); value["extra"] = 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid keys"):
                MODULE.load_result(path, "0" * 64, 32000)
            value = result(); value["Config"]["TiePolicy"] = 3
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tie policy"):
                MODULE.load_result(path, "0" * 64, 32000)


if __name__ == "__main__":
    unittest.main()
