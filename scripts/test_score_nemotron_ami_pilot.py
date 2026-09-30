"""Offline validation tests for the single AMI Nemotron pilot scorer."""
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("score_nemotron_ami_pilot", ROOT / "score_nemotron_ami_pilot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AMIPilotScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = Path(self.temp.name)
        self.paths = [p / name for name in (
            "contract.json", "manifest.json", "ES2004a-350-410.wav", "ES2004a-350-410.rttm",
            "ES2004a-350-410.words.json", "transcript.txt", "spans.json")]
        contract, manifest, audio, rttm, words, transcript, spans = self.paths
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 960000)
        rttm.write_text("".join(
            f"SPEAKER ES2004a-350-410 1 {i * 2:.6f} 1.000000 <NA> <NA> {'ABCD'[i % 4]} <NA> <NA>\n"
            for i in range(28)), encoding="utf-8")
        words.write_text(json.dumps({
            "schema": 1, "corpus": "AMI Meeting Corpus", "meeting": "ES2004a", "source_interval": [350, 410],
            "sample_rate": 16000, "speakers": list("ABCD"),
            "words": [{"start": i / 3, "end": i / 3 + .1, "speaker": "ABCD"[i % 4], "text": "hello",
                       "source_id": f"w{i}", "punctuation": False} for i in range(179)],
        }), encoding="utf-8")
        transcript.write_text("Hello.\n", encoding="utf-8")
        spans.write_text(json.dumps([{"Start": 0, "End": 1, "Speaker": 0}]), encoding="utf-8")
        manifest.write_text(json.dumps({
            "schema": 1, "meeting": "ES2004a", "source_interval": [350, 410], "source": {}, "licence": {},
            "output": {
                "audio": {"path": audio.name, "sha256": MODULE.CORPUS.sha256(audio), "samples": 960000},
                "rttm": {"path": rttm.name, "sha256": MODULE.CORPUS.sha256(rttm), "turns": 28, "uri": "ES2004a-350-410"},
                "words": {"path": words.name, "sha256": MODULE.CORPUS.sha256(words), "count": 179},
            },
        }), encoding="utf-8")
        contract.write_bytes(manifest.read_bytes())

    def run_score(self):
        with patch.object(MODULE.CORPUS, "score_turns", return_value=[{"collar": 0., "der": {}, "jer": {}}]) as scorer:
            result = MODULE.score(*self.paths)
        return result, scorer

    def test_transcript_only_metrics_and_metric_conventions(self):
        result, scorer = self.run_score()
        self.assertEqual(result["wer"]["reference_tokens"], 179)
        self.assertEqual(result["wer"]["errors"], 178)
        self.assertEqual(result["wer"]["deletions"], 178)
        self.assertEqual(result["wer"]["rate"], 178 / 179)
        self.assertFalse(result["qualified"])
        self.assertNotIn("cp_sawer", result)
        self.assertEqual(result["diarization"]["hypothesis_spans"], 1)
        self.assertFalse(result["diarization"]["skip_overlap"])
        self.assertEqual(scorer.call_args.args[2:], ([[0, 60]], [(0., 1., "0")], [0., .25]))

    def test_changed_reference_and_spans_rejected(self):
        self.paths[4].write_text(self.paths[4].read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            self.run_score()
        self.paths[4].write_text(self.paths[4].read_text()[:-1], encoding="utf-8")
        self.paths[6].write_text(json.dumps([{"Start": 0, "End": 61, "Speaker": 0}]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "turn outside"):
            self.run_score()

    def test_invalid_transcript_and_unsupported_pilot(self):
        self.paths[5].write_text(" ... ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid transcript"):
            self.run_score()
        data = json.loads(self.paths[1].read_text())
        data["source_interval"] = [351, 411]
        self.paths[1].write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from pinned contract"):
            self.run_score()


if __name__ == "__main__":
    unittest.main()
