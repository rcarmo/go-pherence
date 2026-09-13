"""Offline contracts for permutation-invariant speaker-attributed WER."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("score_speaker_words", ROOT / "score_speaker_words.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(words):
    return {
        "schema": 1,
        "corpus": "fixture",
        "meeting": "XX0000a",
        "source_interval": [0, 10],
        "sample_rate": 16000,
        "speakers": ["A", "B"],
        "words": [
            {"start": i, "end": i + .5, "speaker": speaker, "text": text, "source_id": f"w{i}", "punctuation": False}
            for i, (speaker, text) in enumerate(words)
        ],
    }


def hypothesis(words):
    labelled = sum(speaker >= 0 for speaker, _ in words)
    return {
        "schema": 2,
        "experimental": True,
        "transcript_key": "0" * 64,
        "diarization_key": "1" * 64,
        "policy": "exclusive-turns-maximum-positive-overlap-per-word-v3",
        "labelled_cues": 0,
        "unlabelled_cues": 1,
        "labelled_words": labelled,
        "unlabelled_words": len(words) - labelled,
        "transcript": {
            "schema": 2,
            "sample_rate": 16000,
            "total_samples": 160000,
            "language": "en",
            "source_timing": {},
            "cues": [{"start_sample": 0, "end_sample": 160000, "speaker": -1, "text": "fixture"}],
            "words": [
                {"start_sample": i * 16000, "end_sample": i * 16000 + 8000, "speaker": speaker, "text": text}
                for i, (speaker, text) in enumerate(words)
            ],
        },
    }


class SpeakerWordScorerContract(unittest.TestCase):
    def test_edit_counts(self):
        self.assertEqual(MODULE.edit_counts(["a", "b"], ["a", "b"]), {"errors": 0, "substitutions": 0, "deletions": 0, "insertions": 0})
        self.assertEqual(MODULE.edit_counts(["a", "b"], ["a", "c", "d"])["errors"], 2)
        self.assertEqual(MODULE.edit_counts([], ["a"])["insertions"], 1)
        self.assertEqual(MODULE.edit_counts(["a"], [])["deletions"], 1)

    def test_normalization(self):
        self.assertEqual(MODULE.normalize("L’École's—42!"), ["l'école's", "42"])
        self.assertEqual(MODULE.normalize("ＡＢＣ"), ["abc"])
        self.assertEqual(MODULE.normalize("..."), [])

    def test_permutation_is_free_but_wrong_attribution_is_not(self):
        ref = MODULE.reference_words(reference([("A", "one"), ("B", "two"), ("A", "three")]))
        permuted = MODULE.hypothesis_words(hypothesis([(7, "one"), (2, "two"), (7, "three")]))
        cost, mapping = MODULE.assignment_cost(ref, permuted)
        self.assertEqual(cost, 0)
        self.assertEqual(mapping, [
            {"reference": "A", "hypothesis": 7, "errors": 0},
            {"reference": "B", "hypothesis": 2, "errors": 0},
        ])
        wrong = MODULE.hypothesis_words(hypothesis([(7, "one"), (7, "two"), (2, "three")]))
        cost, _ = MODULE.assignment_cost(ref, wrong)
        self.assertGreater(cost, 0)

    def test_unlabelled_word_is_separate_error(self):
        ref = MODULE.reference_words(reference([("A", "one"), ("B", "two")]))
        hyp = MODULE.hypothesis_words(hypothesis([(3, "one"), (-1, "two")]))
        lexical = MODULE.edit_counts([word[3] for word in ref], [word[3] for word in hyp])
        stream_errors, _ = MODULE.assignment_cost(ref, hyp)
        self.assertEqual(lexical["errors"], 0)
        self.assertEqual(stream_errors, 1)
        self.assertEqual(stream_errors + sum(word[2] < 0 for word in hyp), 2)

    def test_unknown_keys_duplicate_keys_and_accounting_rejected(self):
        value = hypothesis([(0, "one")])
        value["extra"] = 1
        with self.assertRaisesRegex(ValueError, "invalid keys"):
            MODULE.hypothesis_words(value)
        value = hypothesis([(0, "one")])
        value["labelled_words"] = 0
        with self.assertRaisesRegex(ValueError, "accounting"):
            MODULE.hypothesis_words(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema":1,"schema":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                MODULE.read_json(path)

    def test_cli_result_is_absolute_and_unqualified(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ref, hyp, out = directory / "reference.json", directory / "hypothesis.json", directory / "result.json"
            ref.write_text(json.dumps(reference([("A", "one"), ("B", "two")])), encoding="utf-8")
            hyp.write_text(json.dumps(hypothesis([(4, "one"), (9, "two")])), encoding="utf-8")
            old = __import__("sys").argv
            try:
                __import__("sys").argv = ["score_speaker_words.py", "--reference-words", str(ref), "--speaker-transcript", str(hyp), "--output", str(out)]
                MODULE.main()
            finally:
                __import__("sys").argv = old
            result = json.loads(out.read_text())
            self.assertEqual(result["wer"]["rate"], 0)
            self.assertEqual(result["cp_sawer"]["rate"], 0)
            self.assertFalse(result["qualified"])


if __name__ == "__main__":
    unittest.main()
