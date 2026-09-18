"""Contracts for the deterministic AMI corpus excerpt preparer."""
import importlib.util
import json
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("prepare_ami", ROOT / "prepare_ami_corpus.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def xml(kind, meeting, speaker):
    if kind == "segments":
        root_id = f"{meeting}.{speaker}.segs"
        body = '<segment nite:id="s0" transcriber_start="0.5" transcriber_end="2.5"><nite:child href="words/x"/></segment>'
    else:
        root_id = f"{meeting}.{speaker}.words"
        body = '<w nite:id="w0" starttime="0.75" endtime="1.25">hello</w><gap nite:id="g0" starttime="1.25" endtime="1.25"/>'
    return f'<?xml version="1.0"?><nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="{root_id}">{body}</nite:root>'.encode()


class AMIPreparerContract(unittest.TestCase):
    def test_parse_clip_and_write(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "annotations.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for speaker in MODULE.SPEAKERS:
                    zf.writestr(f"segments/ES2004a.{speaker}.segments.xml", xml("segments", "ES2004a", speaker))
                    zf.writestr(f"words/ES2004a.{speaker}.words.xml", xml("words", "ES2004a", speaker))
            turns, words = MODULE.parse_annotations(archive, "ES2004a")
            self.assertEqual(len(turns), 4)
            self.assertEqual(len(words), 4)
            self.assertEqual(words[0][3], "hello")
            self.assertEqual(MODULE.clip_interval(.5, 2.5, 1, 2), (0, 1))
            source, output = directory / "source.wav", directory / "out.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b"\x00\x00" * 32000)
            MODULE.write_wav(source, output, 8000, 16000)
            with wave.open(str(output), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 16000, 16000))

    def test_rejects_hash_geometry_and_archive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"value")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                MODULE.checked_hash(path, "0" * 64, "fixture")
            self.assertIsNone(MODULE.clip_interval(0, 1, 1, 2))
            with zipfile.ZipFile(Path(directory) / "bad.zip", "w") as zf:
                zf.writestr("../bad", b"x")
            with zipfile.ZipFile(Path(directory) / "bad.zip") as zf:
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    MODULE.archive_member(zf, "../bad")

    def test_rejects_unexpected_xml_and_missing_timed_word(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "annotations.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for speaker in MODULE.SPEAKERS:
                    zf.writestr(f"segments/ES2004a.{speaker}.segments.xml", xml("segments", "ES2004a", speaker))
                    value = xml("words", "ES2004a", speaker)
                    if speaker == "A":
                        value = value.replace(b' endtime="1.25"', b"")
                    zf.writestr(f"words/ES2004a.{speaker}.words.xml", value)
            with self.assertRaisesRegex(ValueError, "invalid word end"):
                MODULE.parse_annotations(archive, "ES2004a")


if __name__ == "__main__":
    unittest.main()
