#!/usr/bin/env python3
"""Generate the pinned Transformers 4.57.1 multilingual word oracle.

Run with the workspace's PyTorch 2.14 CPU environment and a local Whisper Tiny:
  PYTHONPATH=../whisper-stt/runtime/whisper-oracle-4.57.1 \
  ../whisper-stt/runtime/diar-export/bin/python3 transformers-oracle.py \
    --model ../models/whisper-tiny-169d4a4 --fixtures tmp/minds-final-20260913
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import _combine_tokens_into_words

FIXTURES = [
    ("minds-pt-0", "pt", "pt-real-0-source.wav", "fc084982ad50c6ea6cf066f08374b9b3aaa628d9a9accb167be5ae9376dbd275"),
    ("minds-pt-1", "pt", "pt-real-1-source.wav", "aacee91914f902b0949425ee29ac984c48a34df55e5c195e65e0c8ff66977484"),
    ("minds-fr-0", "fr", "fr-real-0-source.wav", "84defdc828ef59cec10364354fbc284bc2cc683fdd4a5edd5863b7bb2c6123a8"),
]

def pcm(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"])
    return np.frombuffer(raw, dtype="<f4").copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", required=True)
    args = parser.parse_args()
    processor = WhisperProcessor.from_pretrained(args.model, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, local_files_only=True, attn_implementation="eager").eval()
    for name, language, filename, want_hash in FIXTURES:
        path = Path(args.fixtures) / filename
        got_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if got_hash != want_hash:
            raise SystemExit(f"{name}: sha256 {got_hash}, expected {want_hash}")
        audio = pcm(path)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            out = model.generate(inputs.input_features, attention_mask=inputs.attention_mask, language=language, task="transcribe", max_new_tokens=96, return_token_timestamps=True, return_dict_in_generate=True)
        ids = out["sequences"][0].tolist()
        times = out["token_timestamps"][0].tolist()
        text_ids = ids[4:-1]
        words, _, indices = _combine_tokens_into_words(processor.tokenizer, text_ids, language=language)
        spans = []
        for word, positions in zip(words, indices):
            first, last = 4 + positions[0], 4 + positions[-1] + 1
            spans.append({"word": word.strip(), "token_start": positions[0], "token_end": positions[-1] + 1, "start": round(times[first], 6), "end": round(times[last], 6)})
        print(json.dumps({"fixture": name, "language": language, "sha256": got_hash, "samples": len(audio), "ids": ids, "token_timestamps": [round(float(x), 6) for x in times], "words": spans, "text": processor.tokenizer.decode(ids, skip_special_tokens=True)}, ensure_ascii=False, separators=(",", ":")))

if __name__ == "__main__":
    main()
