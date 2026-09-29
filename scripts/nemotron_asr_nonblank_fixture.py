"""Pin a bounded nonblank RNNT decode from the released full JFK encoder.

Only the PyTorch encoder processes the full 11-second recording. The saved
five projected rows are the independent input for the native decoder test;
this script does not claim native long-window encoder parity or WER.
"""
import argparse
import gzip
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import Nemotron3_5AsrRNNTDecoderCache

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/asr"


def save(path, tensor):
    values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as compressed:
            compressed.write(values.tobytes())
    print(path.name, values.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    audio, rate = sf.read(ROOT / "testdata/jfk.wav", dtype="float32")
    if rate != 16000 or len(audio) != 176000:
        raise ValueError("unexpected JFK waveform")
    processed = processor(audio, sampling_rate=rate, return_tensors="pt")
    with torch.inference_mode():
        output = model.get_audio_features(
            input_features=processed.input_features,
            attention_mask=processed.attention_mask,
            num_lookahead_tokens=3,
            prompt_ids=torch.tensor([model.config.default_prompt_id]),
        )
        encoded = output.pooler_output
        if encoded.shape != (1, 139, 640):
            raise ValueError(f"unexpected projected shape {encoded.shape}")
        start, length = 10, 5
        window = encoded[:, start:start+length]
        save(args.out / "asr_jfk_nonblank_encoder.f32.gz", window[0])
        cache = Nemotron3_5AsrRNNTDecoderCache(model.config)
        frame, token, symbols = 0, model.config.blank_token_id, 0
        logits_by_step, tokens, frames = [], [], []
        while frame < length:
            predictor = model.decoder(torch.tensor([[token]]), cache=cache)
            logits = model.joint(window[:, frame:frame+1, None, :], predictor[:, None, :, :]).reshape(-1)
            logits_by_step.append(logits)
            frames.append(frame)
            token = int(logits.argmax())
            tokens.append(token)
            symbols = 0 if token == model.config.blank_token_id else symbols + 1
            if token == model.config.blank_token_id or symbols >= model.max_symbols_per_step:
                frame, symbols = frame+1, 0
            if len(tokens) > length*model.max_symbols_per_step:
                raise ValueError("unbounded fixture decode")
        if not any(token != model.config.blank_token_id for token in tokens):
            raise ValueError("slice had no nonblank token")
        save(args.out / "asr_jfk_nonblank_logits.f32.gz", torch.stack(logits_by_step))
        result = {"encoder_start": start, "encoder_rows": length, "lookahead": 3,
                  "prompt_id": model.config.default_prompt_id, "tokens": tokens, "frames": frames}
        (args.out / "asr_jfk_nonblank.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result)


if __name__ == "__main__":
    main()
