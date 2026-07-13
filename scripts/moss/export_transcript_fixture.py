#!/usr/bin/env python3
"""Export a real-speech MOSS transcript fixture from pinned Transformers."""

import argparse
import hashlib
import json
import math
import re
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import Qwen3Config, WhisperConfig, WhisperFeatureExtractor
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from export_transformers_fixtures import load_encoder, run_adaptor  # noqa: E402

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as source:
        if source.getframerate() != 16000 or source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("fixture exporter requires mono 16-bit 16 kHz WAV")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def audio_span(audio_tokens: int, tokenizer: Tokenizer) -> list[int]:
    audio_id = 151671
    digits = [tokenizer.encode(str(i), add_special_tokens=False).ids[0] for i in range(10)]
    output, consumed = [], 0
    tokens_per_marker = int(12.5 * 5)
    duration = audio_tokens / 12.5
    for second in range(5, int(duration) + 1, 5):
        position = second // 5 * tokens_per_marker
        output.extend([audio_id] * (position - consumed))
        consumed = position
        output.extend(digits[int(ch)] for ch in str(second))
    output.extend([audio_id] * (audio_tokens - consumed))
    return output


def load_decoder(model_dir: Path, config: Qwen3Config, dtype: torch.dtype) -> Qwen3ForCausalLM:
    decoder = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    state = {}
    with safe_open(model_dir / "model-00000-of-00001.safetensors", framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("model.language_model."):
                state["model." + name.removeprefix("model.language_model.")] = source.get_tensor(name).to(dtype=dtype)
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    decoder.load_state_dict(state, strict=True)
    return decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    args = parser.parse_args()

    root = json.loads((args.model_dir / "config.json").read_text())
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    samples = read_wav(args.audio)
    padded = np.zeros(480000, dtype=np.float32)
    padded[: len(samples)] = samples
    extractor = WhisperFeatureExtractor(feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
    features = extractor._np_extract_fbank_features(np.array([padded]), "cpu")[0]
    audio_tokens = math.ceil(len(samples) / 1280)

    encoder = load_encoder(args.model_dir, WhisperConfig(**root["audio_config"]), dtype)
    with torch.inference_mode():
        encoded = encoder(torch.from_numpy(features).unsqueeze(0).to(dtype=dtype), return_dict=True).last_hidden_state
    _, adapted = run_adaptor(args.model_dir, encoded, dtype, audio_tokens=audio_tokens)
    del encoder, encoded

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n"
        + DEFAULT_PROMPT
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    base = tokenizer.encode(prompt, add_special_tokens=False).ids
    at = base.index(151671)
    input_ids = base[:at] + audio_span(audio_tokens, tokenizer) + base[at + 1 :]

    decoder = load_decoder(args.model_dir, Qwen3Config(**root["text_config"]), dtype)
    ids = torch.tensor([input_ids], dtype=torch.long)
    embeds = decoder.model.embed_tokens(ids).to(dtype=dtype)
    embeds[ids == 151671] = adapted.reshape(-1, adapted.shape[-1])
    with torch.inference_mode():
        generated = decoder.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=151645,
            pad_token_id=151643,
            use_cache=True,
        )[0].tolist()
    # inputs_embeds generation returns only continuation IDs in current Transformers.
    if generated[: len(input_ids)] == input_ids:
        generated = generated[len(input_ids) :]
    if 151645 in generated:
        generated = generated[: generated.index(151645)]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    segments = [
        {"start": float(start), "end": float(end), "speaker": speaker, "text": segment_text.strip()}
        for start, speaker, segment_text, end in re.findall(
            r"\[([0-9.]+)\]\[(S[0-9]+)\](.*?)\[([0-9.]+)\]", text
        )
    ]
    result = {
        "contract": "moss-real-transcript-v1",
        "transformers": "4.57.1",
        "checkpoint_revision": "0b0295acf3e6282a1692e1f6226faa32a453f7a2",
        "dtype": args.dtype,
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "audio_samples": len(samples),
        "audio_tokens": audio_tokens,
        "prompt_tokens": len(input_ids),
        "generated_ids": generated,
        "text": text,
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
