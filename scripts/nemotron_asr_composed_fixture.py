"""Pin a 32-frame composed ASR encoder and prompt projection on released weights.

The input features come from the independently saved JFK processor fixture.
This script does not invoke the RNN-T token decoder or produce transcription.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForRNNT
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import Nemotron3_5AsrRNNTDecoderCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import chunked_limited_mask_function

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/asr"
FEATURES = ROOT / "model/nemotronasr/testdata/features.f32.gz"


def save(path, tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    print(path.name, array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(FEATURES, "rb") as f:
        features = torch.from_numpy(np.frombuffer(f.read(), dtype="<f4").copy().reshape(1, 32, 128))
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    with torch.inference_mode():
        for lookahead in (0, 3):
            projected = model.encoder.subsampling(features, torch.ones(1, 32, dtype=torch.bool)) * model.encoder.input_scale
            left_ctx, right_ctx = model.encoder._resolve_attn_context(lookahead)
            mask = create_bidirectional_mask(
                config=model.encoder.config,
                inputs_embeds=projected,
                attention_mask=torch.ones(1, projected.shape[1], dtype=torch.bool),
                and_mask_function=chunked_limited_mask_function(left_ctx, right_ctx),
            )
            positions = model.encoder.encode_positions(projected)
            hidden = projected
            for layer in model.encoder.layers:
                hidden = layer(hidden, attention_mask=mask, position_embeddings=positions)
            save(args.out / f"asr_composed_tower_look{lookahead}.f32.gz", hidden[0])
            prompt = torch.nn.functional.one_hot(torch.tensor([model.config.default_prompt_id]), num_classes=model.config.num_prompts).to(hidden.dtype)
            fused = model.prompt_projector(torch.cat([hidden, prompt[:, None, :].expand(-1, hidden.shape[1], -1)], dim=-1))
            encoded = model.encoder_projector(fused)
            save(args.out / f"asr_composed_prompt_look{lookahead}.f32.gz", fused[0])
            save(args.out / f"asr_composed_encoder_look{lookahead}.f32.gz", encoded[0])
            decoder_cache = Nemotron3_5AsrRNNTDecoderCache(model.config)
            frame, token, symbols = 0, model.config.blank_token_id, 0
            step_logits, tokens, frames = [], [], []
            while frame < encoded.shape[1]:
                decoder_state = model.decoder(torch.tensor([[token]]), cache=decoder_cache)
                logits = model.joint(encoded[:, frame:frame+1, None, :], decoder_state[:, None, :, :]).reshape(-1)
                step_logits.append(logits)
                frames.append(frame)
                token = int(logits.argmax())
                tokens.append(token)
                symbols = 0 if token == model.config.blank_token_id else symbols + 1
                if token == model.config.blank_token_id or symbols >= model.max_symbols_per_step:
                    frame, symbols = frame + 1, 0
                if len(tokens) > encoded.shape[1] * model.max_symbols_per_step:
                    raise ValueError("unbounded fixture decode")
            save(args.out / f"asr_composed_greedy_logits_look{lookahead}.f32.gz", torch.stack(step_logits))
            print("greedy", lookahead, "frames", frames, "tokens", tokens)


if __name__ == "__main__":
    main()
