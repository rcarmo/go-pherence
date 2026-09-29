"""Independent bounded Nemotron 3.5 ASR prompt/projector/joint operator fixtures.

Inputs are the five-row first-block reference and a released token embedding;
these are not outputs of the full 24-layer encoder or a decoded hypothesis.
Requires the pinned checkpoint and CPU Transformers environment recorded in
../docs/validation/nemotron-speech-reference-2026-09-28.md.
"""
import argparse
import gzip
from pathlib import Path

import torch
from transformers import AutoModelForRNNT

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/asr"
BLOCK = ROOT / "model/nemotronasr/testdata/encoder0_block_output.f32.gz"


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
    with gzip.open(BLOCK, "rb") as f:
        import numpy as np
        block = torch.from_numpy(np.frombuffer(f.read(), dtype="<f4").copy().reshape(1, 5, 1024))
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    if model.config.num_prompts != 128 or model.config.default_prompt_id != 101 or model.config.hidden_act != "relu":
        raise ValueError("unexpected released ASR prompt/joint configuration")
    with torch.inference_mode():
        decoder = model.decoder.embedding(torch.tensor([3]))  # fixed released token vector, not an LSTM prediction
        save(args.out / "rnnt_joint_decoder_input.f32.gz", decoder)
        for prompt in (101, 7):
            one_hot = torch.nn.functional.one_hot(torch.tensor([prompt]), num_classes=128).float()
            fused = torch.cat([block, one_hot[:, None, :].expand(-1, 5, -1)], dim=-1)
            hidden = model.prompt_projector.linear_1(fused)
            save(args.out / f"rnnt_prompt{prompt}_linear1.f32.gz", hidden[0])
            prompt_out = model.prompt_projector.linear_2(torch.relu(hidden))
            save(args.out / f"rnnt_prompt{prompt}_output.f32.gz", prompt_out[0])
            encoded = model.encoder_projector(prompt_out)
            save(args.out / f"rnnt_prompt{prompt}_encoder.f32.gz", encoded[0])
            logits = model.joint(encoded, decoder[:, None, :])
            save(args.out / f"rnnt_prompt{prompt}_joint_logits.f32.gz", logits[0])


if __name__ == "__main__":
    main()
