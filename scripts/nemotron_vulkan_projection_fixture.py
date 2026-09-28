"""Generate temporary F32 power/filter inputs for the opt-in Vulkan mel-projection gate.

Use the pinned Transformers CPU environment from docs/validation/nemotron-speech-reference-2026-09-28.md.
The output files are independent model-processor inputs, not acceptance hashes.
"""
import argparse
import hashlib
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "testdata/jfk.wav"
SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(INPUT.read_bytes()).hexdigest() != SHA:
        raise ValueError("JFK input provenance changed")
    audio, rate = sf.read(INPUT, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected audio input")
    extractor = AutoProcessor.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True).feature_extractor
    waveform = torch.tensor(audio).unsqueeze(0)
    pre = torch.cat([waveform[:, :1], waveform[:, 1:] - extractor.preemphasis * waveform[:, :-1]], dim=1)
    window = torch.hann_window(extractor.win_length, periodic=False)
    spectrum = torch.stft(pre, extractor.n_fft, hop_length=extractor.hop_length,
                          win_length=extractor.win_length, window=window,
                          pad_mode="constant", center=True, return_complex=True)
    # Match the reference frontend's intermediate operations and F32 roundings.
    components = torch.view_as_real(spectrum)
    magnitudes = torch.sqrt(components.pow(2).sum(-1))
    power = magnitudes.pow(2).permute(0, 2, 1)[0].contiguous().numpy().astype("<f4", copy=False)
    filters = extractor.mel_filters.contiguous().numpy().astype("<f4", copy=False)
    if power.shape != (1101, 257) or filters.shape != (128, 257):
        raise ValueError(f"unexpected shapes {power.shape} {filters.shape}")
    (args.out / "power.f32").write_bytes(power.tobytes())
    (args.out / "filters.f32").write_bytes(filters.tobytes())
    print(f"wrote power {power.shape} and filters {filters.shape} to {args.out}")


if __name__ == "__main__":
    main()
