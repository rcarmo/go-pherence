#!/usr/bin/env python3
"""Generate pinned Kaldi/pyannote WeSpeaker frontend fixtures with synthetic PCM.

CPU only, one thread, no checkpoint/private audio/pipeline. Imports only pinned
kaldi module plus an AST-extracted WeSpeaker compute_fbank method. No network.
"""
import argparse
import ast
from functools import partial
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

KALDI_SHA = "5cbea1a584ddea748f6f68a621d794e13334e88d9faa1d40986f7af32f196d29"
WESPEAKER_SHA = "a2c13a792c50d97f7a7583b69fcabd5eb5efc34d1374c2b6e5757bbcf3b62139"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaldi", required=True)
    parser.add_argument("--wespeaker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path, want in [(args.kaldi, KALDI_SHA), (args.wespeaker, WESPEAKER_SHA)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != want:
            raise SystemExit("reference checksum mismatch")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    spec = importlib.util.spec_from_file_location("pinned_kaldi", args.kaldi)
    kaldi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaldi)
    tree = ast.parse(Path(args.wespeaker).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseWeSpeakerResNet")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_fbank")
    method.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), args.wespeaker, "exec"), namespace)
    fbank = partial(kaldi.fbank, num_mel_bins=80, frame_length=25, round_to_power_of_two=True,
                    frame_shift=10, snip_edges=True, dither=0.0, sample_frequency=16000,
                    window_type="hamming", use_energy=False)
    instance = SimpleNamespace(_fbank=fbank, hparams=SimpleNamespace(fbank_centering_span=None))
    mel, _ = kaldi.get_mel_banks(80, 512, 16000, 20, 0, 100, -500, 1)
    mel = torch.nn.functional.pad(mel, (0, 1))
    hamming = kaldi._feature_window_function("hamming", 400, .42, torch.device("cpu"), torch.float32)
    cases = []
    with torch.no_grad():
        for kind, length in [("silence", 720), ("constant", 1000), ("impulse", 1001),
                             ("broadband", 2000), ("tone", 1600), ("odd-tail", 559),
                             ("single-frame", 400), ("very-low", 1200)]:
            n = torch.arange(length, dtype=torch.int64)
            x = (((n * 1103515245 + 12345) % 65536).float() / 32768 - 1) * .17
            if kind == "silence": x.zero_()
            elif kind == "constant": x.fill_(.2)
            elif kind == "impulse":
                x.zero_(); x[399] = .75
            elif kind == "tone": x = torch.sin(n.float() * (2 * torch.pi * 440 / 16000)) * .13
            elif kind == "very-low": x *= 1e-8
            waveform = x * 32768
            windows, _ = kaldi._get_window(waveform, 512, 400, 160, "hamming", .42, True, True, 1.0, 0.0, True, .97)
            power = torch.fft.rfft(windows).abs().pow(2)
            logmel = fbank(waveform[None, :])
            expected_logmel = torch.maximum(power @ mel.T, torch.tensor(torch.finfo(torch.float32).eps)).log()
            if not torch.equal(logmel, expected_logmel):
                raise RuntimeError("manual intermediates mismatch")
            # Execute the actual extracted pyannote method, including torch.vmap.
            output = namespace["compute_fbank"](instance, x[None, None, :])
            centered = logmel - logmel.mean(dim=0, keepdim=True)
            if not torch.allclose(output[0], centered, atol=1e-6, rtol=1e-6):
                raise RuntimeError("vmap WeSpeaker centering mismatch")
            means = waveform.unfold(0,400,160).mean(dim=1)
            cases.append({"name": kind, "input": x.tolist(), "means": means.tolist(), "frames": windows.shape[0],
                          "window": windows.flatten().tolist(), "power": power.flatten().tolist(),
                          "log_mel": logmel.flatten().tolist(), "output": output.flatten().tolist()})
    output = {"schema": 1, "reference": {"kaldi_sha256": KALDI_SHA, "wespeaker_sha256": WESPEAKER_SHA,
              "pyannote_commit": "b749285c5cdd4636b2edc7f766f1352c8dde9369", "torchaudio": kaldi.torchaudio.__version__,
              "torch": torch.__version__, "torch_commit": torch.version.git_version, "cpu_threads": 1,
              "sum_kernel_url": "https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/SumKernel.cpp",
              "sum_kernel_sha256": "9b88aa14e626dd708e50816a5ca89acfc66798226dc82501cf1857caebe96f95",
              "cpu_capability": torch.backends.cpu.get_cpu_capability()},
              "scope": "fixed default Fbank contract, synthetic input, no trained-model quality or speed claim",
              "hamming": hamming.tolist(), "mel": mel.flatten().tolist(), "cases": cases}
    encoded = (json.dumps(output, separators=(",", ":")) + "\n").encode()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_bytes(gzip.compress(encoded, mtime=0))
    print(f"wrote {len(cases)} WeSpeaker Fbank cases")


if __name__ == "__main__":
    main()
