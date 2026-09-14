#!/usr/bin/env python3
"""Offline pinned WeSpeaker checkpoint conversion and bounded embedding oracle.

No download or serving. Outputs contain model weights/public audio tensors and
must remain in an external cache. The output directory must not exist.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave

REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
CHECKPOINT_SHA = "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929"
PCM_SHA = "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e"
TORCH_COMMIT = "08187d9e0fba026dc8217405802ab5381dc88d90"
HASHES = {
    "models/embedding/wespeaker/__init__.py": "a2c13a792c50d97f7a7583b69fcabd5eb5efc34d1374c2b6e5757bbcf3b62139",
    "models/embedding/wespeaker/resnet.py": "2de7673e14e8c74d6c430e0b6e6cc844157f47ac35b67ff6238ca2d2c559eba9",
    "models/blocks/pooling.py": "8cb687441630e6759fb6ca545d649b41dbec1d42f869954c0a03ae191c7cbd82",
    "core/task.py": "e2b0cc845a1234fe2f211ba2a0084c3e9be43fbdb82e41e6ca775f4746bbe444",
}
KALDI_SHA = "5cbea1a584ddea748f6f68a621d794e13334e88d9faa1d40986f7af32f196d29"
CONFIG = dict(BaseChannels=32, MelBins=80, EmbedDim=256)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"checksum mismatch: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--public-wav", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    checked(args.checkpoint, CHECKPOINT_SHA)
    checked(args.public_wav, PCM_SHA)
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("output directory already exists")
    import torch
    import torch.nn.functional as F
    import numpy as np
    import torchaudio.compliance.kaldi as kaldi
    import pyannote.audio
    from safetensors.torch import save_file
    base = Path(pyannote.audio.__file__).parent
    for name, digest in HASHES.items():
        checked(base / name, digest)
    checked(kaldi.__file__, KALDI_SHA)
    if torch.version.git_version != TORCH_COMMIT:
        raise ValueError("Torch reference revision changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
    expected = {"pyannote.audio.core.task." + n for n in ("Problem", "Resolution", "Specifications")}
    if set(torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)) != expected:
        raise ValueError("unexpected checkpoint globals")
    with torch.serialization.safe_globals([Problem, Resolution, Specifications]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"state_dict", "pyannote.audio", "pytorch-lightning_version"}:
        raise ValueError("unexpected checkpoint root contract")
    meta = checkpoint["pyannote.audio"]
    if meta["architecture"] != dict(module="pyannote.audio.models.embedding.wespeaker", **{"class": "WeSpeakerResNet34"}):
        raise ValueError("architecture changed")
    specification = meta["specifications"]
    if specification.problem != Problem.REPRESENTATION or specification.resolution != Resolution.CHUNK or specification.duration != 5.0:
        raise ValueError("embedding specification changed")
    # This checkpoint has no hyper_parameters. Pin the source's constructor
    # defaults explicitly; validate exact state_dict inventory below.
    model = WeSpeakerResNet34(sample_rate=16000, num_channels=1, num_mel_bins=80, frame_length=25,
                             frame_shift=10, dither=0., window_type="hamming", use_energy=False).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if len(model.state_dict()) != 218 or model.resnet.embed_dim != 256:
        raise ValueError("tensor count/dimension changed")
    with wave.open(args.public_wav, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) != (1, 2, 16000, 480000):
            raise ValueError("public PCM geometry changed")
        pcm = torch.from_numpy(np.frombuffer(w.readframes(480000), dtype="<i2").copy()).float() / 32768
    output.mkdir(parents=True)
    files = {}
    def save(name, tensors):
        values = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
        if any(v.dtype not in (torch.float32, torch.int64) for v in values.values()):
            raise ValueError("unexpected dtype")
        save_file(values, str(output / name), metadata={"provenance": json.dumps(dict(revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA), sort_keys=True, separators=(",", ":"))})
        files[name] = dict(sha256=sha(output / name), bytes=(output / name).stat().st_size,
                           tensors={k: dict(shape=list(v.shape), dtype="F32" if v.dtype == torch.float32 else "I64") for k, v in values.items()})
    cases = []
    with torch.no_grad():
        save("embedding.safetensors", model.state_dict())
        noise = (((torch.arange(16000, dtype=torch.int64) * 1103515245 + 12345) % 65536).float()/32768-1)*.2
        inputs = [("silence-1s", torch.zeros(16000)), ("broadband-1s", noise),
                  ("public-6-7s", pcm[96000:112000]), ("public-6-11s", pcm[96000:176000])]
        for name, x in inputs:
            trace = {"pcm": x}
            fbank = model.compute_fbank(x[None, None, :])
            trace["fbank"] = fbank[0]
            y = F.relu(model.resnet.bn1(model.resnet.conv1(fbank.permute(0, 2, 1).unsqueeze(1))))
            trace["trunk.-1.-1"] = y[0]
            for i, stage in enumerate([model.resnet.layer1, model.resnet.layer2, model.resnet.layer3, model.resnet.layer4]):
                for j, block in enumerate(stage):
                    y = block(y)
                    trace[f"trunk.{i}.{j}"] = y[0]
            if not torch.equal(y, model.forward_frames(x[None, None, :])):
                raise ValueError("manual trunk differs from wrapper")
            masks = torch.tensor([[1., .3, 0., .8, 1., .2, .7], [0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0.]])[None]
            trace["masks"] = masks[0]
            for label, weights in [("unweighted", None), ("masked", masks)]:
                stats = model.resnet.pool(y, weights)
                emb = model.forward_embedding(y, weights)
                if not torch.equal(emb, model.resnet.seg_1(stats)):
                    raise ValueError("pool/projection differs from embedding method")
                if label == "masked" and not torch.equal(emb, model(x[None, None, :], weights)):
                    raise ValueError("shared trunk differs from complete forward")
                trace[label + ".stats"] = stats.reshape(-1, 5120)
                trace[label + ".embedding"] = emb.reshape(-1, 256)
                aligned = torch.ones(1, 1, y.shape[-1]) if weights is None else F.interpolate(weights, size=y.shape[-1], mode="nearest")
                trace[label + ".weight_sum"] = aligned.sum(-1).flatten()
                trace[label + ".nonzero_frames"] = (aligned > 0).sum(-1).flatten()
            filename = name + ".safetensors"
            save(filename, trace)
            cases.append(dict(name=name, file=filename, samples=x.numel(), frames=fbank.shape[1], cnn_frames=y.shape[-1]))
    manifest = dict(schema=1, model_revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA, public_pcm_sha256=PCM_SHA,
                    config=CONFIG, prefix="resnet", source_hashes=HASHES, kaldi_sha256=KALDI_SHA, torch_commit=TORCH_COMMIT,
                    cpu_threads=1, mkldnn=False, frontend=dict(sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=10,
                    snip_edges=True, round_to_power_of_two=True, dither=0, window_type="hamming", use_energy=False, centering="whole-window"),
                    scope="Bounded offline trained embedding reference; no DER/global speaker or performance qualification", files=files, cases=cases)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(output=str(output), tensors=len(model.state_dict()), cases=cases)))


if __name__ == "__main__":
    main()
