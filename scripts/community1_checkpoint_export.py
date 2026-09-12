#!/usr/bin/env python3
"""Lower the pinned local Community-1 segmentation checkpoint for Go qualification.

Offline conversion/reference only. No network, serving, or runtime fallback.
Requires an explicit output directory that must not already exist. Model weights,
public audio tensors and traces belong in a local cache, never in source control.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.util
from itertools import pairwise
import json
from pathlib import Path
from types import SimpleNamespace
import wave

REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
CHECKPOINT_SHA = "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e"
PCM_SHA = "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e"
TASK_SHA = "e2b0cc845a1234fe2f211ba2a0084c3e9be43fbdb82e41e6ca775f4746bbe444"
HASHES = {
    "pyannet": "3ceebc8c00e83d96747a706212102e7e99c44732ff1fc769e241e8de47d3d6af",
    "sincnet": "3f151a2482c3f8c266b1efd9bdcab297238c52b4792a7265261f2da063e5dade",
    "receptive_field": "021df3fc249a3c5146e15f879740a0ca02ad13fad6269191c8392a1ad61cd488",
    "filterbank": "2df0d1e6f109985c00efcc60970ebceff6a9665c3e65ebae73ea4848e48d8eae",
    "encoder": "4b385912861a60c6ebcadc170a3ef747637952e03ab6eab303c523788cb3beee",
}
TORCH_COMMIT = "08187d9e0fba026dc8217405802ab5381dc88d90"
CONFIG = dict(SincNetStride=10, LSTM=dict(InputSize=60, HiddenSize=128, NumLayers=4, Bidirectional=True),
              Head=dict(InputSize=256, HiddenSize=128, NumLayers=2, Speakers=3, MaxActive=2), SplitLSTM=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return Path(path)


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
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from safetensors.torch import save_file
    import asteroid_filterbanks.param_sinc_fb as fb
    import asteroid_filterbanks.enc_dec as enc
    from einops import rearrange
    spec = importlib.util.find_spec("pyannote.audio")
    base = Path(spec.origin).parent
    paths = dict(pyannet=base / "models/segmentation/PyanNet.py", sincnet=base / "models/blocks/sincnet.py",
                 receptive_field=base / "utils/receptive_field.py", filterbank=Path(fb.__file__), encoder=Path(enc.__file__))
    for key, path in paths.items():
        checked(path, HASHES[key])
    checked(base / "core/task.py", TASK_SHA)
    if torch.version.git_version != TORCH_COMMIT:
        raise ValueError("reference torch revision changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    # Only three checksum-pinned metadata types are allowlisted. Never use
    # weights_only=False; no arbitrary checkpoint global is accepted.
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    expected_globals = {"pyannote.audio.core.task." + n for n in ("Problem", "Resolution", "Specifications")}
    if set(torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)) != expected_globals:
        raise ValueError("unexpected checkpoint globals")
    with torch.serialization.safe_globals([Problem, Resolution, Specifications]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"pytorch-lightning_version", "state_dict", "hparams_name", "hyper_parameters", "pyannote.audio"}:
        raise ValueError("unexpected checkpoint root inventory")
    architecture = checkpoint["pyannote.audio"]["architecture"]
    if architecture != dict(module="pyannote.audio.models.segmentation.PyanNet", **{"class": "PyanNet"}):
        raise ValueError("checkpoint architecture changed")
    hp = checkpoint["hyper_parameters"]
    expected_hp = dict(sample_rate=16000, num_channels=1, sincnet=dict(stride=10, sample_rate=16000),
                       lstm=dict(hidden_size=128, num_layers=4, bidirectional=True, monolithic=True, dropout=0.5),
                       linear=dict(hidden_size=128, num_layers=2))
    # PyanNet writes batch_first during construction; checkpoints may retain it.
    clean_hp = dict(hp, lstm={k: v for k, v in hp["lstm"].items() if k != "batch_first"})
    if clean_hp != expected_hp or hp["lstm"].get("batch_first", True) is not True:
        raise ValueError(f"checkpoint hyperparameter contract changed: {hp}")
    specification = checkpoint["pyannote.audio"]["specifications"]
    if (specification.problem != Problem.MONO_LABEL_CLASSIFICATION or specification.resolution != Resolution.FRAME
            or specification.duration != 10.0 or len(specification.classes) != 3 or specification.powerset_max_classes != 2
            or not specification.permutation_invariant):
        raise ValueError("checkpoint powerset/window contract changed")

    class Model(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def save_hyperparameters(self, *names):
            import inspect
            local = inspect.currentframe().f_back.f_locals
            self.hparams = SimpleNamespace(**{name: local[name] for name in names})
        def default_activation(self):
            return nn.LogSoftmax(dim=-1)

    scope = dict(torch=torch, nn=nn, F=F, Model=Model, lru_cache=lru_cache, pairwise=pairwise, rearrange=rearrange,
                 merge_dict=lambda default, override: dict(default, **(override or {})), Encoder=enc.Encoder, ParamSincFB=fb.ParamSincFB)
    spec = importlib.util.spec_from_file_location("reference_rf", paths["receptive_field"])
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    for name in ("multi_conv_num_frames", "multi_conv_receptive_field_center", "multi_conv_receptive_field_size"):
        scope[name] = getattr(rf, name)
    for key, name in (("sincnet", "SincNet"), ("pyannet", "PyanNet")):
        cls = next(n for n in ast.parse(paths[key].read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
        mod = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, str(paths[key]), "exec"), scope)
    model = scope["PyanNet"](**hp)
    model.specifications = specification
    model.build()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with wave.open(args.public_wav, "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, 480000):
            raise ValueError("public PCM contract changed")
        pcm = torch.from_numpy(np.frombuffer(wav.readframes(480000), dtype="<i2").copy()).float() / 32768
    output.mkdir(parents=True)
    files = {}
    def save(name, tensors):
        # Clone to avoid alias constraints and make layout explicit.
        values = {k: v.detach().cpu().float().contiguous().clone() for k, v in tensors.items()}
        save_file(values, str(output / name), metadata={"provenance": json.dumps(dict(model_revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA), sort_keys=True, separators=(",", ":"))})
        files[name] = dict(sha256=sha(output / name), bytes=(output / name).stat().st_size,
                           tensors={k: dict(shape=list(v.shape), dtype="F32") for k, v in values.items()})
    cases = []
    with torch.no_grad():
        save("segmentation.safetensors", model.state_dict())
        save("filters.safetensors", {"sincnet.filters": model.sincnet.conv1d[0].filterbank.filters().reshape(80, 251)})
        inputs = [("silence-1s", torch.zeros(16000)), ("wave-short", torch.sin(torch.arange(1600).float() * .071) * .13),
                  ("public-0-10s", pcm[:160000]), ("public-10-20s", pcm[160000:320000])]
        for name, x in inputs:
            trace = {"pcm": x}
            y = model.sincnet.wav_norm1d(x[None, None, :])
            trace["sincnet.-1"] = y[0]
            for i, (conv, pool, norm) in enumerate(zip(model.sincnet.conv1d, model.sincnet.pool1d, model.sincnet.norm1d)):
                trace[f"conv_input.{i}"] = y[0]
                y = conv(y)
                trace[f"conv.{i}"] = y[0]
                if i == 0:
                    y = y.abs()
                pooled = pool(y)
                trace[f"pool.{i}"] = pooled[0]
                y = F.leaky_relu(norm(pooled))
                trace[f"sincnet.{i}"] = y[0]
            features = y.transpose(1, 2).contiguous()
            sequence = features
            for i in range(4):
                layer = nn.LSTM(sequence.shape[-1], 128, num_layers=1, bidirectional=True, batch_first=True).eval()
                layer.load_state_dict({k.replace(f"_l{i}", "_l0"): v for k, v in model.lstm.state_dict().items() if f"_l{i}" in k})
                sequence, _ = layer(sequence)
                trace[f"lstm.{i}"] = sequence[0]
            recurrent, _ = model.lstm(features)
            if not torch.equal(sequence, recurrent):
                raise ValueError("split reference recurrent boundaries differ from monolithic model")
            for i, linear in enumerate(model.linear):
                sequence = F.leaky_relu(linear(sequence))
                trace[f"head.{i}"] = sequence[0]
            logits = model.classifier(sequence)
            trace["head.2"] = logits[0]
            trace["log_probabilities"] = model.activation(logits)[0]
            actual = model(x[None, None, :])[0]
            if not torch.equal(actual, trace["log_probabilities"]):
                raise ValueError("reference manual boundaries differ from full PyanNet")
            fname = name + ".safetensors"
            save(fname, trace)
            cases.append(dict(name=name, file=fname, samples=x.numel(), frames=actual.shape[0],
                              source="synthetic" if not name.startswith("public") else "pyannote tutorial public PCM"))
    manifest = dict(schema=1, model_revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA, public_pcm_sha256=PCM_SHA,
                    source_hashes=HASHES, task_sha256=TASK_SHA, torch_commit=TORCH_COMMIT, cpu_threads=1, mkldnn=False,
                    config=CONFIG, sample_rate=16000, channels=1, window_samples=160000,
                    scope="Offline conversion and bounded segmentation oracle; not full diarization or quality qualification",
                    files=files, cases=cases)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), model_tensors=len(files["segmentation.safetensors"]["tensors"]), cases=cases)))


if __name__ == "__main__":
    main()
