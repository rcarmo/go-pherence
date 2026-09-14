#!/usr/bin/env python3
"""Generate small pinned SincNet CPU fixtures, without trained weights/audio.

Loads only the checksum-pinned SincNet class, receptive-field functions and
asteroid filterbank. One CPU thread, MKLDNN disabled. No pipeline/network/GPU.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path

SINC_SHA = "3f151a2482c3f8c266b1efd9bdcab297238c52b4792a7265261f2da063e5dade"
RF_SHA = "021df3fc249a3c5146e15f879740a0ca02ad13fad6269191c8392a1ad61cd488"
FB_SHA = "2df0d1e6f109985c00efcc60970ebceff6a9665c3e65ebae73ea4848e48d8eae"
ENC_SHA = "4b385912861a60c6ebcadc170a3ef747637952e03ab6eab303c523788cb3beee"


def checked(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"source checksum mismatch: {path}")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sincnet", required=True)
    p.add_argument("--receptive-field", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trace-output", help="optional diagnostic raw convolution/pool boundaries")
    p.add_argument("--norm-output", help="optional isolated normalization inputs/outputs")
    args = p.parse_args()
    source = checked(args.sincnet, SINC_SHA)
    checked(args.receptive_field, RF_SHA)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import asteroid_filterbanks.param_sinc_fb as fb
    import asteroid_filterbanks.enc_dec as enc
    checked(fb.__file__, FB_SHA)
    checked(enc.__file__, ENC_SHA)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    spec = importlib.util.spec_from_file_location("reference_rf", args.receptive_field)
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    scope = dict(torch=torch, nn=nn, F=F, lru_cache=lru_cache,
                 Encoder=enc.Encoder, ParamSincFB=fb.ParamSincFB)
    for name in ["multi_conv_num_frames", "multi_conv_receptive_field_center", "multi_conv_receptive_field_size"]:
        scope[name] = getattr(rf, name)
    tree = ast.parse(source)
    only_class = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SincNet"], type_ignores=[])
    exec(compile(only_class, args.sincnet, "exec"), scope)
    model = scope["SincNet"](sample_rate=16000, stride=10).eval()
    with torch.no_grad():
        for index, (name, param) in enumerate(model.named_parameters()):
            v = torch.arange(param.numel(), dtype=torch.float32)
            if name.endswith("low_hz_"):
                values = 30 + v * 187
                values[2] = -values[2]
            elif name.endswith("band_hz_"):
                values = 65 + v * 4
                values[3] = -values[3]
                values[-1] = 1200  # exercise Nyquist clipping
            elif "norm" in name and name.endswith("weight"):
                values = 0.8 + torch.sin(v * 0.3) * 0.1
            elif "norm" in name:
                values = torch.sin(v * 0.2) * 0.03
            else:
                values = torch.sin(v * 0.17 + index) * 0.013
            param.copy_(values.reshape(param.shape))
        def norm(n):
            return {"Weight": n.weight.tolist(), "Bias": n.bias.tolist()}
        weights = {"WaveNorm": norm(model.wav_norm1d),
                   "LowHz": model.conv1d[0].filterbank.low_hz_.flatten().tolist(),
                   "BandHz": model.conv1d[0].filterbank.band_hz_.flatten().tolist(),
                   "Norm": [norm(n) for n in model.norm1d],
                   "Conv": [{"Weight": c.weight.flatten().tolist(), "Bias": c.bias.tolist()} for c in model.conv1d[1:]]}
        filters = model.conv1d[0].filterbank.filters().flatten().tolist()
        cases = []
        traces = []
        norms = []
        for stride, length, kind in [(10, 1531, "wave"), (10, 1540, "impulse"),
                                      (10, 2000, "silence"), (10, 2000, "constant"), (1, 600, "wave"),
                                      (10, 2000, "broadband"), (1, 600, "broadband")]:
            model.stride = stride
            model.conv1d[0].stride = stride
            x = torch.zeros(length, dtype=torch.float32)
            if kind == "wave":
                n = torch.arange(length, dtype=torch.float32)
                x = (torch.sin(n * 0.071) + torch.cos(n * 0.127)) * 0.13
            elif kind == "broadband":
                n = torch.arange(length, dtype=torch.int64)
                x = (((n * 1103515245 + 12345) % 65536).float() / 32768 - 1) * 0.2
            elif kind == "impulse":
                x[length // 2] = 0.7
            elif kind == "constant":
                x.fill_(0.2)
            y = model.wav_norm1d(x[None, None, :])
            if args.norm_output:
                norms.append({"stride": stride, "kind": kind, "stage": -1, "channels": 1, "frames": length,
                              "input": x.tolist(), "output": y.flatten().tolist(), "norm": norm(model.wav_norm1d)})
            boundaries = [{"stage": -1, "channels": 1, "frames": length, "values": y.flatten().tolist()}]
            stages = []
            for index, (conv, pool, normalization) in enumerate(zip(model.conv1d, model.pool1d, model.norm1d)):
                before = y
                y = conv(y)
                raw = y
                if index == 0:
                    y = y.abs()
                pooled = pool(y)
                if args.trace_output:
                    stages.append({"input": before.flatten().tolist(), "input_frames": before.shape[2],
                                   "convolution": raw.flatten().tolist(), "convolution_frames": raw.shape[2],
                                   "pooled": pooled.flatten().tolist(), "pooled_frames": pooled.shape[2]})
                normalized = normalization(pooled)
                if args.norm_output:
                    norms.append({"stride": stride, "kind": kind, "stage": index, "channels": pooled.shape[1], "frames": pooled.shape[2],
                                  "input": pooled.flatten().tolist(), "output": normalized.flatten().tolist(), "norm": norm(normalization)})
                y = F.leaky_relu(normalized)
                boundaries.append({"stage": index, "channels": y.shape[1], "frames": y.shape[2], "values": y.flatten().tolist()})
            actual = model(x[None, None, :])
            if not torch.equal(actual, y):
                raise RuntimeError("manual boundaries differ from SincNet.forward")
            kernels, steps = [251, 3, 5, 3, 5, 3], [stride, 3, 1, 3, 1, 3]
            grid = {"Frames": rf.multi_conv_num_frames(length, kernels, steps, [0] * 6, [1] * 6),
                    "Step": 27 * stride,
                    "FirstCenter": rf.multi_conv_receptive_field_center(0, kernels, steps, [0] * 6, [1] * 6),
                    "ReceptiveField": rf.multi_conv_receptive_field_size(1, kernels, steps, [0] * 6, [1] * 6)}
            if args.trace_output:
                traces.append({"stride": stride, "kind": kind, "stages": stages})
            cases.append({"stride": stride, "kind": kind, "input": x.tolist(), "grid": grid,
                          "boundaries": boundaries, "output": y.transpose(1, 2).flatten().tolist()})
    result = {"schema": 1, "reference": {"pyannote_commit": "b749285c5cdd4636b2edc7f766f1352c8dde9369",
              "sincnet_sha256": SINC_SHA, "receptive_field_sha256": RF_SHA,
              "filterbank_sha256": FB_SHA, "enc_dec_sha256": ENC_SHA,
              "asteroid_filterbanks": "0.4.0", "torch": torch.__version__,
              "torch_commit": torch.version.git_version, "cpu_threads": 1, "mkldnn": False},
              "scope": "synthetic frontend only, no trained checkpoint or quality/speed claim",
              "weights": weights, "filters": filters, "cases": cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, separators=(",", ":")) + "\n").encode()
    Path(args.output).write_bytes(gzip.compress(encoded, mtime=0))
    if args.trace_output:
        Path(args.trace_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace_output).write_bytes(gzip.compress((json.dumps({"schema": 1, "reference": result["reference"], "cases": traces}, separators=(",", ":")) + "\n").encode(), mtime=0))
    if args.norm_output:
        Path(args.norm_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.norm_output).write_bytes(gzip.compress((json.dumps({"schema": 1, "reference": result["reference"], "cases": norms}, separators=(",", ":")) + "\n").encode(), mtime=0))
    print(f"wrote {len(cases)} SincNet synthetic cases")


if __name__ == "__main__":
    main()
