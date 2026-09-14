#!/usr/bin/env python3
"""Generate pinned WeSpeaker BasicBlock CPU fixtures, no trained model or audio.

Only the BasicBlock class is extracted from the Apache-2.0 reference. CPU one
thread, MKLDNN disabled, synthetic parameters and running statistics. No pipeline.
"""
import argparse
import ast
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

SHA = "2de7673e14e8c74d6c430e0b6e6cc844157f47ac35b67ff6238ca2d2c559eba9"
REV = "b749285c5cdd4636b2edc7f766f1352c8dde9369"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.reference).read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA:
        raise SystemExit("WeSpeaker ResNet source checksum mismatch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "BasicBlock")
    scope = dict(torch=torch, nn=nn, F=F, lru_cache=lru_cache)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), args.reference, "exec"), scope)
    cases = []
    geometries = [(2, 2, 1, 5, 7), (3, 5, 2, 7, 9), (3, 4, 1, 4, 6),
                  (1, 2, 2, 1, 1), (3, 3, 1, 2, 3), (8, 16, 2, 10, 11)]
    with torch.no_grad():
        for index, (ins, outs, stride, freq, frames) in enumerate(geometries):
            block = scope["BasicBlock"](ins, outs, stride).eval()
            for pindex, (name, param) in enumerate(block.named_parameters()):
                n = torch.arange(param.numel(), dtype=torch.float32)
                values = torch.sin(n*.17+pindex*.3)*.055
                if param.ndim == 1 and name.endswith("weight"):
                    values = torch.cos(n*.29+pindex*.4)*.7
                param.copy_(values.reshape(param.shape))
            for bindex, (name, buffer) in enumerate(block.named_buffers()):
                n = torch.arange(buffer.numel(), dtype=torch.float32)
                if name.endswith("running_mean"):
                    buffer.copy_((torch.sin(n*.21+bindex)*.19).reshape(buffer.shape))
                elif name.endswith("running_var"):
                    buffer.copy_((.2 + (n % 7)*.13).reshape(buffer.shape))
            def bn(x):
                return {"Weight": x.weight.tolist(), "Bias": x.bias.tolist(),
                        "RunningMean": x.running_mean.tolist(), "RunningVariance": x.running_var.tolist()}
            weights = {"Conv1": block.conv1.weight.flatten().tolist(), "Conv2": block.conv2.weight.flatten().tolist(),
                       "BN1": bn(block.bn1), "BN2": bn(block.bn2)}
            if len(block.shortcut):
                weights["Shortcut"] = block.shortcut[0].weight.flatten().tolist()
                weights["ShortcutBN"] = bn(block.shortcut[1])
            x = (torch.sin(torch.arange(ins*freq*frames, dtype=torch.float32)*.37 + index)*.3).reshape(1, ins, freq, frames)
            boundaries = []
            def save(name, value):
                boundaries.append({"stage": name, "shape": {"Channels": value.shape[1], "Frequency": value.shape[2], "Frames": value.shape[3]}, "values": value.flatten().tolist()})
                return value
            y = save("conv1", block.conv1(x))
            y = save("bn1", block.bn1(y))
            y = save("relu1", F.relu(y))
            y = save("conv2", block.conv2(y))
            y = save("bn2", block.bn2(y))
            shortcut = x
            if len(block.shortcut):
                shortcut = save("shortcut_conv", block.shortcut[0](x))
                shortcut = block.shortcut[1](shortcut)
            save("shortcut", shortcut)
            y = save("residual", y + shortcut)
            y = save("output", F.relu(y))
            if not torch.equal(block(x), y):
                raise RuntimeError("boundary composition differs from BasicBlock.forward")
            cases.append({"config": {"InChannels": ins, "OutChannels": outs, "Stride": stride},
                          "shape": {"Channels": ins, "Frequency": freq, "Frames": frames}, "weights": weights,
                          "input": x.flatten().tolist(), "output": y.flatten().tolist(), "boundaries": boundaries})
    result = {"schema": 1, "absolute_tolerance": 2e-6, "relative_tolerance": 2e-6,
              "reference": {"url": f"https://raw.githubusercontent.com/pyannote/pyannote-audio/{REV}/src/pyannote/audio/models/embedding/wespeaker/resnet.py",
                            "sha256": SHA, "license": "Apache-2.0", "torch": torch.__version__,
                            "torch_commit": torch.version.git_version, "cpu_threads": 1, "mkldnn": False},
              "scope": "synthetic single-block intermediate parity, not full ResNet or trained-model quality/speed",
              "cases": cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, separators=(",", ":")) + "\n").encode()
    Path(args.output).write_bytes(gzip.compress(encoded, mtime=0))
    print(f"wrote {len(cases)} WeSpeaker BasicBlock cases")


if __name__ == "__main__":
    main()
