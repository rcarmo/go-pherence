#!/usr/bin/env python3
"""Full-depth synthetic WeSpeaker ResNet34 graph oracle with reduced channel widths.

No trained checkpoint/audio/GPU. CPU one thread, MKLDNN off. Extract only
BasicBlock/TSTP/ResNet from pinned Apache2 source plus pinned MIT StatsPool.
"""
import argparse
import ast
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Optional

RESNET_SHA = "2de7673e14e8c74d6c430e0b6e6cc844157f47ac35b67ff6238ca2d2c559eba9"
POOL_SHA = "8cb687441630e6759fb6ca545d649b41dbec1d42f869954c0a03ae191c7cbd82"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet", required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path, expected in [(args.resnet, RESNET_SHA), (args.pooling, POOL_SHA)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise SystemExit("source checksum mismatch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    spec = importlib.util.spec_from_file_location("pinned_pool", args.pooling)
    pooling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pooling)
    scope = dict(torch=torch, nn=nn, F=F, lru_cache=lru_cache, Optional=Optional,
                 rearrange=rearrange, StatsPool=pooling.StatsPool)
    tree = ast.parse(Path(args.resnet).read_text())
    for name in ["BasicBlock", "TSTP", "ResNet"]:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), args.resnet, "exec"), scope)
        if name == "TSTP":
            scope["POOLING_LAYERS"] = {"TSTP": scope[name]}
    def bn(module):
        return {"Weight": module.weight.tolist(), "Bias": module.bias.tolist(),
                "RunningMean": module.running_mean.tolist(), "RunningVariance": module.running_var.tolist()}
    def block_weights(block):
        w = {"Conv1": block.conv1.weight.flatten().tolist(), "Conv2": block.conv2.weight.flatten().tolist(),
             "BN1": bn(block.bn1), "BN2": bn(block.bn2)}
        if len(block.shortcut):
            w["Shortcut"] = block.shortcut[0].weight.flatten().tolist()
            w["ShortcutBN"] = bn(block.shortcut[1])
        return w
    cases = []
    with torch.no_grad():
        for base, mel, embed, frames in [(1, 8, 5, 17), (2, 16, 7, 19), (1, 80, 9, 9)]:
            model = scope["ResNet"](scope["BasicBlock"], [3, 4, 6, 3], m_channels=base,
                                    feat_dim=mel, embed_dim=embed, pooling_func="TSTP", two_emb_layer=False).eval()
            for index, (name, param) in enumerate(model.named_parameters()):
                n = torch.arange(param.numel(), dtype=torch.float32)
                values = torch.sin(n * .31 + index * .17) * .08
                if name.endswith("weight") and param.ndim == 1:
                    values = .8 + torch.cos(n * .2 + index) * .13
                param.copy_(values.reshape(param.shape))
            for index, (name, buf) in enumerate(model.named_buffers()):
                n = torch.arange(buf.numel(), dtype=torch.float32)
                if name.endswith("running_mean"):
                    buf.copy_((torch.sin(n * .4 + index) * .07).reshape(buf.shape))
                elif name.endswith("running_var"):
                    buf.copy_((.7 + (n % 5) * .09).reshape(buf.shape))
            stages = [model.layer1, model.layer2, model.layer3, model.layer4]
            weights = {"Stem": model.conv1.weight.flatten().tolist(), "StemBN": bn(model.bn1),
                       "Stages": [[block_weights(b) for b in stage] for stage in stages],
                       "Projection": {"Weight": model.seg_1.weight.flatten().tolist(), "Bias": model.seg_1.bias.tolist()}}
            x = (torch.sin(torch.arange(frames * mel, dtype=torch.float32) * .13) * .27).reshape(1, frames, mel)
            boundaries = []
            def save(stage, block, value):
                boundaries.append({"stage": stage, "block": block,
                                   "shape": {"Channels": value.shape[1], "Frequency": value.shape[2], "Frames": value.shape[3]},
                                   "values": value.flatten().tolist()})
            y = F.relu(model.bn1(model.conv1(x.permute(0, 2, 1).unsqueeze(1))))
            save(-1, -1, y)
            for i, stage in enumerate(stages):
                for j, block in enumerate(stage):
                    y = block(y)
                    save(i, j, y)
            if not torch.equal(y, model.forward_frames(x)):
                raise RuntimeError("trunk composition mismatch")
            mask_cases = []
            for masks in [None, [[1., .3, 0., .8, 1.], [0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.]]]:
                mask = None if masks is None else torch.tensor(masks)[None]
                stats = model.pool(y, mask)
                embeddings = model.forward_embedding(y, mask)[1]
                if not torch.equal(embeddings, model(x, mask)[1]):
                    raise RuntimeError("combined forward differs from shared trunk reuse")
                aligned = torch.ones(1, 1, y.shape[-1]) if mask is None else F.interpolate(mask, size=y.shape[-1], mode="nearest")
                mask_cases.append({"masks": None if mask is None else mask.flatten().tolist(),
                                   "speakers": 0 if masks is None else len(masks), "mask_frames": 0 if masks is None else len(masks[0]),
                                   "statistics": stats.flatten().tolist(), "embeddings": embeddings.flatten().tolist(),
                                   "weight_sum": aligned.sum(-1).flatten().tolist(), "nonzero_frames": (aligned > 0).sum(-1).flatten().tolist()})
            cases.append({"config": {"BaseChannels": base, "MelBins": mel, "EmbedDim": embed}, "frames": frames,
                          "weights": weights, "input": x.flatten().tolist(), "boundaries": boundaries, "mask_cases": mask_cases})
    result = {"schema": 1, "absolute_tolerance": 2e-6, "relative_tolerance": 2e-6,
              "reference": {"pyannote_commit": "b749285c5cdd4636b2edc7f766f1352c8dde9369", "resnet_sha256": RESNET_SHA,
                            "pooling_sha256": POOL_SHA, "torch": torch.__version__, "torch_commit": torch.version.git_version,
                            "cpu_threads": 1, "mkldnn": False},
              "scope": "full [3,4,6,3] topology with reduced-width synthetic weights, not trained model or production-shape performance",
              "cases": cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_bytes(gzip.compress((json.dumps(result, separators=(",", ":")) + "\n").encode(), mtime=0))
    print(f"wrote {len(cases)} full-depth synthetic ResNet34 cases")


if __name__ == "__main__":
    main()
