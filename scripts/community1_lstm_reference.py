#!/usr/bin/env python3
"""Generate tiny PyTorch LSTM sequence/state/layer fixtures for Community-1.

Offline developer oracle, no audio or checkpoint load. CPU only, one thread,
MKLDNN disabled. PyanNet and installed torch rnn.py checksums are verified before
execution. The Go inference package has no Python or PyTorch dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path

PYANNET_SHA = "3ceebc8c00e83d96747a706212102e7e99c44732ff1fc769e241e8de47d3d6af"
RNN_SHA = "902deeba8fd7d3c92b00e3557071b3bb41cf747bd4b2b9d63757d88562688bde"
REV = "b749285c5cdd4636b2edc7f766f1352c8dde9369"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyannet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if hashlib.sha256(Path(args.pyannet).read_bytes()).hexdigest() != PYANNET_SHA:
        raise SystemExit("PyanNet source checksum mismatch")
    import torch
    import torch.nn.modules.rnn as rnn
    if hashlib.sha256(Path(rnn.__file__).read_bytes()).hexdigest() != RNN_SHA:
        raise SystemExit("PyTorch RNN source checksum mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    cases = []
    # Unidirectional, reverse output, stacking, odd SIMD tails, and long state
    # propagation; these are tiny synthetic geometries, not trained models.
    geometries = [(1, 1, 1, False, 1, False), (3, 2, 1, True, 5, True),
                  (5, 3, 2, True, 7, False), (7, 5, 3, True, 9, True),
                  (9, 7, 2, False, 17, True), (3, 4, 2, True, 65, True)]
    for case_index, (width, hidden, layers, bidirectional, frames, initial) in enumerate(geometries):
        dirs = 2 if bidirectional else 1
        model = torch.nn.LSTM(width, hidden, layers, batch_first=True,
                              bidirectional=bidirectional, dropout=0).eval()
        with torch.no_grad():
            for index, (_, param) in enumerate(model.named_parameters()):
                v = torch.arange(param.numel(), dtype=torch.float32)
                values = torch.sin(v * 0.37 + (index + 1) * 0.61) * 0.23
                param.copy_(values.reshape(param.shape))
            x = (torch.sin(torch.arange(frames * width, dtype=torch.float32) * 0.19 + case_index) * 0.31).reshape(1, frames, width)
            size = layers * dirs * hidden
            h = (torch.cos(torch.arange(size, dtype=torch.float32) * 0.17) * 0.13).reshape(layers * dirs, 1, hidden)
            c = (torch.sin(torch.arange(size, dtype=torch.float32) * 0.23) * 0.29).reshape(layers * dirs, 1, hidden)
            if not initial:
                h.zero_()
                c.zero_()
            output, (hn, cn) = model(x, (h, c))
            parameters, intermediates = [], []
            sequence = x
            for layer in range(layers):
                single = torch.nn.LSTM(width if layer == 0 else dirs * hidden, hidden, 1,
                                       batch_first=True, bidirectional=bidirectional).eval()
                item = {}
                for direction in range(dirs):
                    suffix = "_reverse" if direction else ""
                    values = {}
                    for name, field in [("weight_ih", "WeightIH"), ("weight_hh", "WeightHH"),
                                        ("bias_ih", "BiasIH"), ("bias_hh", "BiasHH")]:
                        param = getattr(model, f"{name}_l{layer}{suffix}")
                        getattr(single, f"{name}_l0{suffix}").copy_(param)
                        values[field] = param.flatten().tolist()
                    item["Reverse" if direction else "Forward"] = values
                parameters.append(item)
                sequence, _ = single(sequence, (h[layer * dirs:(layer + 1) * dirs], c[layer * dirs:(layer + 1) * dirs]))
                intermediates.append(sequence.flatten().tolist())
            if not torch.allclose(sequence, output, atol=1e-7, rtol=0):
                raise RuntimeError("single-layer reconstruction differs from stacked LSTM")
            cases.append({"config": {"InputSize": width, "HiddenSize": hidden,
                          "NumLayers": layers, "Bidirectional": bidirectional},
                          "frames": frames, "weights": parameters,
                          "input": x.flatten().tolist(),
                          "initial_hidden": h.flatten().tolist() if initial else [],
                          "initial_cell": c.flatten().tolist() if initial else [],
                          "output": output.flatten().tolist(), "hidden": hn.flatten().tolist(),
                          "cell": cn.flatten().tolist(), "layer_outputs": intermediates})
    result = {"schema": 1, "absolute_tolerance": 2e-6,
              "reference": {"pyannet_url": f"https://raw.githubusercontent.com/pyannote/pyannote-audio/{REV}/src/pyannote/audio/models/segmentation/PyanNet.py",
                            "pyannet_sha256": PYANNET_SHA, "torch_rnn_sha256": RNN_SHA,
                            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                            "device": "cpu", "threads": 1, "mkldnn": False},
              "scope": "synthetic LSTM forward/state/intermediates, not Community-1 model qualification",
              "cases": cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(cases)} synthetic LSTM cases")


if __name__ == "__main__":
    main()
