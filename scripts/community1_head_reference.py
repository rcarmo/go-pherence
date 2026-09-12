#!/usr/bin/env python3
"""Generate synthetic PyanNet head and LSTM/head/powerset component fixtures.

Offline CPU-only oracle, one thread, MKLDNN disabled. No trained weights/audio
or full segmentation pipeline. All supplied reference files are checksum pinned.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PYANNET_SHA = "3ceebc8c00e83d96747a706212102e7e99c44732ff1fc769e241e8de47d3d6af"
MODEL_SHA = "baef26b0d3bcb0027b179f9d472cc5795f77073fa8e5210bdd2ce53a267275e3"
POWERSET_SHA = "7eeb5691c20337bd24462f6a4ee2e56e279564bfef7875776df0a41f245b0325"
RNN_SHA = "902deeba8fd7d3c92b00e3557071b3bb41cf747bd4b2b9d63757d88562688bde"
LSTM_SHA = "3eb82a9d1f5b0d6b7dee6418de62a47ae388298b158e0c776fc74600d538bdac"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pyannet", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--powerset", required=True)
    p.add_argument("--lstm-fixture", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    for path, expected in [(args.pyannet, PYANNET_SHA), (args.model, MODEL_SHA), (args.powerset, POWERSET_SHA), (args.lstm_fixture, LSTM_SHA)]:
        if sha(path) != expected:
            raise SystemExit("reference source checksum mismatch")
    import torch
    import torch.nn.functional as F
    import torch.nn.modules.rnn as rnn
    if sha(rnn.__file__) != RNN_SHA:
        raise SystemExit("torch RNN source checksum mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    torch.backends.mkldnn.enabled = False
    spec = importlib.util.spec_from_file_location("pinned_powerset", args.powerset)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def evaluate(x, width, hidden, layers, speakers, active):
        powerset = module.Powerset(speakers, active)
        dimensions = [width] + [hidden] * layers + [powerset.num_powerset_classes]
        weights, intermediates = [], []
        y = x
        for index, (ins, outs) in enumerate(zip(dimensions, dimensions[1:])):
            w = torch.sin(torch.arange(ins * outs, dtype=torch.float32) * 0.19 + index + 1).reshape(outs, ins) * 0.17
            b = torch.cos(torch.arange(outs, dtype=torch.float32) * 0.31 + index) * 0.09
            weights.append({"Weight": w.flatten().tolist(), "Bias": b.tolist()})
            y = F.linear(y, w, b)
            if index < layers:
                y = F.leaky_relu(y)  # PyanNet default negative_slope=0.01
            intermediates.append(y.flatten().tolist())
        logp = F.log_softmax(y, dim=-1)
        return {"config": {"InputSize": width, "HiddenSize": hidden, "NumLayers": layers,
                "Speakers": speakers, "MaxActive": active}, "frames": x.shape[1],
                "input": x.flatten().tolist(), "layers": weights[:-1], "classifier": weights[-1],
                "intermediates": intermediates, "log_probabilities": logp.flatten().tolist(),
                "hard": powerset.to_multilabel(logp).flatten().tolist(),
                "soft": powerset.to_multilabel(logp, soft=True).flatten().tolist()}

    with torch.no_grad():
        cases = []
        for width, hidden, layers, speakers, active, frames in [(1, 0, 0, 1, 1, 1), (5, 7, 1, 3, 2, 4),
                                                                 (6, 9, 2, 3, 2, 7), (7, 3, 3, 4, 2, 5)]:
            x = (torch.sin(torch.arange(width * frames, dtype=torch.float32) * 0.27) * 0.35).reshape(1, frames, width)
            cases.append(evaluate(x, width, hidden, layers, speakers, active))
        recurrent = json.loads(Path(args.lstm_fixture).read_text())["cases"][2]
        cfg = recurrent["config"]
        lstm = torch.nn.LSTM(cfg["InputSize"], cfg["HiddenSize"], cfg["NumLayers"],
                             batch_first=True, bidirectional=cfg["Bidirectional"]).eval()
        for i, layer in enumerate(recurrent["weights"]):
            for direction in ["Forward", "Reverse"]:
                suffix = "_reverse" if direction == "Reverse" else ""
                for key, stem in [("WeightIH", "weight_ih"), ("WeightHH", "weight_hh"), ("BiasIH", "bias_ih"), ("BiasHH", "bias_hh")]:
                    param = getattr(lstm, f"{stem}_l{i}{suffix}")
                    param.copy_(torch.tensor(layer[direction][key]).reshape(param.shape))
        sequence = torch.tensor(recurrent["input"]).reshape(1, recurrent["frames"], cfg["InputSize"])
        encoded, _ = lstm(sequence)
        composition = {"lstm_case_index": 2, "head": evaluate(encoded, 2 * cfg["HiddenSize"], 5, 2, 3, 2)}
    output = {"schema": 1, "absolute_tolerance": 2e-6,
              "reference": {"pyannote_commit": "b749285c5cdd4636b2edc7f766f1352c8dde9369",
                            "pyannet_sha256": PYANNET_SHA, "model_sha256": MODEL_SHA,
                            "powerset_sha256": POWERSET_SHA, "torch_rnn_sha256": RNN_SHA,
                            "lstm_fixture_sha256": sha(args.lstm_fixture),
                            "torch": torch.__version__, "torch_commit": torch.version.git_version,
                            "cpu_threads": 1, "mkldnn": False},
              "scope": "synthetic head and recurrent-feature composition; no SincNet/full segmentation or trained model",
              "cases": cases, "composition": composition}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print("wrote four head cases plus LSTM/head/powerset composition")


if __name__ == "__main__":
    main()
