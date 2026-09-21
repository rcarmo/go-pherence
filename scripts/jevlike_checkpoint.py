#!/usr/bin/env python3
"""Convert upstream JEV-like PyTorch checkpoints and generate deterministic fixtures.

Offline CPU-only utility. It supports:
- .pt -> native Go v1 JSON checkpoints
- native Go v1 JSON checkpoints -> upstream-compatible .pt payloads
- deterministic tiny reference fixture generation for Go parity tests

Scope note: the reference fixture qualifies the tiny byte-level scorer and its
attention head only. It does not claim arbitrary Hugging Face backbone parity.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "torch is required; run this script with an environment that has torch installed, "
        "for example /workspace/projects/go-pherence/.venv-speaker/bin/python"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path("/tmp/jevlike")
ALLOWED_UPSTREAM_CONFIG_KEYS = {
    "encoder",
    "hf_model",
    "width",
    "rank",
    "context_tokens",
    "option_tokens",
}
TOPLEVEL_PT_KEYS = {"config", "state_dict"}
TINY_VOCAB_SIZE = 257
DEFAULT_FIXTURE_OUTPUT = ROOT / "model" / "jevlike" / "testdata" / "pytorch_reference.json"


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


def fail(message: str) -> None:
    raise ValueError(message)


def require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{name} must be an object")
    for key in value:
        if not isinstance(key, str):
            fail(f"{name} keys must be strings")
    return value


def require_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{name} must be an array")
    return value


def require_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        fail(f"{name} must be a string")
    if value == "":
        fail(f"{name} must not be empty")
    return value


def require_int(name: str, value: Any, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        fail(f"{name} must be >= {minimum}")
    return value


def require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        fail(f"{name} must be finite")
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def head_parameter_specs(width: int, rank: int) -> "OrderedDict[str, list[int]]":
    return OrderedDict([
        ("head.context_norm.weight", [width]),
        ("head.context_norm.bias", [width]),
        ("head.option_norm.weight", [width]),
        ("head.option_norm.bias", [width]),
        ("head.query.weight", [rank, width]),
        ("head.key.weight", [rank, width]),
        ("head.value.weight", [rank, width]),
    ])


def tiny_parameter_specs(width: int, rank: int, context_tokens: int) -> "OrderedDict[str, list[int]]":
    specs = OrderedDict([
        ("embedding.weight", [TINY_VOCAB_SIZE, width]),
        ("position.weight", [context_tokens, width]),
    ])
    specs.update(head_parameter_specs(width, rank))
    return specs


def load_pt(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return require_mapping("checkpoint", payload)


def tensor_to_values(name: str, tensor: Any, expected_shape: list[int]) -> list[float]:
    if not isinstance(tensor, torch.Tensor):
        fail(f"state_dict[{name!r}] must be a tensor")
    if list(tensor.shape) != expected_shape:
        fail(
            f"state_dict[{name!r}] shape {list(tensor.shape)} does not match expected {expected_shape}"
        )
    if tensor.dtype == torch.bool or tensor.is_complex() or not tensor.dtype.is_floating_point:
        fail(f"state_dict[{name!r}] must be a real floating tensor")
    value = tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not torch.isfinite(value).all().item():
        fail(f"state_dict[{name!r}] contains non-finite values")
    return value.reshape(-1).tolist()


def parse_upstream_config(raw: Any) -> dict[str, Any]:
    config = dict(require_mapping("config", raw))
    extras = set(config) - ALLOWED_UPSTREAM_CONFIG_KEYS
    if extras:
        fail(f"unsupported config keys: {sorted(extras)}")
    encoder = require_str("config.encoder", config.get("encoder"))
    rank = require_int("config.rank", config.get("rank"), minimum=1)
    context_tokens = require_int("config.context_tokens", config.get("context_tokens"), minimum=1)
    option_tokens = require_int("config.option_tokens", config.get("option_tokens"), minimum=0)
    if encoder == "tiny":
        width = require_int("config.width", config.get("width"), minimum=1)
        if "hf_model" in config and config["hf_model"] is not None and not isinstance(config["hf_model"], str):
            fail("config.hf_model must be a string when present")
        return {
            "encoder": "tiny",
            "width": width,
            "rank": rank,
            "context_tokens": context_tokens,
            "option_tokens": option_tokens,
        }
    if encoder == "hf":
        if "width" in config and config["width"] is not None:
            require_int("config.width", config["width"], minimum=1)
        return {
            "encoder": "hf",
            "hf_model": require_str("config.hf_model", config.get("hf_model")),
            "rank": rank,
            "context_tokens": context_tokens,
            "option_tokens": option_tokens,
        }
    fail(f"unsupported upstream encoder {encoder!r}")


def parse_state_dict(raw: Any) -> Mapping[str, torch.Tensor]:
    state = require_mapping("state_dict", raw)
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            fail(f"state_dict[{name!r}] must be a tensor")
    return state  # type: ignore[return-value]


def derive_frozen_width_and_rank(state: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    query = state.get("head.query.weight")
    if not isinstance(query, torch.Tensor):
        fail("frozen checkpoint requires tensor head.query.weight")
    if query.ndim != 2:
        fail(f"head.query.weight must be rank-2, got {list(query.shape)}")
    rank, width = map(int, query.shape)
    if rank <= 0 or width <= 0:
        fail(f"invalid frozen head shape {list(query.shape)}")
    return width, rank


def validate_and_collect_parameters(
    state: Mapping[str, torch.Tensor],
    specs: "OrderedDict[str, list[int]]",
) -> list[dict[str, Any]]:
    missing = [name for name in specs if name not in state]
    extras = [name for name in state if name not in specs]
    if missing or extras:
        fail(f"state_dict mismatch: missing={missing}, unexpected={extras}")
    parameters: list[dict[str, Any]] = []
    for name, shape in specs.items():
        parameters.append({
            "name": name,
            "shape": shape,
            "values": tensor_to_values(name, state[name], shape),
        })
    return parameters


def checkpoint_from_pt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != TOPLEVEL_PT_KEYS:
        fail(f"checkpoint root keys must be {sorted(TOPLEVEL_PT_KEYS)}, got {sorted(payload)}")
    config = parse_upstream_config(payload["config"])
    state = parse_state_dict(payload["state_dict"])
    if config["encoder"] == "tiny":
        specs = tiny_parameter_specs(config["width"], config["rank"], config["context_tokens"])
        parameters = validate_and_collect_parameters(state, specs)
        return {
            "version": 1,
            "encoder": "tiny",
            "config": {
                "width": config["width"],
                "rank": config["rank"],
                "context_tokens": config["context_tokens"],
                "option_tokens": config["option_tokens"],
            },
            "parameters": parameters,
        }
    width, derived_rank = derive_frozen_width_and_rank(state)
    if derived_rank != config["rank"]:
        fail(
            f"config.rank={config['rank']} does not match head.query.weight rank={derived_rank}"
        )
    specs = head_parameter_specs(width, derived_rank)
    parameters = validate_and_collect_parameters(state, specs)
    return {
        "version": 1,
        "encoder": "frozen",
        "encoder_reference": config["hf_model"],
        "config": {
            "width": width,
            "rank": derived_rank,
            "context_tokens": config["context_tokens"],
            "option_tokens": config["option_tokens"],
        },
        "parameters": parameters,
    }


def parse_native_checkpoint(raw: Any) -> dict[str, Any]:
    checkpoint = dict(require_mapping("checkpoint", raw))
    allowed = {"version", "encoder", "encoder_reference", "config", "parameters"}
    extras = set(checkpoint) - allowed
    if extras:
        fail(f"unsupported checkpoint keys: {sorted(extras)}")
    version = require_int("checkpoint.version", checkpoint.get("version"), minimum=1)
    if version != 1:
        fail(f"unsupported checkpoint version {version}")
    encoder = require_str("checkpoint.encoder", checkpoint.get("encoder"))
    config = dict(require_mapping("checkpoint.config", checkpoint.get("config")))
    if set(config) != {"width", "rank", "context_tokens", "option_tokens"}:
        fail(f"unsupported checkpoint config keys: {sorted(config)}")
    parsed_config = {
        "width": require_int("checkpoint.config.width", config.get("width"), minimum=1),
        "rank": require_int("checkpoint.config.rank", config.get("rank"), minimum=1),
        "context_tokens": require_int(
            "checkpoint.config.context_tokens", config.get("context_tokens"), minimum=1
        ),
        "option_tokens": require_int(
            "checkpoint.config.option_tokens", config.get("option_tokens"), minimum=0
        ),
    }
    parameters_raw = require_list("checkpoint.parameters", checkpoint.get("parameters"))
    if encoder == "tiny":
        specs = tiny_parameter_specs(
            parsed_config["width"], parsed_config["rank"], parsed_config["context_tokens"]
        )
        encoder_reference = None
        if checkpoint.get("encoder_reference") not in (None, ""):
            fail("tiny checkpoint must not carry encoder_reference")
    elif encoder == "frozen":
        specs = head_parameter_specs(parsed_config["width"], parsed_config["rank"])
        encoder_reference = require_str(
            "checkpoint.encoder_reference", checkpoint.get("encoder_reference")
        )
    else:
        fail(f"unsupported checkpoint encoder {encoder!r}")

    seen: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(parameters_raw):
        param = dict(require_mapping(f"checkpoint.parameters[{index}]", item))
        if set(param) != {"name", "shape", "values"}:
            fail(f"checkpoint.parameters[{index}] keys must be name/shape/values")
        name = require_str(f"checkpoint.parameters[{index}].name", param.get("name"))
        if name in seen:
            fail(f"duplicate parameter {name!r}")
        shape_raw = require_list(f"checkpoint.parameters[{index}].shape", param.get("shape"))
        shape = [require_int(f"checkpoint.parameters[{index}].shape[{i}]", n, minimum=1) for i, n in enumerate(shape_raw)]
        values_raw = require_list(f"checkpoint.parameters[{index}].values", param.get("values"))
        values = [require_number(f"checkpoint.parameters[{index}].values[{i}]", v) for i, v in enumerate(values_raw)]
        seen[name] = {"name": name, "shape": shape, "values": values}
    if list(seen) != list(specs) or len(seen) != len(specs):
        missing = [name for name in specs if name not in seen]
        extras = [name for name in seen if name not in specs]
        fail(f"checkpoint parameter mismatch: missing={missing}, unexpected={extras}")
    for name, expected_shape in specs.items():
        got = seen[name]
        if got["shape"] != expected_shape:
            fail(f"checkpoint parameter {name!r} shape {got['shape']} does not match expected {expected_shape}")
        want_len = math.prod(expected_shape)
        if len(got["values"]) != want_len:
            fail(
                f"checkpoint parameter {name!r} has {len(got['values'])} values, want {want_len}"
            )
    out = {
        "version": 1,
        "encoder": encoder,
        "config": parsed_config,
        "parameters": [seen[name] for name in specs],
    }
    if encoder_reference is not None:
        out["encoder_reference"] = encoder_reference
    return out


def pt_payload_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_native_checkpoint(checkpoint)
    config = parsed["config"]
    state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for parameter in parsed["parameters"]:
        state_dict[parameter["name"]] = torch.tensor(
            parameter["values"], dtype=torch.float32
        ).reshape(parameter["shape"])
    if parsed["encoder"] == "tiny":
        upstream_config = {
            "encoder": "tiny",
            "width": config["width"],
            "rank": config["rank"],
            "context_tokens": config["context_tokens"],
            "option_tokens": config["option_tokens"],
        }
    else:
        upstream_config = {
            "encoder": "hf",
            "hf_model": parsed["encoder_reference"],
            "width": config["width"],
            "rank": config["rank"],
            "context_tokens": config["context_tokens"],
            "option_tokens": config["option_tokens"],
        }
    return {"config": upstream_config, "state_dict": state_dict}


def json_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == json.dumps(
        b, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def save_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fill_tensor(tensor: torch.Tensor, *, scale: float, offset: float, amplitude: float, bias: float = 0.0, fn: str = "sin") -> None:
    values = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
    base = values * scale + offset
    wave = torch.sin(base) if fn == "sin" else torch.cos(base)
    tensor.copy_(wave * amplitude + bias)


def import_upstream() -> tuple[Any, Any, Any]:
    if not UPSTREAM_ROOT.exists():
        fail(f"upstream checkout {UPSTREAM_ROOT} does not exist")
    sys.path.insert(0, str(UPSTREAM_ROOT))
    from jevlike.data import ByteCollator, ChoiceExample
    from jevlike.model import TinyScorer

    return ByteCollator, ChoiceExample, TinyScorer


def make_reference_examples(ChoiceExample: Any) -> list[Any]:
    return [
        ChoiceExample(
            "café 🙂 / 青",
            ("naïve", "猫", "🙂 ok"),
            2,
        ),
        ChoiceExample(
            "ASCII + accents: jalapeño",
            ("短", "longer option"),
            1,
        ),
        ChoiceExample(
            "pad µ mix",
            ("Å", "bbb", "zebra🐾", "Ωmega"),
            0,
        ),
    ]


def initialise_reference_model(model: Any) -> None:
    with torch.no_grad():
        fill_tensor(model.embedding.weight, scale=0.071, offset=0.13, amplitude=0.19, fn="sin")
        model.embedding.weight[0].zero_()
        fill_tensor(model.position.weight, scale=0.113, offset=0.29, amplitude=0.11, fn="cos")
        fill_tensor(model.head.context_norm.weight, scale=0.41, offset=0.07, amplitude=0.17, bias=0.97, fn="sin")
        fill_tensor(model.head.context_norm.bias, scale=0.23, offset=0.31, amplitude=0.05, fn="cos")
        fill_tensor(model.head.option_norm.weight, scale=0.37, offset=0.17, amplitude=0.14, bias=1.03, fn="cos")
        fill_tensor(model.head.option_norm.bias, scale=0.19, offset=0.43, amplitude=0.04, fn="sin")
        fill_tensor(model.head.query.weight, scale=0.089, offset=0.11, amplitude=0.16, fn="sin")
        fill_tensor(model.head.key.weight, scale=0.097, offset=0.23, amplitude=0.15, fn="cos")
        fill_tensor(model.head.value.weight, scale=0.101, offset=0.41, amplitude=0.18, fn="sin")


def tensor_to_nested(value: torch.Tensor) -> Any:
    return value.detach().to(device="cpu").tolist()


def batch_to_jsonable(batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "context_ids": tensor_to_nested(batch["context_ids"]),
        "context_mask": tensor_to_nested(batch["context_mask"]),
        "option_ids": tensor_to_nested(batch["option_ids"]),
        "option_token_mask": tensor_to_nested(batch["option_token_mask"]),
        "option_mask": tensor_to_nested(batch["option_mask"]),
        "labels": tensor_to_nested(batch["labels"]),
    }


def examples_to_jsonable(examples: list[Any]) -> list[dict[str, Any]]:
    return [
        {"context": item.context, "options": list(item.options), "label": int(item.label)}
        for item in examples
    ]


def build_reference_dlogits(option_mask: torch.Tensor) -> torch.Tensor:
    d_logits = torch.zeros(option_mask.shape, dtype=torch.float32)
    index = 0
    for row in range(option_mask.shape[0]):
        for option in range(option_mask.shape[1]):
            if not bool(option_mask[row, option]):
                continue
            value = math.sin((index + 1) * 0.47) * 0.6 + math.cos((row + 1) * (option + 2) * 0.29) * 0.25
            d_logits[row, option] = float(value)
            index += 1
    return d_logits


def parameter_gradients_to_jsonable(names: list[str], grads: list[torch.Tensor]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, grad in zip(names, grads, strict=True):
        grad = grad.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(grad).all().item():
            fail(f"gradient {name} contains non-finite values")
        out.append({
            "name": name,
            "shape": list(grad.shape),
            "values": grad.reshape(-1).tolist(),
        })
    return out


def save_pt(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    torch.save(dict(payload), path)


def assert_pt_payload_equivalent(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    if dict(a["config"]) != dict(b["config"]):
        fail(f"roundtrip config mismatch: {a['config']} != {b['config']}")
    state_a = parse_state_dict(a["state_dict"])
    state_b = parse_state_dict(b["state_dict"])
    if list(state_a) != list(state_b):
        fail(f"roundtrip state_dict keys mismatch: {list(state_a)} != {list(state_b)}")
    for name in state_a:
        ta = state_a[name].detach().to(device="cpu", dtype=torch.float32)
        tb = state_b[name].detach().to(device="cpu", dtype=torch.float32)
        if ta.shape != tb.shape or not torch.equal(ta, tb):
            fail(f"roundtrip tensor mismatch for {name}")


def generate_reference(output: Path) -> None:
    configure_torch()
    ByteCollator, ChoiceExample, TinyScorer = import_upstream()
    config = {
        "encoder": "tiny",
        "width": 5,
        "rank": 3,
        "context_tokens": 24,
        "option_tokens": 16,
    }
    model = TinyScorer(config["width"], config["rank"], config["context_tokens"]).eval()
    initialise_reference_model(model)
    examples = make_reference_examples(ChoiceExample)
    collator = ByteCollator(config["context_tokens"], config["option_tokens"])
    batch = collator(examples)

    with torch.no_grad():
        logits = model(batch)
        probabilities = logits.softmax(-1)
        positions = torch.arange(batch["context_ids"].shape[1], device="cpu")
        context = model.embedding(batch["context_ids"]) + model.position(positions)
        option_tokens = model.embedding(batch["option_ids"])
        weights = batch["option_token_mask"].unsqueeze(-1)
        options = (option_tokens * weights).sum(2) / weights.sum(2).clamp_min(1)
        head_logits = model.head(context, batch["context_mask"], options, batch["option_mask"])
        if not torch.equal(head_logits, logits):
            fail("manual tiny forward does not match model(batch)")

    d_logits = build_reference_dlogits(batch["option_mask"])
    context_in = context.detach().clone().requires_grad_(True)
    options_in = options.detach().clone().requires_grad_(True)
    head_logits = model.head(context_in, batch["context_mask"], options_in, batch["option_mask"])
    objective = (head_logits * d_logits).sum()
    head_parameters = list(model.head.named_parameters())
    grads = torch.autograd.grad(
        objective,
        [context_in, options_in] + [parameter for _, parameter in head_parameters],
    )
    d_context = grads[0]
    d_options = grads[1]
    d_parameters = list(grads[2:])
    prefixed_names = [f"head.{name}" for name, _ in head_parameters]

    pt_payload = {
        "config": dict(config),
        "state_dict": OrderedDict(
            (name, parameter.detach().cpu().clone())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
    }
    native_checkpoint = checkpoint_from_pt_payload(pt_payload)
    roundtrip_pt = pt_payload_from_checkpoint(native_checkpoint)
    assert_pt_payload_equivalent(pt_payload, roundtrip_pt)
    if not json_equal(native_checkpoint, checkpoint_from_pt_payload(roundtrip_pt)):
        fail("checkpoint JSON roundtrip is not stable")

    fixture = {
        "schema": 1,
        "absolute_tolerance": 1e-5,
        "generator": {
            "script": "scripts/jevlike_checkpoint.py",
            "reproduce": f"python3 scripts/jevlike_checkpoint.py generate-reference --output {output.as_posix()}",
            "upstream_root": str(UPSTREAM_ROOT),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cpu_threads": 1,
            "mkldnn": False,
        },
        "scope": "Deterministic tiny scorer fixture only; no training changes and no claim of arbitrary Hugging Face backbone parity.",
        "checkpoint": native_checkpoint,
        "examples": examples_to_jsonable(examples),
        "batch": batch_to_jsonable(batch),
        "predict": {
            "logits": tensor_to_nested(logits),
            "probabilities": tensor_to_nested(probabilities),
        },
        "head_inputs": {
            "context": tensor_to_nested(context),
            "context_mask": tensor_to_nested(batch["context_mask"]),
            "options": tensor_to_nested(options),
            "option_mask": tensor_to_nested(batch["option_mask"]),
        },
        "head_gradients": {
            "d_logits": tensor_to_nested(d_logits),
            "parameters": parameter_gradients_to_jsonable(prefixed_names, d_parameters),
            "d_context": tensor_to_nested(d_context),
            "d_options": tensor_to_nested(d_options),
        },
    }
    save_json(output, fixture)
    print(json.dumps({"output": str(output), "examples": len(examples), "encoder": "tiny"}))


def convert_to_json(input_path: Path, output_path: Path) -> None:
    configure_torch()
    payload = load_pt(input_path)
    checkpoint = checkpoint_from_pt_payload(payload)
    save_json(output_path, checkpoint)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "encoder": checkpoint["encoder"]}))


def convert_to_pt(input_path: Path, output_path: Path) -> None:
    configure_torch()
    checkpoint = json.loads(input_path.read_text(encoding="utf-8"))
    payload = pt_payload_from_checkpoint(checkpoint)
    save_pt(output_path, payload)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "encoder": payload["config"]["encoder"]}))


def roundtrip_smoke(kind: str) -> None:
    configure_torch()
    with tempfile.TemporaryDirectory(prefix="jevlike-roundtrip-") as tmp:
        tmpdir = Path(tmp)
        if kind == "tiny":
            pt_payload = {
                "config": {
                    "encoder": "tiny",
                    "width": 4,
                    "rank": 3,
                    "context_tokens": 9,
                    "option_tokens": 7,
                },
                "state_dict": OrderedDict(),
            }
            specs = tiny_parameter_specs(4, 3, 9)
        elif kind == "frozen":
            pt_payload = {
                "config": {
                    "encoder": "hf",
                    "hf_model": "synthetic/frozen-head",
                    "width": 99,
                    "rank": 2,
                    "context_tokens": 11,
                    "option_tokens": 5,
                },
                "state_dict": OrderedDict(),
            }
            specs = head_parameter_specs(6, 2)
        else:
            fail(f"unknown smoke kind {kind}")
        for index, (name, shape) in enumerate(specs.items()):
            tensor = torch.empty(shape, dtype=torch.float32)
            fill_tensor(
                tensor,
                scale=0.17 + index * 0.01,
                offset=0.03 * (index + 1),
                amplitude=0.2,
                fn="sin" if index % 2 == 0 else "cos",
            )
            pt_payload["state_dict"][name] = tensor
        pt_path = tmpdir / f"{kind}.pt"
        json_path = tmpdir / f"{kind}.json"
        back_path = tmpdir / f"{kind}.roundtrip.pt"
        save_pt(pt_path, pt_payload)
        convert_to_json(pt_path, json_path)
        convert_to_pt(json_path, back_path)
        restored = load_pt(back_path)
        expected = pt_payload_from_checkpoint(checkpoint_from_pt_payload(pt_payload))
        assert_pt_payload_equivalent(expected, restored)
        print(json.dumps({"smoke": kind, "json": str(json_path), "pt": str(back_path)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_json = subparsers.add_parser("to-json", help="convert upstream .pt checkpoint to native Go v1 JSON")
    to_json.add_argument("input", type=Path)
    to_json.add_argument("output", type=Path)

    to_pt = subparsers.add_parser("to-pt", help="convert native Go v1 JSON checkpoint to upstream-compatible .pt")
    to_pt.add_argument("input", type=Path)
    to_pt.add_argument("output", type=Path)

    reference = subparsers.add_parser("generate-reference", help="generate deterministic tiny PyTorch reference JSON")
    reference.add_argument("--output", type=Path, default=DEFAULT_FIXTURE_OUTPUT)

    smoke = subparsers.add_parser("roundtrip-smoke", help="run an offline roundtrip smoke test")
    smoke.add_argument("kind", choices=("tiny", "frozen"))

    args = parser.parse_args()
    if args.command == "to-json":
        convert_to_json(args.input, args.output)
    elif args.command == "to-pt":
        convert_to_pt(args.input, args.output)
    elif args.command == "generate-reference":
        generate_reference(args.output)
    else:
        roundtrip_smoke(args.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
