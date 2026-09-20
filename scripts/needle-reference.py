#!/usr/bin/env python3
"""Create a deterministic Needle 3 CPU parity fixture from pinned upstream.

This script is intentionally offline-only:
- it asserts /workspace/projects/needle-upstream is pinned to
  fc5bae0f9b6138828fe7589f6b531fb9a26968de
- it imports the local upstream model files directly without importing the top-
  level needle package
- it never downloads weights, libraries, or metadata
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

PIN = "fc5bae0f9b6138828fe7589f6b531fb9a26968de"
SEED = 0
TOKENS = [2, 7, 4, 9, 3]


def _configure_env() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


_configure_env()

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax.traverse_util import flatten_dict, unflatten_dict  # noqa: E402


PACKAGE_NAME = "needle_fixture_upstream"


def _assert_upstream_pin(upstream: Path) -> str:
    head = subprocess.run(
        ["git", "-C", os.fspath(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != PIN:
        raise SystemExit(f"expected upstream pin {PIN}, found {head}")
    return head


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_architecture(upstream: Path):
    model_dir = upstream / "needle" / "model"
    if not (model_dir / "architecture.py").is_file():
        raise SystemExit(f"missing {model_dir / 'architecture.py'}")
    if not (model_dir / "quantize.py").is_file():
        raise SystemExit(f"missing {model_dir / 'quantize.py'}")

    root_pkg = types.ModuleType(PACKAGE_NAME)
    root_pkg.__path__ = [os.fspath(upstream / "needle")]
    sys.modules[PACKAGE_NAME] = root_pkg

    model_pkg_name = f"{PACKAGE_NAME}.model"
    model_pkg = types.ModuleType(model_pkg_name)
    model_pkg.__path__ = [os.fspath(model_dir)]
    sys.modules[model_pkg_name] = model_pkg

    _load_module(f"{model_pkg_name}.quantize", model_dir / "quantize.py")
    return _load_module(f"{model_pkg_name}.architecture", model_dir / "architecture.py")


def _unwrap_leaf(value: Any) -> Any:
    out = value
    for _ in range(4):
        if hasattr(out, "value") and not isinstance(out, (np.ndarray, jax.Array)):
            out = out.value
        else:
            break
    return out


def _tensor_payload(value: Any, *, dtype=np.float32, integer: bool = False) -> dict[str, Any]:
    array = np.asarray(jax.device_get(_unwrap_leaf(value)))
    if integer:
        return {
            "shape": [int(v) for v in array.shape],
            "data": [int(v) for v in array.reshape(-1).tolist()],
        }
    cast = array.astype(dtype, copy=False)
    return {
        "shape": [int(v) for v in cast.shape],
        "data": [float(v) for v in cast.reshape(-1).tolist()],
    }


def _config_payload(config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(config.__dataclass_fields__):
        value = getattr(config, key)
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _pattern(shape: tuple[int, ...], *, scale: float, phase: float, bias: float = 0.0) -> np.ndarray:
    x = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (bias + scale * np.sin(x + np.float32(phase))).astype(np.float32)


def _taps_pattern(shape: tuple[int, ...], *, phase: float) -> np.ndarray:
    taps = _pattern(shape, scale=0.07, phase=phase)
    taps[0] += 0.82
    if shape[0] > 1:
        taps[1] += 0.18
    if shape[0] > 2:
        taps[2] -= 0.11
    if shape[0] > 3:
        taps[3] += 0.05
    return taps.astype(np.float32)


def _mutate_params(params: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(jax.tree_util.tree_map(_unwrap_leaf, params))
    for key, value in list(flat.items()):
        name = "/".join(key)
        array = np.asarray(jax.device_get(value), dtype=np.float32)
        leaf = name.rsplit("/", 1)[-1]

        if leaf == "scale":
            flat[key] = jnp.asarray(_pattern(array.shape, scale=0.06, phase=0.1 * len(name)))
            continue

        if leaf in {"q_taps", "k_taps", "v_taps", "taps"}:
            flat[key] = jnp.asarray(_taps_pattern(array.shape, phase=0.2 * len(name)))
            continue

        if leaf == "attn_gate":
            flat[key] = jnp.asarray(np.array([-0.35, 0.45], dtype=np.float32))
            continue

        if "/hadamard_mlp/" not in name:
            continue
        if leaf in {"w1a", "w1b", "w2a", "w2b", "w3a", "w3b"}:
            flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.035, phase=0.3 * len(name)))
        elif leaf == "d1":
            flat[key] = jnp.asarray(1.0 + _pattern(array.shape, scale=0.09, phase=0.4))
        elif leaf == "d2":
            flat[key] = jnp.asarray(0.95 + _pattern(array.shape, scale=0.08, phase=0.9))
        elif leaf == "b2":
            flat[key] = jnp.asarray(_pattern(array.shape, scale=0.05, phase=1.7))
        elif leaf == "d3":
            flat[key] = jnp.asarray(1.05 + _pattern(array.shape, scale=0.07, phase=2.1))
        elif leaf == "d4":
            flat[key] = jnp.asarray(0.025 + _pattern(array.shape, scale=0.012, phase=2.7))
        elif leaf == "cond_u":
            flat[key] = jnp.asarray(_pattern(array.shape, scale=0.06, phase=3.3))
    return unflatten_dict(flat)


def _named_tensor_map(tree: dict[str, Any], *, integer: bool = False) -> dict[str, Any]:
    flat = flatten_dict(jax.tree_util.tree_map(_unwrap_leaf, tree))
    out: dict[str, Any] = {}
    for key in sorted(flat):
        out["/".join(key)] = _tensor_payload(flat[key], integer=integer)
    return out


def _build_fixture(arch: Any, upstream: Path, head: str) -> dict[str, Any]:
    if jax.default_backend() != "cpu":
        raise SystemExit(f"expected cpu backend, got {jax.default_backend()}")

    config = arch.TransformerConfig(
        vocab_size=16,
        d_model=8,
        num_heads=2,
        num_kv_heads=1,
        qk_head_dim=4,
        v_head_dim=4,
        num_layers=2,
        max_seq_len=16,
        embedding_dim=4,
        embedding_probes=2,
        embedding_queries=2,
        confidence_probes=2,
        confidence_queries=2,
        router_probes=2,
        router_queries=2,
        dtype="float32",
        flash=False,
        engram_orders=(2, 3),
        engram_heads=1,
        engram_slots=8,
        engram_layers=(0,),
        global_layers=(1,),
        sliding_window=3,
        mhc_lanes=2,
        qkv_conv_taps=3,
        remat=False,
    )
    model = arch.SimpleAttentionNetwork(config)
    tokens = jnp.asarray([TOKENS], dtype=jnp.int32)
    params = model.init(jax.random.key(SEED), tokens, quant=False)["params"]
    params = _mutate_params(params)

    def loss_fn(p):
        logits = model.apply({"params": p}, tokens, quant=False)
        next_ids = tokens[:, 1:]
        log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, next_ids[..., None], axis=-1)[..., 0]
        loss = -jnp.mean(token_log_probs)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    padded_dim = 1 << (config.d_model - 1).bit_length()
    p1, p2 = arch._hada_perms(padded_dim, split=bool(getattr(config, "ladder_widths", ())))

    logits_2d = np.asarray(jax.device_get(logits[0]), dtype=np.float32)
    loss_value = float(np.asarray(jax.device_get(loss), dtype=np.float32))
    next_tokens = TOKENS[1:]

    return {
        "schema": "go-pherence/needle3-cpu-parity/v1",
        "generator": {
            "script": "scripts/needle-reference.py",
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": flax.__version__,
            "seed": SEED,
            "device": "cpu",
            "dtype": "float32",
            "quant": False,
            "upstream": os.fspath(upstream),
            "upstream_pin": head,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "thread_env": {
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
            "loss": "mean cross entropy over logits[:-1] against next-token ids",
            "notes": [
                "heads are not separately initialized; fixture covers the default forward path",
                "parameters were deterministically perturbed after init to avoid identity-only taps/scales/cond_u coverage",
                "imports architecture.py and quantize.py directly from local upstream checkout; no downloads",
            ],
        },
        "config": _config_payload(config),
        "tokens": TOKENS,
        "targets": next_tokens,
        "permutations": {
            "padded_dim": padded_dim,
            "p1": _tensor_payload(p1, integer=True),
            "p2": _tensor_payload(p2, integer=True),
        },
        "tensors": _named_tensor_map(params),
        "logits": {
            "shape": [int(v) for v in logits_2d.shape],
            "data": [float(v) for v in logits_2d.reshape(-1).tolist()],
        },
        "loss": loss_value,
        "gradients": _named_tensor_map(grads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="local pinned needle-upstream checkout")
    parser.add_argument("--output", required=True, type=Path, help="output fixture JSON path")
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    output = args.output.resolve()
    head = _assert_upstream_pin(upstream)
    arch = _load_architecture(upstream)
    fixture = _build_fixture(arch, upstream, head)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {output}")
    print(f"upstream_pin={head}")
    print(f"params={len(fixture['tensors'])} gradients={len(fixture['gradients'])}")
    print(f"logits_shape={fixture['logits']['shape']} loss={fixture['loss']:.9f}")


if __name__ == "__main__":
    main()
