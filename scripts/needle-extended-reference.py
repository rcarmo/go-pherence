#!/usr/bin/env python3
"""Create an extended deterministic Needle 3 CPU reference fixture.

This script is intentionally offline-only:
- it asserts the local upstream checkout is pinned to fc5bae0f9b6138828fe7589f6b531fb9a26968de
- it imports upstream architecture.py/quantize.py directly via scripts/needle-reference.py
- it never uses the network, GPU, or git writes
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REF_PATH = Path(__file__).with_name("needle-reference.py")
SEED = 0
BASE_TOKENS = [2, 7, 0, 9, 3]
SLICE_TOKENS = [2, 7, 4, 9, 3]
AB_TOKENS = [2, 7, 4, 9, 3]


def _load_ref_module():
    spec = importlib.util.spec_from_file_location("needle_reference_ref", REF_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {REF_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load_ref_module()
flax = ref.flax
jax = ref.jax
jnp = ref.jnp
np = ref.np
flatten_dict = ref.flatten_dict
unflatten_dict = ref.unflatten_dict


if jax.default_backend() != "cpu":
    raise SystemExit(f"expected cpu backend, got {jax.default_backend()}")


def _clone_tree(tree: Any) -> Any:
    return copy.deepcopy(flax.core.unfreeze(tree))


def _merge_trees(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict):
            current = dst.get(key)
            if not isinstance(current, dict):
                current = {}
                dst[key] = current
            _merge_trees(current, value)
        else:
            dst[key] = value
    return dst


def _unbox_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(ref._unwrap_leaf, tree)


def _pattern(shape: tuple[int, ...], *, scale: float, phase: float, bias: float = 0.0) -> np.ndarray:
    x = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (bias + scale * np.sin(x + np.float32(phase))).astype(np.float32)


def _taps_pattern(shape: tuple[int, ...], *, phase: float) -> np.ndarray:
    taps = _pattern(shape, scale=0.06, phase=phase)
    taps[0] += 0.85
    if shape[0] > 1:
        taps[1] += 0.15
    if shape[0] > 2:
        taps[2] -= 0.08
    if shape[0] > 3:
        taps[3] += 0.04
    return taps.astype(np.float32)


def _mutate_params(params: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(_unbox_tree(params))
    for key, value in list(flat.items()):
        name = "/".join(key)
        leaf = key[-1]
        array = np.asarray(jax.device_get(value), dtype=np.float32)

        if name == "embedding/embedding" or leaf == "embedding":
            flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.018, phase=0.11 * len(name)))
            continue
        if leaf.startswith("mhc_phi"):
            flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.02, phase=0.07 * len(name)))
            continue
        if leaf == "kernel":
            flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.017, phase=0.05 * len(name)))
            continue
        if leaf == "scale":
            flat[key] = jnp.asarray(_pattern(array.shape, scale=0.05, phase=0.09 * len(name)))
            continue
        if leaf in {"q_taps", "k_taps", "v_taps", "taps"}:
            flat[key] = jnp.asarray(_taps_pattern(array.shape, phase=0.13 * len(name)))
            continue
        if leaf == "attn_gate":
            gate = -0.4 + 0.22 * np.sin(np.arange(array.size, dtype=np.float32) + 0.3)
            flat[key] = jnp.asarray(gate.reshape(array.shape).astype(np.float32))
            continue
        if "/hadamard_mlp/" in name:
            if leaf in {"w1a", "w1b", "w2a", "w2b", "w3a", "w3b"}:
                flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.03, phase=0.17 * len(name)))
            elif leaf == "d1":
                flat[key] = jnp.asarray(1.0 + _pattern(array.shape, scale=0.08, phase=0.4))
            elif leaf == "d2":
                flat[key] = jnp.asarray(0.93 + _pattern(array.shape, scale=0.07, phase=0.8))
            elif leaf == "b2":
                flat[key] = jnp.asarray(_pattern(array.shape, scale=0.045, phase=1.2))
            elif leaf == "d3":
                flat[key] = jnp.asarray(1.04 + _pattern(array.shape, scale=0.065, phase=1.7))
            elif leaf == "d4":
                flat[key] = jnp.asarray(0.03 + _pattern(array.shape, scale=0.015, phase=2.2))
            elif leaf == "cond_u":
                flat[key] = jnp.asarray(_pattern(array.shape, scale=0.055, phase=2.8))
            elif leaf == "cond_v":
                flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.02, phase=3.1))
            continue
        if "_head/" in name:
            if leaf in {"probes", "query"}:
                flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.025, phase=0.03 * len(name)))
            elif leaf == "gain":
                flat[key] = jnp.asarray(0.95 + _pattern(array.shape, scale=0.08, phase=0.04 * len(name)))
            elif leaf == "row_bias":
                flat[key] = jnp.asarray(_pattern(array.shape, scale=0.05, phase=0.05 * len(name)))
            elif leaf == "bias":
                if array.ndim == 0:
                    flat[key] = jnp.asarray(np.float32(float(array) + 0.35))
                else:
                    flat[key] = jnp.asarray(array + _pattern(array.shape, scale=0.04, phase=0.06 * len(name)))
            elif leaf == "log_temp":
                flat[key] = jnp.asarray(np.float32(math.log(7.0)))
            continue
        if leaf in {"mhc_a_pre", "mhc_a_post", "mhc_a_res"}:
            flat[key] = jnp.asarray(0.85 + _pattern(array.shape, scale=0.11, phase=0.12 * len(name)))
            continue
        if leaf in {"mhc_b_pre", "mhc_b_post", "mhc_b_res"}:
            flat[key] = jnp.asarray(_pattern(array.shape, scale=0.06, phase=0.14 * len(name)))
            continue
    return unflatten_dict(flat)


def _config() -> Any:
    return arch.TransformerConfig(
        vocab_size=16,
        d_model=8,
        num_heads=2,
        num_kv_heads=1,
        qk_head_dim=4,
        v_head_dim=4,
        num_layers=4,
        max_seq_len=16,
        pad_token_id=0,
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
        engram_layers=(0, 2, 3),
        global_layers=(3,),
        sliding_window=3,
        mhc_lanes=2,
        qkv_conv_taps=3,
        remat=False,
    )


arch = None


def _make_model_and_params(tokens: list[int]) -> tuple[Any, dict[str, Any], Any]:
    config = _config()
    model = arch.SimpleAttentionNetwork(config)
    batch = jnp.asarray([tokens], dtype=jnp.int32)
    key = jax.random.key(SEED)
    params = _clone_tree(model.init(key, batch, quant=False)["params"])
    for method in (model.forward_embedding, model.forward_confidence, model.forward_router):
        extra = _clone_tree(model.init(key, batch, method=method, quant=False)["params"])
        _merge_trees(params, extra)
    params = _mutate_params(params)
    return model, params, config


def _named_tensor_map(tree: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(_unbox_tree(tree))
    out: dict[str, Any] = {}
    for key in sorted(flat):
        out["/".join(key)] = ref._tensor_payload(flat[key])
    return out


def _logits_payload(value: Any) -> dict[str, Any]:
    array = np.asarray(jax.device_get(ref._unwrap_leaf(value)), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return {
        "shape": [int(v) for v in array.shape],
        "data": [float(v) for v in array.reshape(-1).tolist()],
    }


def _head_array(value: Any) -> list[float]:
    array = np.asarray(jax.device_get(ref._unwrap_leaf(value)), dtype=np.float32)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return [float(v) for v in array.reshape(-1).tolist()]


def _head_scalar(value: Any) -> float:
    array = np.asarray(jax.device_get(ref._unwrap_leaf(value)), dtype=np.float32)
    return float(array.reshape(-1)[0])


def _head_outputs(model: Any, params: dict[str, Any], tokens: list[int], *, quant: bool) -> dict[str, Any]:
    batch = jnp.asarray([tokens], dtype=jnp.int32)
    return {
        "embedding": _head_array(model.apply({"params": params}, batch, method=model.forward_embedding, quant=quant)),
        "confidence": _head_scalar(model.apply({"params": params}, batch, method=model.forward_confidence, quant=quant)),
        "router": _head_array(model.apply({"params": params}, batch, method=model.forward_router, quant=quant)),
    }


def _base_section(model: Any, params: dict[str, Any], config: Any) -> dict[str, Any]:
    batch = jnp.asarray([BASE_TOKENS], dtype=jnp.int32)
    fp32_logits = model.apply({"params": params}, batch, quant=False)

    arch._quantize.configure_deploy(act_bits=8, kv_bits=8)
    cq_params = arch._quantize.cq_ste_params(params, 4)
    cq_logits = model.apply({"params": cq_params}, batch, quant=True)

    return {
        "config": ref._config_payload(config),
        "tensors": _named_tensor_map(params),
        "tokens": BASE_TOKENS,
        "fp32": {
            "logits": _logits_payload(fp32_logits),
            "heads": _head_outputs(model, params, BASE_TOKENS, quant=False),
        },
        "cq": {
            "logits": _logits_payload(cq_logits),
            "heads": _head_outputs(model, cq_params, BASE_TOKENS, quant=True),
        },
        "slices": {
            "depth3": _slice_section(params, config, 3),
            "depth2": _slice_section(params, config, 2),
            "nested3then2": _nested_slice_section(params, config),
        },
    }


def _slice_outputs(config: Any, params: dict[str, Any], tokens: list[int]) -> dict[str, Any]:
    model = arch.SimpleAttentionNetwork(config)
    batch = jnp.asarray([tokens], dtype=jnp.int32)
    return {
        "config": ref._config_payload(config),
        "tensors": _named_tensor_map(params),
        "tokens": tokens,
        "logits": _logits_payload(model.apply({"params": params}, batch, quant=False)),
        "heads": _head_outputs(model, params, tokens, quant=False),
    }


def _slice_section(params: dict[str, Any], config: Any, depth: int) -> dict[str, Any]:
    child_config = arch.ladder_config(config, depth)
    child_params = arch.ladder_slice(params, config, depth)
    return _slice_outputs(child_config, child_params, SLICE_TOKENS)


def _nested_slice_section(params: dict[str, Any], config: Any) -> dict[str, Any]:
    mid_config = arch.ladder_config(config, 3)
    mid_params = arch.ladder_slice(params, config, 3)
    child_config = arch.ladder_config(mid_config, 2)
    child_params = arch.ladder_slice(mid_params, mid_config, 2)
    return _slice_outputs(child_config, child_params, SLICE_TOKENS)


def _ab_scales() -> dict[str, Any]:
    return {
        "embedding/embedding": {
            "a": jnp.asarray(np.float32(0.9)),
            "b": jnp.asarray((1.1 + 0.07 * np.sin(np.arange(8, dtype=np.float32) + 0.1)).astype(np.float32)),
        },
        "stack/layers/block/self_attn/q_proj/kernel": {
            "a": jnp.asarray((0.9 + 0.03 * np.sin(np.arange(4 * 8, dtype=np.float32).reshape(4, 8, 1) + 0.2)).astype(np.float32)),
            "b": jnp.asarray((1.1 + 0.05 * np.sin(np.arange(8, dtype=np.float32) + 0.3)).astype(np.float32)),
        },
        "stack/layers/block/self_attn/k_proj/kernel": {
            "a": jnp.asarray((0.88 + 0.025 * np.sin(np.arange(4 * 4, dtype=np.float32).reshape(4, 4, 1) + 0.4)).astype(np.float32)),
            "b": jnp.asarray((1.07 + 0.04 * np.sin(np.arange(8, dtype=np.float32).reshape(1, 8) + 0.5)).astype(np.float32)),
        },
        "stack/mhc_phi_pre": {
            "a": jnp.asarray(np.float32(0.93)),
            "b": jnp.asarray((1.09 + 0.03 * np.sin(np.arange(16, dtype=np.float32) + 0.6)).astype(np.float32)),
        },
        "engrams_1/key_proj/kernel": {
            "a": jnp.asarray((0.91 + 0.02 * np.sin(np.arange(8, dtype=np.float32).reshape(8, 1) + 0.7)).astype(np.float32)),
            "b": jnp.asarray((1.08 + 0.06 * np.sin(np.arange(8, dtype=np.float32) + 0.8)).astype(np.float32)),
        },
    }


def _ab_section(model: Any, params: dict[str, Any], config: Any) -> dict[str, Any]:
    ab_params = _clone_tree(params)
    ab_params["ab_scales"] = _ab_scales()
    batch = jnp.asarray([AB_TOKENS], dtype=jnp.int32)
    targets = jnp.asarray([AB_TOKENS[1:]], dtype=jnp.int32)

    arch._quantize.configure_deploy(act_bits=8, kv_bits=8)

    def loss_fn(p):
        qparams = arch._quantize.cq_ste_params(p, 4)
        logits = model.apply({"params": qparams}, batch, quant=True)
        log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        loss = -jnp.mean(token_log_probs)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(ab_params)
    return {
        "config": ref._config_payload(config),
        "tensors": _named_tensor_map(ab_params),
        "tokens": AB_TOKENS,
        "logits": _logits_payload(logits),
        "loss": float(np.asarray(jax.device_get(loss), dtype=np.float32)),
        "gradients": _named_tensor_map(grads),
    }


def _fixture(upstream: Path, head: str) -> dict[str, Any]:
    model, params, config = _make_model_and_params(BASE_TOKENS)
    return {
        "schema": "go-pherence/needle3-extended-reference/v1",
        "generator": {
            "script": "scripts/needle-extended-reference.py",
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": flax.__version__,
            "seed": SEED,
            "device": "cpu",
            "dtype": "float32",
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
            "notes": [
                "base uses pad-bearing tokens to exercise head masking; slice and AB sections use non-pad tokens",
                "auxiliary heads are initialized separately and merged into the main params tree before deterministic perturbation",
                "slices use upstream ladder_config and ladder_slice directly; AB numerics use upstream CQ W4/A8/KV8 helpers",
            ],
        },
        "base": _base_section(model, params, config),
        "ab": _ab_section(model, params, config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="local pinned needle-upstream checkout")
    parser.add_argument("--output", required=True, type=Path, help="output fixture JSON path")
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    output = args.output.resolve()
    head = ref._assert_upstream_pin(upstream)
    global arch
    arch = ref._load_architecture(upstream)
    fixture = _fixture(upstream, head)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {output}")
    print(f"upstream_pin={head}")
    print(f"base_tensors={len(fixture['base']['tensors'])} ab_tensors={len(fixture['ab']['tensors'])} ab_gradients={len(fixture['ab']['gradients'])}")
    print(f"size_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
