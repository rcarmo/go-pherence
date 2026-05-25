#!/usr/bin/env python3
"""Generate Hunyuan3D low-step latent-loop fixture summaries.

This optional script runs the upstream Hunyuan3DDiTFlowMatching denoising loop
for a small number of steps and writes compact summaries for initial/final
latents. It deliberately stops before VAE decode or mesh export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require_deps(hunyuan3d_src: str):
    if hunyuan3d_src:
        sys.path.insert(0, hunyuan3d_src)
    missing = []
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
        missing.append("pyyaml")
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
        missing.append("torch")
    try:
        import safetensors.torch  # type: ignore
    except Exception:
        safetensors_torch = None
        missing.append("safetensors")
    else:
        safetensors_torch = safetensors.torch
    try:
        from hy3dgen.shapegen.pipelines import instantiate_from_config, retrieve_timesteps  # type: ignore
    except Exception:
        instantiate_from_config = None
        retrieve_timesteps = None
        missing.append("hy3dgen from --hunyuan3d-src")
    if missing:
        print(
            "missing Python low-step latent fixture dependencies: "
            + ", ".join(missing)
            + "\ninstall Hunyuan3D deps and pass --hunyuan3d-src to a local Tencent-Hunyuan/Hunyuan3D-2 checkout",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return np, yaml, torch, safetensors_torch, instantiate_from_config, retrieve_timesteps


def strip_group(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = {}
    dot = prefix + "."
    for key, value in state.items():
        if key.startswith(dot):
            out[key[len(dot):]] = value
    return out


def load_state(torch, safetensors_torch, checkpoint: Path) -> dict[str, Any]:
    if checkpoint.suffix == ".safetensors":
        return safetensors_torch.load_file(str(checkpoint), device="cpu")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        return state["state_dict"]
    if any(k in state for k in ("model", "conditioner")):
        flat = {}
        for group in ("model", "conditioner"):
            if group in state:
                flat.update({group + "." + k: v for k, v in state[group].items()})
        return flat
    return state


def tensor_summary(torch, name: str, value) -> dict[str, Any]:
    v = value.detach().to("cpu", dtype=torch.float32).contiguous()
    raw = v.numpy().astype("<f4", copy=False).tobytes()
    flat = v.reshape(-1)
    sample_count = min(16, flat.numel())
    return {
        "name": name,
        "dtype": "float32",
        "shape": list(v.shape),
        "sha256_le_f32": hashlib.sha256(raw).hexdigest(),
        "min": float(v.min().item()) if flat.numel() else 0.0,
        "max": float(v.max().item()) if flat.numel() else 0.0,
        "mean": float(v.mean().item()) if flat.numel() else 0.0,
        "first_values": [float(x) for x in flat[:sample_count].tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hunyuan3d-src", default="/workspace/tmp/Hunyuan3D-2-info")
    parser.add_argument("--config", required=True, help="Hunyuan3D shape config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Hunyuan3D model safetensors/ckpt containing model.* and conditioner.* weights")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", default="testdata/hunyuan3d/lowstep-latent-fixture.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    args = parser.parse_args()

    np, yaml, torch, safetensors_torch, instantiate_from_config, retrieve_timesteps = require_deps(args.hunyuan3d_src)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_processor = instantiate_from_config(cfg["image_processor"])
    conditioner = instantiate_from_config(cfg["conditioner"])
    model = instantiate_from_config(cfg["model"])
    scheduler = instantiate_from_config(cfg["scheduler"])

    state = load_state(torch, safetensors_torch, Path(args.checkpoint))
    conditioner_state = strip_group(state, "conditioner")
    model_state = strip_group(state, "model")
    if not model_state:
        raise SystemExit(f"no model.* tensors found in {args.checkpoint}")
    if not conditioner_state:
        raise SystemExit(f"no conditioner.* tensors found in {args.checkpoint}")
    cond_missing, cond_unexpected = conditioner.load_state_dict(conditioner_state, strict=False)
    model_missing, model_unexpected = model.load_state_dict(model_state, strict=False)

    conditioner.to(device=args.device, dtype=dtype).eval()
    model.to(device=args.device, dtype=dtype).eval()

    cond_inputs = image_processor(args.image)
    image = cond_inputs.pop("image").to(device=args.device, dtype=dtype)
    cond_inputs = {k: v.to(device=args.device, dtype=dtype) if hasattr(v, "to") else v for k, v in cond_inputs.items()}

    batch_size = image.shape[0]
    do_cfg = args.guidance_scale >= 0 and not (hasattr(model, "guidance_embed") and model.guidance_embed is True)
    sigmas = np.linspace(0, 1, args.steps)
    timesteps, _ = retrieve_timesteps(scheduler, args.steps, args.device, sigmas=sigmas)

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    latents = torch.randn((batch_size, cfg["vae"]["params"]["num_latents"], getattr(model, "in_channels")), generator=generator, device=args.device, dtype=dtype)
    initial_latents = latents.detach().clone()

    guidance = None
    if hasattr(model, "guidance_embed") and model.guidance_embed is True:
        guidance = torch.tensor([args.guidance_scale] * batch_size, device=args.device, dtype=dtype)

    step_summaries = []
    with torch.inference_mode():
        cond = conditioner(image=image, **cond_inputs)
        for i, t in enumerate(timesteps):
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype) / scheduler.config.num_train_timesteps
            noise_pred = model(latent_model_input, timestep, cond, guidance=guidance)
            if do_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            outputs = scheduler.step(noise_pred, t, latents)
            latents = outputs.prev_sample
            step_summaries.append({
                "index": i,
                "timestep": float(t.detach().cpu().item()) if hasattr(t, "detach") else float(t),
                "model_timestep_input": float((t / scheduler.config.num_train_timesteps).detach().cpu().item()),
                "latents": tensor_summary(torch, f"latents_after_step_{i}", latents),
            })

    fixture = {
        "schema": "go-pherence-hunyuan3d-lowstep-latents-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "hunyuan3d_src": args.hunyuan3d_src,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "image": args.image,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
        },
        "load_state": {
            "conditioner_missing": list(cond_missing),
            "conditioner_unexpected": list(cond_unexpected),
            "model_missing": list(model_missing),
            "model_unexpected": list(model_unexpected),
            "conditioner_tensor_count": len(conditioner_state),
            "model_tensor_count": len(model_state),
        },
        "scheduler": {
            "timesteps": [float(x.detach().cpu().item()) for x in timesteps],
            "sigmas": [float(x) for x in scheduler.sigmas.detach().cpu().tolist()],
            "num_train_timesteps": scheduler.config.num_train_timesteps,
            "step_formula": "prev_sample = sample + (sigma_next - sigma) * model_output",
        },
        "outputs": [
            tensor_summary(torch, "initial_latents", initial_latents),
            tensor_summary(torch, "final_latents", latents),
        ] + [tensor_summary(torch, f"conditioned.{k}", v) for k, v in sorted(cond.items())],
        "steps": step_summaries,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
