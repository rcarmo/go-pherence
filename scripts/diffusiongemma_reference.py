#!/usr/bin/env python3
"""Generate DiffusionGemma reference outputs with Hugging Face Transformers.

This is a fixture-capture helper, not a Go test. It can run in --dry-run mode
without loading the 26B model to validate paths/config availability.
"""
import argparse
import json
import pathlib
import time


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_env(out: pathlib.Path | None) -> dict:
    report = {"check_env": True}
    try:
        import torch
        report["torch"] = {"present": True, "version": getattr(torch, "__version__", None), "cuda_available": bool(torch.cuda.is_available())}
    except Exception as e:
        report["torch"] = {"present": False, "error": str(e)}
    try:
        import transformers
        report["transformers"] = {"present": True, "version": getattr(transformers, "__version__", None)}
        try:
            from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor  # noqa: F401
            report["diffusiongemma"] = {"present": True}
        except Exception as e:
            report["diffusiongemma"] = {"present": False, "error": str(e)}
    except Exception as e:
        report["transformers"] = {"present": False, "error": str(e)}
        report["diffusiongemma"] = {"present": False, "error": "transformers unavailable"}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def dry_run(model_id: str, out: pathlib.Path | None) -> dict:
    p = pathlib.Path(model_id)
    report = {"model": model_id, "dry_run": True, "exists": p.exists()}
    if p.exists():
        for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "model.safetensors.index.json"]:
            fp = p / name
            report[name] = {"present": fp.exists(), "bytes": fp.stat().st_size if fp.exists() else 0}
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            data = load_json(idx)
            shards = sorted(set(data.get("weight_map", {}).values()))
            report["tensors"] = len(data.get("weight_map", {}))
            report["shards"] = len(shards)
            report["missing_shards"] = [s for s in shards if not (p / s).exists()]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_messages(args) -> list[dict]:
    if args.messages_json and args.messages_file:
        raise ValueError("use only one of --messages-json or --messages-file")
    if args.messages_file:
        return json.loads(args.messages_file.read_text(encoding="utf-8"))
    if args.messages_json:
        return json.loads(args.messages_json)
    return [{"role": "user", "content": args.prompt}]


def run_reference(args) -> dict:
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model,
        dtype="auto",
        device_map=args.device_map,
    )
    message = load_messages(args)
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        max_denoising_steps=args.max_denoising_steps,
    )
    text = processor.decode(output[0], skip_special_tokens=False)
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "messages": message,
        "max_new_tokens": args.max_new_tokens,
        "max_denoising_steps": args.max_denoising_steps,
        "input_ids": inputs["input_ids"][0].detach().cpu().tolist() if "input_ids" in inputs else None,
        "output_ids": output[0].detach().cpu().tolist(),
        "decoded": text,
        "seconds": time.time() - started,
        "torch_dtype": str(getattr(model, "dtype", "")),
        "device": str(model.device),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it", help="HF repo ID or local model directory")
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--messages-json", default="", help="JSON array of chat messages; overrides --prompt")
    ap.add_argument("--messages-file", type=pathlib.Path, help="path to JSON array of chat messages; overrides --prompt")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-denoising-steps", type=int, default=48)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true", help="inspect local metadata/index only; do not import/load Transformers")
    ap.add_argument("--check-env", action="store_true", help="check Python/PyTorch/Transformers DiffusionGemma availability without loading weights")
    args = ap.parse_args()
    result = check_env(args.out) if args.check_env else (dry_run(args.model, args.out) if args.dry_run else run_reference(args))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
