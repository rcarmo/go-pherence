#!/usr/bin/env python3
"""Opt-in independent Qwen3 final-hidden-state fixture, local pinned weights only.

Runs Qwen3Model (no vocabulary projection). Uses CPU float32 eager attention as
an oracle for Go's float32 arithmetic; downloaded tensors remain BF16 on disk.
No dataset evaluation or model training is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--repository", required=True, choices=["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B"])
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--layer-trace", default="", help="Optional new directory of long-case F32 layer input/output diagnostics")
    args = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    out = Path(args.output)
    if out.exists():
        raise ValueError("output must not exist")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3Model.from_pretrained(args.model, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    loaded = time.monotonic()
    texts = [
        "Yes", "No", "Lisboa não é Madrid. 東京 is in Japan.",
        "Task: Select the supported statement. Evidence: Ada lives in Lisbon, not Paris.",
        "The evidence says the red box contains a key. " * 40,
    ]
    fixtures = []
    trace = Path(args.layer_trace) if args.layer_trace else None
    if trace:
        if trace.exists():
            raise ValueError("layer trace directory must not exist")
        trace.mkdir(parents=True)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 512:
            raise ValueError("reference text exceeds fixed 512-token contract")
        begun = time.monotonic()
        hooks = []
        if trace and text == texts[-1]:
            def save_layer(index):
                def save(module, inputs, kwargs, output):
                    inp = inputs[0] if inputs else kwargs["hidden_states"]
                    outp = output[0] if isinstance(output, tuple) else output
                    inp.detach().float().cpu().numpy().tofile(trace / f"layer-{index:02d}-input.f32")
                    outp.detach().float().cpu().numpy().tofile(trace / f"layer-{index:02d}-output.f32")
                return save
            for index, layer in enumerate(model.layers):
                hooks.append(layer.register_forward_hook(save_layer(index), with_kwargs=True))
            (trace / "config.json").write_text(json.dumps({"tokens": ids, "width": model.config.hidden_size, "layers": len(model.layers), "dtype": "little-endian-f32"}))
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([ids]), use_cache=False, return_dict=True)
            hidden = result.last_hidden_state[0].float().cpu()
        for hook in hooks:
            hook.remove()
        if not torch.isfinite(hidden).all():
            raise ValueError("nonfinite reference")
        fixtures.append({"text": text, "tokens": ids, "hidden": hidden.tolist(),
            "elapsed_seconds": time.monotonic()-begun})
    record = {"version":1,"repository":args.repository,"revision":args.revision,
        "contract":"causal/final-rmsnorm/all-token-rows/no-bos-no-eos/no-logits",
        "precision":"CPU float32 arithmetic from BF16 source weights", "attention":"eager",
        "torch":torch.__version__,"transformers":transformers.__version__,"threads":args.threads,
        "width":model.config.hidden_size,"load_seconds":loaded-start,
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds":time.monotonic()-start,"fixtures":fixtures}
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+".part")
    with tmp.open("w") as f:
        json.dump(record,f,separators=(",",":"),allow_nan=False)
        f.flush();os.fsync(f.fileno())
    tmp.rename(out)
    print(json.dumps({k:v for k,v in record.items() if k!="fixtures"}),flush=True)
    print(json.dumps({"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"tokens":[len(f["tokens"])for f in fixtures]}),flush=True)


if __name__ == "__main__":
    main()
