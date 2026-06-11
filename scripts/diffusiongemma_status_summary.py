#!/usr/bin/env python3
"""Print a compact summary from diffusiongemmainspect -json output."""
import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("status", type=pathlib.Path)
    args = ap.parse_args()
    data = json.loads(args.status.read_text())
    caps = data.get("capabilities", {})
    summary = data.get("summary") or {}
    shards = data.get("shards") or {}
    readiness = data.get("readiness") or {}
    tensors = data.get("tensors") or {}
    ops = data.get("operation_status") or []
    missing = caps.get("missing_for_reference") or []
    print(f"model={data.get('model_path')}")
    if summary:
        print(f"summary_text_scaffold={summary.get('text_scaffold_ready')} summary_shards={summary.get('shards_ready')} summary_reference_complete={summary.get('reference_complete')} summary_runtime_ready={summary.get('runtime_ready')}")
        if summary.get("missing"):
            print("summary_missing=" + ",".join(summary.get("missing")))
    print(f"text_scaffold={caps.get('text_only_scaffold_ready')} text_sparse={caps.get('text_full_stack_sparse_ready')} sparse_topk_lm={caps.get('sparse_topk_lm_head')} reference_complete={caps.get('reference_complete')} runtime_ready={caps.get('runtime_ready')}")
    print(f"ops={caps.get('implemented_ops')}/{caps.get('total_ops')} reference_ops={caps.get('reference_complete_ops')}/{caps.get('total_ops')} op_entries={len(ops)}")
    print(f"text_ready={readiness.get('text_ready')} vision_inventory={readiness.get('vision_inventory_ready')}")
    size_bytes = tensors.get('total_size_bytes') or 0
    size_gib = size_bytes / (1024 ** 3) if size_bytes else 0
    print(f"parameters={tensors.get('total_parameters')} size_bytes={size_bytes} size_gib={size_gib:.2f}")
    print(f"shards_ready={shards.get('ready')} present={shards.get('present_shards')}/{shards.get('expected_shards')} percent={shards.get('present_percent')} bytes={shards.get('present_bytes')}/{shards.get('expected_bytes')} byte_percent={shards.get('present_byte_percent')}")
    if missing:
        print("missing_reference=" + ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
