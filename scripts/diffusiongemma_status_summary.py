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
    shards = data.get("shards") or {}
    readiness = data.get("readiness") or {}
    ops = data.get("operation_status") or []
    missing = caps.get("missing_for_reference") or []
    print(f"model={data.get('model_path')}")
    print(f"text_scaffold={caps.get('text_only_scaffold_ready')} reference_complete={caps.get('reference_complete')} runtime_ready={caps.get('runtime_ready')}")
    print(f"ops={caps.get('implemented_ops')}/{caps.get('total_ops')} reference_ops={caps.get('reference_complete_ops')}/{caps.get('total_ops')} op_entries={len(ops)}")
    print(f"text_ready={readiness.get('text_ready')} vision_inventory={readiness.get('vision_inventory_ready')}")
    print(f"shards_ready={shards.get('ready')} present={shards.get('present_shards')}/{shards.get('expected_shards')}")
    if missing:
        print("missing_reference=" + ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
