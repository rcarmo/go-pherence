#!/usr/bin/env python3
"""Print a compact summary from DiffusionGemma download plan JSON."""
import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=pathlib.Path)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    print(f"repo={plan.get('repo')}")
    print(f"out={plan.get('out')}")
    print(f"shards={plan.get('shards')} parameters={plan.get('total_parameters')}")
    print(f"total_size_bytes={plan.get('total_size_bytes')} total_size_gib={plan.get('total_size_gib'):.2f}")
    print(f"target_free_bytes={plan.get('target_free_bytes')} target_free_gib={plan.get('target_free_gib'):.2f} enough_space={plan.get('enough_space')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
