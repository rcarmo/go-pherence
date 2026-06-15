#!/usr/bin/env python3
"""Build a GGUF expert prewarm plan from DiffusionGemma active-trace logs.

Input lines come from GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_ACTIVE_TRACE_TOP=N
under -dispatch-progress, for example:

  gguf_expert_active_trace: layer=1 active=60 work=736 ... top=88:80!,35:75!,0:62!

The output is suitable for:

  GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_PLAN="$(...script...)"

By default the first trace row per layer is used, which corresponds to the
prompt encoder pass when the log contains both encoder and denoise-forward
traces. Use --include-repeated-layers to keep later repeated layer rows too.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TRACE_RE = re.compile(r"gguf_expert_active_trace:\s+layer=(\d+)\b.*?\btop=([^\s]+)")
TOP_RE = re.compile(r"(\d+):(\d+)(!)?")


def parse_layer_set(text: str) -> set[int] | None:
    text = (text or "").strip()
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def extract_plan(path: Path, top: int, q8_layers: set[int] | None, include_repeated_layers: bool, order: str = "layer") -> list[tuple[int, list[int]]]:
    seen_layers: set[int] = set()
    plan: list[tuple[int, list[int]]] = []
    flat: list[tuple[int, int, int, int]] = []  # (-work, layer, rank, expert)
    for line in path.read_text(errors="ignore").splitlines():
        m = TRACE_RE.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        if q8_layers is not None and layer not in q8_layers:
            continue
        if not include_repeated_layers and layer in seen_layers:
            continue
        seen_layers.add(layer)
        experts: list[int] = []
        seen_experts: set[int] = set()
        for rank, entry in enumerate(m.group(2).split(",")):
            mm = TOP_RE.fullmatch(entry.strip())
            if not mm:
                continue
            expert = int(mm.group(1))
            work = int(mm.group(2))
            if expert in seen_experts:
                continue
            seen_experts.add(expert)
            experts.append(expert)
            flat.append((-work, layer, rank, expert))
            if len(experts) >= top:
                break
        if experts:
            plan.append((layer, experts))
    if order == "global-work":
        flat.sort()
        return [(layer, [expert]) for _, layer, _, expert in flat]
    return plan


def format_plan(plan: list[tuple[int, list[int]]]) -> str:
    return ";".join(f"{layer}:{','.join(str(e) for e in experts)}" for layer, experts in plan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="dispatch-progress log containing gguf_expert_active_trace lines")
    ap.add_argument("--top", type=int, default=6, help="experts per layer to emit (default: 6)")
    ap.add_argument("--q8-layers", default="", help="optional comma/range list of pointer-compatible Q8_0 down layers, e.g. 0-2,5,8,11")
    ap.add_argument("--include-repeated-layers", action="store_true", help="include repeated layer traces instead of only the first row per layer")
    ap.add_argument("--order", choices=("layer", "global-work"), default="layer", help="emit layer-major groups or globally sort layer/expert entries by observed work (default: layer)")
    args = ap.parse_args(argv)

    if args.top <= 0:
        ap.error("--top must be positive")
    q8_layers = parse_layer_set(args.q8_layers)
    plan = extract_plan(args.log, args.top, q8_layers, args.include_repeated_layers, args.order)
    sys.stdout.write(format_plan(plan))
    if plan:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
