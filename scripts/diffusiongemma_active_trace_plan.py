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


def expert_cost_bytes(layer: int, q8_layers: set[int] | None, q5_layers: set[int] | None) -> int:
    # DiffusionGemma-26B-A4B GGUF expert device representation:
    # Q4_K gate/up [1408,2816]: 1408 * 11 blocks * (128 q + 8 scale f32 + 8 min f32)
    q4_gate_up = 1408 * 11 * (128 + 8 * 4 + 8 * 4)
    q8_down = 2816 * (704 + 22 * 4)
    q5_down = 2816 * (704 // 2 + 22 * (4 + 4))
    if q5_layers is not None and layer in q5_layers:
        return q4_gate_up + q5_down
    if q8_layers is not None and layer in q8_layers:
        return q4_gate_up + q8_down
    # If no layer typing was supplied, use the conservative larger estimate.
    return q4_gate_up + q8_down


def extract_plan(path: Path, top: int, q8_layers: set[int] | None, q5_layers: set[int] | None, include_repeated_layers: bool, order: str = "layer", missing_only: bool = False) -> list[tuple[int, list[int]]]:
    seen_layers: set[int] = set()
    plan: list[tuple[int, list[int]]] = []
    flat: list[tuple[float, int, int, int]] = []  # sort-key, layer, rank, expert
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
            missing = bool(mm.group(3))
            if missing_only and not missing:
                continue
            if expert in seen_experts:
                continue
            seen_experts.add(expert)
            experts.append(expert)
            if order == "efficiency":
                cost = expert_cost_bytes(layer, q8_layers, q5_layers)
                flat.append((-float(work) / float(cost), layer, rank, expert))
            else:
                flat.append((-float(work), layer, rank, expert))
            if len(experts) >= top:
                break
        if experts:
            plan.append((layer, experts))
    if order in {"global-work", "efficiency"}:
        flat.sort()
        return [(layer, [expert]) for _, layer, _, expert in flat]
    return plan


def flatten_plan(plan: list[tuple[int, list[int]]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for layer, experts in plan:
        out.extend((layer, expert) for expert in experts)
    return out


def group_flat_plan(entries: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    grouped: list[tuple[int, list[int]]] = []
    index: dict[int, int] = {}
    for layer, expert in entries:
        if layer not in index:
            index[layer] = len(grouped)
            grouped.append((layer, []))
        grouped[index[layer]][1].append(expert)
    return grouped


def apply_budget(plan: list[tuple[int, list[int]]], budget_mb: int, q8_layers: set[int] | None, q5_layers: set[int] | None) -> tuple[list[tuple[int, list[int]]], int, int]:
    if budget_mb <= 0:
        return plan, 0, len(flatten_plan(plan))
    budget = budget_mb * 1024 * 1024
    used = 0
    kept: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for layer, expert in flatten_plan(plan):
        key = (layer, expert)
        if key in seen:
            continue
        seen.add(key)
        cost = expert_cost_bytes(layer, q8_layers, q5_layers)
        if used + cost > budget:
            break
        used += cost
        kept.append(key)
    return group_flat_plan(kept), used, len(kept)


def format_plan(plan: list[tuple[int, list[int]]]) -> str:
    return ";".join(f"{layer}:{','.join(str(e) for e in experts)}" for layer, experts in plan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="dispatch-progress log containing gguf_expert_active_trace lines")
    ap.add_argument("--top", type=int, default=6, help="experts per layer to emit (default: 6)")
    ap.add_argument("--q8-layers", default="", help="optional comma/range list of Q8_0 down layers, e.g. 0-2,5,8,11")
    ap.add_argument("--q5-layers", default="", help="optional comma/range list of Q5_0 down layers for cost-aware planning")
    ap.add_argument("--include-repeated-layers", action="store_true", help="include repeated layer traces instead of only the first row per layer")
    ap.add_argument("--missing-only", action="store_true", help="only emit experts marked missing with ! in the active trace; useful for incremental plan refinement")
    ap.add_argument("--order", choices=("layer", "global-work", "efficiency"), default="layer", help="emit layer-major groups, globally sort by work, or sort by work/estimated resident byte (default: layer)")
    ap.add_argument("--budget-mb", type=int, default=0, help="truncate the emitted plan to the prefix that fits this expert-cache budget")
    ap.add_argument("--summary", action="store_true", help="print budget/entry summary to stderr")
    args = ap.parse_args(argv)

    if args.top <= 0:
        ap.error("--top must be positive")
    q8_layers = parse_layer_set(args.q8_layers)
    q5_layers = parse_layer_set(args.q5_layers)
    plan = extract_plan(args.log, args.top, q8_layers, q5_layers, args.include_repeated_layers, args.order, args.missing_only)
    original_entries = len(flatten_plan(plan))
    used = 0
    kept_entries = original_entries
    if args.budget_mb > 0:
        plan, used, kept_entries = apply_budget(plan, args.budget_mb, q8_layers, q5_layers)
    if args.summary:
        if args.budget_mb > 0:
            print(f"entries={kept_entries}/{original_entries} used_mib={used / (1024 * 1024):.1f}/{args.budget_mb}", file=sys.stderr)
        else:
            print(f"entries={original_entries}", file=sys.stderr)
    sys.stdout.write(format_plan(plan))
    if plan:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
