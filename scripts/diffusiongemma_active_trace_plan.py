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


def allowed_layers(q8_layers: set[int] | None, q5_layers: set[int] | None) -> set[int] | None:
    if q8_layers is None and q5_layers is None:
        return None
    out: set[int] = set()
    if q8_layers is not None:
        out.update(q8_layers)
    if q5_layers is not None:
        out.update(q5_layers)
    return out


def trace_work_map(path: Path, q8_layers: set[int] | None, q5_layers: set[int] | None, include_repeated_layers: bool, missing_only: bool = False) -> dict[tuple[int, int], int]:
    layer_filter = allowed_layers(q8_layers, q5_layers)
    seen_layers: set[int] = set()
    work: dict[tuple[int, int], int] = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = TRACE_RE.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        if layer_filter is not None and layer not in layer_filter:
            continue
        if not include_repeated_layers and layer in seen_layers:
            continue
        seen_layers.add(layer)
        for entry in m.group(2).split(","):
            mm = TOP_RE.fullmatch(entry.strip())
            if not mm:
                continue
            expert = int(mm.group(1))
            missing = bool(mm.group(3))
            if missing_only and not missing:
                continue
            work[(layer, expert)] = max(work.get((layer, expert), 0), int(mm.group(2)))
    return work


def extract_plan(path: Path, top: int, q8_layers: set[int] | None, q5_layers: set[int] | None, include_repeated_layers: bool, order: str = "layer", missing_only: bool = False, ensure_layer_coverage: bool = False) -> list[tuple[int, list[int]]]:
    layer_filter = allowed_layers(q8_layers, q5_layers)
    seen_layers: set[int] = set()
    plan: list[tuple[int, list[int]]] = []
    flat: list[tuple[float, int, int, int]] = []  # sort-key, layer, rank, expert
    layer_first: list[tuple[int, int]] = []
    for line in path.read_text(errors="ignore").splitlines():
        m = TRACE_RE.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        if layer_filter is not None and layer not in layer_filter:
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
            layer_first.append((layer, experts[0]))
    if order in {"global-work", "efficiency"}:
        flat.sort()
        if ensure_layer_coverage:
            emitted = set(layer_first)
            ordered = list(layer_first)
            ordered.extend((layer, expert) for _, layer, _, expert in flat if (layer, expert) not in emitted)
            return [(layer, [expert]) for layer, expert in ordered]
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


def plan_work(plan: list[tuple[int, list[int]]], work: dict[tuple[int, int], int]) -> int:
    return sum(work.get((layer, expert), 0) for layer, expert in flatten_plan(plan))


def optimize_budget(plan: list[tuple[int, list[int]]], budget_mb: int, q8_layers: set[int] | None, q5_layers: set[int] | None, work: dict[tuple[int, int], int], ensure_layer_coverage: bool = False) -> tuple[list[tuple[int, list[int]]], int, int]:
    if budget_mb <= 0:
        return plan, 0, len(flatten_plan(plan))
    full_budget = budget_mb * 1024 * 1024
    mandatory: list[tuple[int, int, int, int]] = []
    mandatory_layers: set[int] = set()
    if ensure_layer_coverage:
        for layer, expert in flatten_plan(plan):
            if layer in mandatory_layers:
                continue
            key = (layer, expert)
            bytes_ = expert_cost_bytes(layer, q8_layers, q5_layers)
            mandatory.append((layer, expert, work.get(key, 0), bytes_))
            mandatory_layers.add(layer)
    mandatory_used = sum(x[3] for x in mandatory)
    if mandatory_used > full_budget:
        return group_flat_plan([(layer, expert) for layer, expert, _, _ in mandatory if False]), 0, 0
    candidates: list[tuple[int, int, int, int]] = []  # layer, expert, work, bytes
    seen: set[tuple[int, int]] = {(layer, expert) for layer, expert, _, _ in mandatory}
    for layer, expert in flatten_plan(plan):
        key = (layer, expert)
        if key in seen:
            continue
        seen.add(key)
        bytes_ = expert_cost_bytes(layer, q8_layers, q5_layers)
        candidates.append((layer, expert, work.get(key, 0), bytes_))
    # Exact 0/1 knapsack over traced work values. Total traced work is small
    # (tens of thousands), so minimizing bytes for each achievable work is cheap
    # and avoids coarse byte-bucket artifacts.
    budget = full_budget - mandatory_used
    max_work = sum(max(0, c[2]) for c in candidates)
    inf = 1 << 62
    dp = [inf] * (max_work + 1)
    dp[0] = 0
    parent: list[dict[int, int]] = []
    for i, (_, _, w, bytes_) in enumerate(candidates):
        changes: dict[int, int] = {}
        if w <= 0:
            parent.append(changes)
            continue
        for cur in range(max_work - w, -1, -1):
            if dp[cur] == inf:
                continue
            new_bytes = dp[cur] + bytes_
            if new_bytes < dp[cur+w]:
                dp[cur+w] = new_bytes
                changes[cur+w] = cur
        parent.append(changes)
    best_work = max(w for w, bytes_ in enumerate(dp) if bytes_ <= budget)
    selected: list[tuple[int, int, int, int]] = []
    cur = best_work
    for i in range(len(candidates) - 1, -1, -1):
        prev = parent[i].get(cur)
        if prev is not None:
            layer, expert, w, bytes_ = candidates[i]
            selected.append((layer, expert, w, bytes_))
            cur = prev
    selected.reverse()
    selected = mandatory + selected
    used = sum(x[3] for x in selected)
    selected.sort(key=lambda x: (-x[2], x[0], x[1]))
    return group_flat_plan([(layer, expert) for layer, expert, _, _ in selected]), used, len(selected)


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
    ap.add_argument("--ensure-layer-coverage", action="store_true", help="for global/efficiency ordering, emit each traced layer's hottest expert before the remaining ranked entries")
    ap.add_argument("--order", choices=("layer", "global-work", "efficiency"), default="layer", help="emit layer-major groups, globally sort by work, or sort by work/estimated resident byte (default: layer)")
    ap.add_argument("--budget-mb", type=int, default=0, help="truncate the emitted plan to the prefix that fits this expert-cache budget")
    ap.add_argument("--optimize-budget", action="store_true", help="with --budget-mb, choose entries that maximize traced work instead of taking a sorted prefix")
    ap.add_argument("--summary", action="store_true", help="print budget/entry summary to stderr")
    args = ap.parse_args(argv)

    if args.top <= 0:
        ap.error("--top must be positive")
    q8_layers = parse_layer_set(args.q8_layers)
    q5_layers = parse_layer_set(args.q5_layers)
    work = trace_work_map(args.log, q8_layers, q5_layers, args.include_repeated_layers, args.missing_only)
    plan = extract_plan(args.log, args.top, q8_layers, q5_layers, args.include_repeated_layers, args.order, args.missing_only, args.ensure_layer_coverage)
    original_entries = len(flatten_plan(plan))
    original_work = plan_work(plan, work)
    used = 0
    kept_entries = original_entries
    if args.budget_mb > 0:
        if args.optimize_budget:
            plan, used, kept_entries = optimize_budget(plan, args.budget_mb, q8_layers, q5_layers, work, args.ensure_layer_coverage)
        else:
            plan, used, kept_entries = apply_budget(plan, args.budget_mb, q8_layers, q5_layers)
    kept_work = plan_work(plan, work)
    if args.summary:
        if args.budget_mb > 0:
            print(f"entries={kept_entries}/{original_entries} work={kept_work}/{original_work} used_mib={used / (1024 * 1024):.1f}/{args.budget_mb}", file=sys.stderr)
        else:
            print(f"entries={original_entries} work={original_work}", file=sys.stderr)
    sys.stdout.write(format_plan(plan))
    if plan:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
