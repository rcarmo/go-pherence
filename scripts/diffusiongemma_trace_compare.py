#!/usr/bin/env python3
"""Compare phase-aligned DiffusionGemma row traces from Go and llama.cpp.

The script expects:
  * Go trace lines from GO_PHERENCE_DIFFUSIONGEMMA_LAYER_TRACE_OPS=1 with
    `step=<N> enc_seq=<P>` labels.
  * llama.cpp trace lines produced by the temporary/diagnostic row-summary
    callback format: `DiffusionGemma llama_row_summary: tensor=<name>-<layer> ...`.

It reports per-layer/op RMS and max-abs deltas, then exits non-zero if any
matched boundary exceeds the selected tolerances. It deliberately compares
structural graph boundaries (attn_out, ffn_mlp, ffn_moe, ffn_moe_combined,
ffn_post_norm, l_out) rather than refreshing divergence fixtures.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LLAMA_RE = re.compile(r"tensor=([A-Za-z_]+)-(\d+).*rms=([0-9.eE+-]+).*max_abs=([0-9.eE+-]+)")
GO_RE = re.compile(r"stage=op/([A-Za-z_]+) layer=(\d+).*rms=([0-9.eE+-]+).*max_abs=([0-9.eE+-]+)")

GO_OP_MAP = {
    "post_attention_norm": "attn_out",
    "dense_mlp_out": "ffn_mlp",
    "moe_out": "ffn_moe",
    "ffn_combined": "ffn_moe_combined",
    "ffn_post_norm": "ffn_post_norm",
    "layer_scalar": "l_out",
}

DEFAULT_OPS = ["attn_out", "ffn_mlp", "ffn_moe", "ffn_moe_combined", "ffn_post_norm", "l_out"]


def parse_llama(path: Path, phase: str | None) -> dict[tuple[int, str], tuple[float, float]]:
    out: dict[tuple[int, str], tuple[float, float]] = {}
    active = phase is None
    for line in path.read_text(errors="ignore").splitlines():
        if phase is not None and "DiffusionGemma llama_phase:" in line:
            active = phase in line
            continue
        if not active or "DiffusionGemma llama_row_summary:" not in line:
            continue
        m = LLAMA_RE.search(line)
        if not m:
            continue
        op, layer, rms, max_abs = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        out[(layer, op)] = (rms, max_abs)
    return out


def parse_go(path: Path, step: int | None, enc_seq: int | None) -> dict[tuple[int, str], tuple[float, float]]:
    out: dict[tuple[int, str], tuple[float, float]] = {}
    step_s = f"step={step}" if step is not None else None
    enc_s = f"enc_seq={enc_seq}" if enc_seq is not None else None
    for line in path.read_text(errors="ignore").splitlines():
        if "DiffusionGemma row_trace:" not in line:
            continue
        if step_s is not None and step_s not in line:
            continue
        if enc_s is not None and enc_s not in line:
            continue
        m = GO_RE.search(line)
        if not m:
            continue
        go_op = m.group(1)
        op = GO_OP_MAP.get(go_op)
        if op is None:
            continue
        layer, rms, max_abs = int(m.group(2)), float(m.group(3)), float(m.group(4))
        out[(layer, op)] = (rms, max_abs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama", required=True, type=Path, help="llama.cpp stderr trace")
    ap.add_argument("--go", required=True, type=Path, help="Go stderr trace")
    ap.add_argument("--llama-phase", default=None, help="substring selecting llama phase, e.g. 'decode step=48 enc_seq=17'")
    ap.add_argument("--go-step", type=int, default=None, help="Go trace step to select")
    ap.add_argument("--go-enc-seq", type=int, default=None, help="Go trace enc_seq to select")
    ap.add_argument("--rms-tol", type=float, default=0.15)
    ap.add_argument("--max-tol", type=float, default=10.0)
    ap.add_argument("--ops", default=",".join(DEFAULT_OPS), help="comma-separated op names to compare")
    args = ap.parse_args()

    ops = [x for x in args.ops.split(",") if x]
    llama = parse_llama(args.llama, args.llama_phase)
    go = parse_go(args.go, args.go_step, args.go_enc_seq)
    first_bad = None
    matched = 0
    print("layer\top\tllama_rms\tgo_rms\trms_delta\tllama_max\tgo_max\tmax_delta")
    for layer in range(1000):
        any_layer = False
        for op in ops:
            key = (layer, op)
            if key not in llama and key not in go:
                continue
            any_layer = True
            if key not in llama or key not in go:
                print(f"{layer}\t{op}\tMISSING\tllama={key in llama}\tgo={key in go}")
                if first_bad is None:
                    first_bad = (layer, op, "missing")
                continue
            matched += 1
            lr, lm = llama[key]
            gr, gm = go[key]
            dr, dm = abs(gr - lr), abs(gm - lm)
            print(f"{layer}\t{op}\t{lr:.9g}\t{gr:.9g}\t{dr:.9g}\t{lm:.9g}\t{gm:.9g}\t{dm:.9g}")
            if first_bad is None and (dr > args.rms_tol or dm > args.max_tol):
                first_bad = (layer, op, dr, dm)
        if layer > 0 and not any_layer and all((l > layer for (l, _) in set(llama) | set(go))):
            break
        if layer > 200 and not any_layer:
            break
    print(f"matched={matched}")
    if first_bad is not None:
        print(f"FIRST_MISMATCH={first_bad}")
        return 1
    print("FIRST_MISMATCH=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
