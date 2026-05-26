#!/usr/bin/env python3
"""Lightweight O-Voxel metadata inspector scaffold.

This tool intentionally inspects container metadata only. It does not run
O-Voxel conversion, rendering, postprocess, or mesh/GLB export kernels.
Supported lightweight paths:
- .npz: list contained .npy arrays with dtype/shape by parsing headers
- .npy: parse dtype/shape header
- other files: report size, extension, and first-byte signature
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import zipfile
from pathlib import Path
from typing import Any, BinaryIO


def parse_npy_header(f: BinaryIO) -> dict[str, Any]:
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not an npy stream")
    major, minor = f.read(2)
    if major == 1:
        header_len = struct.unpack("<H", f.read(2))[0]
    elif major in (2, 3):
        header_len = struct.unpack("<I", f.read(4))[0]
    else:
        raise ValueError(f"unsupported npy version {major}.{minor}")
    header = f.read(header_len).decode("latin1").strip()
    meta = ast.literal_eval(header)
    return {
        "format": "npy",
        "version": [major, minor],
        "dtype": meta.get("descr"),
        "fortran_order": bool(meta.get("fortran_order")),
        "shape": list(meta.get("shape") or []),
    }


def inspect_npz(path: Path) -> dict[str, Any]:
    arrays = []
    with zipfile.ZipFile(path) as zf:
        for info in sorted(zf.infolist(), key=lambda x: x.filename):
            entry: dict[str, Any] = {
                "name": info.filename,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
            }
            if info.filename.endswith(".npy"):
                with zf.open(info) as f:
                    entry.update(parse_npy_header(f))
            arrays.append(entry)
    return {"kind": "npz", "arrays": arrays}


def inspect_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".npz":
        report.update(inspect_npz(path))
    elif path.suffix.lower() == ".npy":
        with path.open("rb") as f:
            report.update(parse_npy_header(f))
        report["kind"] = "npy"
    else:
        with path.open("rb") as f:
            sig = f.read(32)
        report.update({"kind": "unknown", "signature_hex": sig.hex()})
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="O-Voxel related files to inspect (.npz/.npy supported for shape metadata)")
    ap.add_argument("--out", default="/workspace/tmp/trellis2-ovoxel-inspect.json")
    args = ap.parse_args()

    reports = []
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            reports.append({"path": raw, "error": "not found"})
            continue
        try:
            reports.append(inspect_file(path))
        except Exception as exc:  # keep inspector best-effort
            reports.append({"path": raw, "error": str(exc)})

    output = {
        "schema": "go-pherence-trellis2-ovoxel-inspect-v1",
        "note": "metadata-only inspector; no O-Voxel conversion/render/postprocess runtime support",
        "files": reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(reports)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
