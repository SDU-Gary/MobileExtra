#!/usr/bin/env python3
"""Export *raw forward-only* ONNX graph.

This is used for fair forward-only benchmarking. No pre/post-processing is
included in the graph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.common.artifacts import default_artifacts_dir  # noqa: E402
from tools.common.model_loading import load_model_from_ckpt, load_model_from_pth  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Export raw (forward-only) model to ONNX")
    ap.add_argument("--model", required=True, help="ckpt or pth path")
    ap.add_argument("--from-ckpt", action="store_true")
    ap.add_argument("--network-type", default="v2")
    ap.add_argument("--base-channels", type=int, default=24)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--artifacts-dir", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else default_artifacts_dir(ROOT)
    export_dir = artifacts_dir / "exports" / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (export_dir / f"{args.network_type}_{args.height}x{args.width}_raw.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_ckpt:
        model = load_model_from_ckpt(args.model, args.network_type, args.base_channels, device)
    else:
        model = load_model_from_pth(args.model, args.network_type, args.base_channels, device)

    dummy = torch.randn(1, 7, args.height, args.width, device=device)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=int(args.opset),
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    print(f"✅ Exported raw ONNX: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

